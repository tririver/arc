from __future__ import annotations

import json

import pytest

from arc_companion.intent_guidance import (
    IntentGuidanceAmbiguousError,
    IntentGuidanceError,
    build_intent_guidance,
    resolve_worker_evidence_requests,
    validate_worker_query,
    worker_guidance_payload,
    worker_guidance_prompt_prefix,
    worker_policy_descriptor,
    _controller_page,
)
from arc_llm import EvidenceRequest


def _getters(*, body: str = "SECRET SECTION BODY", source_id: str = "book-cache"):
    def parsed(requested: str, *, include_document: bool):
        assert requested == source_id
        assert include_document is False
        return {
            "ok": True,
            "data": {
                "paper_id": source_id,
                "source_hash": "source-hash-v1",
                "document_hash": "document-hash-v1",
                "metadata": {
                    "title": "Reference Translation",
                    "authors": ["A. Translator"],
                    "year": 2024,
                    "abstract": "SECRET ABSTRACT",
                    "unsafe": "SECRET METADATA",
                },
                "sections": [{"section_id": "ch-2", "text": body}],
                "document": {"blocks": [{"text": body}]},
            },
        }

    def toc(requested: str):
        assert requested == source_id
        return {"ok": True, "data": [
            {"id": "ch-1", "title": "Foundations", "level": 1, "text": "TOC BODY SECRET"},
            {"id": "ch-2", "title": "Renormalization", "level": 1},
        ]}

    return parsed, toc


def _resolved(_prompt, _schema, _path, _label):
    return {
        "guidance": "Use the reference chapter's established Chinese terminology.",
        "resolution_status": "resolved",
        "reference_targets": [{
            "source_id": "book-cache",
            "locator": "ch-2",
            "purpose": "Terminology and idiomatic phrasing",
            "lanes": ["glossary", "translation", "review"],
        }],
    }


def _build(tmp_path, call_model=_resolved, **overrides):
    parsed, toc = _getters()
    arguments = {
        "source_language": "English",
        "target_language": "Chinese",
        "document_type": "book",
        "context_paper_ids": ["book-cache"],
        "project_dir": tmp_path,
        "call_model": call_model,
        "parsed_getter": parsed,
        "toc_getter": toc,
    }
    arguments.update(overrides)
    return build_intent_guidance(
        "Follow the terminology of chapter Renormalization in the cached translation.",
        **arguments,
    )


def test_one_model_call_is_reused_and_reference_hash_changes_identity(tmp_path):
    calls = []

    def model(*args):
        calls.append(args)
        return _resolved(*args)

    first = _build(tmp_path, model)
    second = _build(tmp_path, model)
    assert first == second
    assert len(calls) == 1
    artifacts = list((tmp_path / ".arc-companion" / "intent-guidance").glob("*/artifact.json"))
    assert len(artifacts) == 1
    assert len(first["user_intent_sha256"]) == 64

    parsed, toc = _getters()

    def changed_hash(source_id, *, include_document):
        result = parsed(source_id, include_document=include_document)
        result["data"]["document_hash"] = "document-hash-v2"
        return result

    _build(tmp_path, model, parsed_getter=changed_hash, toc_getter=toc)
    assert len(calls) == 2
    assert len(list((tmp_path / ".arc-companion" / "intent-guidance").glob("*/artifact.json"))) == 2


def test_empty_intent_skips_getters_and_model(tmp_path):
    def fail(*_args, **_kwargs):
        raise AssertionError("empty intent must be a zero-call path")

    assert build_intent_guidance(
        "  ", source_language="English", target_language="Chinese",
        document_type="book", context_paper_ids=["missing"], project_dir=tmp_path,
        call_model=fail, parsed_getter=fail, toc_getter=fail,
    ) is None


def test_invalid_cache_id_stops_before_model(tmp_path):
    def missing(source_id, **_kwargs):
        return {"ok": False, "error": {"message": f"No parsed source found for {source_id}"}}

    with pytest.raises(IntentGuidanceError, match="local ARC cache"):
        _build(tmp_path, lambda *_: pytest.fail("model must not run"), parsed_getter=missing)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_id", "other-cache", "unauthorized source_id"),
        ("locator", "ch-missing", "locator is missing or non-unique"),
    ],
)
def test_invalid_model_reference_source_or_locator_is_rejected(tmp_path, field, value, message):
    def invalid(*_args):
        result = _resolved(*_args)
        result["reference_targets"][0][field] = value
        return result

    with pytest.raises(IntentGuidanceError, match=message):
        _build(tmp_path, invalid)
    assert not list((tmp_path / ".arc-companion" / "intent-guidance").glob("*/artifact.json"))


def test_ambiguous_resolution_is_persisted_then_stops_without_second_call(tmp_path):
    calls = []

    def ambiguous(*_args):
        calls.append(1)
        return {
            "guidance": "The requested chapter is not uniquely identifiable.",
            "resolution_status": "ambiguous",
        }

    with pytest.raises(IntentGuidanceAmbiguousError):
        _build(tmp_path, ambiguous)
    with pytest.raises(IntentGuidanceAmbiguousError):
        _build(tmp_path, ambiguous)
    assert calls == [1]
    artifact_path = next((tmp_path / ".arc-companion" / "intent-guidance").glob("*/artifact.json"))
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["resolution_status"] == "ambiguous"


def test_all_workers_receive_identical_payload_and_strict_read_only_policy(tmp_path):
    artifact = _build(tmp_path)
    payloads = [worker_guidance_payload(artifact) for _lane in (
        "glossary", "title_translation", "guide", "translation", "commentary", "review"
    )]
    assert all(payload == payloads[0] for payload in payloads)
    payloads[0]["guidance"] = "mutated"
    assert worker_guidance_payload(artifact)["guidance"] != "mutated"
    assert worker_guidance_prompt_prefix(artifact) == worker_guidance_prompt_prefix(artifact)

    policy = worker_policy_descriptor(artifact)
    assert policy["allowed_operations"] == [
        "artifact-read", "get-parsed-toc", "get-parsed-section"
    ]
    assert policy["reference_operations"] == ["get-parsed-toc", "get-parsed-section"]
    assert policy["network"] is False
    assert validate_worker_query(
        artifact, operation="get-parsed-section", source_id="book-cache", locator="ch-2"
    )["locator"] == "ch-2"
    with pytest.raises(IntentGuidanceError, match="not read-only or allowed"):
        validate_worker_query(artifact, operation="parse", source_id="book-cache")
    with pytest.raises(IntentGuidanceError, match="exact guidance target"):
        validate_worker_query(
            artifact, operation="get-parsed-section", source_id="book-cache", locator="ch-1"
        )

    assert worker_policy_descriptor(artifact, lane="translation")[
        "authorized_section_targets"
    ] == [{"source_id": "book-cache", "locator": "ch-2"}]
    assert worker_policy_descriptor(artifact, lane="commentary")[
        "authorized_section_targets"
    ] == []
    with pytest.raises(IntentGuidanceError, match="not authorized"):
        validate_worker_query(
            artifact, operation="get-parsed-section", source_id="book-cache",
            locator="ch-2", lane="commentary",
        )


def test_oversized_guidance_is_rejected(tmp_path):
    def oversized(*_args):
        result = _resolved(*_args)
        result["guidance"] = "术" * 8001
        return result

    with pytest.raises(IntentGuidanceError, match="oversized"):
        _build(tmp_path, oversized)


def test_generation_prompt_contains_only_sanitized_metadata_and_compact_toc(tmp_path):
    observed = {}

    def inspect(prompt, schema, _path, _label):
        observed["prompt"] = prompt
        observed["schema"] = schema
        return _resolved(prompt, schema, _path, _label)

    _build(tmp_path, inspect)
    prompt = observed["prompt"]
    assert "Reference Translation" in prompt
    assert "Renormalization" in prompt
    assert "source-hash-v1" in prompt
    assert "SECRET SECTION BODY" not in prompt
    assert "TOC BODY SECRET" not in prompt
    assert "SECRET ABSTRACT" not in prompt
    assert "SECRET METADATA" not in prompt
    assert observed["schema"]["properties"]["resolution_status"]["enum"] == [
        "resolved", "ambiguous"
    ]


def test_controller_fallback_enforces_targets_and_pages_cached_section(tmp_path):
    artifact = _build(tmp_path)
    requests = (
        EvidenceRequest(
            "ok", "get-parsed-section",
            {"source_id": "book-cache", "locator": "ch-2", "limit": 12},
        ),
        EvidenceRequest(
            "bad", "get-parsed-section",
            {"source_id": "book-cache", "locator": "ch-1"},
        ),
    )
    responses = resolve_worker_evidence_requests(
        artifact, requests, round_number=1,
        toc_getter=lambda _source: {"ok": True, "data": []},
        section_getter=lambda _source, _section: {
            "ok": True, "data": {"text": "术语" * 100},
        },
    )
    assert responses[0].ok is True
    assert responses[0].data["eof"] is False
    assert responses[0].data["next_offset"] <= 12
    assert responses[0].provenance["provider"] == "local-cache"
    assert responses[1].ok is False
    assert "exact guidance target" in responses[1].error

    tiny = resolve_worker_evidence_requests(
        artifact,
        (EvidenceRequest(
            "tiny", "get-parsed-section",
            {"source_id": "book-cache", "locator": "ch-2", "limit": 1},
        ),),
        round_number=2,
        toc_getter=lambda _source: {"ok": True, "data": []},
        section_getter=lambda _source, _section: {
            "ok": True, "data": {"text": "术语"},
        },
    )[0]
    assert tiny.ok is True
    assert tiny.data["next_offset"] > tiny.data["offset"]
    tiny.data["content"].encode("utf-8").decode("utf-8")
    multibyte = _controller_page("术", offset=1, limit=1)
    assert multibyte["content"] == "术"
    assert multibyte["next_offset"] == 4
