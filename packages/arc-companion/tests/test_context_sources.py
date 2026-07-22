from __future__ import annotations

from pathlib import Path

import pytest

from arc_companion import cli
from arc_companion.context_sources import ContextSourceError, load_context_evidence
from arc_companion.evidence import validate_annotation_citations, validate_evidence_record
from arc_companion.pipeline import (
    CONTEXT_SEGMENT_CHARS_PER_SOURCE,
    CONTEXT_SEGMENT_CHARS_TOTAL,
    BuildOptions,
    _evidence_for_segment,
)
from arc_companion.prompts import annotation_prompt
from arc_companion.reader_text import clean_reader_annotation


def _parsed_getter(paper_id: str, *, include_document: bool) -> dict:
    if include_document:
        return {"ok": False, "error": {"message": "no rich cache"}}
    return {
        "ok": True,
        "data": {
            "paper_id": paper_id,
            "source_hash": "a" * 64,
            "title": f"Title {paper_id}",
            "sections": [
                {"section_id": "s1", "title": "Free field", "text": "canonical field momentum"},
                {"section_id": "s2", "title": "Scattering", "text": "amplitude cross section"},
            ],
            "equations": [
                {
                    "id": "eq1",
                    "before": "canonical relation",
                    "normalized_latex": r"[\phi(x),\pi(y)]=i\delta(x-y)",
                    "after": "equal time",
                }
            ],
        },
    }


def test_context_sources_use_only_local_cache_reader_and_fall_back_to_compact_cache() -> None:
    calls: list[tuple[str, bool]] = []

    def getter(paper_id: str, *, include_document: bool) -> dict:
        calls.append((paper_id, include_document))
        return _parsed_getter(paper_id, include_document=include_document)

    records = load_context_evidence(["isbn:one", "local:two"], parsed_getter=getter)

    assert calls == [
        ("isbn:one", True), ("isbn:one", False),
        ("local:two", True), ("local:two", False),
    ]
    assert len({record["evidence_id"] for record in records}) == 2
    assert all(record["relation"] == "context" for record in records)
    assert all(record["context_role"].endswith("only") for record in records)
    assert all(validate_evidence_record(record) is record for record in records)
    assert {piece["block_id"] for piece in records[0]["blocks"]} == {"s1", "s2", "eq1"}
    assert records[0]["context_index"]["version"].endswith("v3")
    assert next(piece for piece in records[0]["blocks"] if piece["block_id"] == "s1")["section_title"] == "Free field"


def test_rich_context_blocks_inherit_nearest_heading_title() -> None:
    def getter(paper_id: str, *, include_document: bool) -> dict:
        return {"ok": True, "data": {
            "paper_id": paper_id,
            "title": "Reference Book",
            "document": {"blocks": [
                {"block_id": "h1", "kind": "heading", "heading_level": 1, "text": "2 Free Fields"},
                {"block_id": "p1", "kind": "prose", "text": "canonical field momentum"},
                {"block_id": "h2", "kind": "heading", "heading_level": 2, "text": "2.1 Scattering"},
                {"block_id": "p2", "kind": "prose", "text": "amplitude cross section"},
            ]},
        }}

    record = load_context_evidence(["isbn:rich"], parsed_getter=getter)[0]

    by_id = {piece["block_id"]: piece for piece in record["blocks"]}
    assert by_id["h1"]["section_title"] == "2 Free Fields"
    assert by_id["p1"]["section_title"] == "2 Free Fields"
    assert by_id["h2"]["section_title"] == "2.1 Scattering"
    assert by_id["p2"]["section_title"] == "2.1 Scattering"


def test_missing_explicit_context_source_fails_instead_of_fetching() -> None:
    calls = 0

    def missing(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"ok": False, "error": {"message": "not cached"}}

    with pytest.raises(ContextSourceError, match="local arc-paper cache"):
        load_context_evidence(["isbn:missing"], parsed_getter=missing)
    assert calls == 2


def test_context_selection_is_relevant_bounded_and_kept_out_of_chronology() -> None:
    records = load_context_evidence(
        [f"isbn:{index}" for index in range(5)], parsed_getter=_parsed_getter
    )
    segment = {"segment_id": "seg", "block_ids": ["b"]}
    selected = _evidence_for_segment(
        segment,
        {"b": {"block_id": "b", "text": "canonical field momentum"}},
        {"related_papers": records},
    )["papers"]

    assert len(selected) == 5
    assert all(item["relation"] == "context" for item in selected)
    assert all(item["snippets"][0]["block_id"] == "s1" for item in selected)
    assert all(item["snippets"][0]["section_title"] == "Free field" for item in selected)
    assert all(sum(len(piece["text"]) for piece in item["snippets"]) <= CONTEXT_SEGMENT_CHARS_PER_SOURCE for item in selected)
    assert sum(
        len(piece["text"]) for item in selected for piece in item["snippets"]
    ) <= CONTEXT_SEGMENT_CHARS_TOTAL
    assert all(validate_evidence_record(item) is item for item in selected)
    assert all(item["context_selection"]["version"].endswith("v2") for item in selected)

    cleaned = clean_reader_annotation(
        {
            "commentary": f"解释【{selected[0]['evidence_id']}】",
            "explanation": "解释",
            "prior_work": "",
            "later_work": "",
            "evidence_ids": [selected[0]["evidence_id"]],
        },
        evidence_records=[selected[0]],
        language="zh-CN",
    )
    assert cleaned["commentary"] == "解释（参考：《Title isbn:0》，Free field）"

    annotation = {
        "explanation": "Title isbn:0 clarifies the canonical relation.",
        "commentary": "Conceptual context.",
        "prior_work": "",
        "later_work": "",
        "evidence_ids": [selected[0]["evidence_id"]],
    }
    assert validate_annotation_citations(annotation, selected) == annotation["evidence_ids"]
    bad = {**annotation, "prior_work": "This established priority."}
    with pytest.raises(ValueError, match="registered prior evidence"):
        validate_annotation_citations(bad, selected)


def test_context_evidence_is_selected_once_without_request_protocol_fields() -> None:
    record = load_context_evidence(["isbn:one"], parsed_getter=_parsed_getter)[0]
    segment = {"segment_id": "seg", "block_ids": ["b"]}

    selected = _evidence_for_segment(
        segment,
        {"b": {"block_id": "b", "text": "canonical field momentum"}},
        {"related_papers": [record]},
    )["papers"]

    assert [item["evidence_id"] for item in selected] == [record["evidence_id"]]
    assert "supported_request_keys" not in selected[0]


def test_context_index_keeps_late_relevant_blocks_until_segment_selection() -> None:
    def getter(paper_id: str, *, include_document: bool) -> dict:
        blocks = [
            {"block_id": f"b-{index:04d}", "text": "generic introductory material"}
            for index in range(800)
        ]
        blocks.append({"block_id": "late-hit", "text": "Schwinger Dyson spectral kernel"})
        return {"ok": True, "data": {
            "paper_id": paper_id,
            "source_hash": "b" * 64,
            "document": {"blocks": blocks},
        }}

    records = load_context_evidence(["isbn:long"], parsed_getter=getter)
    assert records[0]["context_index"]["block_count"] == 801

    selected = _evidence_for_segment(
        {"segment_id": "seg", "block_ids": ["source"]},
        {"source": {"block_id": "source", "text": "spectral kernel"}},
        {"related_papers": records},
    )["papers"]
    assert selected[0]["snippets"][0]["block_id"] == "late-hit"
    assert sum(len(item["text"]) for item in selected[0]["snippets"]) <= 3_000


def test_cli_accepts_repeatable_context_paper_ids(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_build(options):
        captured["options"] = options
        return {"ok": True, "data": {"status": "complete"}, "meta": {}}

    monkeypatch.setattr(cli, "build_companion", fake_build)
    code = cli.main([
        "build", "local:seed", "--project-dir", str(tmp_path),
        "--context-paper-id", "isbn:one",
        "--context-paper-id", "isbn:two",
        "--user-intent", "Use chapter two terminology exactly.",
        "--json",
    ])

    assert code == 0
    assert captured["options"].context_paper_ids == ("isbn:one", "isbn:two")
    assert captured["options"].user_intent == "Use chapter two terminology exactly."


def test_build_options_reject_duplicate_or_self_context(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        BuildOptions(
            paper_id="local:seed", project_dir=tmp_path,
            context_paper_ids=("isbn:one", "isbn:one"),
        )
    with pytest.raises(ValueError, match="cannot also"):
        BuildOptions(
            paper_id="local:seed", project_dir=tmp_path,
            context_paper_ids=("local:seed",),
        )


def test_prompt_limits_context_to_explanation() -> None:
    prompt = annotation_prompt(
        {"segment_id": "s", "block_ids": ["b"]},
        [{"block_id": "b", "text": "source"}],
        language="zh-CN",
        metadata={},
        evidence={"papers": [{"relation": "context"}]},
        glossary={},
        protected_names=[],
        paper_context={},
    )
    assert "BOUNDED SOURCES" in prompt
    assert "direct HTTP(S) URL" in prompt
    assert "never expose hashes, internal IDs, or controller labels" in prompt
    assert "same turn" in prompt
