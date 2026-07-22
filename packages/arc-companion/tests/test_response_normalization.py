from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from arc_companion.io import sha256_file, sha256_json
from arc_companion import pipeline as pipeline_module
from arc_companion.pipeline import BuildOptions, SourceBundle
from arc_companion.response_normalization import (
    NORMALIZER_VERSION,
    RECEIPT_SCHEMA_VERSION,
    ResponseNormalizationError,
    normalize_complete_response,
    normalize_complete_response_with_receipt,
)


SCHEMA = {
    "type": "object",
    "required": ["repairs", "meta"],
    "properties": {
        "repairs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["block_id", "value"],
                "properties": {
                    "block_id": {"type": "string", "minLength": 1},
                    "value": {},
                },
                "additionalProperties": False,
            },
        },
        "meta": {"type": "string"},
    },
    "additionalProperties": False,
}


def _normalize(value, expected=("a", "b"), *, invariant=lambda value: True):
    return normalize_complete_response(
        value,
        "repairs",
        expected,
        lambda item: item.get("block_id"),
        lambda candidate: jsonschema.validate(candidate, SCHEMA),
        invariant,
        "test.validator.v1",
    )


def test_exact_and_reordered_responses_preserve_top_level_fields() -> None:
    exact = {
        "meta": "keep",
        "repairs": [
            {"block_id": "a", "value": 1},
            {"block_id": "b", "value": 2},
        ],
    }
    projected, receipt = _normalize(exact)
    assert projected == exact
    assert receipt["reason_code"] == "exact_expected_ids"
    assert receipt["schema_version"] == RECEIPT_SCHEMA_VERSION
    assert receipt["normalizer_version"] == NORMALIZER_VERSION

    reordered = {"meta": "keep", "repairs": list(reversed(exact["repairs"]))}
    projected, receipt = _normalize(reordered)
    assert projected == exact
    assert receipt["reason_code"] == "reordered_expected_ids"
    assert receipt["returned_ids"] == ["b", "a"]


def test_unknown_and_duplicate_extras_are_discarded_in_response_order() -> None:
    value = {
        "meta": "keep",
        "repairs": [
            {"block_id": "x", "value": "first-secret"},
            {"block_id": "b", "value": 2},
            {"block_id": "x", "value": "second-secret"},
            {"block_id": "a", "value": 1},
        ],
    }
    projected, receipt = _normalize(value)
    assert projected["repairs"] == [
        {"block_id": "a", "value": 1},
        {"block_id": "b", "value": 2},
    ]
    assert receipt["reason_code"] == "projected_unknown_ids"
    assert receipt["discarded_ids"] == ["x", "x"]
    assert receipt["discarded_item_sha256s"] == [
        sha256_json(value["repairs"][0]),
        sha256_json(value["repairs"][2]),
    ]
    assert "first-secret" not in json.dumps(receipt)
    assert "second-secret" not in json.dumps(receipt)


def test_canonical_duplicate_expected_items_collapse_but_conflicts_reject() -> None:
    duplicate = {"block_id": "a", "value": {"flag": True}}
    projected, receipt = _normalize({
        "meta": "keep",
        "repairs": [duplicate, {"value": {"flag": True}, "block_id": "a"},
                    {"block_id": "b", "value": 2}],
    })
    assert projected["repairs"][0] == duplicate
    assert receipt["collapsed_ids"] == ["a"]
    assert receipt["reason_code"] == "collapsed_duplicate_ids"

    with pytest.raises(ResponseNormalizationError) as raised:
        _normalize({
            "meta": "keep",
            "repairs": [
                {"block_id": "a", "value": True},
                {"block_id": "a", "value": 1},
                {"block_id": "b", "value": 2},
            ],
        })
    assert raised.value.code == "conflicting_duplicate_ids"
    assert raised.value.receipt["conflicting_ids"] == ["a"]


@pytest.mark.parametrize(
    ("value", "expected", "code"),
    [
        ({"meta": "x", "repairs": [{"block_id": "a", "value": 1}]},
         ("a", "b"), "missing_expected_ids"),
        ({"meta": "x", "repairs": "not-a-list"}, ("a",), "invalid_collection"),
        ({"meta": "x", "repairs": ["not-an-object"]}, ("a",),
         "invalid_returned_item"),
        ({"meta": "x", "repairs": [{"block_id": "", "value": 1}]}, ("a",),
         "invalid_returned_id"),
        ({"meta": "x", "repairs": [{"block_id": True, "value": 1}]}, ("a",),
         "invalid_returned_id"),
        ({"meta": "x", "repairs": []}, (), "invalid_expected_ids"),
        ({"meta": "x", "repairs": []}, ("a", "a"), "invalid_expected_ids"),
        ({"meta": "x", "repairs": []}, "ab", "invalid_expected_ids"),
    ],
)
def test_invalid_ids_items_collections_and_missing_ids_reject(
    value, expected, code,
) -> None:
    with pytest.raises(ResponseNormalizationError) as raised:
        _normalize(value, expected)
    assert raised.value.code == code
    assert raised.value.receipt["decision"] == "rejected"


def test_schema_and_domain_invariants_are_both_mandatory() -> None:
    with pytest.raises(ResponseNormalizationError) as schema_failure:
        _normalize({
            "meta": 3,
            "repairs": [
                {"block_id": "a", "value": 1},
                {"block_id": "b", "value": 2},
            ],
        })
    assert schema_failure.value.code == "schema_validation_failed"
    assert schema_failure.value.receipt["schema_valid"] is False
    assert schema_failure.value.receipt["invariant_valid"] is None

    with pytest.raises(ResponseNormalizationError) as invariant_failure:
        _normalize({
            "meta": "keep",
            "repairs": [
                {"block_id": "a", "value": 1},
                {"block_id": "b", "value": 2},
            ],
        }, invariant=lambda value: False)
    assert invariant_failure.value.code == "invariant_validation_failed"
    assert invariant_failure.value.receipt["schema_valid"] is True
    assert invariant_failure.value.receipt["invariant_valid"] is False


def test_atomic_receipt_replay_matches_and_tamper_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "attempt" / "response-normalization.json"
    value = {
        "meta": "keep",
        "repairs": [
            {"block_id": "extra", "value": "not-in-receipt"},
            {"block_id": "a", "value": 1},
            {"block_id": "b", "value": 2},
        ],
    }
    arguments = (
        value,
        "repairs",
        ("a", "b"),
        lambda item: item.get("block_id"),
        lambda candidate: jsonschema.validate(candidate, SCHEMA),
        lambda candidate: True,
        "test.validator.v1",
    )
    first = normalize_complete_response_with_receipt(*arguments, receipt_path=path)
    second = normalize_complete_response_with_receipt(*arguments, receipt_path=path)
    assert first == second
    assert path.is_file()
    assert "not-in-receipt" not in path.read_text(encoding="utf-8")

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["discarded_ids"] = []
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ResponseNormalizationError) as raised:
        normalize_complete_response_with_receipt(*arguments, receipt_path=path)
    assert raised.value.code == "receipt_replay_mismatch"
    assert json.loads(path.read_text(encoding="utf-8")) == tampered


def test_rejected_receipt_replay_only_recomputes_local_validators(tmp_path: Path) -> None:
    path = tmp_path / "response-normalization.json"
    calls = 0

    def invariant(value):
        nonlocal calls
        calls += 1
        return False

    arguments = (
        {"meta": "keep", "repairs": [{"block_id": "a", "value": 1}]},
        "repairs",
        ("a",),
        lambda item: item.get("block_id"),
        lambda candidate: jsonschema.validate(candidate, SCHEMA),
        invariant,
        "test.validator.v1",
    )
    with pytest.raises(ResponseNormalizationError):
        normalize_complete_response_with_receipt(*arguments, receipt_path=path)
    with pytest.raises(ResponseNormalizationError):
        normalize_complete_response_with_receipt(*arguments, receipt_path=path)
    # Receipt replay deliberately recomputes local validation. The helper has
    # no provider/formatter callback and therefore cannot repeat paid work.
    assert calls == 2
    assert json.loads(path.read_text())["reason_code"] == "invariant_validation_failed"


def test_pipeline_receipt_paths_never_escape_checkpoint_root(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    marker = checkpoint / "attempts" / "marker.json"
    canonical_owner = checkpoint / "llm" / "attempt"
    canonical = pipeline_module._repair_response_normalization_path(
        canonical_owner,
        marker,
        checkpoint_dir=checkpoint,
        persisted_response=False,
    )
    assert canonical == canonical_owner / "response-normalization.json"

    fallback = pipeline_module._repair_response_normalization_path(
        checkpoint / "llm" / "historical-missing-attempt",
        marker,
        checkpoint_dir=checkpoint,
        persisted_response=True,
    )
    assert fallback == checkpoint / "attempts" / "marker.response-normalization.json"

    with pytest.raises(RuntimeError, match="outside the checkpoint root"):
        pipeline_module._repair_response_normalization_path(
            tmp_path / "outside" / "retry-offset-1",
            tmp_path / "outside" / "marker.json",
            checkpoint_dir=checkpoint,
            persisted_response=True,
        )

    with pytest.raises(RuntimeError, match="outside the checkpoint root"):
        pipeline_module._repair_response_normalization_path(
            tmp_path / "outside" / "new-attempt",
            marker,
            checkpoint_dir=checkpoint,
            persisted_response=False,
        )


def test_pipeline_receipt_paths_reject_symlink_ownership_escape(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    real_attempt = checkpoint / "real-attempt"
    real_attempt.mkdir()
    linked_attempt = checkpoint / "linked-attempt"
    linked_attempt.symlink_to(real_attempt, target_is_directory=True)
    marker = checkpoint / "markers" / "marker.json"

    with pytest.raises(RuntimeError, match="contains a symlink"):
        pipeline_module._repair_response_normalization_path(
            linked_attempt,
            marker,
            checkpoint_dir=checkpoint,
            persisted_response=False,
        )

    owned_attempt = checkpoint / "owned-attempt"
    owned_attempt.mkdir()
    target = checkpoint / "other-receipt.json"
    target.write_text("{}", encoding="utf-8")
    (owned_attempt / "response-normalization.json").symlink_to(target)
    with pytest.raises(RuntimeError, match="contains a symlink"):
        pipeline_module._repair_response_normalization_path(
            owned_attempt,
            marker,
            checkpoint_dir=checkpoint,
            persisted_response=False,
        )

    real_marker_parent = checkpoint / "real-markers"
    real_marker_parent.mkdir()
    linked_marker_parent = checkpoint / "linked-markers"
    linked_marker_parent.symlink_to(real_marker_parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="contains a symlink"):
        pipeline_module._repair_response_normalization_path(
            checkpoint / "llm" / "missing-historical",
            linked_marker_parent / "marker.json",
            checkpoint_dir=checkpoint,
            persisted_response=True,
        )


def _inline_run(kind: str, text: str, ordinal: int, **extra) -> dict:
    digest = hashlib.sha256(f"{kind}:{text}".encode()).hexdigest()
    return {
        "kind": kind,
        "content": text,
        "token_id": f"tok-{ordinal}",
        "content_hash": digest,
        "order": ordinal,
        **extra,
    }


def _barthes_626_block() -> dict:
    return {
        "block_id": "md-line-626",
        "type": "text",
        "text": "Before x after.",
        "inline_runs": [
            _inline_run("text", "Before ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", " after.", 3),
        ],
    }


def _current_offset_repair(block: dict) -> dict:
    slot_ids = pipeline_module._translation_repair_slot_ids(block)
    return {
        "block_id": block["block_id"],
        "slots": [
            {"slot_id": slot_ids[0], "start_offset": 0, "end_offset": 2},
            {"slot_id": slot_ids[1], "start_offset": 2, "end_offset": 3},
        ],
    }


def _persist_token_attempt(
    checkpoint: Path,
    block: dict,
    raw_response: dict,
    *,
    segment_id: str = "seg-ch38",
    input_sha256: str = "ch38-input",
) -> Path:
    marker = pipeline_module._translation_token_attempt_path(checkpoint, segment_id)
    pipeline_module.write_json(marker, {
        "schema_version": "arc.companion.translation-token-attempt.v2",
        "segment_id": segment_id,
        "generation": 1,
        "input_sha256": input_sha256,
        "prompt_version": pipeline_module.TRANSLATION_RETRY_PROMPT_VERSION,
        "response_schema_version": pipeline_module.TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
        "model_tier": pipeline_module.TRANSLATION_RETRY_TIER,
        "block_ids": [block["block_id"]],
        "status": "response_received",
        "started_at": "2026-07-01T00:00:00+00:00",
        "response_received_at": "2026-07-01T00:01:00+00:00",
        "raw_response": raw_response,
    })
    return marker


def test_ch38_shaped_current_repair_discards_621_and_replays_without_llm(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    artifact_dir = checkpoint / "llm" / "translation"
    (artifact_dir / "retry-offset-1").mkdir(parents=True)
    block = _barthes_626_block()
    extra = {
        "block_id": "md-line-621",
        "slots": [{
            "slot_id": "md-line-621.repair-slot-0000",
            "start_offset": 0,
            "end_offset": 0,
        }],
    }
    raw_response = {"repairs": [extra, _current_offset_repair(block)]}
    marker = _persist_token_attempt(checkpoint, block, raw_response)

    def forbidden_llm(prompt: str, **kwargs):
        raise AssertionError("persisted complete response must not call a provider")

    arguments = dict(
        segment={"segment_id": "seg-ch38", "block_ids": ["md-line-626"]},
        translation={"blocks": [{"block_id": "md-line-626", "text": "译文。"}]},
        blocks_by_id={"md-line-626": block},
        protected_names=[],
        options=BuildOptions(paper_id="local:barthes", project_dir=tmp_path),
        checkpoint_dir=checkpoint,
        artifact_dir=artifact_dir,
        input_sha256="ch38-input",
        llm=forbidden_llm,
    )
    first, provenance = pipeline_module._repair_translation_token_placement(**arguments)
    second, _ = pipeline_module._repair_translation_token_placement(**arguments)

    token = pipeline_module._opaque_inline_tokens(block)[0]
    assert first == second
    assert first["blocks"][0]["text"] == f"译文{token}。"
    assert provenance["response_normalization"]["path"].endswith(
        "retry-offset-1/response-normalization.json"
    )
    receipt_path = artifact_dir / "retry-offset-1" / "response-normalization.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["decision"] == "accepted"
    assert receipt["discarded_ids"] == ["md-line-621"]
    assert receipt["expected_ids"] == ["md-line-626"]
    validated_marker = json.loads(marker.read_text(encoding="utf-8"))
    assert validated_marker["status"] == "validated"
    assert validated_marker["raw_response"] == raw_response
    draft = pipeline_module._matching_translation_token_repair_draft(
        checkpoint, "seg-ch38", "ch38-input",
    )
    assert draft is not None
    assert draft["raw_response"] == raw_response
    assert draft["response_normalization"]["sha256"] == sha256_file(receipt_path)


def test_ch38_legacy_full_slot_remains_supervised_without_llm(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    block = _barthes_626_block()
    raw_response = {"repairs": [{
        "block_id": "md-line-626",
        "slots": [{
            "slot_id": "repair-md-line-626-full",
            "start_offset": 0,
            "end_offset": 3,
        }],
    }]}
    marker = _persist_token_attempt(checkpoint, block, raw_response)

    with pytest.raises(
        pipeline_module.TranslationRepairNeedsSupervision,
        match="invariant_validation_failed",
    ):
        pipeline_module._repair_translation_token_placement(
            {"segment_id": "seg-ch38", "block_ids": ["md-line-626"]},
            {"blocks": [{"block_id": "md-line-626", "text": "译文。"}]},
            blocks_by_id={"md-line-626": block},
            protected_names=[],
            options=BuildOptions(paper_id="local:barthes", project_dir=tmp_path),
            checkpoint_dir=checkpoint,
            artifact_dir=checkpoint / "llm" / "translation",
            input_sha256="ch38-input",
            llm=lambda *args, **kwargs: pytest.fail("must not call a provider"),
        )
    persisted = json.loads(marker.read_text(encoding="utf-8"))
    assert persisted["status"] == "response_received"
    assert persisted["raw_response"] == raw_response
    receipt = json.loads(marker.with_name(
        f"{marker.stem}.response-normalization.json"
    ).read_text(encoding="utf-8"))
    assert receipt["decision"] == "rejected"
    assert receipt["reason_code"] == "invariant_validation_failed"


def test_persisted_token_marker_ids_bind_current_expected_set_without_llm(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    block = _barthes_626_block()
    raw_response = {"repairs": [_current_offset_repair(block)]}
    marker = _persist_token_attempt(checkpoint, block, raw_response)
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    marker_value["block_ids"] = ["md-line-621"]
    pipeline_module.write_json(marker, marker_value)
    calls = 0

    def forbidden_llm(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("mismatched persisted marker must not call a provider")

    with pytest.raises(
        pipeline_module.TranslationRepairNeedsSupervision,
        match="owning marker",
    ):
        pipeline_module._repair_translation_token_placement(
            {"segment_id": "seg-ch38", "block_ids": ["md-line-626"]},
            {"blocks": [{"block_id": "md-line-626", "text": "译文。"}]},
            blocks_by_id={"md-line-626": block},
            protected_names=[],
            options=BuildOptions(paper_id="local:barthes", project_dir=tmp_path),
            checkpoint_dir=checkpoint,
            artifact_dir=checkpoint / "llm" / "translation",
            input_sha256="ch38-input",
            llm=forbidden_llm,
        )
    assert calls == 0


def test_persisted_token_response_is_not_reused_after_expected_set_changes(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    first = _barthes_626_block()
    second = {
        **_barthes_626_block(),
        "block_id": "md-line-627",
    }
    _persist_token_attempt(
        checkpoint, first, {"repairs": [_current_offset_repair(first)]},
    )
    with pytest.raises(
        pipeline_module.TranslationRepairNeedsSupervision,
        match="owning marker",
    ):
        pipeline_module._repair_translation_token_placement(
            {
                "segment_id": "seg-ch38",
                "block_ids": ["md-line-626", "md-line-627"],
            },
            {"blocks": [
                {"block_id": "md-line-626", "text": "甲。"},
                {"block_id": "md-line-627", "text": "乙。"},
            ]},
            blocks_by_id={"md-line-626": first, "md-line-627": second},
            protected_names=[],
            options=BuildOptions(paper_id="local:barthes", project_dir=tmp_path),
            checkpoint_dir=checkpoint,
            artifact_dir=checkpoint / "llm" / "translation",
            input_sha256="ch38-input",
            llm=lambda *args, **kwargs: pytest.fail("must not reinterpret paid response"),
        )


def test_persisted_token_draft_provenance_ids_cannot_be_reinterpreted(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    artifact_dir = checkpoint / "llm" / "translation"
    (artifact_dir / "retry-offset-1").mkdir(parents=True)
    block = _barthes_626_block()
    _persist_token_attempt(
        checkpoint, block, {"repairs": [_current_offset_repair(block)]},
    )
    arguments = dict(
        segment={"segment_id": "seg-ch38", "block_ids": ["md-line-626"]},
        translation={"blocks": [{"block_id": "md-line-626", "text": "译文。"}]},
        blocks_by_id={"md-line-626": block}, protected_names=[],
        options=BuildOptions(paper_id="local:barthes", project_dir=tmp_path),
        checkpoint_dir=checkpoint, artifact_dir=artifact_dir,
        input_sha256="ch38-input",
        llm=lambda *args, **kwargs: pytest.fail("persisted response must not call LLM"),
    )
    pipeline_module._repair_translation_token_placement(**arguments)
    draft_path = pipeline_module._translation_token_repair_draft_path(
        checkpoint, "seg-ch38",
    )
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    draft["repair_provenance"]["repaired_block_ids"] = ["md-line-621"]
    pipeline_module.write_json(draft_path, draft)

    with pytest.raises(
        pipeline_module.TranslationRepairNeedsSupervision,
        match="draft provenance",
    ):
        pipeline_module._repair_translation_token_placement(**arguments)


def test_paid_repair_finalizer_projects_complete_extra_response(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    block = _barthes_626_block()
    pipeline_module.write_json(
        checkpoint / "document.json",
        {"paper_id": "local:barthes", "document": {"blocks": [block]}},
    )
    ledger_path = checkpoint / "chapters" / "ch-38" / "translation-ledger.json"
    pipeline_module.initialize_lane_ledger(
        ledger_path,
        chapter_id="ch-38",
        lane="translation",
        segment_ids=["seg-ch38"],
    )
    pipeline_module.mark_submitted(ledger_path, segment_id="seg-ch38")
    pipeline_module.mark_response_received(ledger_path, segment_id="seg-ch38")
    draft_path = pipeline_module._translation_draft_path(checkpoint, "seg-ch38")
    pipeline_module.write_json(draft_path, {
        "schema_version": "arc.companion.translation-primary-draft.v1",
        "segment_id": "seg-ch38",
        "generation": 1,
        "input_sha256": "ch38-input",
        "translation": {
            "blocks": [{"block_id": "md-line-626", "text": "译文。"}],
        },
    })
    raw_response = {"repairs": [
        {
            "block_id": "md-line-621",
            "slots": [{
                "slot_id": "md-line-621.repair-slot-0000",
                "start_offset": 0,
                "end_offset": 0,
            }],
        },
        _current_offset_repair(block),
    ]}
    marker = _persist_token_attempt(checkpoint, block, raw_response)
    attempt_dir = (
        checkpoint
        / "llm"
        / "translations"
        / pipeline_module._segment_checkpoint_name("seg-ch38")
        / "retry-offset-1"
    )
    pipeline_module.write_json(attempt_dir / "call-checkpoints" / "call.json", {
        "submission_state": "submitted",
        "logical_identity": {"idempotency_key": "ch38-paid-repair"},
    })

    entries = pipeline_module._finalize_paid_translation_repairs(checkpoint)

    assert len(entries) == 1
    assert entries[0]["recovery_action"] == "deterministic-replay"
    assert entries[0]["blocking_reason"] == ""
    assert json.loads(marker.read_text(encoding="utf-8"))["raw_response"] == raw_response
    receipt_path = attempt_dir / "response-normalization.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["decision"] == "accepted"
    assert receipt["discarded_ids"] == ["md-line-621"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["needs_supervision"] is None


def test_persisted_coverage_response_discards_extra_and_uses_no_llm(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    block = {"block_id": "missing", "type": "text", "text": "Missing source."}
    document = {
        "front_matter": {},
        "blocks": [block],
        "equations": [],
        "figures": [],
        "tables": [],
        "bibliography": [],
        "assets": [],
        "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="local:coverage",
        parsed={"document": document},
        document=document,
        metadata={},
        references=[],
        citers=[],
    )
    segment = {"segment_id": "seg-coverage", "block_ids": ["missing"]}
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1)
    draft_path = pipeline_module._seed_translation_coverage_draft(
        segment,
        options=options,
        bundle=bundle,
        glossary={"entries": []},
        protected_names=[],
        checkpoint_dir=checkpoint,
        translation={"blocks": []},
    )
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    raw_response = {"repairs": [
        {"block_id": "md-line-621", "slots": [{
            "slot_id": "md-line-621.coverage-slot-0000", "text": "discard",
        }]},
        {"block_id": "missing", "slots": [{
            "slot_id": pipeline_module._translation_coverage_slot_ids(block)[0],
            "text": "补齐译文。",
        }]},
    ]}
    attempt_path = pipeline_module._translation_coverage_attempt_path(
        checkpoint, "seg-coverage",
    )
    pipeline_module.write_json(attempt_path, {
        "schema_version": pipeline_module.TRANSLATION_COVERAGE_ATTEMPT_SCHEMA_VERSION,
        "segment_id": "seg-coverage",
        "generation": 1,
        "input_sha256": draft["input_sha256"],
        "prompt_version": pipeline_module.TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION,
        "response_schema_version": pipeline_module.TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION,
        "model_tier": pipeline_module.TRANSLATION_COVERAGE_REPAIR_TIER,
        "missing_block_ids": ["missing"],
        "status": "response_received",
        "started_at": "2026-07-01T00:00:00+00:00",
        "response_received_at": "2026-07-01T00:01:00+00:00",
        "raw_response": raw_response,
    })
    artifact_dir = (
        checkpoint
        / "llm"
        / "translations"
        / pipeline_module._segment_checkpoint_name("seg-coverage")
    )
    (artifact_dir / "coverage-repair-1").mkdir(parents=True)

    result = pipeline_module._generate_translations(
        [segment],
        options=options,
        bundle=bundle,
        glossary={"entries": []},
        protected_names=[],
        checkpoint_dir=checkpoint,
        llm=lambda *args, **kwargs: pytest.fail("persisted response must not call LLM"),
    )
    assert result["seg-coverage"]["blocks"] == [
        {"block_id": "missing", "text": "补齐译文。"}
    ]
    marker = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert marker["status"] == "validated"
    assert marker["raw_response"] == raw_response
    receipt_path = artifact_dir / "coverage-repair-1" / "response-normalization.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["discarded_ids"] == ["md-line-621"]
    assert marker["response_normalization"]["sha256"] == sha256_file(receipt_path)


def _coverage_case(tmp_path: Path, *, segment_id: str):
    checkpoint = tmp_path / "checkpoint"
    block = {"block_id": "missing", "type": "text", "text": "Missing source."}
    document = {
        "front_matter": {}, "blocks": [block], "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [],
        "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="local:coverage-failure",
        parsed={"document": document},
        document=document,
        metadata={}, references=[], citers=[],
    )
    segment = {"segment_id": segment_id, "block_ids": ["missing"]}
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1)
    return checkpoint, block, bundle, segment, options


@pytest.mark.parametrize("failure", ["missing", "conflict", "invariant"])
def test_persisted_coverage_failures_are_supervised_without_llm(
    tmp_path: Path,
    failure: str,
) -> None:
    checkpoint, block, bundle, segment, options = _coverage_case(
        tmp_path, segment_id=f"seg-{failure}",
    )
    draft_path = pipeline_module._seed_translation_coverage_draft(
        segment,
        options=options,
        bundle=bundle,
        glossary={"entries": []},
        protected_names=[],
        checkpoint_dir=checkpoint,
        translation={"blocks": []},
    )
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    valid = {
        "block_id": "missing",
        "slots": [{
            "slot_id": pipeline_module._translation_coverage_slot_ids(block)[0],
            "text": "补齐译文。",
        }],
    }
    if failure == "missing":
        repairs = [{
            "block_id": "unrequested",
            "slots": [{"slot_id": "unrequested.slot", "text": "extra"}],
        }]
    elif failure == "conflict":
        repairs = [valid, {
            **valid,
            "slots": [{**valid["slots"][0], "text": "冲突译文。"}],
        }]
    else:
        repairs = [{
            **valid,
            "slots": [{**valid["slots"][0], "slot_id": "missing.invalid-slot"}],
        }]
    raw_response = {"repairs": repairs}
    attempt_path = pipeline_module._translation_coverage_attempt_path(
        checkpoint, segment["segment_id"],
    )
    pipeline_module.write_json(attempt_path, {
        "schema_version": pipeline_module.TRANSLATION_COVERAGE_ATTEMPT_SCHEMA_VERSION,
        "segment_id": segment["segment_id"],
        "generation": 1,
        "input_sha256": draft["input_sha256"],
        "prompt_version": pipeline_module.TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION,
        "response_schema_version": pipeline_module.TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION,
        "model_tier": pipeline_module.TRANSLATION_COVERAGE_REPAIR_TIER,
        "missing_block_ids": ["missing"],
        "status": "response_received",
        "started_at": "2026-07-01T00:00:00+00:00",
        "response_received_at": "2026-07-01T00:01:00+00:00",
        "raw_response": raw_response,
    })
    provider_calls = 0

    def forbidden_llm(*args, **kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("persisted coverage response must not call a provider")

    with pytest.raises(pipeline_module.CompanionLaneError) as raised:
        pipeline_module._generate_translations(
            [segment], options=options, bundle=bundle,
            glossary={"entries": []}, protected_names=[],
            checkpoint_dir=checkpoint, llm=forbidden_llm,
        )
    assert provider_calls == 0
    assert pipeline_module._chapter_failure_requires_supervision(raised.value)
    assert "refusing resubmission" in str(raised.value)

    ledger_path = checkpoint / "chapters" / "ch" / "translation-ledger.json"
    pipeline_module.initialize_lane_ledger(
        ledger_path,
        chapter_id="ch",
        lane="translation",
        segment_ids=[segment["segment_id"]],
    )
    pipeline_module.mark_submitted(ledger_path, segment_id=segment["segment_id"])
    pipeline_module.mark_response_received(ledger_path, segment_id=segment["segment_id"])
    assert pipeline_module._mark_translation_repair_supervision(
        ledger_path, segment_id=segment["segment_id"], exc=raised.value,
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["needs_supervision"]["segment_id"] == segment["segment_id"]
    assert ledger["needs_supervision"]["recovery_context"]["recovery_action"] == (
        "operator-supervision"
    )

    with pytest.raises(pipeline_module.CompanionLaneError):
        pipeline_module._generate_translations(
            [segment], options=options, bundle=bundle,
            glossary={"entries": []}, protected_names=[],
            checkpoint_dir=checkpoint, llm=forbidden_llm,
        )
    assert provider_calls == 0
    assert json.loads(attempt_path.read_text(encoding="utf-8"))["raw_response"] == (
        raw_response
    )


def test_new_paid_invalid_coverage_is_persisted_then_never_resubmitted(
    tmp_path: Path,
) -> None:
    checkpoint, block, bundle, segment, options = _coverage_case(
        tmp_path, segment_id="seg-new-paid",
    )
    calls: list[str] = []

    def invalid_llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        calls.append(label)
        if label.endswith("coverage-repair-1"):
            return {"repairs": [{
                "block_id": "missing",
                "slots": [{
                    "slot_id": "missing.invalid-slot",
                    "text": "invalid",
                }],
            }]}
        return {"blocks": []}

    with pytest.raises(pipeline_module.CompanionLaneError) as raised:
        pipeline_module._generate_translations(
            [segment], options=options, bundle=bundle,
            glossary={"entries": []}, protected_names=[],
            checkpoint_dir=checkpoint, llm=invalid_llm,
        )
    assert calls == [
        "companion-translation-seg-new-paid",
        "companion-translation-seg-new-paid-coverage-repair-1",
    ]
    assert pipeline_module._chapter_failure_requires_supervision(raised.value)
    attempt_path = pipeline_module._translation_coverage_attempt_path(
        checkpoint, segment["segment_id"],
    )
    marker = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert marker["status"] == "response_received"

    with pytest.raises(pipeline_module.CompanionLaneError):
        pipeline_module._generate_translations(
            [segment], options=options, bundle=bundle,
            glossary={"entries": []}, protected_names=[],
            checkpoint_dir=checkpoint,
            llm=lambda *args, **kwargs: pytest.fail("paid response must not be resent"),
        )


@pytest.mark.parametrize("change", ["marker_tamper", "current_set_change"])
def test_paid_coverage_marker_ids_bind_current_missing_set_without_provider(
    tmp_path: Path,
    change: str,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    blocks = [
        {"block_id": "kept", "type": "text", "text": "Kept source."},
        {"block_id": "missing", "type": "text", "text": "Missing source."},
    ]
    document = {
        "front_matter": {}, "blocks": blocks, "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [],
        "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="local:coverage-id-binding",
        parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    segment = {
        "segment_id": f"seg-{change}",
        "block_ids": ["kept", "missing"],
    }
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1)
    draft_path = pipeline_module._seed_translation_coverage_draft(
        segment,
        options=options,
        bundle=bundle,
        glossary={"entries": []},
        protected_names=[],
        checkpoint_dir=checkpoint,
        translation={"blocks": [{"block_id": "kept", "text": "保留译文。"}]},
    )
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    raw_response = {"repairs": [{
        "block_id": "missing",
        "slots": [{
            "slot_id": pipeline_module._translation_coverage_slot_ids(blocks[1])[0],
            "text": "补齐译文。",
        }],
    }]}
    marker_path = pipeline_module._translation_coverage_attempt_path(
        checkpoint, segment["segment_id"],
    )
    marker = {
        "schema_version": pipeline_module.TRANSLATION_COVERAGE_ATTEMPT_SCHEMA_VERSION,
        "segment_id": segment["segment_id"],
        "generation": 1,
        "input_sha256": draft["input_sha256"],
        "prompt_version": pipeline_module.TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION,
        "response_schema_version": pipeline_module.TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION,
        "model_tier": pipeline_module.TRANSLATION_COVERAGE_REPAIR_TIER,
        "missing_block_ids": ["missing"],
        "status": "response_received",
        "started_at": "2026-07-01T00:00:00+00:00",
        "response_received_at": "2026-07-01T00:01:00+00:00",
        "raw_response": raw_response,
    }
    if change == "marker_tamper":
        marker["missing_block_ids"] = ["kept"]
    else:
        draft["translation"] = {"blocks": []}
        pipeline_module.write_json(draft_path, draft)
    pipeline_module.write_json(marker_path, marker)
    provider_calls = 0

    def forbidden_llm(*args, **kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("mismatched paid coverage IDs must not call a provider")

    with pytest.raises(pipeline_module.CompanionLaneError) as raised:
        pipeline_module._generate_translations(
            [segment], options=options, bundle=bundle,
            glossary={"entries": []}, protected_names=[],
            checkpoint_dir=checkpoint, llm=forbidden_llm,
        )
    assert provider_calls == 0
    assert pipeline_module._chapter_failure_requires_supervision(raised.value)
    assert "owning marker" in str(raised.value)
    persisted = json.loads(marker_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "response_received"
    assert persisted["raw_response"] == raw_response
