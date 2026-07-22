from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

from arc_companion.recovery_units import (
    PIPELINE_LEDGER_LANES,
    PIPELINE_LANE_REGISTRY,
    RECOVERY_UNIT_REGISTRY,
    call_model_with_recovery_descriptor,
    recovery_unit_for_ledger,
)
from arc_companion.callsite_inventory import (
    EXPLICIT_NONRECOVERABLE_EXEMPTIONS,
    assert_paid_call_inventory_complete,
    inventory_paid_calls,
)
from arc_companion.ledger import initialize_lane_ledger
import arc_companion.pipeline as pipeline
import pytest
from arc_companion.glossary import validate_glossary_acceptance_checkpoint
from arc_companion.prompts import CUT_SCHEMA, GLOSSARY_SCHEMA
from arc_companion.segmentation import validate_segmentation_acceptance_checkpoint


def test_descriptor_wrapper_fails_closed_for_runtime_callback_without_keyword(
    tmp_path: Path,
) -> None:
    def legacy_callback(_prompt, _schema, _artifact_dir, _call_label):
        return {"ok": True}

    with pytest.raises(TypeError, match="must accept the recovery_descriptor"):
        call_model_with_recovery_descriptor(
            legacy_callback,
            "prompt",
            {"type": "object"},
            tmp_path / "llm",
            "paid-call",
            {"unit": "segmentation"},
        )


def test_segmentation_acceptance_replays_full_window_invariants(tmp_path: Path) -> None:
    window = {
        "window_id": "w-0001",
        "start_ordinal": 1,
        "end_ordinal": 3,
        "owned_blocks": [{"ordinal": value} for value in (1, 2, 3)],
    }
    value = {
        "schema_version": "arc.companion.segmentation.v6",
        "accepted": True,
        "response": {"cut_after_ordinals": [1]},
        "window_sha256": pipeline.sha256_json(window),
        "window": window,
        "total_blocks": 3,
        "refinement": False,
        "validated_cuts": [1],
    }
    path = tmp_path / "attempt.json"
    pipeline.write_json(path, value)
    receipt = {
        "input_sha256": value["window_sha256"],
        "schema": CUT_SCHEMA,
    }

    assert validate_segmentation_acceptance_checkpoint(value)
    assert pipeline._pipeline_acceptance_checkpoint_valid(
        "segmentation", path, receipt, checkpoint_dir=tmp_path,
    )

    for mutation in (
        {**value, "validated_cuts": [2]},
        {**value, "response": {"cut_after_ordinals": [3]}},
        {key: item for key, item in value.items() if key != "window"},
    ):
        assert not validate_segmentation_acceptance_checkpoint(mutation)


def test_glossary_acceptance_replays_normalization_and_name_invariants(
    tmp_path: Path,
) -> None:
    entry = {
        "source_term": "Poisson bracket",
        "target_term": "泊松括号（Poisson）",
        "brief_explanation": "A canonical bracket operation.",
        "aliases": [],
        "protected_names": ["Poisson"],
        "first_block_id": "b1",
    }
    value = {
        "input_sha256": "a" * 64,
        "result": {"entries": [entry]},
        "business_validation": {
            "kind": "window",
            "blocks": [{
                "block_id": "b1", "type": "text",
                "text": "The Poisson bracket is bilinear.",
            }],
            "language": "zh-CN",
            "protected_names": ["Poisson"],
            "require_exact_source": True,
            "entry_limit": 50,
        },
    }
    path = tmp_path / "glossary-window.json"
    pipeline.write_json(path, value)
    receipt = {"input_sha256": "a" * 64, "schema": GLOSSARY_SCHEMA}

    assert validate_glossary_acceptance_checkpoint(value)
    assert pipeline._pipeline_acceptance_checkpoint_valid(
        "glossary", path, receipt, checkpoint_dir=tmp_path,
    )

    missing_name = deepcopy(value)
    missing_name["result"]["entries"][0]["target_term"] = "泊松括号"
    duplicate = deepcopy(value)
    duplicate["result"]["entries"].append(deepcopy(entry))
    unknown_source = deepcopy(value)
    unknown_source["result"]["entries"][0]["source_term"] = "Unknown term"
    empty_consolidation = deepcopy(value)
    empty_consolidation["result"]["entries"] = []
    empty_consolidation["business_validation"].update({
        "kind": "consolidation", "input_entry_count": 1,
    })
    for mutation in (
        missing_name, duplicate, unknown_source, empty_consolidation,
    ):
        assert not validate_glossary_acceptance_checkpoint(mutation)


def test_recovery_registry_covers_every_pipeline_structured_unit() -> None:
    package_root = Path(pipeline.__file__).parent
    descriptor_units: set[str] = set()
    non_pipeline_descriptor_counts: dict[str, int] = {}
    non_pipeline_helper_counts: dict[str, int] = {}
    for source_path in package_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        helper_calls = 0
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "call_model_with_recovery_descriptor"
            ):
                helper_calls += 1
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "submission_descriptor"
            ):
                continue
            keyword = next((item for item in node.keywords if item.arg == "unit"), None)
            assert keyword is not None, f"{source_path.name} has an unowned descriptor"
            assert isinstance(keyword.value, ast.Constant) and isinstance(
                keyword.value.value, str,
            ), f"{source_path.name} has a non-literal recovery unit"
            descriptor_units.add(keyword.value.value)
            if source_path.name != "pipeline.py":
                non_pipeline_descriptor_counts[source_path.name] = (
                    non_pipeline_descriptor_counts.get(source_path.name, 0) + 1
                )
        if source_path.name != "pipeline.py":
            non_pipeline_helper_counts[source_path.name] = helper_calls
    assert all(
        non_pipeline_helper_counts[name] >= count
        for name, count in non_pipeline_descriptor_counts.items()
    )

    pipeline_tree = ast.parse(Path(pipeline.__file__).read_text(encoding="utf-8"))
    stateful_receipts = [
        node for node in ast.walk(pipeline_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_ledger_submission_receipt"
    ]
    receipt_units = [
        next(item.value for item in node.keywords if item.arg == "recovery_unit")
        for node in stateful_receipts
    ]
    literal_receipt_units = {
        item.value for item in receipt_units
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }
    dynamic_receipt_units = {
        item.attr for item in receipt_units
        if isinstance(item, ast.Attribute)
        and isinstance(item.value, ast.Name)
        and item.value.id == "lane_binding"
    }
    assert "guide" in literal_receipt_units
    assert "recovery_unit" in dynamic_receipt_units
    run_lane_values = {
        node.args[2].value
        for node in ast.walk(pipeline_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_lane"
        and len(node.args) >= 3
        and isinstance(node.args[2], ast.Constant)
    }
    assert run_lane_values | literal_receipt_units == PIPELINE_LEDGER_LANES

    paid_sites = inventory_paid_calls(package_root)
    assert paid_sites
    assert_paid_call_inventory_complete(paid_sites)
    assert EXPLICIT_NONRECOVERABLE_EXEMPTIONS == {}
    # Stateful chapter adapters write their production receipts dynamically by
    # real ledger lane. Every other structured producer must declare a literal
    # descriptor at its actual submission site.
    assert descriptor_units | set(PIPELINE_LEDGER_LANES) == set(
        RECOVERY_UNIT_REGISTRY
    )
    assert {
        spec.ledger_lane for spec in RECOVERY_UNIT_REGISTRY.values()
        if spec.ledger_lane is not None
    } == PIPELINE_LEDGER_LANES
    for lane in PIPELINE_LEDGER_LANES:
        spec = recovery_unit_for_ledger(lane)
        assert spec is not None
        assert spec.owner and spec.validator
        assert spec.application
        assert spec.side_effect_policy == "no_unproven_external_side_effects"
        binding = PIPELINE_LANE_REGISTRY[lane]
        assert binding.public_lane == lane
        assert binding.recovery_unit == lane
        assert binding.validator == spec.validator
        assert binding.application == spec.application


def test_every_pipeline_llm_call_has_the_central_descriptor_hook() -> None:
    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_llm_call"
    ]
    assert calls
    assert all(
        any(keyword.arg == "recovery_descriptor" for keyword in node.keywords)
        for node in calls
    )


def test_paid_repairs_have_distinct_schema_and_application_handlers() -> None:
    token = RECOVERY_UNIT_REGISTRY["translation-token-repair"]
    coverage = RECOVERY_UNIT_REGISTRY["translation-coverage-repair"]
    translation = RECOVERY_UNIT_REGISTRY["translation"]

    assert token.validator != coverage.validator
    assert token.application != coverage.application
    assert token.validator != translation.validator
    assert coverage.validator != translation.validator
    assert token.ledger_lane is None
    assert coverage.ledger_lane is None


def test_repair_acceptance_requires_validated_marker_normalization_and_final_binding(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    segment_id = "seg-1"
    input_sha = "a" * 64
    raw_response = {"repairs": [{
        "block_id": "b1",
        "slots": [{"slot_id": "slot-1", "start_offset": 0, "end_offset": 1}],
    }]}
    normalization_path = checkpoint / "normalization.json"
    normalization = {
        "schema_version": "arc.companion.response-normalization-receipt.v1",
        "normalizer_version": "arc.companion.response-normalizer.v1",
        "validator_version": pipeline.TRANSLATION_REPAIR_NORMALIZATION_VALIDATOR_VERSION,
        "decision": "accepted",
        "expected_ids": ["b1"],
        "original_response_sha256": pipeline.sha256_json(raw_response),
    }
    pipeline.write_json(normalization_path, normalization)
    normalization_reference = {
        "path": normalization_path.relative_to(checkpoint).as_posix(),
        "sha256": pipeline.sha256_file(normalization_path),
    }
    translation = {"blocks": [{"block_id": "b1", "text": "译"}]}
    final_path = checkpoint / "translations" / "seg-1.json"
    pipeline.write_json(final_path, {
        "schema_version": "arc.companion.translation-checkpoint.v2",
        "segment_id": segment_id,
        "generation": 1,
        "input_sha256": input_sha,
        "generation_provenance": {"repairs": [{
            "kind": "token-placement",
            "repaired_block_ids": ["b1"],
            "response_normalization": normalization_reference,
        }]},
        "translation": translation,
    })
    marker_path = pipeline._translation_token_attempt_path(
        checkpoint, segment_id, 1,
    )
    marker = {
        "schema_version": "arc.companion.translation-token-attempt.v2",
        "status": "validated",
        "segment_id": segment_id,
        "generation": 1,
        "input_sha256": input_sha,
        "prompt_version": pipeline.TRANSLATION_RETRY_PROMPT_VERSION,
        "response_schema_version": pipeline.TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
        "block_ids": ["b1"],
        "raw_response": raw_response,
        "validated_translation_sha256": pipeline.sha256_json(translation),
        "response_normalization": normalization_reference,
        "final_translation_checkpoint": {
            "path": final_path.relative_to(checkpoint).as_posix(),
            "sha256": pipeline.sha256_file(final_path),
        },
    }
    pipeline.write_json(marker_path, marker)
    receipt = {
        "recovery_unit": "translation-token-repair",
        "logical_unit": f"{segment_id}:token-repair",
        "generation": 1,
        "input_sha256": input_sha,
    }

    assert pipeline._pipeline_acceptance_checkpoint_valid(
        "translation-token-repair",
        final_path,
        receipt,
        checkpoint_dir=checkpoint,
    ) is True

    pipeline.write_json(marker_path, {**marker, "status": "response_received"})
    assert pipeline._pipeline_acceptance_checkpoint_valid(
        "translation-token-repair",
        final_path,
        receipt,
        checkpoint_dir=checkpoint,
    ) is False

    pipeline.write_json(marker_path, marker)
    external_normalization = tmp_path / "external-normalization.json"
    pipeline.write_json(external_normalization, normalization)
    normalization_path.unlink()
    normalization_path.symlink_to(external_normalization)
    assert pipeline._pipeline_acceptance_checkpoint_valid(
        "translation-token-repair",
        final_path,
        receipt,
        checkpoint_dir=checkpoint,
    ) is False

    normalization_path.unlink()
    pipeline.write_json(normalization_path, normalization)
    external_marker = tmp_path / "external-marker.json"
    pipeline.write_json(external_marker, marker)
    marker_path.unlink()
    marker_path.symlink_to(external_marker)
    assert pipeline._pipeline_acceptance_checkpoint_valid(
        "translation-token-repair",
        final_path,
        receipt,
        checkpoint_dir=checkpoint,
    ) is False

def test_automatic_restart_uses_structural_owner_and_fails_closed(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    guide = checkpoint / "chapters" / "ch-0001" / "guide-ledger.json"
    ledger = initialize_lane_ledger(
        guide, chapter_id="ch-0001", lane="guide", segment_ids=["guide"],
    )
    entry = {
        "ledger_path": str(guide),
        "session_key": "ch-0001:guide",
        "segment_id": "guide",
        "recovery_context": {"failure_category": "timeout"},
    }

    assert pipeline._automatic_restart_blocker(entry, ledger, checkpoint) is None
    assert "ledger is unavailable" in pipeline._automatic_restart_blocker({
        "session_key": "run:segmentation",
        "segment_id": "window-2",
        "recovery_unit": "segmentation",
    })


def test_caller_validated_batch_accepts_every_logical_unit(
    tmp_path, monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    receipts = []
    ledger_paths = []
    for index in range(2):
        logical = f"section-review-{index}"
        ledger_path = (
            checkpoint / "recovery-controls" / "section-review"
            / f"{index}-ledger.json"
        )
        pipeline.initialize_lane_ledger(
            ledger_path, chapter_id=f"review-{index}",
            lane="section-review", segment_ids=[logical],
        )
        session_key = f"review-{index}:section-review"
        idempotency_key = f"{session_key}:{logical}:generation-1"
        pipeline._guarded_mark_transport_state(
            ledger_path,
            checkpoint_dir=checkpoint,
            session_key=session_key,
            logical_unit=logical,
            idempotency_key=idempotency_key,
            response_received=True,
        )
        acceptance = checkpoint / "section-reviews" / f"{index:04d}.json"
        pipeline.write_json(acceptance, {"validated": True})
        receipts.append((checkpoint / f"receipt-{index}.json", {
            "sealed": True, "recovery_unit": "section-review",
            "logical_unit": logical,
            "session_key": session_key,
            "generation": 1,
            "idempotency_key": idempotency_key,
            "ledger_path": ledger_path.relative_to(checkpoint).as_posix(),
            "acceptance_checkpoint": acceptance.relative_to(checkpoint).as_posix(),
            "input_sha256": f"{index:064x}",
        }))
        ledger_paths.append(ledger_path)
    monkeypatch.setattr(
        pipeline, "discover_submission_receipts", lambda _root: receipts,
    )
    receipt_by_path = {str(path): value for path, value in receipts}

    def receipt_reference(path, *, checkpoint_dir):
        return {
            "path": str(path),
            "sha256": "a" * 64,
            "identity_sha256": "b" * 64,
        }

    def validate_reference(reference, **kwargs):
        receipt = dict(receipt_by_path[str(reference["path"])])
        _ledger, digest = pipeline.read_registered_lane_ledger(
            checkpoint, kwargs["ledger_path"],
        )
        receipt["current_registered_ledger_sha256"] = digest
        return receipt

    monkeypatch.setattr(pipeline, "submission_receipt_reference", receipt_reference)
    monkeypatch.setattr(
        pipeline, "_validate_pipeline_submission_reference", validate_reference,
    )

    accepted = pipeline._accept_completed_pipeline_controls(
        checkpoint,
        caller_validated_units=frozenset({"section-review"}),
    )

    assert accepted == 2
    assert all(
        pipeline.read_json(path)["blocks"][0]["state"] == "accepted"
        for path in ledger_paths
    )


def test_ordered_control_ledger_preserves_prefix_and_invalidates_exact_suffix(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    group = "a" * 64
    first = pipeline.submission_descriptor(
        unit="title-translation",
        logical_unit="title-a",
        checkpoint_dir=checkpoint,
        artifact_root=checkpoint / "title-a",
        acceptance_checkpoint=checkpoint / "title-a.json",
        input_sha256="1" * 64,
        group_sha256=group,
        ordered_siblings=["title-a", "title-b", "title-c"],
        suffix=["title-a", "title-b", "title-c"],
    )
    control = pipeline._prepare_pipeline_recovery_control(
        first, artifact_dir=checkpoint / "title-a",
    )
    ledger_path = Path(control["ledger_path"])
    for state in ("submitted", "response_received", "schema_valid", "invariant_valid"):
        pipeline.advance_block(ledger_path, segment_id="title-a", state=state)
    pipeline.advance_block(
        ledger_path,
        segment_id="title-a",
        state="accepted",
        input_sha256="1" * 64,
        output_sha256="2" * 64,
    )

    changed = pipeline.submission_descriptor(
        unit="title-translation",
        logical_unit="title-d",
        checkpoint_dir=checkpoint,
        artifact_root=checkpoint / "title-d",
        acceptance_checkpoint=checkpoint / "title-d.json",
        input_sha256="3" * 64,
        group_sha256=group,
        ordered_siblings=["title-a", "title-d", "title-e"],
        suffix=["title-d", "title-e"],
    )
    rebound = pipeline._prepare_pipeline_recovery_control(
        changed, artifact_dir=checkpoint / "title-d",
    )
    ledger = pipeline.read_json(Path(rebound["ledger_path"]))

    assert rebound["ledger_path"] == control["ledger_path"]
    assert rebound["ordered_siblings"] == ["title-a", "title-d", "title-e"]
    assert rebound["suffix"] == ["title-d", "title-e"]
    assert ledger["generation"] == 2
    assert [item["segment_id"] for item in ledger["blocks"]] == [
        "title-a", "title-d", "title-e",
    ]
    assert [item["state"] for item in ledger["blocks"]] == [
        "accepted", "prepared", "prepared",
    ]
    assert {item["generation"] for item in ledger["blocks"]} == {2}


def test_index_glossary_acceptance_recomputes_business_identity(tmp_path) -> None:
    import hashlib

    checkpoint = tmp_path / "checkpoint"
    expected_ids = ["entry-a", "entry-b"]
    logical_unit = (
        "index-glossary-batch-0001-"
        + pipeline.sha256_json(expected_ids)[:16]
    )
    prompt_sha256 = hashlib.sha256(b"prompt").hexdigest()
    schema_sha256 = pipeline.sha256_json({"type": "object"})
    source_sha256 = "a" * 64
    identity = {
        "source_sha256": source_sha256,
        "language": "zh-CN",
        "expected_entry_ids": expected_ids,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_sha256 = pipeline.sha256_json(identity)
    path = checkpoint / "index-glossary-batches" / f"{logical_unit}.json"
    response = {"entries": [
        {"entry_id": entry_id, "target": "译", "explanation": "释"}
        for entry_id in expected_ids
    ]}
    value = {
        "schema_version": pipeline.INDEX_GLOSSARY_BATCH_VERSION,
        "logical_unit": logical_unit,
        "input_sha256": input_sha256,
        **identity,
        "response": response,
    }
    receipt = {
        "logical_unit": logical_unit,
        "input_sha256": input_sha256,
        "group_sha256": source_sha256,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
        "schema": {
            "type": "object",
            "required": ["entries"],
            "properties": {"entries": {"type": "array"}},
        },
    }
    pipeline.write_json(path, value)
    assert pipeline._pipeline_acceptance_checkpoint_valid(
        "glossary-index", path, receipt, checkpoint_dir=checkpoint,
    )

    for field, replacement in (
        ("expected_entry_ids", ["entry-a", "forged"]),
        ("source_sha256", "b" * 64),
        ("prompt_sha256", "c" * 64),
        ("schema_sha256", "d" * 64),
    ):
        pipeline.write_json(path, {**value, field: replacement})
        assert not pipeline._pipeline_acceptance_checkpoint_valid(
            "glossary-index", path, receipt, checkpoint_dir=checkpoint,
        )
