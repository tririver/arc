from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from arc_companion.artifact_store import (
    AcceptedArtifactStore,
    ArtifactStoreError,
    canonical_sha256,
)
from arc_companion.artifact_ids import allocate_artifact_dir
from arc_companion.migration import (
    accepted_translation_projection_candidates,
    import_accepted_checkpoint_objects,
)
from arc_companion.regeneration import (
    REGENERATABLE_LANES,
    RegenerationRequestError,
    normalize_regeneration_lanes,
    reject_broad_force,
)
from arc_companion.reuse import (
    ReuseRequest,
    build_reuse_plan,
    lane_recipe_sha256,
    lane_semantic_sha256,
)


EMPTY_CHAIN = hashlib.sha256(b"").hexdigest()


def _accepted_block(output: object, *, input_sha: str | None = None) -> dict:
    input_sha = input_sha or canonical_sha256({"source": "one"})
    output_sha = canonical_sha256(output)
    block = {
        "segment_id": "ch-0001.seg-0001",
        "state": "accepted",
        "generation": 1,
        "input_sha256": input_sha,
        "output_sha256": output_sha,
        "predecessor_accepted_chain_sha256": EMPTY_CHAIN,
        "validation_receipt": {"local_validation": True},
        "logical_receipt": {"provider": "stub", "call_id": "call-1"},
    }
    block["accepted_chain_sha256"] = hashlib.sha256(json.dumps({
        "predecessor": EMPTY_CHAIN,
        "segment_id": block["segment_id"],
        "input_sha256": input_sha,
        "output_sha256": output_sha,
        "generation": 1,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return block


def test_object_store_accepts_only_hash_bound_accepted_output(tmp_path: Path) -> None:
    store = AcceptedArtifactStore(tmp_path)
    output = {"blocks": [{"block_id": "b1", "text": "译文"}]}
    block = _accepted_block(output)
    record = store.put_accepted(
        kind="translation",
        semantic_input_sha256=block["input_sha256"],
        recipe_sha256=canonical_sha256({"prompt": 1}),
        contract_version="translation.v1",
        output=output,
        ledger_block=block,
        provider_receipt={
            "provider": "stub", "model": "test", "call_id": "call-1", "usage": {},
        },
        provenance={"checkpoint": "old"},
    )

    assert store.read("translation", record["artifact_id"])["output"] == output
    assert record["predecessor_accepted_chain_sha256"] == EMPTY_CHAIN
    assert record["provider_receipt"]["call_id"] == "call-1"


def test_translation_projection_candidates_do_not_collide_across_source_or_language(
    tmp_path: Path,
) -> None:
    store = AcceptedArtifactStore(tmp_path)
    for index, (source_hash, language, text) in enumerate((
        ("paper-a", "zh-CN", "译文 A"),
        ("paper-b", "zh-CN", "译文 B"),
        ("paper-a", "fr", "Traduction A"),
    ), 1):
        checkpoint = tmp_path / f"checkpoint-{index}"
        chapter = checkpoint / "chapters" / "ch-0001"
        chapter.mkdir(parents=True)
        (checkpoint / "migration-metadata.json").write_text(json.dumps({
            "source_hash": source_hash, "language": language,
        }))
        (chapter / "segmentation.json").write_text(json.dumps({
            "segments": [{"segment_id": "seg-0001", "block_ids": ["h1", "b1"]}],
        }))
        output = {"blocks": [
            {"block_id": "h1", "text": "Heading"},
            {"block_id": "b1", "text": text},
        ]}
        block = _accepted_block(output, input_sha=canonical_sha256({"candidate": index}))
        block["segment_id"] = "seg-0001"
        block["accepted_chain_sha256"] = hashlib.sha256(json.dumps({
            "predecessor": EMPTY_CHAIN,
            "segment_id": block["segment_id"],
            "input_sha256": block["input_sha256"],
            "output_sha256": block["output_sha256"],
            "generation": 1,
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        store.put_accepted(
            kind="translation", semantic_input_sha256=block["input_sha256"],
            recipe_sha256=canonical_sha256({"recipe": index}),
            contract_version="translation.v1", output=output, ledger_block=block,
            provider_receipt={
                "provider": "stub", "model": "test", "call_id": f"call-{index}",
                "usage": {},
            },
            provenance={
                "checkpoint_dir": str(checkpoint),
                "ledger": str(chapter / "translation-ledger.json"),
            },
        )

    all_candidates = accepted_translation_projection_candidates(store)
    selected = accepted_translation_projection_candidates(
        store, source_hash="paper-a", language="zh-CN",
    )

    assert len(all_candidates) == 3
    assert len(selected) == 1
    candidate = next(iter(selected.values()))
    assert candidate["source_hash"] == "paper-a"
    assert candidate["language"] == "zh-CN"
    assert candidate["block_ids"] == ["h1", "b1"]

    submitted = {**block, "state": "submitted"}
    with pytest.raises(ArtifactStoreError, match="only an accepted"):
        store.put_accepted(
            kind="translation", semantic_input_sha256=block["input_sha256"],
            recipe_sha256=canonical_sha256({"prompt": 1}),
            contract_version="translation.v1", output=output, ledger_block=submitted,
            provider_receipt={"provider": "p", "model": "m", "call_id": "c", "usage": {}},
            provenance={"checkpoint": "old"},
        )
    with pytest.raises(ArtifactStoreError, match="output hash"):
        store.put_accepted(
            kind="translation", semantic_input_sha256=block["input_sha256"],
            recipe_sha256=canonical_sha256({"prompt": 1}),
            contract_version="translation.v1", output={"changed": True}, ledger_block=block,
            provider_receipt={"provider": "p", "model": "m", "call_id": "c", "usage": {}},
            provenance={"checkpoint": "old"},
        )


def test_tampered_object_is_never_reused(tmp_path: Path) -> None:
    store = AcceptedArtifactStore(tmp_path)
    output = {"commentary": "accepted"}
    block = _accepted_block(output)
    record = store.put_accepted(
        kind="commentary", semantic_input_sha256=block["input_sha256"],
        recipe_sha256=canonical_sha256("recipe"), contract_version="commentary.v1",
        output=output, ledger_block=block,
        provider_receipt={"provider": "p", "model": "m", "call_id": "c", "usage": {}},
        provenance={"run_id": "run"},
    )
    path = store.path_for("commentary", record["artifact_id"])
    raw = json.loads(path.read_text())
    raw["output"] = {"commentary": "tampered"}
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ArtifactStoreError, match="modified"):
        store.read("commentary", record["artifact_id"])


def test_accepted_block_read_is_bound_to_ledger_hashes_and_segment(
    tmp_path: Path,
) -> None:
    store = AcceptedArtifactStore(tmp_path)
    output = {
        "explanation": "accepted explanation",
        "commentary": "accepted commentary",
        "commentary_sources": [],
        "prior_work": [],
        "later_work": [],
    }
    block = _accepted_block(output)
    record = store.put_accepted(
        kind="commentary",
        semantic_input_sha256=block["input_sha256"],
        recipe_sha256=canonical_sha256("recipe"),
        contract_version="commentary.v1",
        output=output,
        ledger_block=block,
        provider_receipt={
            "provider": "p", "model": "m", "call_id": "c", "usage": {},
        },
        provenance={"run_id": "run"},
    )

    restored = store.read_for_accepted_block(
        kind="commentary",
        contract_version="commentary.v1",
        ledger_block=block,
        output_validator=lambda value: value.get("commentary") == "accepted commentary",
    )
    assert restored["artifact_id"] == record["artifact_id"]
    assert restored["output"] == output

    rebound_block = {
        **block,
        "input_sha256": canonical_sha256({"source": "current"}),
        "validation_receipt": {
            "local_validation": True,
            "object_store_revalidated": True,
        },
        "logical_receipt": {
            "kind": "accepted_artifact_reuse",
            "artifact_id": record["artifact_id"],
            "provider_calls": 0,
        },
    }
    rebound_block["accepted_chain_sha256"] = hashlib.sha256(json.dumps({
        "predecessor": rebound_block["predecessor_accepted_chain_sha256"],
        "segment_id": rebound_block["segment_id"],
        "input_sha256": rebound_block["input_sha256"],
        "output_sha256": rebound_block["output_sha256"],
        "generation": rebound_block["generation"],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rebound_before = json.loads(json.dumps(rebound_block))

    rebound = store.read_for_accepted_block(
        kind="commentary",
        contract_version="commentary.v1",
        ledger_block=rebound_block,
        output_validator=lambda value: value.get("commentary") == "accepted commentary",
    )

    assert rebound["artifact_id"] != record["artifact_id"]
    assert rebound["semantic_input_sha256"] == rebound_block["input_sha256"]
    assert rebound["output_sha256"] == rebound_block["output_sha256"]
    assert rebound["provenance"]["derived_from_artifact_id"] == record["artifact_id"]
    assert store.path_for("commentary", rebound["artifact_id"]).is_file()
    assert rebound_block == rebound_before

    wrong_output = {**block, "output_sha256": "f" * 64}
    wrong_output["logical_receipt"] = {
        "kind": "accepted_artifact_reuse",
        "artifact_id": record["artifact_id"], "provider_calls": 0,
    }
    wrong_output["validation_receipt"] = {
        "local_validation": True, "object_store_revalidated": True,
    }
    wrong_output["accepted_chain_sha256"] = hashlib.sha256(json.dumps({
        "predecessor": wrong_output["predecessor_accepted_chain_sha256"],
        "segment_id": wrong_output["segment_id"],
        "input_sha256": wrong_output["input_sha256"],
        "output_sha256": wrong_output["output_sha256"],
        "generation": wrong_output["generation"],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ArtifactStoreError, match="output does not match"):
        store.read_for_accepted_block(
            kind="commentary", contract_version="commentary.v1",
            ledger_block=wrong_output,
        )

    wrong_segment = {**block, "segment_id": "ch-0001.seg-9999"}
    wrong_segment["logical_receipt"] = {
        "kind": "accepted_artifact_reuse",
        "artifact_id": record["artifact_id"], "provider_calls": 0,
    }
    wrong_segment["validation_receipt"] = {
        "local_validation": True, "object_store_revalidated": True,
    }
    wrong_segment["accepted_chain_sha256"] = hashlib.sha256(json.dumps({
        "predecessor": wrong_segment["predecessor_accepted_chain_sha256"],
        "segment_id": wrong_segment["segment_id"],
        "input_sha256": wrong_segment["input_sha256"],
        "output_sha256": wrong_segment["output_sha256"],
        "generation": wrong_segment["generation"],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ArtifactStoreError, match="segment does not match"):
        store.read_for_accepted_block(
            kind="commentary", contract_version="commentary.v1",
            ledger_block=wrong_segment,
        )

    with pytest.raises(ArtifactStoreError, match="current output contract"):
        store.read_for_accepted_block(
            kind="commentary", contract_version="commentary.v1",
            ledger_block=block, output_validator=lambda _value: False,
        )


@pytest.mark.parametrize(
    ("logical_kind", "validation_receipt", "message"),
    [
        (
            "provider_call", {
                "local_validation": True, "object_store_revalidated": True,
            }, "accepted-artifact reuse receipt",
        ),
        (
            "accepted_artifact_reuse", {"local_validation": True},
            "explicit local object revalidation",
        ),
        (
            "accepted_artifact_reuse", {"object_store_revalidated": True},
            "explicit local object revalidation",
        ),
    ],
)
def test_historical_rebind_requires_explicit_reuse_validation_receipts(
    tmp_path: Path,
    logical_kind: str,
    validation_receipt: dict[str, bool],
    message: str,
) -> None:
    store = AcceptedArtifactStore(tmp_path)
    output = {
        "explanation": "accepted explanation",
        "commentary": "accepted commentary",
        "commentary_sources": [],
        "prior_work": [],
        "later_work": [],
    }
    original = _accepted_block(output)
    record = store.put_accepted(
        kind="commentary",
        semantic_input_sha256=original["input_sha256"],
        recipe_sha256=canonical_sha256("recipe"),
        contract_version="commentary.v1",
        output=output,
        ledger_block=original,
        provider_receipt={
            "provider": "p", "model": "m", "call_id": "c", "usage": {},
        },
        provenance={"run_id": "run"},
    )
    rebound = {
        **original,
        "input_sha256": canonical_sha256({
            "source": logical_kind,
            "validation_receipt": validation_receipt,
        }),
        "validation_receipt": validation_receipt,
        "logical_receipt": {
            "kind": logical_kind,
            "artifact_id": record["artifact_id"],
            "provider_calls": 0,
        },
    }
    rebound["accepted_chain_sha256"] = hashlib.sha256(json.dumps({
        "predecessor": rebound["predecessor_accepted_chain_sha256"],
        "segment_id": rebound["segment_id"],
        "input_sha256": rebound["input_sha256"],
        "output_sha256": rebound["output_sha256"],
        "generation": rebound["generation"],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    with pytest.raises(ArtifactStoreError, match=message):
        store.read_for_accepted_block(
            kind="commentary",
            contract_version="commentary.v1",
            ledger_block=rebound,
            output_validator=lambda _value: True,
        )


def test_recipe_change_is_stale_zero_call_but_semantic_change_is_miss(tmp_path: Path) -> None:
    store = AcceptedArtifactStore(tmp_path)
    output = {"blocks": [{"block_id": "b1", "text": "译文"}]}
    block = _accepted_block(output)
    old_recipe = canonical_sha256({"model": "old"})
    store.put_accepted(
        kind="translation", semantic_input_sha256=block["input_sha256"],
        recipe_sha256=old_recipe, contract_version="translation.v1",
        output=output, ledger_block=block,
        provider_receipt={"provider": "p", "model": "m", "call_id": "c", "usage": {}},
        provenance={"run_id": "run"},
    )
    requests = [
        ReuseRequest(
            chapter_id="ch-0001", segment_id="ch-0001.seg-0001", lane="translation",
            semantic_input_sha256=block["input_sha256"],
            recipe_sha256=canonical_sha256({"model": "new"}),
            contract_version="translation.v1",
        ),
        ReuseRequest(
            chapter_id="ch-0002", segment_id="ch-0002.seg-0001", lane="translation",
            semantic_input_sha256=canonical_sha256({"source": "changed"}),
            recipe_sha256=old_recipe, contract_version="translation.v1",
        ),
    ]
    plan = build_reuse_plan(store, requests, validators={"translation": lambda value: True})

    assert [item["status"] for item in plan["entries"]] == ["recipe_stale", "miss"]
    assert [item["estimated_provider_calls"] for item in plan["entries"]] == [0, 1]
    assert plan["estimated_provider_calls"] == 1


def test_lane_identity_excludes_runtime_and_render_options() -> None:
    shared = {
        "source_segment": {"blocks": ["b1"]}, "target_language": "Chinese",
        "glossary": {"mass": "质量"}, "protected_names": ["Einstein"],
        "guide": {"main": "guide"}, "static_context": {"paper": "p"},
        "predecessor_accepted_chain_sha256": EMPTY_CHAIN,
    }
    first = lane_semantic_sha256("translation", {**shared, "workers": 1, "font_size": 10})
    second = lane_semantic_sha256("translation", {**shared, "workers": 24, "font_size": 14})
    changed = lane_semantic_sha256("translation", {**shared, "target_language": "Japanese"})
    changed_navigation = lane_semantic_sha256("translation", {
        **shared,
        "guide": {"main": "changed"},
        "static_context": {"paper": "changed"},
        "predecessor_accepted_chain_sha256": "f" * 64,
    })

    assert first == second
    assert first == changed_navigation
    assert changed != first
    assert lane_recipe_sha256(
        "translation", prompt="p", model="m1", tier="high"
    ) != lane_recipe_sha256("translation", prompt="p", model="m2", tier="high")


def test_intent_guidance_changes_only_content_lane_semantics() -> None:
    guidance = {
        "user_intent_sha256": "intent", "output_sha256": "guidance",
        "references": [{"source_id": "book", "document_hash": "v2", "locator": "c2"}],
    }
    contexts = {
        "segmentation": {"source": {}, "chapter": {}, "limits": {}},
        "glossary": {"source": {}, "target_language": "zh-CN", "index": [], "protected_names": []},
        "title_translation": {"source_titles": [], "source_language": "en", "target_language": "zh-CN", "glossary": {}, "protected_names": []},
        "guide": {"chapter_source": {}, "target_language": "zh-CN", "verified_evidence": {}},
        "translation": {"source_segment": {}, "target_language": "zh-CN", "glossary": {}, "protected_names": []},
        "commentary": {"source_segment": {}, "guide": {}, "metadata": {}, "selected_evidence": {}, "selected_domain_context": {}, "access_policy": {}, "predecessor_accepted_chain_sha256": ""},
        "review": {"translation_artifacts": {}, "commentary_artifacts": {}, "review_contract": {}},
    }
    for lane, context in contexts.items():
        old_hash = lane_semantic_sha256(lane, context)
        assert old_hash == lane_semantic_sha256(
            lane, {**context, "intent_guidance": None},
        )
        changed = lane_semantic_sha256(
            lane, {**context, "intent_guidance": guidance},
        )
        assert (changed == old_hash) is (lane == "segmentation")


def test_scoped_regeneration_requires_confirmation_for_all_and_rejects_force() -> None:
    assert normalize_regeneration_lanes(["commentary", "translation", "commentary"]) == (
        "translation", "commentary"
    )
    with pytest.raises(RegenerationRequestError, match="requires"):
        normalize_regeneration_lanes(["all"])
    assert normalize_regeneration_lanes(["all"], confirm_expensive_all=True) == REGENERATABLE_LANES
    with pytest.raises(RegenerationRequestError, match="no longer"):
        reject_broad_force(True, ())


def test_all_fingerprint_migration_imports_only_strong_accepted_receipts(tmp_path: Path) -> None:
    project = tmp_path / "project"
    checkpoint_identity = "a" * 64
    checkpoint = allocate_artifact_dir(
        project / ".arc-companion" / "checkpoints",
        checkpoint_identity,
        kind="checkpoint",
    ).path
    annotation_dir = checkpoint / "annotations"
    chapter_dir = checkpoint / "chapters" / "ch-0001"
    annotation_dir.mkdir(parents=True)
    chapter_dir.mkdir(parents=True)
    output = {"commentary": "accepted", "commentary_sources": []}
    block = _accepted_block(output)
    (annotation_dir / "candidate.json").write_text(json.dumps({
        "schema_version": "legacy.annotation.v1",
        "segment_id": block["segment_id"],
        "input_sha256": "different-cache-identity-is-not-the-ledger-identity",
        "annotation": output,
    }), encoding="utf-8")
    ledger = {
        "schema_version": "arc.companion.chapter-lane-ledger.v1",
        "chapter_id": "ch-0001", "lane": "companion", "generation": 1,
        "blocks": [block], "accepted_chain_sha256": block["accepted_chain_sha256"],
    }
    (chapter_dir / "companion-ledger.json").write_text(json.dumps(ledger), encoding="utf-8")

    report = import_accepted_checkpoint_objects(
        project,
        validators={"commentary": lambda value: value.get("commentary") == "accepted"},
        contract_versions={"commentary": "commentary.v1"},
    )

    assert report["provider_calls"] == 0
    assert len(report["imported_artifact_ids"]) == 1
    assert report["receipts"][0]["accepted"] is True
    assert report["receipts"][0]["checkpoint"] == checkpoint_identity

    ledger["blocks"][0]["accepted_chain_sha256"] = "0" * 64
    (chapter_dir / "companion-ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    second = import_accepted_checkpoint_objects(
        project, validators={"commentary": lambda value: True},
        contract_versions={"commentary": "commentary.v1"},
    )
    assert second["receipts"][0]["reason"] == "accepted_chain_hash_mismatch"


def test_migration_validates_review_binding_and_imports_valid_reader_final(tmp_path: Path) -> None:
    project = tmp_path / "project"
    checkpoint = (
        project / ".arc-companion" / "checkpoints" / ("a" * 64)
    )
    chapter_dir = checkpoint / "chapters" / "ch-0001"
    chapter_dir.mkdir(parents=True)
    output = {"commentary": "base", "commentary_sources": []}
    block = _accepted_block(output)
    ledger = {
        "schema_version": "arc.companion.chapter-lane-ledger.v1",
        "chapter_id": "ch-0001", "lane": "companion", "generation": 1,
        "blocks": [block], "accepted_chain_sha256": block["accepted_chain_sha256"],
    }
    (chapter_dir / "companion-ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    overlay = {
        "schema_version": "arc.companion.chapter-review-overlay.v1",
        "chapter_id": "ch-0001", "lane": "companion",
        "base_accepted_chain_sha256": block["accepted_chain_sha256"],
        "reviewed_output_sha256": canonical_sha256({"review": "all"}),
        "validation_receipt": {"review_output_matches_sha256": True},
        "blocks": [{
            "segment_id": block["segment_id"],
            "accepted_chain_sha256": block["accepted_chain_sha256"],
            "base_output_sha256": block["output_sha256"],
            "reviewed_output_sha256": canonical_sha256({"commentary": "reviewed"}),
            "validation_receipt": {"review_applied": True},
        }],
    }
    (chapter_dir / "companion-review-overlay.json").write_text(
        json.dumps(overlay), encoding="utf-8"
    )
    segment_id = block["segment_id"]
    reader = {
        "schema_version": "arc.companion.reader-final.v1",
        "final_overrides": {
            "document": {"blocks": [{"block_id": "b1", "text": "source"}]},
            "chapters": [{"chapter_id": "ch-0001", "block_ids": ["b1"]}],
            "segments": [{"chapter_id": "ch-0001", "segment_id": segment_id, "block_ids": ["b1"]}],
            "chapter_guides": {"ch-0001": {"main_content": "guide"}}, "translations": None,
            "annotations": {segment_id: {"commentary": "reviewed"}},
            "glossary": {}, "metadata": {}, "language": "Chinese",
            "translation_mode": "skipped",
        },
    }
    (checkpoint / "reader-final.json").write_text(json.dumps(reader), encoding="utf-8")

    report = import_accepted_checkpoint_objects(
        project,
        validators={"commentary": lambda value: True},
        contract_versions={"commentary": "commentary.v1"},
    )
    review_receipt = next(item for item in report["receipts"] if item["lane"] == "review")
    content_receipt = next(item for item in report["receipts"] if item["lane"] == "reader-content")
    assert review_receipt["base_binding_valid"] is True
    assert review_receipt["reason"] == "reviewed_output_checkpoint_missing"
    assert content_receipt["accepted"] is True, content_receipt
    assert report["imported_content_sha256"] == [content_receipt["content_sha256"]]
