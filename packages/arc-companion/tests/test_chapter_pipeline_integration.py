from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import sys
import threading

import pytest

import arc_companion.pipeline as pipeline
from arc_companion.artifact_store import (
    AcceptedArtifactStore,
    artifact_id_for,
    canonical_sha256,
)
from arc_companion.chapter_scheduler import run_chapter_pipeline
from arc_companion.ledger import initialize_lane_ledger, mark_needs_supervision
from arc_companion.ledger_registry import (
    mutate_registered_lane_ledger,
    read_registered_lane_ledger,
)
from arc_companion.pipeline import BuildOptions, build_companion
from arc_companion.source import SourceBundle
from arc_companion.io import sha256_json
from arc_companion.reuse import lane_recipe_sha256
from arc_llm.attempt_diagnostics import AttemptDiagnostics
from arc_llm.call_checkpoint import (
    checkpoint_path,
    prepare_call,
    record_failure,
    record_submitted,
)
from arc_llm.providers.base import LLMSubmissionState, LLMWorkerTimeout
from arc_llm.progress_journal import ProgressJournal
from arc_llm.recovery_context import read_recovery_context
from arc_llm.sessions import LLMSessionManager


def _title_response(prompt: str) -> dict[str, object]:
    payload = json.loads(prompt[prompt.index("{"):])
    return {"titles": [
        {"title_id": item["title_id"], "text": item["source_text"]}
        for item in payload["titles"]
    ]}


def _segment_review_response(prompt: str) -> dict[str, object]:
    marker = "PORTION:\n" if "PORTION:\n" in prompt else "COMPANION:\n"
    payload = json.loads(prompt.split(marker, 1)[1])
    return {
        "reviewed_segment_ids": [
            item["segment"]["segment_id"]
            for item in payload["segments"]
        ],
        "findings": [],
        "patches": [],
    }


def test_reader_final_checkpoint_is_shared_by_both_pipeline_shapes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    overrides = {
        "document": {"blocks": []}, "chapters": [], "segments": [],
        "chapter_guides": {}, "translations": {}, "annotations": {},
        "glossary": {"entries": []}, "metadata": {},
        "translation_mode": "enabled",
    }
    path = pipeline._write_reader_final_checkpoint(checkpoint, overrides)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {
        "schema_version": "arc.companion.reader-final.v4",
        "final_overrides": overrides,
    }


def test_resume_v2_compacts_absolute_and_relative_ledger_entries(tmp_path: Path) -> None:
    from arc_companion.resume_transaction import load_transaction

    project = tmp_path / "compact"
    ledger_path = project / "checkpoint" / "production" / "translation-ledger.json"
    ledger_path.parent.mkdir(parents=True)
    relative_path = os.path.relpath(ledger_path, Path.cwd())
    key = "ch-0001:translation:companion-translation-s1:generation-1"
    journal = project / ".arc-companion" / "resume-transaction.json"
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps({
        "schema_version": "arc.companion.resume-transaction.v2",
        "action": "resume-native", "status": "continuation_failed",
        "recovery_options": {"paper_id": "local:book"},
        "entries": [
            {
                "ledger_path": relative_path, "session_key": "ch-0001:translation",
                "segment_id": "s1", "status": "pending",
                "blocking_reason": "older supervision detail",
                "recovery_context": {"submission_state": "unknown"},
            },
            {
                "ledger_path": str(ledger_path.resolve()),
                "session_key": "ch-0001:translation", "segment_id": "s1",
                "idempotency_key": key, "status": "resolved",
                "output_sha256": "accepted-output", "generation": 1,
            },
        ],
        "native_resume_contexts": [
            {"idempotency_key": key, "session_key": "ch-0001:translation"},
            {"idempotency_key": key, "generation": 1},
        ],
    }))

    compacted = load_transaction(project)

    assert compacted is not None
    assert len(compacted["entries"]) == 1
    entry = compacted["entries"][0]
    assert entry["ledger_path"] == str(ledger_path.resolve())
    assert entry["idempotency_key"] == key
    assert entry["status"] == "resolved"
    assert entry["output_sha256"] == "accepted-output"
    assert entry["blocking_reason"] == "older supervision detail"
    assert entry["recovery_context"] == {"submission_state": "unknown"}
    assert compacted["native_resume_contexts"] == [{
        "idempotency_key": key,
        "session_key": "ch-0001:translation",
        "generation": 1,
    }]
    assert json.loads(journal.read_text()) == compacted


def test_resume_local_repair_failure_does_not_block_other_native_lanes() -> None:
    calls: list[str] = []
    lock = threading.Lock()

    def record(label: str) -> dict[str, str]:
        with lock:
            calls.append(label)
        return {"logical_key": label}

    def translation(prepared, segment):
        label = f"{prepared.chapter['chapter_id']}:translation:{segment['segment_id']}"
        record(label)
        if prepared.chapter["chapter_id"] == "ch-0008":
            raise ValueError("local token repair reconstruction failed")
        return {"ok": "translation"}

    def companion(prepared, segment):
        return record(
            f"{prepared.chapter['chapter_id']}:companion:{segment['segment_id']}"
        )

    with pytest.raises(ValueError, match="local token repair reconstruction failed"):
        run_chapter_pipeline(
            [{"chapter_id": "ch-0008"}, {"chapter_id": "ch-0009"}],
            workers=4,
            prepare_guide=lambda chapter: record(f"{chapter['chapter_id']}:guide"),
            prepare_segments=lambda chapter: [{
                "segment_id": f"{chapter['chapter_id']}.seg-0001",
            }],
            run_translation=translation,
            run_companion=companion,
        )

    expected = {
        "ch-0008:guide",
        "ch-0008:translation:ch-0008.seg-0001",
        "ch-0008:companion:ch-0008.seg-0001",
        "ch-0009:guide",
        "ch-0009:translation:ch-0009.seg-0001",
        "ch-0009:companion:ch-0009.seg-0001",
    }
    assert set(calls) == expected
    assert len(calls) == len(expected)

def test_incremental_reader_failure_preserves_last_published_state(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "run"
    project.mkdir()
    state_path = project / "state.json"
    pipeline._state(
        state_path, status="active", fingerprint="source",
        output_html=str(project / "reader" / "index.html"),
        output_html_sha256="accepted-html",
        web_render_version="accepted-version",
    )
    prior = json.loads(state_path.read_text(encoding="utf-8"))
    fake_coordinator = SimpleNamespace(
        request=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("render failed")
        ),
    )
    monkeypatch.setattr(
        pipeline, "_create_reader_coordinator",
        lambda *_args, **_kwargs: fake_coordinator,
    )

    result = pipeline._publish_reader_update(
        project, state_path, threading.RLock(),
    )

    assert result is None
    after = json.loads(state_path.read_text(encoding="utf-8"))
    assert after["output_html"] == prior["output_html"]
    assert after["output_html_sha256"] == prior["output_html_sha256"]
    assert after["web_render_version"] == prior["web_render_version"]
    assert after["reader_dirty"] is True
    assert after["reader_publish_state_version"] == (
        "arc.companion.reader-publish-state.v1"
    )


def test_chaptered_skip_translation_omits_lane_artifacts_and_migration(
    tmp_path: Path,
) -> None:
    blocks = [
        {"block_id": "c1", "type": "section", "title": "Chapter One"},
        {"block_id": "p1", "type": "text", "text": "Energy is conserved."},
        {
            "block_id": "idx", "type": "text", "source_role": "index",
            "text": "Source Index Marker, 1",
        },
    ]
    document = {
        "schema_version": "arc.paper.document.v2",
        "front_matter": {},
        "blocks": blocks,
        "equations": [],
        "figures": [],
        "tables": [],
        "assets": [],
        "links": [],
        "bibliography": [],
        "integrity": {"status": "complete", "document_hash": "skip-chapter"},
    }
    bundle = SourceBundle(
        paper_id="local:skip-chapter",
        document=document,
        parsed={
            "paper_id": "local:skip-chapter",
            "document": document,
            "source_hash": "skip-chapter",
            "structure": {
                "document_kind": "book",
                "index_block_ids": ["idx"],
                "chapters": [{"title": "Chapter One", "block_ids": ["c1", "p1"]}],
            },
            "index_entries": {
                "schema_version": "arc.paper.index_entries.v1",
                "entries": [{"term": "Source Index Marker", "pages": [1]}],
            },
        },
        metadata={"title": "Book"},
        references=[],
        citers=[],
    )
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({
        "metadata": {"source_hash": "skip-chapter", "language": "en"},
        "translations": {
            "old": {"segment_id": "old", "translation": {"blocks": [
                {"block_id": "p1", "text": "This must not migrate."},
            ]}},
        },
        "glossary": {"entries": [{
            "source_term": "OLD_GLOSSARY_SENTINEL",
            "target_term": "旧术语",
        }]},
    }), encoding="utf-8")
    labels: list[str] = []
    prompts: list[str] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        labels.append(label)
        prompts.append(prompt)
        if "translation" in label:
            raise AssertionError(f"translation call was submitted: {label}")
        if label.startswith("companion-glossary-"):
            return {"entries": []}
        if label.startswith("companion-index-glossary-"):
            entries = json.loads(prompt.rsplit("\n", 1)[-1])
            return {"entries": [{
                "entry_id": item["entry_id"], "target": item["source"],
                "explanation": "Source index term.",
            } for item in entries]}
        if label.startswith("companion-guide-"):
            return {
                "motivation": "Motivation.", "main_content": "Conservation.",
                "section_logic": None, "prerequisites": None,
                "pedagogical_comparison": None, "historical_context": [],
                "supplementary_reading": [],
            }
        if label.startswith("companion-segmentation-"):
            return {"cut_after_ordinals": []}
        if label.startswith("companion-annotation-"):
            return {
                "explanation": "Commentary remains.", "commentary": "",
                "prior_work": [], "later_work": [], "context_claims": [],
                "evidence_ids": [], "key_points": [], "source_notes": [],
                "evidence_requests": [],
            }
        if label.startswith("companion-review-segment-"):
            return _segment_review_response(prompt)
        raise AssertionError(label)

    result = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=tmp_path / "run",
            annotation_language="en",
            workers=2,
            skip_translation=True,
            stop_after_first_chapter=True,
            regenerate_lanes=("glossary",),
            legacy_checkpoint=legacy,
        ),
        source_loader=lambda *args, **kwargs: bundle,
        llm=llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert result["ok"], result
    checkpoint = Path(result["data"]["checkpoint_dir"])
    assert not any("translation" in label for label in labels)
    assert not any("glossary" in label for label in labels)
    assert all("GLOSSARY:\n" not in prompt for prompt in prompts)
    assert all("OLD_GLOSSARY_SENTINEL" not in prompt for prompt in prompts)
    assert not list((checkpoint / "chapters").rglob("translation-ledger.json"))
    assert not list(checkpoint.rglob("translations*"))
    assert not (checkpoint / "glossary.json").exists()
    assert not (checkpoint / "index-glossary.json").exists()
    assert not (tmp_path / "run" / ".arc-companion" / "objects" / "glossary").exists()
    receipt = json.loads((checkpoint / "legacy-migration.json").read_text())
    assert receipt["glossary"] == {
        "accepted": False, "reason": "glossary_disabled_for_same_language_source",
        "value": None,
    }
    assert receipt["translations"]["ledgers"] == {}
    assert receipt["translations"]["receipts"] == [{
        "status": "skipped",
        "reason": "translation_disabled_for_same_language_source",
    }]
    freeze = json.loads((checkpoint / "first-chapter-freeze.json").read_text())
    assert freeze["translation_mode"] == "skipped"
    assert freeze["pre_review_translation_sha256"] is None
    assert freeze["translation_sha256"] is None
    reuse_plan = json.loads((checkpoint / "reuse-plan.json").read_text())
    glossary_entry = next(
        item for item in reuse_plan["entries"] if item["lane"] == "glossary"
    )
    assert glossary_entry["status"] == "skipped"
    assert glossary_entry["reason"] == "glossary_disabled_for_same_language_source"
    assert glossary_entry["estimated_provider_calls"] == 0
    reader_final = json.loads((checkpoint / "reader-final.json").read_text())
    assert reader_final["final_overrides"]["glossary"] == {}
    assert any(
        block.get("block_id") == "idx"
        for block in reader_final["final_overrides"]["document"]["blocks"]
    )
    tex = Path(result["data"]["output_tex"]).read_text(encoding="utf-8")
    manifest = json.loads(Path(result["data"]["source_manifest_path"]).read_text())
    assert "ARC-TRANSLATION-" not in tex
    assert "Source Index Marker" in tex
    assert manifest["companion_layers"]["translation_mode"] is False
    assert manifest["companion_layers"]["rendered_translation_segment_ids"] == []

    # Raw/custom LLM adapters do not expose provider receipts, so accepted
    # commentary is checkpoint-backed rather than object-store-backed. A
    # partial resume must still rehydrate it without another model call.
    assert not (
        tmp_path / "run" / ".arc-companion" / "objects" / "commentary"
    ).exists()
    annotation_calls_before = len([
        label for label in labels if label.startswith("companion-annotation-")
    ])
    active_state = json.loads((tmp_path / "run" / "state.json").read_text())
    active_state["status"] = "active"
    (tmp_path / "run" / "state.json").write_text(json.dumps(active_state))
    (checkpoint / "first-chapter-freeze.json").unlink()

    resumed = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=tmp_path / "run",
            annotation_language="en",
            workers=2,
            skip_translation=True,
            stop_after_first_chapter=True,
            legacy_checkpoint=legacy,
        ),
        source_loader=lambda *args, **kwargs: bundle,
        llm=llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert resumed["ok"], resumed
    assert len([
        label for label in labels if label.startswith("companion-annotation-")
    ]) == annotation_calls_before


def test_structural_only_chapter_uses_local_receipts_without_provider_calls(
    tmp_path: Path,
) -> None:
    document = {
        "schema_version": "arc.paper.document.v2",
        "front_matter": {},
        "blocks": [{
            "block_id": "part-1", "type": "part", "level": 1,
            "title": "I PART",
        }],
        "equations": [], "figures": [], "tables": [], "assets": [],
        "links": [], "bibliography": [],
        "integrity": {"status": "complete", "document_hash": "structural-only"},
    }
    bundle = SourceBundle(
        paper_id="local:structural-only",
        document=document,
        parsed={
            "paper_id": "local:structural-only",
            "document": document,
            "source_hash": "structural-only",
            "structure": {
                "document_kind": "book",
                "chapters": [{"title": "I PART", "block_ids": ["part-1"]}],
            },
        },
        metadata={"title": "Book"}, references=[], citers=[],
    )
    calls: list[str] = []

    def llm(_prompt: str, **kwargs):
        label = str(kwargs.get("call_label") or "")
        calls.append(label)
        if label.startswith("title-translation-"):
            return _title_response(_prompt)
        raise AssertionError("structural-only chapter submitted a non-title provider call")

    result = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=tmp_path / "run",
            annotation_language="en",
            workers=2,
            stop_after_first_chapter=True,
        ),
        source_loader=lambda *args, **kwargs: bundle,
        llm=llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert result["ok"], result
    assert len(calls) == 1
    assert calls[0].startswith("title-translation-")
    checkpoint = Path(result["data"]["checkpoint_dir"])
    chapters = json.loads((checkpoint / "chapters.json").read_text())
    assert chapters["chapters"][0]["content_block_ids"] == []
    segmentation = json.loads(
        (checkpoint / "chapters" / "ch-0001" / "segmentation.json").read_text()
    )
    assert segmentation["segments"][0]["structural_only"] is True
    for lane in ("guide", "translation", "companion"):
        ledger = json.loads(
            (checkpoint / "chapters" / "ch-0001" / f"{lane}-ledger.json").read_text()
        )
        receipt = ledger["blocks"][0]["logical_receipt"]
        assert receipt["kind"] == "controller_skipped_structural_heading"
        assert receipt["provider_calls"] == 0
        assert ledger["blocks"][0]["submission_state"] == "not_submitted"
    review = json.loads((checkpoint / "chapter-review.json").read_text())
    assert review["reviewed_segment_ids"] == []


def test_targeted_commentary_regeneration_rebinds_deferred_suffix(
    tmp_path: Path,
) -> None:
    blocks = [
        {"block_id": "c1", "type": "section", "title": "Chapter One"},
        {"block_id": "p1", "type": "text", "text": "First claim."},
        {"block_id": "p2", "type": "text", "text": "Second claim."},
    ]
    document = {
        "schema_version": "arc.paper.document.v2", "front_matter": {},
        "blocks": blocks, "equations": [], "figures": [], "tables": [],
        "assets": [], "links": [], "bibliography": [],
        "integrity": {"status": "complete", "document_hash": "deferred-commentary"},
    }
    bundle = SourceBundle(
        paper_id="local:deferred-commentary", document=document,
        parsed={
            "paper_id": "local:deferred-commentary", "document": document,
            "source_hash": "deferred-commentary",
            "structure": {"document_kind": "book", "chapters": [{
                "title": "Chapter One", "block_ids": ["c1", "p1", "p2"],
            }]},
        },
        metadata={"title": "Deferred commentary"}, references=[], citers=[],
    )
    labels: list[str] = []
    annotation_generation = 0

    def llm(_prompt: str, **kwargs):
        nonlocal annotation_generation
        label = str(kwargs["call_label"])
        labels.append(label)
        if label.startswith("companion-guide-"):
            return {
                "motivation": None, "main_content": "Two claims.",
                "section_logic": None, "prerequisites": None,
                "pedagogical_comparison": None, "historical_context": [],
                "supplementary_reading": [],
            }
        if label.startswith("companion-segmentation-"):
            return {"cut_after_ordinals": [2]}
        if label.startswith("companion-annotation-"):
            annotation_generation += 1
            return {
                "explanation": f"Explanation {annotation_generation}.",
                "commentary": "", "prior_work": [], "later_work": [],
                "context_claims": [], "evidence_ids": [], "key_points": [],
                "source_notes": [], "evidence_requests": [],
            }
        if label.startswith("companion-review-segment-"):
            return _segment_review_response(_prompt)
        raise AssertionError(label)

    def result_llm(prompt: str, **kwargs):
        return SimpleNamespace(
            value=llm(prompt, **kwargs),
            logical_receipt={"idempotency_key": kwargs["idempotency_key"]},
        )

    project = tmp_path / "run"
    options = BuildOptions(
        paper_id=bundle.paper_id, project_dir=project,
        annotation_language="en", skip_translation=True, workers=1,
    )
    first = build_companion(
        options, source_loader=lambda *args, **kwargs: bundle,
        llm=llm, result_llm=result_llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )
    assert first["ok"], first
    initial_calls = [
        label for label in labels if label.startswith("companion-annotation-")
    ]
    assert len(initial_calls) == 2

    regenerated = build_companion(
        BuildOptions(
            **{
                **options.__dict__,
                "regenerate_segments": ("commentary:ch-0001.seg-0001",),
            }
        ),
        source_loader=lambda *args, **kwargs: bundle,
        llm=llm, result_llm=result_llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert regenerated["ok"], regenerated
    final_calls = [
        label for label in labels if label.startswith("companion-annotation-")
    ]
    assert len(final_calls) == len(initial_calls) + 1
    checkpoint = Path(regenerated["data"]["checkpoint_dir"])
    ledger = json.loads((
        checkpoint / "chapters" / "ch-0001" / "companion-ledger.json"
    ).read_text())
    suffix = next(
        block for block in ledger["blocks"]
        if block["segment_id"] == "ch-0001.seg-0002"
    )
    assert suffix["state"] == "accepted"
    assert suffix["validation_receipt"]["reuse_status"] == "deferred_hit"
    assert suffix["validation_receipt"]["object_store_revalidated"] is True
    rebound_id = suffix["logical_receipt"]["artifact_id"]
    source_id = suffix["logical_receipt"]["source_artifact_id"]
    assert rebound_id == source_id
    direct_id = artifact_id_for(
        kind="commentary",
        semantic_input_sha256=suffix["input_sha256"],
        output_sha256=suffix["output_sha256"],
        contract_version=pipeline.SCHEMA_VERSION,
        predecessor_accepted_chain_sha256=(
            suffix["predecessor_accepted_chain_sha256"]
        ),
    )
    rebound = AcceptedArtifactStore(project).read("commentary", direct_id)
    assert rebound["provenance"]["derived_from_artifact_id"] == source_id


def test_stateless_targeted_deferred_commentary_resumes_from_current_checkpoint(
    tmp_path: Path,
) -> None:
    blocks = [
        {"block_id": "c1", "type": "section", "title": "Chapter One"},
        {"block_id": "p1", "type": "text", "text": "First claim."},
        {"block_id": "p2", "type": "text", "text": "Second claim."},
    ]
    document = {
        "schema_version": "arc.paper.document.v2", "front_matter": {},
        "blocks": blocks, "equations": [], "figures": [], "tables": [],
        "assets": [], "links": [], "bibliography": [],
        "integrity": {"status": "complete", "document_hash": "stateless-deferred"},
    }
    bundle = SourceBundle(
        paper_id="local:stateless-deferred", document=document,
        parsed={
            "paper_id": "local:stateless-deferred", "document": document,
            "source_hash": "stateless-deferred",
            "structure": {"document_kind": "book", "chapters": [{
                "title": "Chapter One", "block_ids": ["c1", "p1", "p2"],
            }]},
        },
        metadata={"title": "Stateless deferred commentary"},
        references=[], citers=[],
    )
    labels: list[str] = []
    annotation_generation = 0

    def llm(_prompt: str, **kwargs):
        nonlocal annotation_generation
        label = str(kwargs["call_label"])
        labels.append(label)
        if label.startswith("companion-guide-"):
            return {
                "motivation": None, "main_content": "Two claims.",
                "section_logic": None, "prerequisites": None,
                "pedagogical_comparison": None, "historical_context": [],
                "supplementary_reading": [],
            }
        if label.startswith("companion-segmentation-"):
            return {"cut_after_ordinals": [2]}
        if label.startswith("companion-annotation-"):
            annotation_generation += 1
            return {
                "explanation": f"Explanation {annotation_generation}.",
                "commentary": "", "prior_work": [], "later_work": [],
                "context_claims": [], "evidence_ids": [], "key_points": [],
                "source_notes": [], "evidence_requests": [],
            }
        if label.startswith("companion-review-segment-"):
            return _segment_review_response(_prompt)
        raise AssertionError(label)

    project = tmp_path / "run"
    options = BuildOptions(
        paper_id=bundle.paper_id, project_dir=project,
        annotation_language="en", skip_translation=True, workers=1,
    )
    compile_failure = {"enabled": False}

    def compiler(_tex: Path, pdf: Path) -> None:
        if compile_failure["enabled"]:
            raise RuntimeError("injected failure after deferred acceptance")
        pdf.write_bytes(b"%PDF fixture")

    first = build_companion(
        options, source_loader=lambda *args, **kwargs: bundle,
        llm=llm, compiler=compiler,
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )
    assert first["ok"], first
    assert list(AcceptedArtifactStore(project).iter_kind("commentary")) == []
    initial_annotation_calls = len([
        label for label in labels if label.startswith("companion-annotation-")
    ])
    assert initial_annotation_calls == 2

    compile_failure["enabled"] = True
    failed = build_companion(
        BuildOptions(
            **{
                **options.__dict__,
                "regenerate_segments": ("commentary:ch-0001.seg-0001",),
            }
        ),
        source_loader=lambda *args, **kwargs: bundle,
        llm=llm, compiler=compiler,
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )
    assert not failed["ok"]
    assert len([
        label for label in labels if label.startswith("companion-annotation-")
    ]) == initial_annotation_calls + 1

    checkpoint = Path(json.loads((project / "state.json").read_text())["checkpoint_dir"])
    ledger_path = checkpoint / "chapters" / "ch-0001" / "companion-ledger.json"
    failed_ledger = json.loads(ledger_path.read_text())
    suffix = next(
        block for block in failed_ledger["blocks"]
        if block["segment_id"] == "ch-0001.seg-0002"
    )
    assert suffix["state"] == "accepted"
    assert suffix["generation"] == 2
    suffix_checkpoints = list((checkpoint / "annotations" / "generation-2").glob("*.json"))
    assert any(
        json.loads(path.read_text())["segment_id"] == "ch-0001.seg-0002"
        for path in suffix_checkpoints
    )
    accepted_chain = failed_ledger["accepted_chain_sha256"]

    compile_failure["enabled"] = False
    resume_calls: list[str] = []

    def forbid_provider(_prompt: str, **kwargs):
        resume_calls.append(str(kwargs.get("call_label") or ""))
        raise AssertionError("resume must use accepted local checkpoints")

    resumed = build_companion(
        options, source_loader=lambda *args, **kwargs: bundle,
        llm=forbid_provider, compiler=compiler,
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert resumed["ok"], resumed
    assert resume_calls == []
    assert json.loads(ledger_path.read_text())["accepted_chain_sha256"] == accepted_chain


def test_chaptered_interactive_build_runs_only_first_chapter(tmp_path: Path) -> None:
    blocks = [
        {"block_id": "c1", "type": "section", "title": "Chapter One"},
        {"block_id": "p1", "type": "text", "text": "Energy is conserved."},
        {"block_id": "c2", "type": "section", "title": "Chapter Two"},
        {"block_id": "p2", "type": "text", "text": "Momentum is conserved."},
        {"block_id": "idx", "type": "text", "source_role": "index", "text": "Energy, 1"},
    ]
    document = {
        "schema_version": "arc.paper.document.v2", "front_matter": {}, "blocks": blocks,
        "equations": [], "figures": [], "tables": [], "assets": [], "links": [],
        "bibliography": [], "integrity": {"status": "complete", "document_hash": "chapters"},
    }
    structure = {"document_kind": "book", "index_block_ids": ["idx"], "chapters": [
        {"title": "Chapter One", "block_ids": ["c1", "p1"]},
        {"title": "Chapter Two", "block_ids": ["c2", "p2"]},
    ]}
    bundle = SourceBundle(
        paper_id="local:book", document=document,
        parsed={"paper_id": "local:book", "document": document, "structure": structure,
                "index_entries": {"schema_version": "arc.paper.index_entries.v1",
                                  "entries": [{"term": "Energy", "pages": [1]}]}},
        metadata={"title": "Book"}, references=[], citers=[],
    )
    labels: list[str] = []
    result_calls: list[dict[str, object]] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"]); labels.append(label)
        if label.startswith("title-translation-"):
            return _title_response(prompt)
        if label.startswith("companion-index-glossary-"):
            return {"entries": [{"entry_id": "index-00001", "target": "能量", "explanation": "守恒量"}]}
        if label.startswith("companion-guide-"):
            return {"motivation": "理解守恒。", "main_content": "能量守恒。", "section_logic": None,
                    "prerequisites": None, "pedagogical_comparison": None,
                    "historical_context": [], "supplementary_reading": []}
        if label.startswith("companion-segmentation-"):
            return {"cut_after_ordinals": []}
        if label.startswith("companion-translation-"):
            assert "Chapter One" not in prompt
            return {"blocks": [{"block_id": "p1", "text": "能量守恒。"}]}
        if label.startswith("companion-annotation-"):
            assert "Chapter One" not in prompt
            return {"explanation": "解释守恒的意义。", "commentary": "", "prior_work": [],
                    "later_work": [], "context_claims": [], "evidence_ids": [], "key_points": [],
                    "source_notes": [], "evidence_requests": []}
        if label.startswith("companion-review-segment-"):
            return _segment_review_response(prompt)
        raise AssertionError(label)

    def result_llm(prompt: str, **kwargs):
        result_calls.append(dict(kwargs))
        value = llm(prompt, **kwargs)
        return SimpleNamespace(value=value, logical_receipt={"idempotency_key": kwargs["idempotency_key"]})

    result = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "run",
                     workers=4, stop_after_first_chapter=True),
        source_loader=lambda *args, **kwargs: bundle, llm=llm,
        result_llm=result_llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert result["status"] == "first_chapter_ready"
    assert not any("ch-0002" in label for label in labels)
    assert not (
        Path(result["data"]["checkpoint_dir"]) / "chapters" / "ch-0002"
    ).exists()
    lane_calls = [call for call in result_calls if ":translation:" in str(call.get("idempotency_key"))
                  or ":companion:" in str(call.get("idempotency_key"))]
    assert {call["session_key"] for call in lane_calls} == {
        "ch-0001:translation", "ch-0001:companion",
    }
    assert all(call["session_policy"] == "stateful" for call in lane_calls)
    assert all(call["progress_contract_scope"] == "session" for call in lane_calls)
    assert all(call["schema_formatter_enabled"] is False for call in lane_calls)
    assert len({call["idempotency_key"] for call in lane_calls}) == len(lane_calls)
    guide_calls = [call for call in result_calls if str(call.get("session_key", "")).endswith(":guide")]
    assert len(guide_calls) == 1
    assert guide_calls[0]["session_key"] == "ch-0001:guide"
    assert guide_calls[0]["session_policy"] == "stateful"
    assert guide_calls[0]["model_tier"] == "high"
    assert guide_calls[0]["env"]["ARC_CODEX_ALLOW_INTERNET"] == "true"
    assert guide_calls[0]["env"]["ARC_CLAUDE_ALLOW_INTERNET"] == "true"
    assert guide_calls[0]["env"]["ARC_LLM_INHERIT_HOST_TOOLS"] == "false"
    guide_ledger = json.loads(
        (Path(result["data"]["checkpoint_dir"]) / "chapters" / "ch-0001" / "guide-ledger.json").read_text()
    )
    assert guide_ledger["blocks"][0]["state"] == "accepted"
    tex = Path(result["data"]["output_tex"]).read_text(encoding="utf-8")
    assert tex.count("ARC-CHAPTER-GUIDE-BEGIN") == 1
    assert tex.index("Chapter One") < tex.index("章导读") < tex.index("Energy is conserved")
    assert "Energy, 1" not in tex
    assert (tmp_path / "run" / "state.json").is_file()
    assert Path(result["data"]["output_html"]).is_file()
    assert result["data"]["web_render_version"] == "arc.companion.web-render.v5"
    reader_final = json.loads(
        (Path(result["data"]["checkpoint_dir"]) / "reader-final.json").read_text()
    )
    assert reader_final["schema_version"] == "arc.companion.reader-final.v4"
    assert reader_final["final_overrides"]["status"] == "first_chapter_ready"
    freeze = json.loads(
        (Path(result["data"]["checkpoint_dir"]) / "first-chapter-freeze.json").read_text()
    )
    assert freeze["schema_version"] == "arc.companion.first-chapter-freeze.v3"
    assert freeze["translation_mode"] == "enabled"
    assert freeze["translation_sha256"] and freeze["annotation_sha256"]

    checkpoint = Path(result["data"]["checkpoint_dir"])
    commentary_ledger_path = (
        checkpoint / "chapters" / "ch-0001" / "companion-ledger.json"
    )
    commentary_chain = json.loads(
        commentary_ledger_path.read_text()
    )["accepted_chain_sha256"]
    annotation_checkpoints = list((checkpoint / "annotations").glob("*.json"))
    assert len(annotation_checkpoints) == 1
    annotation_checkpoints[0].unlink()
    active_state = json.loads((tmp_path / "run" / "state.json").read_text())
    active_state["status"] = "active"
    (tmp_path / "run" / "state.json").write_text(json.dumps(active_state))
    (checkpoint / "first-chapter-freeze.json").unlink()
    annotation_calls_before = len([
        label for label in labels if label.startswith("companion-annotation-")
    ])

    restored = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id, project_dir=tmp_path / "run",
            workers=4, stop_after_first_chapter=True,
        ),
        source_loader=lambda *args, **kwargs: bundle, llm=llm,
        result_llm=result_llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert restored["status"] == "first_chapter_ready"
    assert len([
        label for label in labels if label.startswith("companion-annotation-")
    ]) == annotation_calls_before
    assert json.loads(
        commentary_ledger_path.read_text()
    )["accepted_chain_sha256"] == commentary_chain

    freeze["guide_sha256"] = "tampered"
    freeze_path = Path(result["data"]["checkpoint_dir"]) / "first-chapter-freeze.json"
    freeze_path.write_text(json.dumps(freeze))
    labels.clear()
    rejected = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "run", workers=4),
        source_loader=lambda *args, **kwargs: bundle, llm=llm, result_llm=result_llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )
    assert not rejected["ok"]
    assert "confirmed first chapter changed" in rejected["error"]["message"]
    assert not any("ch-0002" in label for label in labels)


def test_chaptered_build_applies_read_only_legacy_migration_before_lanes(
    tmp_path: Path,
) -> None:
    blocks = [
        {"block_id": "c1", "type": "section", "title": "Chapter One"},
        {"block_id": "p1", "type": "text", "text": "Energy is conserved."},
        {"block_id": "c2", "type": "section", "title": "Chapter Two"},
        {"block_id": "p2", "type": "text", "text": "Momentum is conserved."},
    ]
    document = {
        "schema_version": "arc.paper.document.v2", "front_matter": {}, "blocks": blocks,
        "equations": [], "figures": [], "tables": [], "assets": [], "links": [],
        "bibliography": [], "integrity": {"status": "complete", "document_hash": "chapters"},
    }
    structure = {"document_kind": "book", "chapters": [
        {"title": "Chapter One", "block_ids": ["c1", "p1"]},
        {"title": "Chapter Two", "block_ids": ["c2", "p2"]},
    ]}
    bundle = SourceBundle(
        paper_id="local:book", document=document,
        parsed={"paper_id": "local:book", "document": document, "structure": structure,
                "source_hash": "chapters"},
        metadata={"title": "Book"}, references=[], citers=[],
    )
    legacy = tmp_path / "legacy"
    (legacy / "translations").mkdir(parents=True)
    prompt_hash = sha256_json({
        "prompt_version": pipeline.PROMPT_VERSION,
        "schema_version": pipeline.SCHEMA_VERSION,
        "workflow_version": pipeline.WORKFLOW_VERSION,
    })
    validator_hash = sha256_json({
        "validator_version": pipeline.LEGACY_MIGRATION_VALIDATOR_VERSION,
        "translation_retry_prompt_version": pipeline.TRANSLATION_RETRY_PROMPT_VERSION,
        "translation_token_repair_version": pipeline.TRANSLATION_TOKEN_REPAIR_VERSION,
    })
    metadata = {
        "source_hash": "chapters", "language": "zh-CN",
        "prompt_hash": prompt_hash, "validator_hash": validator_hash,
    }
    (legacy / "document.json").write_text(json.dumps({"source_hash": "chapters"}))
    (legacy / "migration-metadata.json").write_text(json.dumps(metadata))
    (legacy / "segmentation.json").write_text(json.dumps({
        "cuts": [2], "segments": [
            {"segment_id": "old-1", "block_ids": ["c1", "p1"]},
            {"segment_id": "old-2", "block_ids": ["c2", "p2"]},
        ],
    }))
    (legacy / "glossary.json").write_text(json.dumps({
        **metadata, "entries": [{"source": "Energy", "target": "能量"}],
    }))
    (legacy / "translations" / "old-1.json").write_text(json.dumps({
        "segment_id": "old-1", "translation": {"blocks": [
            {"block_id": "c1", "text": "第一章"},
            {"block_id": "p1", "text": "能量守恒。"},
        ]},
    }))
    # These generated layers are intentionally invalid: the migration reader must not open them.
    (legacy / "chapter-guide.json").write_text("not json")
    (legacy / "review.v4.json").write_text("not json")
    before = {path.relative_to(legacy): path.read_bytes() for path in legacy.rglob("*") if path.is_file()}
    labels: list[str] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"]); labels.append(label)
        if label.startswith("title-translation-"):
            return _title_response(prompt)
        if label.startswith("companion-guide-"):
            return {"motivation": None, "main_content": "能量守恒。", "section_logic": None,
                    "prerequisites": None, "pedagogical_comparison": None,
                    "historical_context": [], "supplementary_reading": []}
        if label.startswith("companion-annotation-"):
            return {"explanation": "解释。", "commentary": "", "prior_work": [],
                    "later_work": [], "context_claims": [], "evidence_ids": [], "key_points": [],
                    "source_notes": [], "evidence_requests": []}
        if label.startswith("companion-translation-"):
            assert "Chapter One" not in prompt
            return {"blocks": [{"block_id": "p1", "text": "能量守恒。"}]}
        if label.startswith("companion-review-segment-"):
            return _segment_review_response(prompt)
        raise AssertionError(f"migration should have avoided call: {label}")

    def result_llm(prompt: str, **kwargs):
        return SimpleNamespace(
            value=llm(prompt, **kwargs),
            logical_receipt={"idempotency_key": kwargs["idempotency_key"]},
        )

    result = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id, project_dir=tmp_path / "run", workers=2,
            stop_after_first_chapter=True, legacy_checkpoint=legacy,
        ),
        source_loader=lambda *args, **kwargs: bundle, llm=llm, result_llm=result_llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert result["status"] == "first_chapter_ready"
    assert not any("segmentation" in label or "glossary" in label for label in labels)
    assert [label for label in labels if label.startswith("companion-translation-")] == []
    checkpoint = Path(result["data"]["checkpoint_dir"])
    receipt = json.loads((checkpoint / "legacy-migration.json").read_text())
    assert receipt["read_only_source"] is True
    assert receipt["cuts"]["reused"]["ch-0001"] == []
    assert receipt["glossary"]["accepted"] is True
    assert receipt["translations"]["receipts"][0]["accepted"] is True
    assert receipt["translations"]["receipts"][0]["reason"] == "translation_revalidated"
    assert receipt["translations"]["receipts"][0]["dropped_structural_block_ids"] == ["c1"]
    assert receipt["never_migrated"] == ["tex", "pdf"]
    ledger = json.loads(
        (checkpoint / "chapters" / "ch-0001" / "translation-ledger.json").read_text()
    )
    assert ledger["blocks"][0]["submission_state"] == "not_submitted"
    assert ledger["blocks"][0]["logical_receipt"] == {
        "kind": "legacy_migration", "provider_calls": 0,
    }
    assert not (checkpoint / "chapters" / "ch-0001" / "chapter-glossary.json").exists()
    migrated_segment = json.loads(
        (checkpoint / "chapters" / "ch-0001" / "segmentation.json").read_text()
    )["segments"][0]
    migrated_segment = {**migrated_segment, "chapter_id": "ch-0001",
                        "segment_id": "ch-0001.seg-0001"}
    translated = ledger["blocks"][0]["translation"]["blocks"]
    assert [item["block_id"] for item in translated] == ["p1"]
    assert {path.relative_to(legacy): path.read_bytes() for path in legacy.rglob("*") if path.is_file()} == before


def test_chaptered_build_projects_accepted_store_translation_without_legacy_option(
    tmp_path: Path,
) -> None:
    blocks = [
        {"block_id": "c1", "type": "section", "title": "Chapter One"},
        {"block_id": "p1", "type": "text", "text": "Energy is conserved."},
    ]
    document = {
        "schema_version": "arc.paper.document.v2", "front_matter": {}, "blocks": blocks,
        "equations": [], "figures": [], "tables": [], "assets": [], "links": [],
        "bibliography": [], "integrity": {"status": "complete", "document_hash": "chapters"},
    }
    bundle = SourceBundle(
        paper_id="local:book", document=document,
        parsed={
            "paper_id": "local:book", "document": document, "source_hash": "chapters",
            "structure": {"document_kind": "book", "chapters": [{
                "title": "Chapter One", "block_ids": ["c1", "p1"],
            }]},
        },
        metadata={"title": "Book"}, references=[], citers=[],
    )
    project = tmp_path / "run"
    old_checkpoint = tmp_path / "accepted-checkpoint"
    old_chapter = old_checkpoint / "chapters" / "ch-0001"
    old_chapter.mkdir(parents=True)
    (old_checkpoint / "migration-metadata.json").write_text(json.dumps({
        "source_hash": "chapters", "language": "zh-CN",
    }))
    (old_chapter / "segmentation.json").write_text(json.dumps({
        "segments": [{
            "segment_id": "seg-0001", "block_ids": ["c1", "p1"],
        }],
    }))
    accepted_output = {"blocks": [
        {"block_id": "c1", "text": "第一章"},
        {"block_id": "p1", "text": "能量守恒。"},
    ]}
    input_sha = canonical_sha256({"legacy": "translation"})
    output_sha = canonical_sha256(accepted_output)
    empty_chain = hashlib.sha256(b"").hexdigest()
    block = {
        "segment_id": "seg-0001", "state": "accepted", "generation": 1,
        "input_sha256": input_sha, "output_sha256": output_sha,
        "predecessor_accepted_chain_sha256": empty_chain,
        "validation_receipt": {"legacy": True},
        "logical_receipt": {"call_id": "old-call"},
    }
    block["accepted_chain_sha256"] = hashlib.sha256(json.dumps({
        "predecessor": empty_chain, "segment_id": "seg-0001",
        "input_sha256": input_sha, "output_sha256": output_sha, "generation": 1,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    AcceptedArtifactStore(project).put_accepted(
        kind="translation", semantic_input_sha256=input_sha,
        recipe_sha256=canonical_sha256({"legacy": "recipe"}),
        contract_version=pipeline.SCHEMA_VERSION, output=accepted_output,
        ledger_block=block,
        provider_receipt={
            "provider": "stub", "model": "old", "call_id": "old-call", "usage": {},
        },
        provenance={
            "checkpoint_dir": str(old_checkpoint),
            "ledger": str(old_chapter / "translation-ledger.json"),
        },
    )
    labels: list[str] = []
    allow_translation_generation = False

    def llm(_prompt: str, **kwargs):
        label = str(kwargs["call_label"]); labels.append(label)
        if label.startswith("title-translation-"):
            return _title_response(_prompt)
        if "translation" in label:
            if not allow_translation_generation:
                raise AssertionError(f"translation call was submitted: {label}")
            return {"blocks": [{"block_id": "p1", "text": "重新生成的翻译。"}]}
        if label.startswith("companion-glossary-"):
            return {"entries": []}
        if label.startswith("companion-segmentation-"):
            return {"cut_after_ordinals": []}
        if label.startswith("companion-guide-"):
            return {
                "motivation": None, "main_content": "Energy conservation.",
                "section_logic": None, "prerequisites": None,
                "pedagogical_comparison": None, "historical_context": [],
                "supplementary_reading": [],
            }
        if label.startswith("companion-annotation-"):
            return {
                "explanation": "Commentary.", "commentary": "", "prior_work": [],
                "later_work": [], "context_claims": [], "evidence_ids": [],
                "key_points": [], "source_notes": [], "evidence_requests": [],
            }
        if label.startswith("companion-review-segment-"):
            return _segment_review_response(_prompt)
        raise AssertionError(label)

    def result_llm(prompt: str, **kwargs):
        return SimpleNamespace(
            value=llm(prompt, **kwargs),
            logical_receipt={"idempotency_key": kwargs["idempotency_key"]},
        )

    result = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id, project_dir=project, workers=1,
            stop_after_first_chapter=True,
        ),
        source_loader=lambda *args, **kwargs: bundle, llm=llm, result_llm=result_llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert result["status"] == "first_chapter_ready"
    assert not any(label.startswith("companion-translation-") for label in labels)
    checkpoint = Path(result["data"]["checkpoint_dir"])
    ledger = json.loads(
        (checkpoint / "chapters" / "ch-0001" / "translation-ledger.json").read_text()
    )
    accepted = ledger["blocks"][0]
    assert accepted["submission_state"] == "not_submitted"
    assert ledger["migration_source"] == "accepted_artifact_store"
    assert accepted["translation"] == {
        "blocks": [{"block_id": "p1", "text": "能量守恒。"}],
    }
    migration = json.loads((checkpoint / "legacy-migration.json").read_text())
    assert migration["source_kind"] == "accepted_artifact_store"
    assert migration["translations"]["receipts"][0]["dropped_structural_block_ids"] == ["c1"]

    guide_ledger_path = checkpoint / "chapters" / "ch-0001" / "guide-ledger.json"
    old_guide_ledger = json.loads(guide_ledger_path.read_text())
    old_generation = old_guide_ledger["generation"]
    current_recipe = lane_recipe_sha256(
        "guide", prompt=pipeline.CHAPTER_GUIDE_VERSION, model=None,
        tier=pipeline.ANNOTATION_TIER,
        access_recipe={
            "provider": "auto", "allow_internet": True,
            "inherit_host_tools": False,
        },
    )
    replacement_guide = {
        "schema_version": pipeline.CHAPTER_GUIDE_VERSION,
        "source_sha256": "replacement-source",
        "chapter_id": "ch-0001",
        "motivation": None,
        "main_content": "A different accepted guide.",
        "section_logic": None,
        "prerequisites": None,
        "pedagogical_comparison": None,
        "historical_context": [],
        "supplementary_reading": [],
    }
    replacement_output_sha = canonical_sha256(replacement_guide)
    replacement_block = {
        "segment_id": "ch-0001:guide", "state": "accepted", "generation": 1,
        "input_sha256": old_guide_ledger["blocks"][0]["input_sha256"],
        "output_sha256": replacement_output_sha,
        "predecessor_accepted_chain_sha256": empty_chain,
        "validation_receipt": {
            "local_validation": True, "recipe_sha256": current_recipe,
        },
        "logical_receipt": {"call_id": "replacement-guide"},
    }
    replacement_block["accepted_chain_sha256"] = hashlib.sha256(json.dumps({
        "predecessor": empty_chain, "segment_id": "ch-0001:guide",
        "input_sha256": replacement_block["input_sha256"],
        "output_sha256": replacement_output_sha, "generation": 1,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    AcceptedArtifactStore(project).put_accepted(
        kind="guide",
        semantic_input_sha256=replacement_block["input_sha256"],
        recipe_sha256=current_recipe,
        contract_version=pipeline.CHAPTER_GUIDE_VERSION,
        output=replacement_guide,
        ledger_block=replacement_block,
        provider_receipt={
            "provider": "stub", "model": "replacement",
            "call_id": "replacement-guide", "usage": {},
        },
        provenance={"checkpoint_dir": str(old_checkpoint), "chapter_id": "ch-0001"},
    )
    _registered_guide, registered_digest = read_registered_lane_ledger(
        checkpoint, guide_ledger_path,
    )

    def invalidate_guide_recipe(ledger):
        ledger["blocks"][0]["validation_receipt"]["recipe_sha256"] = "0" * 64
        return ledger

    mutate_registered_lane_ledger(
        checkpoint,
        guide_ledger_path,
        expected_sha256=registered_digest,
        mutate=invalidate_guide_recipe,
    )
    active_state = json.loads((project / "state.json").read_text())
    active_state["status"] = "active"
    (project / "state.json").write_text(json.dumps(active_state))
    (checkpoint / "first-chapter-freeze.json").unlink()
    guide_calls_before = len([label for label in labels if label.startswith("companion-guide-")])

    resumed = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id, project_dir=project, workers=1,
            stop_after_first_chapter=True,
        ),
        source_loader=lambda *args, **kwargs: bundle, llm=llm, result_llm=result_llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert resumed.get("status") in {"first_chapter_ready", "complete"}, resumed
    guide_calls_after = len([label for label in labels if label.startswith("companion-guide-")])
    assert guide_calls_after == guide_calls_before
    rebound_ledger = json.loads(guide_ledger_path.read_text())
    rebound = rebound_ledger["blocks"][0]
    assert rebound_ledger["generation"] == old_generation + 1
    assert rebound["validation_receipt"]["recipe_sha256"] == current_recipe
    guide = json.loads(
        (checkpoint / "chapters" / "ch-0001" / "chapter-guide.json").read_text()
    )
    assert guide == replacement_guide
    assert rebound["output_sha256"] == canonical_sha256(guide)

    allow_translation_generation = True
    regeneration_state = json.loads((project / "state.json").read_text())
    regeneration_state["status"] = "active"
    (project / "state.json").write_text(json.dumps(regeneration_state))
    (checkpoint / "first-chapter-freeze.json").unlink()
    translation_calls_before = len([
        label for label in labels if label.startswith("companion-translation-")
    ])
    title_calls_before = len([
        label for label in labels if label.startswith("title-translation-")
    ])
    regenerated = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id, project_dir=project, workers=1,
            stop_after_first_chapter=True, regenerate_lanes=("translation",),
        ),
        source_loader=lambda *args, **kwargs: bundle, llm=llm, result_llm=result_llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert regenerated.get("status") in {"first_chapter_ready", "complete"}, regenerated
    translation_calls = [
        label for label in labels if label.startswith("companion-translation-")
    ]
    assert len(translation_calls) == translation_calls_before + 1
    assert len([
        label for label in labels if label.startswith("title-translation-")
    ]) == title_calls_before + 1
    regenerated_ledger = json.loads(
        (checkpoint / "chapters" / "ch-0001" / "translation-ledger.json").read_text()
    )
    assert regenerated_ledger["blocks"][0]["submission_state"] == "submitted"
    assert "translation" not in json.loads(
        (checkpoint / "legacy-migration.json").read_text()
    ).get("source_kind", "")


def test_restart_generation_rotates_session_and_invalidates_suffix(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "run"
    checkpoint = project / ".arc-companion" / "checkpoints" / ("3" * 64)
    ledger_path = checkpoint / "chapters" / "ch-0001" / "translation-ledger.json"
    initialize_lane_ledger(
        ledger_path, chapter_id="ch-0001", lane="translation", segment_ids=["s1", "s2"],
    )
    mark_needs_supervision(
        ledger_path, segment_id="s1", reason="unknown submission",
        recovery_context={"resumable": True, "native_session_id": "native-1"},
    )
    manager = LLMSessionManager(checkpoint / "sessions")
    manager.get_or_create(
        key="ch-0001:translation", provider="codex", model="m",
        runtime_fingerprint="runtime",
    )
    manager.update_native_session_id("ch-0001:translation", "native-1")
    project.mkdir(parents=True, exist_ok=True)
    (project / "state.json").write_text(json.dumps({
        "status": "needs_supervision", "fingerprint": "3" * 64, "checkpoint_dir": str(checkpoint),
        "recovery_options": {"paper_id": "local:book", "workers": 1},
    }))
    captured = {}
    monkeypatch.setattr(
        pipeline, "build_companion",
        lambda options: captured.setdefault("options", options) or {"ok": True},
    )

    result = pipeline.resume_companion(
        project, action="restart-generation", confirm_possible_duplicate_charge=True,
    )

    assert result is captured["options"]  # the stub returns the saved options object
    ledger = json.loads(ledger_path.read_text())
    assert ledger["generation"] == 2
    assert ledger["blocks"][0] == {
        "segment_id": "s1", "state": "prepared",
        "submission_state": "not_submitted", "generation": 2,
    }
    assert LLMSessionManager(checkpoint / "sessions").get_existing(
        "ch-0001:translation"
    ).native_session_id is None


def test_restart_generation_transaction_replays_after_crash_between_session_and_ledger(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "run"
    checkpoint = project / ".arc-companion" / "checkpoints" / ("3" * 64)
    ledger_path = checkpoint / "chapters" / "ch-0001" / "translation-ledger.json"
    initialize_lane_ledger(
        ledger_path, chapter_id="ch-0001", lane="translation", segment_ids=["s1"]
    )
    mark_needs_supervision(
        ledger_path,
        segment_id="s1",
        reason="unknown submission",
        recovery_context={"resumable": True, "native_session_id": "native-1"},
    )
    manager = LLMSessionManager(checkpoint / "sessions")
    manager.get_or_create(
        key="ch-0001:translation", provider="codex", model="m",
        runtime_fingerprint="runtime",
    )
    project.mkdir(parents=True, exist_ok=True)
    (project / "state.json").write_text(json.dumps({
        "status": "needs_supervision", "fingerprint": "3" * 64, "checkpoint_dir": str(checkpoint),
        "recovery_options": {"paper_id": "local:book", "workers": 1},
    }))
    real_invalidate = pipeline.invalidate_suffix
    monkeypatch.setattr(
        pipeline,
        "invalidate_suffix",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("crash after rotate")),
    )

    with pytest.raises(RuntimeError, match="crash after rotate"):
        pipeline.resume_companion(
            project,
            action="restart-generation",
            confirm_possible_duplicate_charge=True,
        )
    assert LLMSessionManager(checkpoint / "sessions").get_existing(
        "ch-0001:translation"
    ).generation == 2

    monkeypatch.setattr(pipeline, "invalidate_suffix", real_invalidate)
    monkeypatch.setattr(pipeline, "build_companion", lambda _options: {"ok": True})
    result = pipeline.resume_companion(
        project,
        action="restart-generation",
        confirm_possible_duplicate_charge=True,
    )

    assert result["ok"] is True
    assert LLMSessionManager(checkpoint / "sessions").get_existing(
        "ch-0001:translation"
    ).generation == 2
    assert json.loads(ledger_path.read_text())["generation"] == 2


def test_resume_native_passes_one_shot_call_authorization_to_build(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "run"
    checkpoint = project / ".arc-companion" / "checkpoints" / ("3" * 64)
    ledger_path = checkpoint / "chapters" / "ch-0001" / "translation-ledger.json"
    initialize_lane_ledger(
        ledger_path, chapter_id="ch-0001", lane="translation", segment_ids=["s1"],
    )
    logical_key = "ch-0001:translation:companion-translation-s1:generation-1"
    mark_needs_supervision(
        ledger_path,
        segment_id="s1",
        reason="unknown submitted response",
        recovery_context={
            "idempotency_key": logical_key,
            "resumable": True,
            "native_session_id": "native-1",
            "submission_state": "submitted",
        },
    )
    manager = LLMSessionManager(checkpoint / "sessions")
    manager.get_or_create(
        key="ch-0001:translation", provider="codex", model="m",
        runtime_fingerprint="runtime",
    )
    manager.update_native_session_id("ch-0001:translation", "native-1")
    project.mkdir(parents=True, exist_ok=True)
    (project / "state.json").write_text(json.dumps({
        "status": "needs_supervision",
        "fingerprint": "3" * 64,
        "checkpoint_dir": str(checkpoint),
        "recovery_options": {"paper_id": "local:book", "workers": 1},
    }))
    captured = {}

    def resumed_build(options):
        captured["options"] = options
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", resumed_build)

    result = pipeline.resume_companion(project, action="resume-native")

    assert result["ok"] is True
    assert captured["options"].supervised_native_resume_keys == (logical_key,)
    assert json.loads(ledger_path.read_text())["needs_supervision"] is not None
    transaction = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert transaction["schema_version"] == "arc.companion.resume-transaction.v3"
    assert transaction["status"] == "continuation_failed"
    assert transaction["entries"][0]["status"] == "reconciling"
    assert manager.get_existing("ch-0001:translation").native_session_id == "native-1"


def _missing_store_native_resume_fixture(tmp_path: Path) -> dict[str, object]:
    project = tmp_path / "run"
    checkpoint = project / ".arc-companion" / "checkpoints" / ("3" * 64)
    session_key = "ch-0008:guide"
    logical_key = "ch-0008:guide:companion-guide-ch-0008:generation-1"
    artifact_dir = checkpoint / "chapters" / "ch-0008" / "llm"
    manager = LLMSessionManager(checkpoint / "sessions")
    manager.get_or_create(
        key=session_key,
        provider="codex-cli",
        model="m",
        runtime_fingerprint="runtime",
    )
    call_path, identity = checkpoint_path(
        artifact_dir,
        prompt="guide prompt",
        schema={"type": "object"},
        provider="codex-cli",
        model="m",
        call_label="companion-guide-ch-0008",
        session_policy="stateful",
        session_key=session_key,
        session_turn=0,
        runtime_fingerprint="runtime",
        idempotency_key=logical_key,
        generation=1,
    )
    prepared = prepare_call(call_path, identity=identity)
    record_submitted(prepared)
    ProgressJournal(
        artifact_dir=artifact_dir,
        call_label="companion-guide-ch-0008",
        provider="codex-cli",
        callback=None,
        identity={
            "idempotency_key": logical_key,
            "session_key": session_key,
            "generation": 1,
            "model": "m",
            "runtime_fingerprint": "runtime",
        },
    )({
        "event": "provider_progress",
        "native_session_id": "native-progress",
        "resumable": True,
    })
    prepared.release_lock()
    recovery = read_recovery_context(
        artifact_dir,
        idempotency_key=logical_key,
        session_manager=manager,
        session_key=session_key,
    )
    ledger_path = checkpoint / "chapters" / "ch-0008" / "guide-ledger.json"
    initialize_lane_ledger(
        ledger_path,
        chapter_id="ch-0008",
        lane="guide",
        segment_ids=["ch-0008:guide"],
    )
    mark_needs_supervision(
        ledger_path,
        segment_id="ch-0008:guide",
        reason="interrupted after provider established a session",
        recovery_context=pipeline._recovery_context_json(recovery),
    )
    project.mkdir(parents=True, exist_ok=True)
    (project / "state.json").write_text(json.dumps({
        "status": "needs_supervision",
        "fingerprint": "3" * 64,
        "checkpoint_dir": str(checkpoint),
        "recovery_options": {"paper_id": "local:book", "workers": 1},
    }))
    return {
        "project": project,
        "checkpoint": checkpoint,
        "ledger_path": ledger_path,
        "manager": manager,
        "session_key": session_key,
        "logical_key": logical_key,
    }


def test_resume_native_restores_valid_progress_session_missing_from_store(
    tmp_path: Path, monkeypatch,
) -> None:
    fixture = _missing_store_native_resume_fixture(tmp_path)
    captured = {}

    def resumed_build(options):
        captured["options"] = options
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", resumed_build)

    result = pipeline.resume_companion(fixture["project"], action="resume-native")

    assert result["ok"] is True
    manager = LLMSessionManager(fixture["checkpoint"] / "sessions")
    assert manager.get_existing(fixture["session_key"]).native_session_id == "native-progress"
    assert captured["options"].supervised_native_resume_keys == (fixture["logical_key"],)
    assert json.loads(fixture["ledger_path"].read_text())["needs_supervision"] is not None


def test_resume_native_recognizes_failed_state_with_supervised_lane(
    tmp_path: Path, monkeypatch,
) -> None:
    fixture = _missing_store_native_resume_fixture(tmp_path)
    state_path = fixture["project"] / "state.json"
    state = json.loads(state_path.read_text())
    state["status"] = "failed"
    state_path.write_text(json.dumps(state))
    captured = {}

    def resumed_build(options):
        captured["options"] = options
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", resumed_build)

    result = pipeline.resume_companion(fixture["project"], action="resume-native")

    assert result["ok"] is True
    assert captured["options"].supervised_native_resume_keys == (fixture["logical_key"],)


def test_reconstructs_cleared_unresolved_authorization_from_durable_state(
    tmp_path: Path,
) -> None:
    fixture = _missing_store_native_resume_fixture(tmp_path)
    manager = LLMSessionManager(fixture["checkpoint"] / "sessions")
    manager.update_native_session_id(fixture["session_key"], "native-progress")
    pipeline.clear_needs_supervision(fixture["ledger_path"])

    automatic = pipeline._reconstruct_unresolved_native_resume_contexts(
        fixture["checkpoint"], session_manager=manager, excluded_keys=set(),
    )
    recovered = pipeline._reconstruct_unresolved_native_resume_contexts(
        fixture["checkpoint"], session_manager=manager,
        # Bare keys are intentionally insufficient for exclusion; only the
        # exact five-field recovery identity may deduplicate this call.
        excluded_keys={fixture["logical_key"]},
        allow_explicit_legacy=True,
    )

    assert automatic == []
    assert recovered == [{
        "session_key": fixture["session_key"],
        "idempotency_key": fixture["logical_key"],
        "provider": "codex-cli",
        "model": "m",
        "runtime_fingerprint": "runtime",
        "generation": 1,
        "native_session_id_to_restore": None,
        "ledger_path": str(fixture["ledger_path"]),
        "segment_id": "ch-0008:guide",
        "reconstructed_from_durable_state": True,
    }]


def test_resume_discovers_submitted_checkpoint_before_ledger_progress(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "barrier-crash"
    checkpoint = project / ".arc-companion" / "checkpoints" / ("3" * 64)
    chapter_id = "ch-0042"
    session_key = f"{chapter_id}:translation"
    segment_id = f"{chapter_id}.seg-0003"
    logical_key = f"{session_key}:companion-translation-{segment_id}:generation-1"
    ledger_path = checkpoint / "production" / "lanes" / chapter_id / "translation-ledger.json"
    ledger = initialize_lane_ledger(
        ledger_path, chapter_id=chapter_id, lane="translation",
        segment_ids=[segment_id],
    )
    assert ledger["blocks"][0]["state"] == "prepared"
    manager = LLMSessionManager(checkpoint / "sessions")
    manager.get_or_create(
        key=session_key, provider="codex-cli", model="m",
        runtime_fingerprint="runtime",
    )
    manager.update_native_session_id(session_key, "native-existing")
    artifact_dir = checkpoint / "production" / "provider-calls" / "nested" / segment_id
    call_path, identity = checkpoint_path(
        artifact_dir, prompt="translate", schema={"type": "object"},
        provider="codex-cli", model="m", call_label=f"call-{segment_id}",
        session_policy="stateful", session_key=session_key, session_turn=0,
        runtime_fingerprint="runtime", idempotency_key=logical_key, generation=1,
    )
    prepared = prepare_call(call_path, identity=identity)
    record_submitted(prepared)
    prepared.release_lock()
    assert not (artifact_dir / "progress.jsonl").exists()
    project.mkdir(parents=True, exist_ok=True)
    (project / "state.json").write_text(json.dumps({
        "status": "needs_supervision", "fingerprint": "3" * 64, "checkpoint_dir": str(checkpoint),
        "recovery_options": {"paper_id": "local:book", "workers": 1},
    }))
    captured = {}

    def resumed_build(options):
        captured["keys"] = options.supervised_native_resume_keys
        pipeline.mark_response_received(ledger_path, segment_id=segment_id)
        pipeline.advance_block(ledger_path, segment_id=segment_id, state="schema_valid")
        pipeline.advance_block(ledger_path, segment_id=segment_id, state="invariant_valid")
        pipeline.advance_block(
            ledger_path, segment_id=segment_id, state="accepted",
            input_sha256="input", output_sha256="output",
        )
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", resumed_build)

    result = pipeline.resume_companion(project, action="resume-native")

    assert result["ok"] is True
    assert captured["keys"] == (logical_key,)
    final_ledger = json.loads(ledger_path.read_text())
    assert final_ledger["blocks"][0]["state"] == "accepted"
    assert final_ledger["needs_supervision"] is None
    transaction = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert transaction["status"] == "complete"
    assert transaction["entries"][0]["status"] == "resolved"
    assert transaction["entries"][0]["idempotency_key"] == logical_key


def test_v1_complete_migration_recovers_all_17_pending_calls(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "v1-seventeen"
    checkpoint = project / ".arc-companion" / "checkpoints" / ("3" * 64)
    manager = LLMSessionManager(checkpoint / "sessions")
    v1_entries = []
    v1_contexts = []
    logical_keys = []
    for index in range(1, 18):
        chapter_id = f"ch-{index:04d}"
        session_key = f"{chapter_id}:translation"
        segment_id = f"{chapter_id}.seg-0001"
        key = f"{session_key}:companion-translation-{segment_id}:generation-1"
        logical_keys.append(key)
        ledger_path = checkpoint / "production" / chapter_id / "translation-ledger.json"
        initialize_lane_ledger(
            ledger_path, chapter_id=chapter_id, lane="translation",
            segment_ids=[segment_id],
        )
        manager.get_or_create(
            key=session_key, provider="codex-cli", model="m",
            runtime_fingerprint="runtime",
        )
        manager.update_native_session_id(session_key, f"native-{index}")
        artifact_dir = checkpoint / "production" / chapter_id / "requests" / segment_id
        call_path, identity = checkpoint_path(
            artifact_dir, prompt="resume", schema={"type": "object"},
            provider="codex-cli", model="m", call_label=f"call-{index}",
            session_policy="stateful", session_key=session_key, session_turn=0,
            runtime_fingerprint="runtime", idempotency_key=key, generation=1,
        )
        prepared = prepare_call(call_path, identity=identity)
        record_submitted(prepared)
        ProgressJournal(
            artifact_dir=artifact_dir, call_label=f"call-{index}",
            provider="codex-cli", callback=None,
            identity={
                "idempotency_key": key, "session_key": session_key,
                "generation": 1, "model": "m", "runtime_fingerprint": "runtime",
            },
        )({"event": "provider_progress", "native_session_id": f"native-{index}"})
        prepared.release_lock()
        recovery = read_recovery_context(
            artifact_dir, idempotency_key=key,
            session_manager=manager, session_key=session_key,
        )
        mark_needs_supervision(
            ledger_path, segment_id=segment_id, reason="interrupted",
            recovery_context=pipeline._recovery_context_json(recovery),
        )
        if index <= 8:
            v1_entries.append({
                "ledger_path": str(ledger_path), "session_key": session_key,
                "segment_id": segment_id, "initial_generation": 1,
                "target_generation": 1, "status": "applied",
            })
            v1_contexts.append({
                "session_key": session_key, "idempotency_key": key,
                "provider": "codex-cli", "model": "m",
                "runtime_fingerprint": "runtime", "generation": 1,
                "native_session_id_to_restore": None,
            })
        else:
            pipeline.clear_needs_supervision(ledger_path)
    project.mkdir(parents=True, exist_ok=True)
    (project / "state.json").write_text(json.dumps({
        "status": "needs_supervision", "fingerprint": "3" * 64, "checkpoint_dir": str(checkpoint),
        "recovery_options": {"paper_id": "local:book", "workers": 1},
    }))
    journal = project / ".arc-companion" / "resume-transaction.json"
    journal.parent.mkdir(exist_ok=True)
    journal.write_text(json.dumps({
        "schema_version": "arc.companion.resume-transaction.v1",
        "action": "resume-native", "status": "complete",
        "recovery_options": {"paper_id": "local:book", "workers": 1},
        "entries": v1_entries, "native_resume_contexts": v1_contexts,
    }))
    captured = {}
    monkeypatch.setattr(
        pipeline, "build_companion",
        lambda options: captured.setdefault("options", options)
        and {"ok": False, "status": "needs_supervision"},
    )

    result = pipeline.resume_companion(project, action="resume-native")

    assert result["ok"] is False
    migrated = json.loads(journal.read_text())
    assert migrated["schema_version"] == "arc.companion.resume-transaction.v3"
    assert migrated["status"] == "continuation_failed"
    assert len(migrated["entries"]) == 17
    assert {item["idempotency_key"] for item in migrated["entries"]} == set(logical_keys)
    assert captured["options"].supervised_native_resume_keys == tuple(logical_keys)


def test_resume_native_isolates_conflicting_entry_and_resolves_other_lane(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "mixed-native"
    checkpoint = project / ".arc-companion" / "checkpoints" / ("3" * 64)
    manager = LLMSessionManager(checkpoint / "sessions")
    ledgers: dict[str, Path] = {}
    keys: dict[str, str] = {}
    for chapter_id in ("ch-good", "ch-conflict"):
        session_key = f"{chapter_id}:translation"
        segment_id = f"{chapter_id}.seg-0001"
        key = f"{session_key}:companion-translation-{segment_id}:generation-1"
        keys[chapter_id] = key
        ledger_path = checkpoint / "production" / chapter_id / "translation-ledger.json"
        ledgers[chapter_id] = ledger_path
        initialize_lane_ledger(
            ledger_path, chapter_id=chapter_id, lane="translation",
            segment_ids=[segment_id],
        )
        manager.get_or_create(
            key=session_key, provider="codex-cli", model="m",
            runtime_fingerprint="runtime",
        )
        manager.update_native_session_id(session_key, f"native-{chapter_id}")
        mark_needs_supervision(
            ledger_path, segment_id=segment_id, reason="interrupted",
            recovery_context={
                "idempotency_key": key, "resumable": True,
                "native_session_id": f"native-{chapter_id}",
                "submission_state": "submitted",
                "provider": (
                    "wrong-provider" if chapter_id == "ch-conflict" else "codex-cli"
                ),
                "model": "m", "runtime_fingerprint": "runtime",
                "session_key": session_key, "generation": 1,
            },
        )
    project.mkdir(parents=True, exist_ok=True)
    (project / "state.json").write_text(json.dumps({
        "status": "needs_supervision", "fingerprint": "3" * 64, "checkpoint_dir": str(checkpoint),
        "recovery_options": {"paper_id": "local:book", "workers": 2},
    }))
    calls = []

    def resumed_build(options):
        calls.append(options.supervised_native_resume_keys)
        chapter_id = "ch-good" if len(calls) == 1 else "ch-later"
        ledger_path = ledgers[chapter_id]
        segment_id = f"{chapter_id}.seg-0001"
        pipeline.mark_response_received(ledger_path, segment_id=segment_id)
        pipeline.advance_block(ledger_path, segment_id=segment_id, state="schema_valid")
        pipeline.advance_block(ledger_path, segment_id=segment_id, state="invariant_valid")
        pipeline.advance_block(
            ledger_path, segment_id=segment_id, state="accepted",
            input_sha256="input", output_sha256="output",
        )
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", resumed_build)

    result = pipeline.resume_companion(project, action="resume-native")

    assert result["ok"] is True
    assert calls == [(keys["ch-good"],)]
    assert json.loads(ledgers["ch-good"].read_text())["needs_supervision"] is None
    assert json.loads(ledgers["ch-conflict"].read_text())["needs_supervision"] is not None
    transaction = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert transaction["status"] == "continuation_failed"
    by_session = {item["session_key"]: item for item in transaction["entries"]}
    assert by_session["ch-good:translation"]["status"] == "resolved"
    conflict = by_session["ch-conflict:translation"]
    assert conflict["status"] == "pending"
    assert "provider" in conflict["blocking_reason"]
    assert conflict["recovery_context"]["provider"] == "wrong-provider"
    assert int(conflict["target_generation"]) == 1

    # A later pending call is still reconciled even though the conflicting
    # entry remains in the same continuation-failed transaction.
    later = "ch-later"
    later_session = f"{later}:translation"
    later_segment = f"{later}.seg-0001"
    later_key = f"{later_session}:companion-translation-{later_segment}:generation-1"
    later_ledger = checkpoint / "new-production-layout" / later / "translation-ledger.json"
    ledgers[later] = later_ledger
    initialize_lane_ledger(
        later_ledger, chapter_id=later, lane="translation", segment_ids=[later_segment],
    )
    manager.get_or_create(
        key=later_session, provider="codex-cli", model="m", runtime_fingerprint="runtime",
    )
    manager.update_native_session_id(later_session, "native-later")
    mark_needs_supervision(
        later_ledger, segment_id=later_segment, reason="interrupted",
        recovery_context={
            "idempotency_key": later_key, "resumable": True,
            "native_session_id": "native-later", "submission_state": "submitted",
            "provider": "codex-cli", "model": "m", "runtime_fingerprint": "runtime",
            "session_key": later_session, "generation": 1,
        },
    )

    second = pipeline.resume_companion(project, action="resume-native")

    assert second["ok"] is True
    assert later_key in calls[1]
    assert keys["ch-conflict"] not in calls[1]
    assert json.loads(later_ledger.read_text())["needs_supervision"] is None
    second_transaction = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert second_transaction["status"] == "continuation_failed"
    assert len(second_transaction["entries"]) == 3


def test_resume_native_applies_many_supervised_ledgers_idempotently(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "many-supervised"
    checkpoint = project / ".arc-companion" / "checkpoints" / ("3" * 64)
    manager = LLMSessionManager(checkpoint / "sessions")
    logical_keys: list[str] = []
    for index in range(1, 18):
        chapter_id = f"ch-{index:04d}"
        session_key = f"{chapter_id}:translation"
        segment_id = f"{chapter_id}.seg-0001"
        logical_key = (
            f"{session_key}:companion-translation-{segment_id}:generation-1"
        )
        logical_keys.append(logical_key)
        ledger_path = checkpoint / "chapters" / chapter_id / "translation-ledger.json"
        initialize_lane_ledger(
            ledger_path, chapter_id=chapter_id, lane="translation",
            segment_ids=[segment_id],
        )
        pipeline._guarded_mark_transport_state(
            ledger_path,
            checkpoint_dir=checkpoint,
            session_key=session_key,
            logical_unit=segment_id,
            idempotency_key=logical_key,
        )
        mark_needs_supervision(
            ledger_path,
            segment_id=segment_id,
            reason="unknown submitted response",
            recovery_context={
                "idempotency_key": logical_key,
                "resumable": True,
                "native_session_id": f"native-{index}",
                "submission_state": "submitted",
            },
        )
        manager.get_or_create(
            key=session_key, provider="codex", model="m",
            runtime_fingerprint="runtime",
        )
        manager.update_native_session_id(session_key, f"native-{index}")
    project.mkdir(parents=True, exist_ok=True)
    (project / "state.json").write_text(json.dumps({
        "status": "needs_supervision",
        "fingerprint": "3" * 64,
        "checkpoint_dir": str(checkpoint),
        "recovery_options": {"paper_id": "local:book", "workers": 1},
    }))

    monkeypatch.setattr(
        pipeline, "build_companion",
        lambda _options: {"ok": False, "status": "needs_supervision"},
    )
    first_result = pipeline.resume_companion(project, action="resume-native")
    assert first_result["ok"] is False

    transaction = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert transaction["status"] == "continuation_failed"
    assert sum(item["status"] == "reconciling" for item in transaction["entries"]) == 17

    captured = {}

    def resumed_build(options):
        captured["options"] = options
        for path in (checkpoint / "chapters").glob("*/translation-ledger.json"):
            ledger = json.loads(path.read_text())
            segment_id = ledger["needs_supervision"]["segment_id"]
            session_key = f"{ledger['chapter_id']}:{ledger['lane']}"
            logical_key = str(
                ledger["needs_supervision"]["recovery_context"]["idempotency_key"]
            )
            pipeline._guarded_mark_transport_state(
                path,
                checkpoint_dir=checkpoint,
                session_key=session_key,
                logical_unit=segment_id,
                idempotency_key=logical_key,
                response_received=True,
            )
            pipeline.advance_block(path, segment_id=segment_id, state="schema_valid")
            pipeline.advance_block(path, segment_id=segment_id, state="invariant_valid")
            pipeline.advance_block(
                path, segment_id=segment_id, state="accepted",
                input_sha256="input", output_sha256="output",
            )
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", resumed_build)
    result = pipeline.resume_companion(project, action="resume-native")

    assert result["ok"] is True
    assert captured["options"].supervised_native_resume_keys == tuple(logical_keys)
    ledgers = list((checkpoint / "chapters").glob("*/translation-ledger.json"))
    assert len(ledgers) == 17
    assert all(json.loads(path.read_text())["needs_supervision"] is None for path in ledgers)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_key", "ch-0007:guide"),
        ("generation", 2),
        ("provider", "claude-cli"),
        ("model", "other-model"),
        ("native_session_id", "tampered-native"),
    ],
)
def test_resume_native_missing_store_id_rejects_mismatched_recovery_context(
    tmp_path: Path, monkeypatch, field: str, value: object,
) -> None:
    fixture = _missing_store_native_resume_fixture(tmp_path)
    ledger = json.loads(fixture["ledger_path"].read_text())
    ledger["needs_supervision"]["recovery_context"][field] = value
    fixture["ledger_path"].write_text(json.dumps(ledger))
    monkeypatch.setattr(
        pipeline,
        "build_companion",
        lambda _options: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    result = pipeline.resume_companion(fixture["project"], action="resume-native")

    assert result["error"]["code"] == "native_resume_context_invalid"
    manager = LLMSessionManager(fixture["checkpoint"] / "sessions")
    assert manager.get_existing(fixture["session_key"]).native_session_id is None
    assert json.loads(fixture["ledger_path"].read_text())["needs_supervision"] is not None


def test_resume_native_rejects_supervision_without_logical_call_key(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "run"
    checkpoint = project / ".arc-companion" / "checkpoints" / ("3" * 64)
    ledger_path = checkpoint / "chapters" / "ch-0001" / "translation-ledger.json"
    initialize_lane_ledger(
        ledger_path, chapter_id="ch-0001", lane="translation", segment_ids=["s1"],
    )
    mark_needs_supervision(
        ledger_path, segment_id="s1", reason="unknown",
        recovery_context={"resumable": True, "native_session_id": "native-1"},
    )
    project.mkdir(parents=True, exist_ok=True)
    (project / "state.json").write_text(json.dumps({
        "status": "needs_supervision",
        "fingerprint": "3" * 64,
        "checkpoint_dir": str(checkpoint),
        "recovery_options": {"paper_id": "local:book", "workers": 1},
    }))
    monkeypatch.setattr(
        pipeline, "build_companion",
        lambda _options: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    result = pipeline.resume_companion(project, action="resume-native")

    assert result["error"]["code"] == "native_resume_idempotency_key_missing"


def test_stateful_lane_timeout_uses_auto_budget_and_stops_other_paid_lane(
    tmp_path: Path,
) -> None:
    blocks = [
        {"block_id": "c1", "type": "section", "title": "One"},
        {"block_id": "p1", "type": "text", "text": "A conserved quantity."},
    ]
    document = {
        "schema_version": "arc.paper.document.v2", "front_matter": {}, "blocks": blocks,
        "equations": [], "figures": [], "tables": [], "assets": [], "links": [],
        "bibliography": [], "integrity": {"status": "complete", "document_hash": "one"},
    }
    bundle = SourceBundle(
        paper_id="local:one", document=document,
        parsed={"paper_id": "local:one", "document": document,
                "structure": {"document_kind": "book", "chapters": [
                    {"title": "One", "block_ids": ["c1", "p1"]},
                ]}},
        metadata={"title": "One"}, references=[], citers=[],
    )
    paid: list[str] = []

    def llm(_prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        if label.startswith("title-translation-"):
            return _title_response(_prompt)
        if label.startswith("companion-glossary"):
            return {"entries": []}
        if label.startswith("companion-segmentation"):
            return {"cut_after_ordinals": []}
        raise AssertionError(label)

    def result_llm(_prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        paid.append(label)
        if label.startswith("companion-guide"):
            kwargs["session_manager"].get_or_create(
                key=kwargs["session_key"], provider="codex-cli",
                model="test-model", runtime_fingerprint="runtime",
            )
            return SimpleNamespace(value={
                "motivation": None, "main_content": "One", "section_logic": None,
                "prerequisites": None, "pedagogical_comparison": None,
                "historical_context": [], "supplementary_reading": [],
            }, logical_receipt={"idempotency_key": kwargs["idempotency_key"]})
        artifact_dir = Path(kwargs["artifact_dir"])
        session = kwargs["session_manager"].get_existing(kwargs["session_key"])
        if session is None:
            session = kwargs["session_manager"].get_or_create(
                key=kwargs["session_key"], provider="codex-cli",
                model="test-model", runtime_fingerprint="runtime",
            )
        call_path, identity = checkpoint_path(
            artifact_dir,
            prompt=_prompt,
            schema=kwargs["schema"],
            provider=session.provider,
            model=session.model,
            call_label=label,
            session_policy="stateful",
            session_key=session.key,
            runtime_fingerprint=session.runtime_fingerprint,
            idempotency_key=kwargs["idempotency_key"],
            generation=session.generation,
            progress_contract_scope="session",
            initial_native_authorization=kwargs["initial_native_authorization"],
        )
        prepared = prepare_call(call_path, identity=identity)
        record_submitted(prepared)
        timeout = LLMWorkerTimeout(
            "typed idle timeout",
            submission_state=LLMSubmissionState.SUBMITTED,
        )
        record_failure(prepared, timeout)
        diagnostics = AttemptDiagnostics(
            artifact_dir,
            provider=session.provider,
            model=session.model,
            fallback_index=0,
            attempt=1,
            call_label=label,
            env={},
        )
        diagnostics.bind_checkpoint_identity(identity)
        diagnostics.mark_submitted()
        attempt_ref = diagnostics.finalize(outcome="timeout", error=timeout)
        timeout.attempt_diagnostic_refs = ({
            "path": attempt_ref.path,
            "sha256": attempt_ref.sha256,
        },)
        journal = ProgressJournal(
            artifact_dir=artifact_dir,
            call_label=label,
            provider=session.provider,
            callback=kwargs.get("progress_callback"),
            identity={
                "idempotency_key": kwargs["idempotency_key"],
                "session_key": session.key,
                "generation": session.generation,
                "runtime_fingerprint": session.runtime_fingerprint,
                "checkpoint_identity": str(identity),
            },
        )
        journal({"event": "submitted"})
        journal({"event": "idle_timeout", "idle_seconds": 1.0})
        raise timeout

    project = tmp_path / "supervised"
    result = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=project, workers=1),
        source_loader=lambda *args, **kwargs: bundle, llm=llm, result_llm=result_llm,
    )

    assert result["error"]["code"] == "automatic_regeneration_exhausted"
    assert len([label for label in paid if "translation" in label]) == 4
    assert not any("annotation" in label for label in paid)
    state = json.loads((project / "state.json").read_text())
    checkpoint = Path(state["checkpoint_dir"])
    ledgers = [json.loads(path.read_text()) for path in
               (checkpoint / "chapters" / "ch-0001").glob("*-ledger.json")]
    supervised = [value for value in ledgers if value.get("needs_supervision")]
    assert len(supervised) == 1
    context = supervised[0]["needs_supervision"]["recovery_context"]
    assert context["idempotency_key"].endswith("generation-4")
    assert context["submission_state"] == "submitted"
    assert state["recovery_options"]["paper_id"] == "local:one"
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert journal["restart_budgets"][0]["attempts_used"] == 3


def test_production_callback_loss_reconstructs_then_runs_one_fresh_task(
    tmp_path: Path,
) -> None:
    """Exercise the real build/recovery controller without replacing its continuation."""

    blocks = [
        {"block_id": "c1", "type": "section", "title": "One"},
        {"block_id": "p1", "type": "text", "text": "A conserved quantity."},
    ]
    document = {
        "schema_version": "arc.paper.document.v2", "front_matter": {},
        "blocks": blocks, "equations": [], "figures": [], "tables": [],
        "assets": [], "links": [], "bibliography": [],
        "integrity": {"status": "complete", "document_hash": "callback-loss"},
    }
    bundle = SourceBundle(
        paper_id="local:callback-loss", document=document,
        parsed={
            "paper_id": "local:callback-loss", "document": document,
            "structure": {"document_kind": "book", "chapters": [{
                "title": "One", "block_ids": ["c1", "p1"],
            }]},
        },
        metadata={"title": "One"}, references=[], citers=[],
    )
    translation_generations: list[int] = []
    supervised_authorizations: list[object] = []

    def llm(_prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        if label.startswith("title-translation-"):
            return _title_response(_prompt)
        if label.startswith("companion-glossary"):
            return {"entries": []}
        if label.startswith("companion-segmentation"):
            return {"cut_after_ordinals": []}
        if label.startswith("companion-review-segment-"):
            return _segment_review_response(_prompt)
        raise AssertionError(label)

    def result_llm(_prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        manager = kwargs["session_manager"]
        session = manager.get_existing(kwargs["session_key"])
        if session is None:
            session = manager.get_or_create(
                key=kwargs["session_key"], provider="fake-provider",
                model="test-model", runtime_fingerprint="runtime",
            )
        supervised_authorizations.append(kwargs.get("supervised_native_resume"))
        if label.startswith("companion-guide"):
            return SimpleNamespace(value={
                "motivation": None, "main_content": "One", "section_logic": None,
                "prerequisites": None, "pedagogical_comparison": None,
                "historical_context": [], "supplementary_reading": [],
            }, logical_receipt={"idempotency_key": kwargs["idempotency_key"]})
        if label.startswith("companion-annotation"):
            kwargs["progress_callback"]({"event": "submitted"})
            return SimpleNamespace(value={
                "explanation": "Commentary.", "commentary": "",
                "prior_work": [], "later_work": [], "context_claims": [],
                "evidence_ids": [], "key_points": [], "source_notes": [],
                "evidence_requests": [],
            }, logical_receipt={"idempotency_key": kwargs["idempotency_key"]})
        assert label.startswith("companion-translation")
        translation_generations.append(session.generation)
        if session.generation > 1:
            kwargs["progress_callback"]({"event": "submitted"})
            return SimpleNamespace(
                value={"blocks": [{"block_id": "p1", "text": "守恒量。"}]},
                logical_receipt={"idempotency_key": kwargs["idempotency_key"]},
            )

        artifact_dir = Path(kwargs["artifact_dir"])
        call_path, identity = checkpoint_path(
            artifact_dir,
            prompt=_prompt,
            schema=kwargs["schema"],
            provider=session.provider,
            model=session.model,
            call_label=label,
            session_policy="stateful",
            session_key=session.key,
            runtime_fingerprint=session.runtime_fingerprint,
            idempotency_key=kwargs["idempotency_key"],
            generation=session.generation,
            progress_contract_scope="session",
            initial_native_authorization=kwargs["initial_native_authorization"],
        )
        prepared = prepare_call(call_path, identity=identity)
        record_submitted(prepared)
        timeout = LLMWorkerTimeout(
            "typed idle callback loss",
            submission_state=LLMSubmissionState.SUBMITTED,
        )
        record_failure(prepared, timeout)
        diagnostics = AttemptDiagnostics(
            artifact_dir, provider=session.provider, model=session.model,
            fallback_index=0, attempt=1, call_label=label, env={},
        )
        diagnostics.bind_checkpoint_identity(identity)
        diagnostics.mark_submitted()
        attempt_ref = diagnostics.finalize(outcome="timeout", error=timeout)
        timeout.attempt_diagnostic_refs = ({
            "path": attempt_ref.path, "sha256": attempt_ref.sha256,
        },)
        journal = ProgressJournal(
            artifact_dir=artifact_dir, call_label=label, provider=session.provider,
            callback=kwargs.get("progress_callback"),
            identity={
                "idempotency_key": kwargs["idempotency_key"],
                "session_key": session.key, "generation": session.generation,
                "runtime_fingerprint": session.runtime_fingerprint,
                "checkpoint_identity": str(identity),
            },
        )
        journal({"event": "submitted"})
        journal({"event": "idle_timeout", "idle_seconds": 1.0})
        raise timeout

    project = tmp_path / "callback-loss"
    result = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id, project_dir=project, workers=1,
            stop_after_first_chapter=True,
        ),
        source_loader=lambda *args, **kwargs: bundle,
        llm=llm,
        result_llm=result_llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert result.get("status") in {"first_chapter_ready", "complete"}, result
    assert translation_generations == [1, 2]
    assert all(value is None for value in supervised_authorizations)
    state = json.loads((project / "state.json").read_text())
    checkpoint = Path(state["checkpoint_dir"])
    ledger = json.loads(
        (checkpoint / "chapters" / "ch-0001" / "translation-ledger.json").read_text()
    )
    block = ledger["blocks"][0]
    assert ledger["generation"] == 2
    assert block["generation"] == 2 and block["state"] == "accepted"
    transaction = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert transaction["entries"][0]["reconstructed_from_durable_state"] is True
