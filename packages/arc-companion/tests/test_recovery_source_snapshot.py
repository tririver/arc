from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import arc_companion.pipeline as pipeline
from arc_companion.artifact_ids import (
    allocate_artifact_dir,
    ensure_artifact_alias_receipt,
)
from arc_companion.pipeline import BuildOptions
from arc_companion.io import sha256_json
from arc_companion.paper_broker import (
    PaperBroker,
    build_paper_broker_policy,
)
from arc_companion.source import SourceBundle, SourceError
from arc_companion.translation_reference import resolve_translation_reference


PAPER_ID = "local:recovery-snapshot"


def _snapshot_fixture(tmp_path: Path) -> tuple[
    BuildOptions, Path, dict[str, object], dict[str, object], str,
]:
    project = tmp_path / "run"
    options = BuildOptions(
        paper_id=PAPER_ID,
        project_dir=project,
        workers=1,
        recovery_policy="auto",
    )
    document: dict[str, object] = {
        "schema_version": "arc.paper.document.v2",
        "front_matter": {"title": "Recovery Snapshot"},
        "blocks": [{"block_id": "p1", "type": "text", "text": "Stable source."}],
        "equations": [],
        "figures": [],
        "tables": [],
        "bibliography": [],
        "assets": [],
        "integrity": {
            "status": "complete",
            "document_hash": "snapshot-document-hash",
        },
    }
    payload: dict[str, object] = {
        "paper_id": PAPER_ID,
        "document": document,
    }
    evidence: dict[str, object] = {
        "references": [],
        "citers": [],
        "diagnostics": [],
        "related_papers": [],
    }
    bundle = SourceBundle(
        paper_id=PAPER_ID,
        parsed=dict(payload),
        document=document,
        metadata={"title": "Recovery Snapshot"},
        references=[],
        citers=[],
    )
    fingerprint = pipeline._fingerprint(
        bundle,
        options,
        evidence=pipeline._evidence(bundle),
        domain_context=None,
    )
    checkpoint = (
        project / ".arc-companion" / "checkpoints" / fingerprint
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "document.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    (checkpoint / "evidence.json").write_text(
        json.dumps(evidence), encoding="utf-8",
    )
    return options, checkpoint, payload, evidence, fingerprint


def _failing_source_loader(*_args, **_kwargs):
    raise SourceError("authoritative source cache is unavailable")


def test_checkpoint_state_resolver_accepts_strict_worker_alias(
    tmp_path: Path,
) -> None:
    project = tmp_path / "alias-run"
    root = project / ".arc-companion" / "checkpoints"
    logical = "d" * 64
    physical = "e" * 64
    checkpoint = root / physical
    checkpoint.mkdir(parents=True)
    alias = ensure_artifact_alias_receipt(
        root,
        logical,
        {
            "schema_version": "arc.companion.checkpoint-alias.v1",
            "kind": "workers-to-total-concurrency-budget",
            "alias_identity": logical,
            "legacy_fingerprint": physical,
            "content_fingerprint": logical,
            "legacy_checkpoint_dir": str(checkpoint),
            "legacy_workers_per_lane": 4097,
        },
    )
    state = {
        "fingerprint": logical,
        "checkpoint_identity": logical,
        "checkpoint_dir": str(checkpoint),
        "checkpoint_alias_identity": logical,
        "checkpoint_alias_receipt_path": str(alias.path),
        "checkpoint_alias_receipt_sha256": alias.sha256,
    }
    resolved = pipeline._resolve_checkpoint_state_identity(
        project, state, checkpoint,
    )
    assert resolved.path == checkpoint.resolve()
    assert resolved.identity == physical


def test_checkpoint_state_rejects_external_identity_receipt_symlink(
    tmp_path: Path,
) -> None:
    project = tmp_path / "identity-run"
    identity = "a" * 64
    allocation = allocate_artifact_dir(
        project / ".arc-companion" / "checkpoints",
        identity,
        kind="checkpoint",
    )
    saved = tmp_path / "identity-receipt-link"
    saved.symlink_to(allocation.receipt_path)
    state = {
        "fingerprint": identity,
        "checkpoint_identity": identity,
        "checkpoint_dir": str(allocation.path),
        "checkpoint_identity_receipt_path": str(saved),
        "checkpoint_identity_receipt_sha256": allocation.receipt_sha256,
    }
    with pytest.raises(RuntimeError, match="receipt state"):
        pipeline._resolve_checkpoint_state_identity(
            project, state, allocation.path,
        )


def test_checkpoint_state_rejects_external_alias_receipt_symlink(
    tmp_path: Path,
) -> None:
    project = tmp_path / "alias-link-run"
    root = project / ".arc-companion" / "checkpoints"
    logical = "b" * 64
    physical = "c" * 64
    checkpoint = root / physical
    checkpoint.mkdir(parents=True)
    alias = ensure_artifact_alias_receipt(
        root,
        logical,
        {
            "schema_version": "arc.companion.checkpoint-alias.v1",
            "kind": "workers-to-total-concurrency-budget",
            "alias_identity": logical,
            "legacy_fingerprint": physical,
            "content_fingerprint": logical,
            "legacy_checkpoint_dir": str(checkpoint),
            "legacy_workers_per_lane": 4097,
        },
    )
    saved = tmp_path / "alias-receipt-link"
    saved.symlink_to(alias.path)
    state = {
        "fingerprint": logical,
        "checkpoint_identity": logical,
        "checkpoint_dir": str(checkpoint),
        "checkpoint_alias_identity": logical,
        "checkpoint_alias_receipt_path": str(saved),
        "checkpoint_alias_receipt_sha256": alias.sha256,
    }
    with pytest.raises(RuntimeError, match="alias receipt state"):
        pipeline._resolve_checkpoint_state_identity(
            project, state, checkpoint,
        )


def test_intent_guidance_recovery_root_has_separate_safe_authority(
    tmp_path: Path,
) -> None:
    project = tmp_path / "intent-run"
    fingerprint = "f" * 64
    root = (
        project / ".arc-companion" / "intent-guidance" / ("1" * 64)
    )
    root.mkdir(parents=True)
    (root / "source-snapshot-receipt.json").write_text(
        json.dumps({
            "schema_version": "arc.companion.source-snapshot-receipt.v2",
            "fingerprint": fingerprint,
            "checkpoint_identity": fingerprint,
        }),
        encoding="utf-8",
    )
    state = {
        "status": "failed",
        "checkpoint_dir": str(root),
        "recovery_root_kind": "intent-guidance",
        "recovery_root_fingerprint": fingerprint,
    }
    assert pipeline._resolve_recovery_state_root(
        project, state, root,
    ) == root.resolve()
    with pytest.raises(RuntimeError, match="intent-guidance"):
        pipeline._resolve_recovery_state_root(
            project, state, tmp_path,
        )


class _ReferenceBroker:
    def __init__(
        self,
        aggregate_broker: PaperBroker,
        structure: dict[str, Any],
        section_payload: dict[str, Any],
    ) -> None:
        self.aggregate_broker = aggregate_broker
        self.structure = structure
        self.section_payload = section_payload

    def resolve_round(self, requests, *, round_number: int):
        del round_number
        operation = requests[0].operation
        if operation == "get-parsed-identity":
            data = {
                "paper_id": "reference",
                "source_hash": "3" * 64,
                "document_hash": "4" * 64,
            }
        elif operation == "get-parsed-structure":
            data = self.structure
        elif operation == "get-parsed-section":
            data = self.section_payload
        else:
            raise AssertionError(operation)
        return (SimpleNamespace(
            ok=True,
            data={"ok": True, "data": data},
            error=None,
            provenance={},
        ),)

    def store_controller_aggregate_json(self, **kwargs):
        return self.aggregate_broker.store_controller_aggregate_json(**kwargs)

    def load_controller_aggregate_json(self, **kwargs):
        return self.aggregate_broker.load_controller_aggregate_json(**kwargs)


def _reference_snapshot_fixture(
    tmp_path: Path,
) -> tuple[BuildOptions, Path, dict[str, object], Path, Path]:
    project = tmp_path / "reference-run"
    options = BuildOptions(
        paper_id=PAPER_ID,
        project_dir=project,
        workers=1,
        recovery_policy="auto",
        reference_translation_id="reference",
    )
    document: dict[str, object] = {
        "schema_version": "arc.paper.document.v2",
        "front_matter": {"title": "Recovery Snapshot"},
        "blocks": [
            {"block_id": "p1", "type": "text", "text": "Stable source."},
        ],
        "equations": [],
        "figures": [],
        "tables": [],
        "bibliography": [],
        "assets": [],
        "integrity": {
            "status": "complete",
            "document_hash": "snapshot-document-hash",
        },
    }
    payload: dict[str, object] = {
        "paper_id": PAPER_ID,
        "parser_version": "arc.paper.rich-document.v1",
        "source_hash": "1" * 64,
        "document_hash": "2" * 64,
        "sections": [
            {"section_id": "source-section-1", "title": "1 First", "level": 1},
        ],
        "structure": {
            "schema_version": "arc.paper.structure.v1",
            "requested_document_kind": "book",
            "document_kind": "book",
            "structure_source": "rich_source_headings",
            "chapters": [{
                "chapter_id": "source-chapter-1",
                "title": "1 First",
                "level": 1,
                "section_ids": ["source-section-1"],
            }],
            "coverage": {
                "status": "complete",
                "expected_count": 1,
                "covered_count": 1,
                "duplicates": [],
                "missing": [],
                "unexpected": [],
                "monotonic_order": True,
            },
        },
        "document": document,
    }
    chapters = {
        "schema_version": "arc.companion.chapters.v2",
        "chapters": [{
            "chapter_id": "source-chapter-1",
            "block_ids": ["p1"],
        }],
    }
    section_payload = {
        "section_id": "reference-section-1",
        "text": "Translated working draft.",
    }
    structure = {
        "schema_version": "arc.paper.parsed-structure-view.v1",
        "requested_source_id": "reference",
        "canonical_source_id": "reference",
        "parser_version": "arc.paper.rich-document.v1",
        "source_hash": "3" * 64,
        "document_hash": "4" * 64,
        "structure_schema_version": "arc.paper.structure.v1",
        "requested_document_kind": "book",
        "document_kind": "book",
        "structure_source": "rich_source_headings",
        "chapters": [{
            "chapter_id": "reference-chapter-1",
            "title": "1 First",
            "level": 1,
            "leading_decimal_ordinal": 1,
            "section_ids": ["reference-section-1"],
        }],
        "sections": [{
            "section_id": "reference-section-1",
            "title": "1 First",
            "level": 1,
            "ordinal": 0,
            "section_payload_sha256": sha256_json(section_payload),
        }],
        "coverage": {
            "status": "complete",
            "expected_count": 1,
            "covered_count": 1,
            "duplicates": [],
            "missing": [],
            "unexpected": [],
            "monotonic_order": True,
        },
    }
    aggregate = PaperBroker(
        checkpoint_root=tmp_path / "broker-run",
        base_cache_root=tmp_path / "cache",
        policy=build_paper_broker_policy(access="none"),
        run_id="reference-recovery",
        generic_internet_allowed=False,
        controller_project_root=project,
    )
    reference = resolve_translation_reference(
        project_dir=project,
        checkpoint_dir=None,
        primary_parsed=payload,
        primary_document=document,
        chapters_pack=chapters,
        requested_reference_id="reference",
        broker=_ReferenceBroker(aggregate, structure, section_payload),
    )
    assert reference is not None
    evidence: dict[str, object] = {
        "references": [],
        "citers": [],
        "diagnostics": [],
        "related_papers": [],
    }
    bundle = SourceBundle(
        paper_id=PAPER_ID,
        parsed=dict(payload),
        document=document,
        metadata={"title": "Recovery Snapshot"},
        references=[],
        citers=[],
    )
    fingerprint = pipeline._fingerprint(
        bundle,
        options,
        evidence=pipeline._evidence(bundle),
        domain_context=None,
        translation_reference_manifest_sha256=reference.manifest_sha256,
    )
    checkpoint = (
        project / ".arc-companion" / "checkpoints" / fingerprint
    )
    checkpoint.mkdir(parents=True)
    for name, value in (
        ("document.json", payload),
        ("evidence.json", evidence),
        ("chapters.json", chapters),
    ):
        (checkpoint / name).write_text(json.dumps(value), encoding="utf-8")
    manifest_path = str(reference.manifest_path.relative_to(project))
    binding = {
        "schema_version": "arc.companion.translation-reference-validation.v1",
        "manifest_path": manifest_path,
        "manifest_sha256": reference.manifest_sha256,
        "compact_provenance": dict(reference.compact_provenance),
    }
    binding_path = checkpoint / "translation-reference.json"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    reference_tuple = {
        "translation_reference_manifest_path": manifest_path,
        "translation_reference_manifest_sha256": reference.manifest_sha256,
        "translation_reference_source_id": "reference",
        "translation_reference_source_hash": "3" * 64,
    }
    receipt = {
        "schema_version": "arc.companion.source-snapshot-receipt.v2",
        "paper_id": PAPER_ID,
        "fingerprint": fingerprint,
        "checkpoint_identity": fingerprint,
        "document_payload_sha256": sha256_json(payload),
        "chapters_pack_sha256": sha256_json(chapters),
        "evidence_sha256": sha256_json(evidence),
        "domain_context_sha256": None,
        **reference_tuple,
    }
    (checkpoint / "source-snapshot-receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8",
    )
    (project / "state.json").write_text(json.dumps({
        "status": "failed",
        "checkpoint_dir": str(checkpoint),
        "fingerprint": fingerprint,
        **reference_tuple,
    }), encoding="utf-8")
    object_path = project / (
        ".arc-companion/paper-broker/controller-objects/"
        "translation-reference/objects/"
        f"{reference.compact_provenance['mappings'][0]['object_id']}.json"
    )
    return options, checkpoint, payload, binding_path, object_path


def test_recovery_source_uses_matching_complete_checkpoint_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, checkpoint, payload, _evidence, fingerprint = _snapshot_fixture(tmp_path)
    (options.project_dir / "state.json").write_text(json.dumps({
        "status": "failed",
        "checkpoint_dir": str(checkpoint),
        "fingerprint": fingerprint,
    }), encoding="utf-8")
    monkeypatch.setattr(pipeline, "load_source_bundle", _failing_source_loader)

    recovered = pipeline._load_recovery_source_bundle(
        options,
        paper_id=PAPER_ID,
        load_kwargs={"refresh": False},
    )

    assert recovered.paper_id == PAPER_ID
    assert recovered.document == payload["document"]
    assert recovered.parsed == payload
    assert recovered.metadata["_arc_companion_metadata_source"] == "checkpoint_snapshot"


def test_reference_v2_recovery_uses_authority_captured_before_state_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, _checkpoint, payload, _binding_path, _object_path = (
        _reference_snapshot_fixture(tmp_path)
    )
    authority = pipeline._capture_recovery_source_authority(options)
    assert authority is not None
    pipeline._state(
        options.project_dir / "state.json",
        status="loading_source",
        paper_id=PAPER_ID,
        translation_reference_manifest_path=None,
        translation_reference_manifest_sha256=None,
        translation_reference_source_id=None,
        translation_reference_source_hash=None,
    )
    monkeypatch.setattr(pipeline, "load_source_bundle", _failing_source_loader)

    recovered = pipeline._load_recovery_source_bundle(
        options,
        paper_id=PAPER_ID,
        load_kwargs={},
        recovery_authority=authority,
    )

    assert recovered.parsed == payload


@pytest.mark.parametrize("mutation", ("binding", "manifest", "object"))
def test_reference_v2_recovery_rejects_tamper_after_authority_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    options, _checkpoint, _payload, binding_path, object_path = (
        _reference_snapshot_fixture(tmp_path)
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    authority = pipeline._capture_recovery_source_authority(options)
    assert authority is not None
    pipeline._state(
        options.project_dir / "state.json",
        status="loading_source",
        paper_id=PAPER_ID,
        translation_reference_manifest_path=None,
        translation_reference_manifest_sha256=None,
        translation_reference_source_id=None,
        translation_reference_source_hash=None,
    )
    if mutation == "binding":
        binding["manifest_sha256"] = "f" * 64
        binding_path.write_text(json.dumps(binding), encoding="utf-8")
    elif mutation == "manifest":
        manifest_path = options.project_dir / binding["manifest_path"]
        manifest_path.write_text("{}", encoding="utf-8")
    else:
        object_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pipeline, "load_source_bundle", _failing_source_loader)

    with pytest.raises(SourceError, match="snapshot receipt"):
        pipeline._load_recovery_source_bundle(
            options,
            paper_id=PAPER_ID,
            load_kwargs={},
            recovery_authority=authority,
        )


@pytest.mark.parametrize("mutation", (None, "binding", "manifest", "object"))
def test_build_auto_recovery_threads_preclear_reference_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str | None,
) -> None:
    options, _checkpoint, payload, binding_path, object_path = (
        _reference_snapshot_fixture(tmp_path)
    )
    state_path = options.project_dir / "state.json"
    generated_state = json.loads(state_path.read_text(encoding="utf-8"))
    state_path.unlink()
    assert pipeline._capture_recovery_source_authority(options) is None
    build_calls = 0
    authoritative_calls = 0

    def transient_source_loader(*_args, **_kwargs) -> SourceBundle:
        nonlocal authoritative_calls
        authoritative_calls += 1
        if authoritative_calls == 1:
            return SourceBundle(
                paper_id=PAPER_ID,
                parsed=dict(payload),
                document=dict(payload["document"]),
                metadata={"title": "Recovery Snapshot"},
                references=[],
                citers=[],
            )
        raise SourceError("authoritative source cache is temporarily unavailable")

    def fake_build(
        active_options: BuildOptions,
        *,
        source_loader,
        **_kwargs,
    ):
        nonlocal build_calls
        build_calls += 1
        if build_calls == 1:
            assert not state_path.exists()
            loaded = source_loader(PAPER_ID, refresh=False)
            assert loaded.parsed == payload
            state_path.write_text(
                json.dumps(generated_state),
                encoding="utf-8",
            )
            return {"ok": False, "status": "needs_supervision"}
        recovered = source_loader(PAPER_ID, refresh=False)
        return {"ok": True, "status": "complete", "data": recovered.parsed}

    def fake_resume(
        _project_dir: Path,
        *,
        continuation,
        source_preflight,
        **_kwargs,
    ):
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if mutation == "binding":
            binding["manifest_sha256"] = "f" * 64
            binding_path.write_text(json.dumps(binding), encoding="utf-8")
        elif mutation == "manifest":
            (options.project_dir / binding["manifest_path"]).write_text(
                "{}", encoding="utf-8",
            )
        elif mutation == "object":
            object_path.write_text("{}", encoding="utf-8")
        pipeline._state(
            options.project_dir / "state.json",
            status="loading_source",
            paper_id=PAPER_ID,
            translation_reference_manifest_path=None,
            translation_reference_manifest_sha256=None,
            translation_reference_source_id=None,
            translation_reference_source_hash=None,
        )
        try:
            source_preflight()
        except SourceError as exc:
            return {
                "ok": False,
                "status": "failed",
                "error": {"code": "companion_source_unavailable", "message": str(exc)},
            }
        return continuation(options)

    monkeypatch.setattr(pipeline, "_build_companion_unlocked", fake_build)
    monkeypatch.setattr(pipeline, "_resume_companion_unlocked", fake_resume)

    result = pipeline.build_companion(
        options,
        source_loader=transient_source_loader,
    )

    if mutation is None:
        assert result["ok"] is True
        assert result["data"] == payload
        assert build_calls == 2
        assert authoritative_calls == 3
    else:
        assert result["ok"] is False
        assert result["error"]["code"] == "companion_source_unavailable"
        assert build_calls == 1
        assert authoritative_calls == 2


def test_failed_state_without_checkpoint_pointer_recovers_root_from_transaction_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, checkpoint, payload, _evidence, fingerprint = _snapshot_fixture(tmp_path)
    (checkpoint / "sessions").mkdir()
    ledger_path = checkpoint / "chapters" / "ch-0001" / "translation-ledger.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text("{}", encoding="utf-8")
    (options.project_dir / "state.json").write_text(json.dumps({
        "status": "failed",
        "fingerprint": fingerprint,
    }), encoding="utf-8")
    journal = options.project_dir / ".arc-companion" / "resume-transaction.json"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps({
        "schema_version": "arc.companion.resume-transaction.v2",
        "action": "auto",
        "status": "continuation_failed",
        "entries": [{"ledger_path": str(ledger_path)}],
    }), encoding="utf-8")
    monkeypatch.setattr(pipeline, "load_source_bundle", _failing_source_loader)

    recovered = pipeline._load_recovery_source_bundle(
        options,
        paper_id=PAPER_ID,
        load_kwargs={},
    )

    assert recovered.document == payload["document"]
    assert recovered.metadata["_arc_companion_metadata_source"] == "checkpoint_snapshot"


def test_recovery_source_receipt_rejects_changed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, checkpoint, payload, evidence, fingerprint = _snapshot_fixture(tmp_path)
    (checkpoint / "source-snapshot-receipt.json").write_text(json.dumps({
        "schema_version": "arc.companion.source-snapshot-receipt.v1",
        "paper_id": PAPER_ID, "fingerprint": fingerprint,
        "document_payload_sha256": sha256_json(payload),
        "evidence_sha256": sha256_json(evidence),
        "domain_context_sha256": None,
    }), encoding="utf-8")
    changed = {**evidence, "references": [{"paper_id": "injected"}]}
    (checkpoint / "evidence.json").write_text(json.dumps(changed), encoding="utf-8")
    (options.project_dir / "state.json").write_text(json.dumps({
        "status": "failed", "checkpoint_dir": str(checkpoint),
        "fingerprint": fingerprint,
    }), encoding="utf-8")
    monkeypatch.setattr(pipeline, "load_source_bundle", _failing_source_loader)

    with pytest.raises(SourceError, match="snapshot receipt"):
        pipeline._load_recovery_source_bundle(
            options, paper_id=PAPER_ID, load_kwargs={},
        )


@pytest.mark.parametrize(
    "mutation",
    ("fingerprint", "missing-document", "paper-id", "incomplete-integrity"),
)
def test_recovery_source_rejects_unverified_checkpoint_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    options, checkpoint, payload, _evidence, fingerprint = _snapshot_fixture(tmp_path)
    state_fingerprint = fingerprint
    if mutation == "fingerprint":
        state_fingerprint = "not-the-authoritative-fingerprint"
    elif mutation == "missing-document":
        (checkpoint / "document.json").unlink()
    else:
        changed = json.loads(json.dumps(payload))
        if mutation == "paper-id":
            changed["paper_id"] = "local:different-paper"
        else:
            changed["document"]["integrity"]["status"] = "partial"
        (checkpoint / "document.json").write_text(
            json.dumps(changed), encoding="utf-8",
        )
    (options.project_dir / "state.json").write_text(json.dumps({
        "status": "failed",
        "checkpoint_dir": str(checkpoint),
        "fingerprint": state_fingerprint,
    }), encoding="utf-8")
    monkeypatch.setattr(pipeline, "load_source_bundle", _failing_source_loader)

    with pytest.raises(SourceError):
        pipeline._load_recovery_source_bundle(
            options,
            paper_id=PAPER_ID,
            load_kwargs={},
        )
