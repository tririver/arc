from __future__ import annotations

import json
from pathlib import Path

import pytest

import arc_companion.pipeline as pipeline
from arc_companion.pipeline import BuildOptions
from arc_companion.io import sha256_json
from arc_companion.source import SourceBundle, SourceError


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
    checkpoint = project / "checkpoints" / fingerprint
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
    journal.parent.mkdir(parents=True)
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
