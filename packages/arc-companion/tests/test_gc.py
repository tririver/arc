from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from arc_companion import cli
from arc_companion.artifact_ids import (
    allocate_artifact_dir,
    render_artifact_identity,
)
import arc_companion.gc as gc
from arc_companion.run_lock import ProjectBuildLock
from arc_companion.pipeline import _merge_gc_state as merge_pipeline_gc_state
from arc_companion.web import WEB_MANIFEST_VERSION


def _write_state(root: Path) -> None:
    (root / "state.json").write_text(
        json.dumps({
            "schema_version": "arc.companion.state.v3",
            "status": "complete",
        }) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_historical_reader(
    root: Path,
    *,
    manifest_version: str = WEB_MANIFEST_VERSION,
    malformed_snapshot_record: bool = False,
) -> tuple[Path, Path, Path]:
    data_dir = root / "reader" / "data"
    data_dir.mkdir(parents=True)
    snapshot_bytes = json.dumps({
        "schema_version": "arc.companion.reader-snapshot.v1",
    }, sort_keys=True).encode()
    snapshot_sha = hashlib.sha256(snapshot_bytes).hexdigest()
    snapshot = data_dir / f"snapshot-{snapshot_sha}.json"
    snapshot.write_bytes(snapshot_bytes)
    script_bytes = b"window.__OLD_READER__ = {};\n"
    script_sha = hashlib.sha256(script_bytes).hexdigest()
    script = data_dir / f"snapshot-{script_sha}.js"
    script.write_bytes(script_bytes)
    manifest_value = {
        "schema_version": manifest_version,
        "snapshot": {
            "path": snapshot.relative_to(root).as_posix(),
            "sha256": snapshot_sha,
            "bytes": len(snapshot_bytes),
        },
        "data_script": {
            "path": script.relative_to(root).as_posix(),
            "sha256": script_sha,
            "bytes": len(script_bytes),
        },
        "index": {
            "path": "reader/index.html",
            "sha256": "0" * 64,
            "bytes": 0,
        },
        "assets": [],
    }
    if malformed_snapshot_record:
        manifest_value["snapshot"].pop("sha256")
    manifest_bytes = json.dumps(
        manifest_value, sort_keys=True, separators=(",", ":"),
    ).encode()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = data_dir / f"manifest-{manifest_sha}.json"
    manifest.write_bytes(manifest_bytes)
    return manifest, snapshot, script


def _write_historical_render(root: Path) -> Path:
    render_root = root / ".arc-companion" / "renders" / "pdf"
    render_root.mkdir(parents=True)
    payload = {
        "content_sha256": "1" * 64,
        "render_recipe_sha256": "2" * 64,
        "validator_version": "test-validator",
        "stem": "paper",
    }
    nonce = "3" * 32
    identity = render_artifact_identity(
        kind="pdf-render", payload=payload, nonce=nonce,
    )
    allocation = allocate_artifact_dir(
        render_root,
        identity,
        kind="pdf-render",
        stem="paper",
        payload=payload,
        nonce=nonce,
        allow_legacy=False,
    )
    (allocation.path / "paper.pdf").write_bytes(b"%PDF-old")
    return allocation.path


def _write_current_legacy_render(root: Path) -> Path:
    render = root / ".arc-companion" / "renders" / "pdf" / ("4" * 64)
    render.mkdir(parents=True)
    tex = render / "paper.tex"
    pdf = render / "paper.pdf"
    manifest = render / "source-manifest.json"
    validation = render / "validation.json"
    tex.write_text("source", encoding="utf-8")
    pdf.write_bytes(b"%PDF-current")
    manifest.write_text("{}\n", encoding="utf-8")
    validation.write_text('{"result":"success"}\n', encoding="utf-8")
    state = {
        "schema_version": "arc.companion.state.v3",
        "status": "complete",
        "published": {
            "pdf": {
                "output_tex": str(tex),
                "output_tex_sha256": _sha(tex),
                "output_pdf": str(pdf),
                "output_pdf_sha256": _sha(pdf),
                "source_manifest_path": str(manifest),
                "source_manifest_sha256": _sha(manifest),
                "validation_path": str(validation),
                "validation_sha256": _sha(validation),
            },
        },
    }
    (root / "state.json").write_text(
        json.dumps(state) + "\n", encoding="utf-8",
    )
    return render


def test_gc_dry_run_no_op_is_deterministic(tmp_path: Path) -> None:
    _write_state(tmp_path)

    first = gc.discover_gc(tmp_path)
    second = gc.discover_gc(tmp_path)

    assert first.status == "no_op"
    assert first.candidates == ()
    assert first.candidate_set_sha256 == second.candidate_set_sha256
    assert first.root_snapshot_sha256 == second.root_snapshot_sha256
    assert first.as_dict()["refusals"] == []


def test_gc_discovers_reader_and_render_history_and_applies(
    tmp_path: Path,
) -> None:
    _write_state(tmp_path)
    reader_paths = _write_historical_reader(tmp_path)
    render_path = _write_historical_render(tmp_path)

    report = gc.discover_gc(tmp_path)

    assert {candidate.category for candidate in report.candidates} == {
        "reader_manifest_history",
        "reader_snapshot_history",
        "reader_data_history",
        "render_history",
    }
    result = gc.apply_gc(
        tmp_path, candidate_digest=report.candidate_set_sha256,
    )
    assert result["status"] == "complete"
    assert result["deleted_count"] == 4
    assert result["reclaimed_bytes"] == report.total_reclaimable_bytes
    assert not render_path.exists()
    assert all(not path.exists() for path in reader_paths)
    assert (tmp_path / result["receipt_path"]).is_file()
    assert gc.discover_gc(tmp_path).status == "no_op"


def test_gc_accepts_immediate_legacy_reader_manifest_history(
    tmp_path: Path,
) -> None:
    _write_state(tmp_path)
    manifest, snapshot, script = _write_historical_reader(
        tmp_path,
        manifest_version="arc.companion.web-manifest.v2",
    )
    with pytest.raises(
        gc.CompanionGCError, match="manifest schema is invalid",
    ):
        gc._manifest_graph(tmp_path, manifest)

    report = gc.discover_gc(tmp_path)

    assert {
        candidate.path for candidate in report.candidates
    } == {
        path.relative_to(tmp_path).as_posix()
        for path in (manifest, snapshot, script)
    }
    result = gc.apply_gc(
        tmp_path, candidate_digest=report.candidate_set_sha256,
    )
    assert result["deleted_count"] == 3
    assert all(not path.exists() for path in (manifest, snapshot, script))


@pytest.mark.parametrize(
    "manifest_version",
    [
        "arc.companion.web-manifest.v1",
        "arc.companion.web-manifest.v4",
    ],
)
def test_gc_rejects_unknown_reader_manifest_history_schema(
    tmp_path: Path,
    manifest_version: str,
) -> None:
    _write_state(tmp_path)
    _write_historical_reader(
        tmp_path, manifest_version=manifest_version,
    )

    with pytest.raises(
        gc.CompanionGCError, match="manifest schema is invalid",
    ) as error:
        gc.discover_gc(tmp_path)
    assert error.value.code == "gc_reader_invalid"


def test_gc_rejects_malformed_legacy_reader_manifest_history(
    tmp_path: Path,
) -> None:
    _write_state(tmp_path)
    _write_historical_reader(
        tmp_path,
        manifest_version="arc.companion.web-manifest.v2",
        malformed_snapshot_record=True,
    )

    with pytest.raises(
        gc.CompanionGCError, match="file identity is incomplete",
    ) as error:
        gc.discover_gc(tmp_path)
    assert error.value.code == "gc_reader_invalid"


def test_gc_apply_recovers_move_before_journal_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_state(tmp_path)
    render_path = _write_historical_render(tmp_path)
    report = gc.discover_gc(tmp_path)
    raised = False

    def crash(label: str) -> None:
        nonlocal raised
        if label == "after_move" and not raised:
            raised = True
            raise RuntimeError("simulated crash")

    monkeypatch.setattr(gc, "_gc_fault_point", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        gc.apply_gc(
            tmp_path, candidate_digest=report.candidate_set_sha256,
        )

    monkeypatch.setattr(gc, "_gc_fault_point", lambda _label: None)
    result = gc.apply_gc(tmp_path)
    assert result["status"] == "complete"
    assert result["deleted_count"] == 1
    assert not render_path.exists()
    assert gc.discover_gc(tmp_path).status == "no_op"


def test_gc_apply_recovers_partial_reader_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_state(tmp_path)
    reader_paths = _write_historical_reader(tmp_path)
    raised = False

    def crash(label: str) -> None:
        nonlocal raised
        if label == "after_move" and not raised:
            raised = True
            raise RuntimeError("reader move crash")

    monkeypatch.setattr(gc, "_gc_fault_point", crash)
    with pytest.raises(RuntimeError, match="reader move crash"):
        gc.apply_gc(tmp_path)
    monkeypatch.setattr(gc, "_gc_fault_point", lambda _label: None)

    result = gc.apply_gc(tmp_path)
    assert result["deleted_count"] == 3
    assert all(not path.exists() for path in reader_paths)


def test_gc_apply_recovers_delete_before_journal_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_state(tmp_path)
    render = _write_historical_render(tmp_path)
    raised = False

    def crash(label: str) -> None:
        nonlocal raised
        if label == "after_delete" and not raised:
            raised = True
            raise RuntimeError("delete crash")

    monkeypatch.setattr(gc, "_gc_fault_point", crash)
    with pytest.raises(RuntimeError, match="delete crash"):
        gc.apply_gc(tmp_path)
    monkeypatch.setattr(gc, "_gc_fault_point", lambda _label: None)

    result = gc.apply_gc(tmp_path)
    assert result["deleted_count"] == 1
    assert not render.exists()


def test_gc_cli_dry_run_and_apply(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _write_state(tmp_path)
    _write_historical_render(tmp_path)

    assert cli.main([
        "gc", "--project-dir", str(tmp_path), "--json",
    ]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    digest = dry_run["data"]["candidate_set_sha256"]
    assert dry_run["data"]["status"] == "ready"

    assert cli.main([
        "gc",
        "--project-dir",
        str(tmp_path),
        "--apply",
        "--candidate-digest",
        digest,
        "--json",
    ]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["data"]["status"] == "complete"
    assert applied["data"]["deleted_count"] == 1


def test_gc_retains_current_render_and_cache_roots(tmp_path: Path) -> None:
    current = _write_current_legacy_render(tmp_path)
    sidecar = current / "paper.validation.txt"
    sidecar.write_text("temporary", encoding="utf-8")
    cache = tmp_path / ".arc-companion" / "objects" / "keep.bin"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"keep")

    report = gc.discover_gc(tmp_path)

    assert [(item.category, item.path) for item in report.candidates] == [(
        "validation_temporary",
        sidecar.relative_to(tmp_path).as_posix(),
    )]
    gc.apply_gc(tmp_path, candidate_digest=report.candidate_set_sha256)
    assert current.is_dir()
    assert (current / "paper.pdf").is_file()
    assert cache.read_bytes() == b"keep"
    assert not sidecar.exists()


def test_gc_refuses_symlink_active_lock_and_digest_mismatch(
    tmp_path: Path,
) -> None:
    _write_state(tmp_path)
    target = tmp_path / "outside"
    target.mkdir()
    reader = tmp_path / "reader"
    reader.mkdir()
    (reader / "data").symlink_to(target, target_is_directory=True)
    with pytest.raises(gc.CompanionGCError) as symlink_error:
        gc.discover_gc(tmp_path)
    assert symlink_error.value.code == "gc_reader_invalid"
    (reader / "data").unlink()
    _write_historical_render(tmp_path)
    with pytest.raises(gc.CompanionGCError) as digest_error:
        gc.apply_gc(tmp_path, candidate_digest="0" * 64)
    assert digest_error.value.code == "gc_candidate_set_changed"
    lock = ProjectBuildLock(tmp_path / ".arc-companion-build.lock")
    lock.acquire()
    try:
        with pytest.raises(gc.CompanionGCError) as lock_error:
            gc.discover_gc(tmp_path)
        assert lock_error.value.code == "gc_build_active"
    finally:
        lock.release()


def test_gc_internal_apply_accepts_caller_owned_render_lock(
    tmp_path: Path,
) -> None:
    _write_state(tmp_path)
    _write_historical_render(tmp_path)
    render_lock = ProjectBuildLock(
        tmp_path / ".arc-companion" / "render.lock",
    )
    render_lock.acquire()
    try:
        with pytest.raises(gc.CompanionGCError) as dry_error:
            gc.discover_gc(tmp_path)
        assert dry_error.value.code == "gc_transaction_active"
        result = gc.apply_gc(tmp_path, lock_already_held=True)
    finally:
        render_lock.release()
    assert result["deleted_count"] == 1


def test_post_publication_gc_success_no_op_and_failure_are_nonfatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_state(tmp_path)
    _write_historical_render(tmp_path)
    updates: list[dict[str, object]] = []

    success = gc.run_post_publication_gc(
        tmp_path,
        state_merger=lambda values: updates.append(dict(values)),
        lock_already_held=False,
    )
    assert success["artifact_gc"]["reclaimed_bytes"] > 0
    no_op = gc.run_post_publication_gc(
        tmp_path,
        state_merger=lambda values: updates.append(dict(values)),
        lock_already_held=False,
    )
    assert no_op["artifact_gc"]["reclaimed_bytes"] == 0

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise gc.CompanionGCError("gc_test_failure", "injected")

    monkeypatch.setattr(gc, "apply_gc", fail)
    warning = gc.run_post_publication_gc(
        tmp_path,
        state_merger=lambda values: updates.append(dict(values)),
        lock_already_held=False,
    )
    assert warning == {
        "artifact_gc": None,
        "artifact_gc_warning": {
            "code": "gc_test_failure",
            "message": "injected",
        },
    }


def test_post_publication_gc_state_merge_failure_is_nonfatal(
    tmp_path: Path,
) -> None:
    _write_state(tmp_path)

    def fail_merge(_values: object) -> None:
        raise OSError("state write failed")

    outcome = gc.run_post_publication_gc(
        tmp_path,
        state_merger=fail_merge,
        lock_already_held=False,
    )

    assert outcome["artifact_gc"]["status"] == "complete"
    assert outcome["artifact_gc_warning"] == {
        "code": "gc_state_update_failed",
        "message": "state write failed",
    }
    assert (tmp_path / outcome["artifact_gc"]["receipt_path"]).is_file()


def test_post_publication_gc_clears_stale_counterpart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_state(tmp_path)
    merge_pipeline_gc_state(
        tmp_path / "state.json",
        {"artifact_gc_warning": {"code": "old", "message": "old"}},
    )
    gc.run_post_publication_gc(
        tmp_path,
        state_merger=lambda values: merge_pipeline_gc_state(
            tmp_path / "state.json", values,
        ),
        lock_already_held=False,
    )
    state = json.loads((tmp_path / "state.json").read_text())
    assert "artifact_gc" in state
    assert "artifact_gc_warning" not in state

    monkeypatch.setattr(
        gc,
        "apply_gc",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            gc.CompanionGCError("gc_failed_test", "failed"),
        ),
    )
    gc.run_post_publication_gc(
        tmp_path,
        state_merger=lambda values: merge_pipeline_gc_state(
            tmp_path / "state.json", values,
        ),
        lock_already_held=False,
    )
    state = json.loads((tmp_path / "state.json").read_text())
    assert "artifact_gc" not in state
    assert state["artifact_gc_warning"]["code"] == "gc_failed_test"


def test_gc_retains_valid_t19_checkpoint(tmp_path: Path) -> None:
    _write_state(tmp_path)
    identity = "5" * 64
    checkpoint_root = tmp_path / ".arc-companion" / "checkpoints"
    allocation = allocate_artifact_dir(
        checkpoint_root,
        identity,
        kind="checkpoint",
        allow_legacy=False,
    )
    state = json.loads((tmp_path / "state.json").read_text())
    state.update({
        "fingerprint": identity,
        "checkpoint_identity": identity,
        "checkpoint_dir": str(allocation.path),
        "checkpoint_identity_receipt_path": str(allocation.receipt_path),
        "checkpoint_identity_receipt_sha256": allocation.receipt_sha256,
    })
    (tmp_path / "state.json").write_text(json.dumps(state) + "\n")

    report = gc.discover_gc(tmp_path)
    assert report.status == "no_op"
    assert allocation.path.is_dir()


def test_gc_rejects_tampered_terminal_receipt(tmp_path: Path) -> None:
    _write_state(tmp_path)
    result = gc.apply_gc(tmp_path)
    receipt = tmp_path / result["receipt_path"]
    value = json.loads(receipt.read_text())
    value["reclaimed_bytes"] = 1
    receipt.write_text(json.dumps(value, sort_keys=True) + "\n")

    with pytest.raises(gc.CompanionGCError) as error:
        gc.apply_gc(tmp_path)
    assert error.value.code == "gc_transaction_invalid"


def test_gc_rejects_tampered_planned_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_state(tmp_path)
    _write_historical_render(tmp_path)

    def crash(label: str) -> None:
        if label == "transaction_planned":
            raise RuntimeError("planned crash")

    monkeypatch.setattr(gc, "_gc_fault_point", crash)
    with pytest.raises(RuntimeError, match="planned crash"):
        gc.apply_gc(tmp_path)
    transaction = next(
        (tmp_path / ".arc-companion" / "gc" / "transactions").iterdir()
    )
    value = json.loads(transaction.read_text())
    value["category_totals"]["render_history"]["bytes"] += 1
    transaction.write_text(json.dumps(value, sort_keys=True) + "\n")
    monkeypatch.setattr(gc, "_gc_fault_point", lambda _label: None)

    with pytest.raises(gc.CompanionGCError) as error:
        gc.apply_gc(tmp_path)
    assert error.value.code == "gc_transaction_invalid"


def test_gc_cli_reports_outside_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _write_state(tmp_path)
    state = json.loads((tmp_path / "state.json").read_text())
    state["checkpoint_dir"] = str(tmp_path.parent / "outside-checkpoint")
    (tmp_path / "state.json").write_text(json.dumps(state) + "\n")

    assert cli.main([
        "gc", "--project-dir", str(tmp_path), "--json",
    ]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["error"]["code"] == "gc_checkpoint_invalid"


def test_gc_cli_preserves_stable_error_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _write_state(tmp_path)
    _write_historical_render(tmp_path)

    assert cli.main([
        "gc",
        "--project-dir",
        str(tmp_path),
        "--apply",
        "--candidate-digest",
        "0" * 64,
        "--json",
    ]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["error"]["code"] == "gc_candidate_set_changed"
