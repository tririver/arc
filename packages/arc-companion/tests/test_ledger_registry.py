from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal

import pytest

import arc_companion.ledger_registry as registry_module
import arc_companion.pipeline as pipeline
from arc_companion.ledger import (
    LaneLedgerError,
    initialize_lane_ledger,
    invalidate_suffix,
    lane_transition_guard,
    mark_needs_supervision,
    mark_response_received,
    mark_submitted,
)
from arc_companion.ledger_registry import (
    LaneLedgerRegistryError,
    REGISTRY_FILE_NAME,
    read_registered_lane_ledger,
    registered_lane_ledger_paths,
)
from arc_companion.resume_transaction import begin_transaction
from arc_llm.sessions import LLMSessionManager


def _ledger_path(checkpoint: Path, chapter_id: str = "ch-0001") -> Path:
    return checkpoint / "chapters" / chapter_id / "translation-ledger.json"


def _create_registered(checkpoint: Path, chapter_id: str = "ch-0001") -> Path:
    path = _ledger_path(checkpoint, chapter_id)
    initialize_lane_ledger(
        path,
        chapter_id=chapter_id,
        lane="translation",
        segment_ids=[f"{chapter_id}.seg-0001"],
        checkpoint_dir=checkpoint,
    )
    return path


def _persist_registered(
    path: Path, ledger: dict, *, checkpoint_dir: Path,
) -> bool:
    return registry_module.persist_lane_ledger(
        path,
        ledger,
        checkpoint_dir=checkpoint_dir,
        expected_existing_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _concurrent_registry_writer(
    checkpoint_text: str, path_text: str, start, result,
) -> None:
    start.wait(10)
    checkpoint = Path(checkpoint_text)
    path = Path(path_text)
    try:
        _persist_registered(
            path,
            {**json.loads(path.read_text(encoding="utf-8")), "updated_at": 222.0},
            checkpoint_dir=checkpoint,
        )
    except BaseException as exc:  # pragma: no cover - surfaced in parent
        result.put((False, repr(exc)))
    else:
        result.put((True, ""))


def _distinct_supervision_writer(
    path_text: str, segment_id: str, start, result,
) -> None:
    start.wait(10)
    try:
        mark_needs_supervision(
            Path(path_text),
            segment_id=segment_id,
            reason=f"worker:{segment_id}",
            recovery_context={"submission_state": "unknown"},
        )
    except BaseException as exc:  # pragma: no cover - surfaced in parent
        result.put((False, repr(exc)))
    else:
        result.put((True, segment_id))


def _racing_identity_initializer(
    checkpoint_text: str, path_text: str, segment_id: str, start, result,
) -> None:
    start.wait(10)
    try:
        initialize_lane_ledger(
            Path(path_text),
            chapter_id="ch-0001",
            lane="translation",
            segment_ids=[segment_id],
            checkpoint_dir=Path(checkpoint_text),
        )
    except BaseException as exc:  # pragma: no cover - surfaced in parent
        result.put((False, segment_id, type(exc).__name__, str(exc)))
    else:
        result.put((True, segment_id, "", ""))


def test_initialize_registers_hash_bound_owned_ledger(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)

    registry = json.loads((checkpoint / REGISTRY_FILE_NAME).read_text(encoding="utf-8"))
    assert registry["schema_version"] == "arc.companion.lane-ledger-registry.v1"
    assert len(registry["entries"]) == 1
    entry = registry["entries"][0]
    assert entry["owner"] == "arc-companion.chapter-lane"
    assert entry["path"] == "chapters/ch-0001/translation-ledger.json"
    assert entry["chapter_id"] == "ch-0001"
    assert entry["lane"] == "translation"
    assert entry["generation"] == 1
    assert entry["ordered_segment_ids"] == ["ch-0001.seg-0001"]
    assert entry["ledger_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert registered_lane_ledger_paths(checkpoint) == [path]
    assert pipeline._all_lane_ledger_paths(checkpoint) == [path]


def test_exact_adoption_registers_without_rewriting_existing_bytes(
    tmp_path: Path,
) -> None:
    source_checkpoint = tmp_path / "source"
    source = _create_registered(source_checkpoint)
    ledger = json.loads(source.read_text())
    checkpoint = tmp_path / "checkpoint"
    path = _ledger_path(checkpoint)
    path.parent.mkdir(parents=True)
    raw = json.dumps(ledger, ensure_ascii=False, indent=2).encode() + b"\n"
    path.write_bytes(raw)
    before = path.stat()

    assert registry_module.adopt_lane_ledger_exact(
        path,
        ledger,
        checkpoint_dir=checkpoint,
        expected_existing_sha256=hashlib.sha256(raw).hexdigest(),
    )

    after = path.stat()
    assert path.read_bytes() == raw
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    adopted, digest = read_registered_lane_ledger(checkpoint, path)
    assert adopted == ledger
    assert digest == hashlib.sha256(raw).hexdigest()


def test_registered_replacement_requires_explicit_exact_digest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    before = path.read_bytes()
    with pytest.raises(LaneLedgerRegistryError, match="exact expected digest"):
        registry_module.persist_lane_ledger(
            path,
            {**json.loads(before), "updated_at": 999.0},
            checkpoint_dir=checkpoint,
        )
    assert path.read_bytes() == before


def test_auto_ignores_schema_compatible_copy_but_explicit_native_can_discover_it(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    registered = _create_registered(checkpoint)
    copied = _ledger_path(checkpoint, "ch-copied")
    copied.parent.mkdir(parents=True)
    shutil.copy2(registered, copied)

    assert pipeline._all_lane_ledger_paths(checkpoint) == [registered]
    assert pipeline._all_lane_ledger_paths(
        checkpoint, include_explicit_legacy=True,
    ) == [registered, copied]


def test_stale_ledger_hash_removes_it_from_automatic_ownership(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["generation"] = 2
    value["blocks"][0]["generation"] = 2
    path.write_text(json.dumps(value), encoding="utf-8")

    assert registered_lane_ledger_paths(checkpoint) == []
    assert pipeline._all_lane_ledger_paths(checkpoint) == []
    assert pipeline._validate_recovery_ledger_address(
        checkpoint_dir=checkpoint,
        ledger_path=path.resolve(),
        ledger=value,
        session_key="ch-0001:translation",
        segment_id="ch-0001.seg-0001",
        generation=2,
    ) == "recovery lane ledger is not registered in the active control index"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_registered_leaf_and_component_symlink_swaps_fail_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    original = path.read_bytes()

    leaf_target = checkpoint / "copied-ledger.json"
    leaf_target.write_bytes(original)
    path.unlink()
    path.symlink_to(leaf_target)
    assert registered_lane_ledger_paths(checkpoint) == []

    path.unlink()
    path.write_bytes(original)
    # Re-register the restored regular file, then replace an intermediate
    # directory with a symlink to an identical tree.
    initialize_lane_ledger(
        path,
        chapter_id="ch-0001",
        lane="translation",
        segment_ids=["ch-0001.seg-0001"],
        checkpoint_dir=checkpoint,
    )
    chapter_dir = path.parent
    moved = checkpoint / "saved-chapter"
    chapter_dir.rename(moved)
    chapter_dir.symlink_to(moved, target_is_directory=True)
    assert registered_lane_ledger_paths(checkpoint) == []


def test_explicit_native_legacy_scan_is_read_only_and_auto_never_mutates_rogue(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    source = _create_registered(tmp_path / "source-checkpoint")
    rogue = _ledger_path(checkpoint, "ch-legacy")
    rogue.parent.mkdir(parents=True)
    shutil.copy2(source, rogue)
    before = rogue.read_bytes()

    assert pipeline._all_lane_ledger_paths(checkpoint) == []
    assert pipeline._all_lane_ledger_paths(
        checkpoint, include_explicit_legacy=True,
    ) == [rogue]
    assert rogue.read_bytes() == before


def test_auto_rejects_preexisting_transaction_bound_to_unregistered_legacy_ledger(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    fingerprint = "a" * 64
    checkpoint = (
        project / ".arc-companion" / "checkpoints" / fingerprint
    )
    source = _create_registered(tmp_path / "source-checkpoint")
    rogue = _ledger_path(checkpoint, "ch-legacy")
    rogue.parent.mkdir(parents=True)
    shutil.copy2(source, rogue)
    before = rogue.read_bytes()
    project.mkdir(parents=True, exist_ok=True)
    (project / "state.json").write_text(json.dumps({
        "status": "needs_supervision",
        "checkpoint_dir": str(checkpoint),
        "fingerprint": fingerprint,
        "recovery_options": {"paper_id": "local:test", "workers": 1},
    }), encoding="utf-8")
    begin_transaction(
        project,
        action="auto",
        recovery_options={"paper_id": "local:test", "workers": 1},
        entries=[{
            "ledger_path": str(rogue),
            "session_key": "ch-0001:translation",
            "segment_id": "ch-0001.seg-0001",
            "initial_generation": 1,
        }],
        checkpoint_path=checkpoint,
        checkpoint_fingerprint=fingerprint,
    )

    result = pipeline._resume_companion_unlocked(project, action="auto")

    assert result["ok"] is False
    assert result["error"]["code"] == "automatic_recovery_ledger_unregistered"
    assert result["meta"]["provider_calls"] == 0
    assert rogue.read_bytes() == before

    value = json.loads(before)
    assert pipeline._validate_recovery_ledger_address(
        checkpoint_dir=checkpoint,
        ledger_path=rogue.resolve(),
        ledger=value,
        session_key="ch-0001:translation",
        segment_id="ch-0001.seg-0001",
        generation=1,
        allow_explicit_legacy=True,
    ) is None
    assert rogue.read_bytes() == before


def test_registry_writer_enforces_entry_bound(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint"
    monkeypatch.setattr(registry_module, "MAX_REGISTRY_ENTRIES", 1)
    _create_registered(checkpoint, "ch-0001")

    with pytest.raises(LaneLedgerRegistryError, match="entry limit"):
        _create_registered(checkpoint, "ch-0002")


def test_ledger_api_refreshes_hash_after_normal_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    mark_needs_supervision(
        path,
        segment_id="ch-0001.seg-0001",
        reason="test",
        recovery_context={"submission_state": "unknown"},
    )

    assert registered_lane_ledger_paths(checkpoint) == [path]
    entry = json.loads(
        (checkpoint / REGISTRY_FILE_NAME).read_text(encoding="utf-8")
    )["entries"][0]
    assert entry["ledger_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_registered_read_rejects_changed_bytes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    original, digest = read_registered_lane_ledger(checkpoint, path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    changed = dict(original)
    changed["generation"] = 2
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(LaneLedgerRegistryError, match="bytes changed"):
        read_registered_lane_ledger(checkpoint, path)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_registered_cas_parent_symlink_swap_never_mutates_external_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    original = path.read_bytes()
    _ledger, digest = read_registered_lane_ledger(checkpoint, path)
    external = tmp_path / "external"
    external.mkdir()
    outside = external / path.name
    outside.write_bytes(b"outside sentinel\n")
    outside_before = outside.read_bytes()
    moved = checkpoint / "saved-chapter"

    def swap_parent(_path: Path) -> None:
        path.parent.rename(moved)
        path.parent.symlink_to(external, target_is_directory=True)

    monkeypatch.setattr(
        registry_module, "_before_registered_ledger_replace", swap_parent,
    )

    with pytest.raises(LaneLedgerError, match="changed before mutation"):
        mark_needs_supervision(
            path,
            segment_id="ch-0001.seg-0001",
            reason="test",
            recovery_context={"submission_state": "unknown"},
            expected_ledger_sha256=digest,
            checkpoint_dir=checkpoint,
        )

    assert outside.read_bytes() == outside_before
    assert (moved / path.name).read_bytes() == original


def test_registered_cas_leaf_swap_never_overwrites_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    _ledger, digest = read_registered_lane_ledger(checkpoint, path)
    stale = b"stale replacement sentinel\n"

    def swap_leaf(_path: Path) -> None:
        path.unlink()
        path.write_bytes(stale)

    monkeypatch.setattr(
        registry_module, "_before_registered_ledger_replace", swap_leaf,
    )

    with pytest.raises(LaneLedgerError, match="changed before mutation"):
        mark_needs_supervision(
            path,
            segment_id="ch-0001.seg-0001",
            reason="test",
            recovery_context={"submission_state": "unknown"},
            expected_ledger_sha256=digest,
            checkpoint_dir=checkpoint,
        )

    assert path.read_bytes() == stale


def test_registered_cas_reconciles_crash_after_ledger_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    _ledger, digest = read_registered_lane_ledger(checkpoint, path)

    class SimulatedCrash(BaseException):
        pass

    def crash_after_replace(_path: Path) -> None:
        raise SimulatedCrash

    monkeypatch.setattr(
        registry_module, "_after_registered_ledger_replace", crash_after_replace,
    )
    with pytest.raises(SimulatedCrash):
        mark_needs_supervision(
            path,
            segment_id="ch-0001.seg-0001",
            reason="test",
            recovery_context={"submission_state": "unknown"},
            expected_ledger_sha256=digest,
            checkpoint_dir=checkpoint,
        )

    changed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert changed_digest != digest
    registry_before = json.loads(
        (checkpoint / REGISTRY_FILE_NAME).read_text(encoding="utf-8")
    )
    assert registry_before["entries"][0]["ledger_sha256"] == digest
    journal = checkpoint / registry_module.MUTATION_JOURNAL_FILE_NAME
    assert journal.is_file()

    recovered, recovered_digest = read_registered_lane_ledger(checkpoint, path)

    assert recovered_digest == changed_digest
    assert recovered["needs_supervision"]["segment_id"] == "ch-0001.seg-0001"
    assert not journal.exists()
    registry_after = json.loads(
        (checkpoint / REGISTRY_FILE_NAME).read_text(encoding="utf-8")
    )
    assert registry_after["entries"][0]["ledger_sha256"] == changed_digest


@pytest.mark.parametrize(
    ("cutpoint", "new_bytes_were_published"),
    [
        ("_after_mutation_journal_publish", False),
        ("_after_registered_ledger_replace", True),
        ("_after_registered_registry_update", True),
        ("_before_mutation_journal_clear", True),
        ("_after_mutation_journal_clear", True),
    ],
)
def test_normal_write_reconciles_every_durable_crash_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cutpoint: str,
    new_bytes_were_published: bool,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    before = path.read_bytes()

    class SimulatedCrash(BaseException):
        pass

    def crash(_path: Path) -> None:
        raise SimulatedCrash

    monkeypatch.setattr(registry_module, cutpoint, crash)
    with pytest.raises(SimulatedCrash):
        # No explicit CAS/checkpoint arguments: this exercises the production
        # ledger writer that previously replaced bytes before registration.
        mark_needs_supervision(
            path,
            segment_id="ch-0001.seg-0001",
            reason="test",
            recovery_context={"submission_state": "unknown"},
        )

    journal = checkpoint / registry_module.MUTATION_JOURNAL_FILE_NAME
    assert journal.exists() is (cutpoint != "_after_mutation_journal_clear")
    recovered, digest = read_registered_lane_ledger(checkpoint, path)

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (path.read_bytes() != before) is new_bytes_were_published
    assert bool(recovered.get("needs_supervision")) is new_bytes_were_published
    assert not journal.exists()


@pytest.mark.parametrize(
    "cutpoint",
    [
        "_after_mutation_journal_publish",
        "_after_registered_ledger_replace",
        "_after_registered_registry_update",
        "_before_mutation_journal_clear",
        "_after_mutation_journal_clear",
    ],
)
def test_initial_creation_reconciles_every_durable_crash_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cutpoint: str,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _ledger_path(checkpoint)

    class SimulatedCrash(BaseException):
        pass

    def crash(_path: Path) -> None:
        raise SimulatedCrash

    monkeypatch.setattr(registry_module, cutpoint, crash)
    with pytest.raises(SimulatedCrash):
        _create_registered(checkpoint)

    journal = checkpoint / registry_module.MUTATION_JOURNAL_FILE_NAME
    if cutpoint == "_after_mutation_journal_publish":
        # The intent was durable but the initial ledger did not exist yet.
        # Reconciliation aborts that uncommitted creation; a retry starts clean.
        assert not path.exists()
        assert registered_lane_ledger_paths(checkpoint) == []
        assert not journal.exists()
        monkeypatch.setattr(registry_module, cutpoint, lambda _path: None)
        _create_registered(checkpoint)
    else:
        # Once the initial bytes were renamed into place, reconciliation always
        # completes registration, including crashes on either side of clear.
        assert path.is_file()
        assert registered_lane_ledger_paths(checkpoint) == [path]

    recovered, digest = read_registered_lane_ledger(checkpoint, path)
    assert recovered["chapter_id"] == "ch-0001"
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not journal.exists()


def test_unregistered_existing_ledger_adoption_is_journaled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_checkpoint = tmp_path / "source"
    source = _create_registered(source_checkpoint)
    checkpoint = tmp_path / "checkpoint"
    path = _ledger_path(checkpoint)
    path.parent.mkdir(parents=True)
    path.write_bytes(source.read_bytes())

    class SimulatedCrash(BaseException):
        pass

    def crash(_path: Path) -> None:
        raise SimulatedCrash

    monkeypatch.setattr(
        registry_module, "_after_registered_registry_update", crash,
    )
    with pytest.raises(SimulatedCrash):
        initialize_lane_ledger(
            path,
            chapter_id="ch-0001",
            lane="translation",
            segment_ids=["ch-0001.seg-0001"],
            checkpoint_dir=checkpoint,
        )

    assert registered_lane_ledger_paths(checkpoint) == [path]
    assert not (checkpoint / registry_module.MUTATION_JOURNAL_FILE_NAME).exists()


def test_root_replacement_after_ledger_publish_never_splits_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    _ledger, digest = read_registered_lane_ledger(checkpoint, path)
    moved = tmp_path / "moved-checkpoint"

    def replace_root(_path: Path) -> None:
        checkpoint.rename(moved)
        checkpoint.mkdir()

    monkeypatch.setattr(
        registry_module, "_after_registered_ledger_replace", replace_root,
    )
    with pytest.raises(LaneLedgerError, match="changed before mutation"):
        mark_needs_supervision(
            path,
            segment_id="ch-0001.seg-0001",
            reason="test",
            recovery_context={"submission_state": "unknown"},
            expected_ledger_sha256=digest,
            checkpoint_dir=checkpoint,
        )

    assert list(checkpoint.iterdir()) == []
    moved_path = moved / "chapters/ch-0001/translation-ledger.json"
    assert moved_path.is_file()
    assert (moved / registry_module.MUTATION_JOURNAL_FILE_NAME).is_file()
    monkeypatch.setattr(
        registry_module, "_after_registered_ledger_replace", lambda _path: None,
    )
    recovered, recovered_digest = read_registered_lane_ledger(moved, moved_path)
    assert recovered["needs_supervision"]["segment_id"] == "ch-0001.seg-0001"
    assert recovered_digest == hashlib.sha256(moved_path.read_bytes()).hexdigest()


def test_registry_leaf_cas_preserves_unexpected_regular_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    sentinel = b'{"unexpected":"registry replacement"}\n'

    def replace_registry(control_path: Path) -> None:
        if control_path.name == registry_module.REGISTRY_FILE_NAME:
            control_path.unlink()
            control_path.write_bytes(sentinel)

    monkeypatch.setattr(
        registry_module, "_before_control_leaf_replace", replace_registry,
    )
    with pytest.raises(LaneLedgerRegistryError, match="changed before atomic"):
        _persist_registered(
            path,
            {**json.loads(path.read_text(encoding="utf-8")), "updated_at": 123.0},
            checkpoint_dir=checkpoint,
        )
    assert (checkpoint / registry_module.REGISTRY_FILE_NAME).read_bytes() == sentinel


def test_registry_exchange_cas_rolls_back_late_regular_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    registry_path = checkpoint / registry_module.REGISTRY_FILE_NAME
    sentinel = b'{"unexpected":"late registry replacement"}\n'

    def replace_at_exchange(control_path: Path) -> None:
        if control_path.name == registry_module.REGISTRY_FILE_NAME:
            registry_path.unlink()
            registry_path.write_bytes(sentinel)

    monkeypatch.setattr(
        registry_module, "_before_control_leaf_exchange", replace_at_exchange,
    )
    with pytest.raises(LaneLedgerRegistryError, match="changed during atomic"):
        _persist_registered(
            path,
            {**json.loads(path.read_text(encoding="utf-8")), "updated_at": 123.0},
            checkpoint_dir=checkpoint,
        )
    assert registry_path.read_bytes() == sentinel


def test_ledger_exchange_cas_rolls_back_late_regular_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    sentinel = b"late ledger replacement\n"

    def replace_at_exchange(_ledger_path: Path) -> None:
        path.unlink()
        path.write_bytes(sentinel)

    monkeypatch.setattr(
        registry_module, "_before_registered_ledger_exchange", replace_at_exchange,
    )
    with pytest.raises(LaneLedgerError, match="changed before mutation"):
        mark_needs_supervision(
            path,
            segment_id="ch-0001.seg-0001",
            reason="test",
            recovery_context={"submission_state": "unknown"},
        )
    assert path.read_bytes() == sentinel


def test_existing_ledger_update_fails_closed_without_atomic_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    before = path.read_bytes()
    monkeypatch.setattr(registry_module, "_rename_exchange", lambda *_args: False)

    with pytest.raises(LaneLedgerError, match="changed before mutation"):
        mark_needs_supervision(
            path,
            segment_id="ch-0001.seg-0001",
            reason="test",
            recovery_context={"submission_state": "unknown"},
        )
    assert path.read_bytes() == before


def test_journal_publish_is_true_cas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    sentinel = b'{"unexpected":"journal replacement"}\n'
    journal_path = checkpoint / registry_module.MUTATION_JOURNAL_FILE_NAME

    def replace_before_publish(control_path: Path) -> None:
        if control_path.name == registry_module.MUTATION_JOURNAL_FILE_NAME:
            control_path.write_bytes(sentinel)

    monkeypatch.setattr(
        registry_module, "_before_control_leaf_replace", replace_before_publish,
    )
    with pytest.raises(LaneLedgerRegistryError, match="changed before atomic"):
        _persist_registered(
            path,
            {**json.loads(path.read_text(encoding="utf-8")), "updated_at": 123.0},
            checkpoint_dir=checkpoint,
        )
    assert journal_path.read_bytes() == sentinel


def test_journal_late_publish_and_clear_cas_preserve_replacements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    journal_path = checkpoint / registry_module.MUTATION_JOURNAL_FILE_NAME
    sentinel = b'{"unexpected":"late journal replacement"}\n'

    def replace_at_publish(control_path: Path) -> None:
        if control_path.name == registry_module.MUTATION_JOURNAL_FILE_NAME:
            journal_path.write_bytes(sentinel)

    monkeypatch.setattr(
        registry_module, "_before_control_leaf_exchange", replace_at_publish,
    )
    with pytest.raises(LaneLedgerRegistryError, match="appeared before atomic"):
        _persist_registered(
            path,
            {**json.loads(path.read_text(encoding="utf-8")), "updated_at": 123.0},
            checkpoint_dir=checkpoint,
        )
    assert journal_path.read_bytes() == sentinel

    journal_path.unlink()
    monkeypatch.setattr(
        registry_module, "_before_control_leaf_exchange", lambda _path: None,
    )

    def replace_at_clear(_control_path: Path) -> None:
        journal_path.unlink()
        journal_path.write_bytes(sentinel)

    monkeypatch.setattr(
        registry_module, "_before_journal_clear_rename", replace_at_clear,
    )
    with pytest.raises(LaneLedgerRegistryError, match="changed during clear"):
        _persist_registered(
            path,
            {**json.loads(path.read_text(encoding="utf-8")), "updated_at": 456.0},
            checkpoint_dir=checkpoint,
        )
    assert journal_path.read_bytes() == sentinel

def test_named_registry_lock_replacement_aborts_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    before = path.read_bytes()
    lock_path = checkpoint / f".{registry_module.REGISTRY_FILE_NAME}.lock"

    def replace_lock(control_path: Path) -> None:
        if control_path.name == registry_module.MUTATION_JOURNAL_FILE_NAME:
            lock_path.unlink()
            lock_path.write_bytes(b"replacement lock\n")

    monkeypatch.setattr(
        registry_module, "_before_control_leaf_replace", replace_lock,
    )
    with pytest.raises(LaneLedgerRegistryError, match="lock inode changed"):
        _persist_registered(
            path,
            {**json.loads(path.read_text(encoding="utf-8")), "updated_at": 123.0},
            checkpoint_dir=checkpoint,
        )
    assert path.read_bytes() == before
    assert lock_path.read_bytes() == b"replacement lock\n"


@pytest.mark.skipif(os.name != "posix", reason="requires spawned POSIX writer")
def test_replaced_named_lock_cannot_authorize_overwrite_of_concurrent_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    lock_path = checkpoint / f".{registry_module.REGISTRY_FILE_NAME}.lock"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    result = context.Queue()
    writer = context.Process(
        target=_concurrent_registry_writer,
        args=(str(checkpoint), str(path), start, result),
    )
    writer.start()

    def replace_lock_and_release_writer(control_path: Path) -> None:
        if control_path.name != registry_module.MUTATION_JOURNAL_FILE_NAME:
            return
        lock_path.unlink()
        lock_path.write_bytes(b"replacement lock\n")
        start.set()
        assert result.get(timeout=15) == (True, "")

    monkeypatch.setattr(
        registry_module,
        "_before_control_leaf_replace",
        replace_lock_and_release_writer,
    )
    try:
        with pytest.raises(LaneLedgerRegistryError, match="lock inode changed"):
            _persist_registered(
                path,
                {**json.loads(path.read_text(encoding="utf-8")), "updated_at": 111.0},
                checkpoint_dir=checkpoint,
            )
    finally:
        writer.join(15)
        if writer.is_alive():
            writer.kill()
            writer.join()
    assert writer.exitcode == 0
    monkeypatch.setattr(
        registry_module, "_before_control_leaf_replace", lambda _path: None,
    )
    recovered, _digest = read_registered_lane_ledger(checkpoint, path)
    assert recovered["updated_at"] == 222.0


def test_named_lock_is_revalidated_before_success_and_unlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    lock_path = checkpoint / f".{registry_module.REGISTRY_FILE_NAME}.lock"

    def replace_after_commit(_ledger_path: Path) -> None:
        lock_path.unlink()
        lock_path.write_bytes(b"late replacement lock\n")

    monkeypatch.setattr(
        registry_module, "_after_mutation_journal_clear", replace_after_commit,
    )
    with pytest.raises(LaneLedgerRegistryError, match="lock inode changed"):
        _persist_registered(
            path,
            {**json.loads(path.read_text(encoding="utf-8")), "updated_at": 333.0},
            checkpoint_dir=checkpoint,
        )
    assert lock_path.read_bytes() == b"late replacement lock\n"
    monkeypatch.setattr(
        registry_module, "_after_mutation_journal_clear", lambda _path: None,
    )
    recovered, _digest = read_registered_lane_ledger(checkpoint, path)
    assert recovered["updated_at"] == 333.0


@pytest.mark.skipif(os.name != "posix", reason="requires spawned POSIX writers")
def test_distinct_production_rmw_operations_reapply_without_lost_update(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _ledger_path(checkpoint)
    initialize_lane_ledger(
        path,
        chapter_id="ch-0001",
        lane="translation",
        segment_ids=["ch-0001.seg-0001", "ch-0001.seg-0002"],
        checkpoint_dir=checkpoint,
    )
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    result = context.Queue()
    writers = [
        context.Process(
            target=_distinct_supervision_writer,
            args=(str(path), segment_id, start, result),
        )
        for segment_id in ("ch-0001.seg-0001", "ch-0001.seg-0002")
    ]
    for writer in writers:
        writer.start()
    start.set()
    outcomes = [result.get(timeout=20) for _writer in writers]
    for writer in writers:
        writer.join(20)
        if writer.is_alive():
            writer.kill()
            writer.join()
        assert writer.exitcode == 0
    assert all(item[0] for item in outcomes), outcomes

    ledger, _digest = read_registered_lane_ledger(checkpoint, path)
    assert {
        item["segment_id"] for item in ledger["supervision_entries"]
    } == {"ch-0001.seg-0001", "ch-0001.seg-0002"}


@pytest.mark.skipif(os.name != "posix", reason="requires spawned POSIX writers")
def test_create_only_race_never_overwrites_a_different_ledger_identity(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _ledger_path(checkpoint)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    result = context.Queue()
    segment_ids = ("ch-0001.seg-A", "ch-0001.seg-B")
    writers = [
        context.Process(
            target=_racing_identity_initializer,
            args=(str(checkpoint), str(path), segment_id, start, result),
        )
        for segment_id in segment_ids
    ]
    for writer in writers:
        writer.start()
    start.set()
    outcomes = [result.get(timeout=20) for _writer in writers]
    for writer in writers:
        writer.join(20)
        if writer.is_alive():
            writer.kill()
            writer.join()
        assert writer.exitcode == 0

    winners = [item for item in outcomes if item[0]]
    losers = [item for item in outcomes if not item[0]]
    assert len(winners) == 1, outcomes
    assert len(losers) == 1, outcomes
    assert "identity changed" in losers[0][3]
    ledger, digest = read_registered_lane_ledger(checkpoint, path)
    assert [item["segment_id"] for item in ledger["blocks"]] == [winners[0][1]]
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_stale_transition_callback_cannot_advance_a_new_generation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    segment_id = "ch-0001.seg-0001"
    stale = lane_transition_guard(
        path,
        segment_id=segment_id,
        session_key="ch-0001:translation",
        idempotency_key="turn-1",
        checkpoint_dir=checkpoint,
    )
    invalidate_suffix(
        path,
        from_segment_id=segment_id,
        generation=2,
        expected_ledger_sha256=stale.expected_ledger_sha256,
        checkpoint_dir=checkpoint,
    )

    with pytest.raises(LaneLedgerError, match="changed before mutation"):
        mark_submitted(
            path,
            segment_id=segment_id,
            expected_generation=stale.expected_generation,
            expected_ledger_sha256=stale.expected_ledger_sha256,
            authorization=stale.authorization,
            checkpoint_dir=checkpoint,
        )
    current, _digest = read_registered_lane_ledger(checkpoint, path)
    assert current["generation"] == 2
    assert current["blocks"][0]["state"] == "prepared"

    submitted_guard = lane_transition_guard(
        path,
        segment_id=segment_id,
        session_key="ch-0001:translation",
        idempotency_key="turn-2",
        checkpoint_dir=checkpoint,
    )
    mark_submitted(
        path,
        segment_id=segment_id,
        expected_generation=submitted_guard.expected_generation,
        expected_ledger_sha256=submitted_guard.expected_ledger_sha256,
        authorization=submitted_guard.authorization,
        checkpoint_dir=checkpoint,
    )
    response_guard = lane_transition_guard(
        path,
        segment_id=segment_id,
        session_key="ch-0001:translation",
        idempotency_key="turn-2",
        checkpoint_dir=checkpoint,
    )
    mark_response_received(
        path,
        segment_id=segment_id,
        expected_generation=response_guard.expected_generation,
        expected_ledger_sha256=response_guard.expected_ledger_sha256,
        authorization=response_guard.authorization,
        checkpoint_dir=checkpoint,
    )
    current, _digest = read_registered_lane_ledger(checkpoint, path)
    assert current["blocks"][0]["state"] == "response_received"
    assert current["blocks"][0]["submission_authorization"] == {
        "control_address": str(path.resolve()),
        "session_key": "ch-0001:translation",
        "logical_unit": segment_id,
        "generation": 2,
        "idempotency_key": "turn-2",
    }


def test_production_transition_rejects_unguarded_or_key_only_callback(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    segment_id = "ch-0001.seg-0001"
    with pytest.raises(LaneLedgerError, match="requires generation, digest"):
        mark_submitted(path, segment_id=segment_id)
    with pytest.raises(LaneLedgerError, match="requires generation, digest"):
        mark_submitted(
            path,
            segment_id=segment_id,
            expected_generation=1,
            authorization=(str(path.resolve()), "session", segment_id, 1, "key"),
        )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX process crash")
@pytest.mark.parametrize(
    "crash_point",
    [
        "ledger_stage",
        "journal_stage",
        "registry_stage",
        "journal_link_cleanup",
        "registry_exchange_cleanup",
        "journal_clear_cleanup",
    ],
)
def test_real_process_crash_stages_are_reconciled(
    tmp_path: Path, crash_point: str,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    path = _create_registered(checkpoint)
    before = path.read_bytes()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child is intentionally SIGKILLed
        if crash_point == "ledger_stage":
            registry_module._before_registered_ledger_replace = (
                lambda _path: os.kill(os.getpid(), signal.SIGKILL)
            )
        elif crash_point == "journal_stage":
            registry_module._before_control_leaf_replace = (
                lambda control_path: (
                    os.kill(os.getpid(), signal.SIGKILL)
                    if control_path.name == registry_module.MUTATION_JOURNAL_FILE_NAME
                    else None
                )
            )
        elif crash_point == "registry_stage":
            registry_module._before_control_leaf_replace = (
                lambda control_path: (
                    os.kill(os.getpid(), signal.SIGKILL)
                    if control_path.name == registry_module.REGISTRY_FILE_NAME
                    else None
                )
            )
        elif crash_point in {"journal_link_cleanup", "registry_exchange_cleanup"}:
            registry_module._after_control_leaf_publish_before_stage_cleanup = (
                lambda control_path: (
                    os.kill(os.getpid(), signal.SIGKILL)
                    if (
                        control_path.name
                        == (
                            registry_module.MUTATION_JOURNAL_FILE_NAME
                            if crash_point == "journal_link_cleanup"
                            else registry_module.REGISTRY_FILE_NAME
                        )
                    )
                    else None
                )
            )
        else:
            registry_module._after_journal_clear_rename = (
                lambda _path: os.kill(os.getpid(), signal.SIGKILL)
            )
        _persist_registered(
            path,
            {**json.loads(path.read_text(encoding="utf-8")), "updated_at": 123.0},
            checkpoint_dir=checkpoint,
        )
        os._exit(99)
    _finished, status = os.waitpid(pid, 0)
    assert os.WIFSIGNALED(status)
    assert os.WTERMSIG(status) == signal.SIGKILL

    recovered, digest = read_registered_lane_ledger(checkpoint, path)
    ledger_was_published = crash_point in {
        "registry_stage", "registry_exchange_cleanup", "journal_clear_cleanup",
    }
    assert (path.read_bytes() != before) is ledger_was_published
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert bool(recovered["updated_at"] == 123.0) is ledger_was_published
    assert not list(checkpoint.rglob("*.arc-stage-*"))


def test_automatic_reconstruction_never_uses_rglob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manager = LLMSessionManager(checkpoint / "sessions")

    def forbidden_rglob(*_args, **_kwargs):
        raise AssertionError("automatic reconstruction must be index-only")

    monkeypatch.setattr(Path, "rglob", forbidden_rglob)
    assert pipeline._reconstruct_unresolved_native_resume_contexts(
        checkpoint, session_manager=manager, excluded_keys=set(),
    ) == []


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_entry_checkpoint_rejects_symlink_and_oversize_exact_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    artifact = checkpoint / "llm" / "call"
    calls = artifact / "call-checkpoints"
    calls.mkdir(parents=True)
    key = "logical-call"
    name = f"idempotency-{hashlib.sha256(key.encode()).hexdigest()}.json"
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    symlink = calls / name
    symlink.symlink_to(target)
    entry = {
        "idempotency_key": key,
        "session_key": "ch:translation",
        "segment_id": "s1",
        "initial_generation": 1,
        "recovery_context": {
            "idempotency_key": key,
            "checkpoint_path": str(symlink),
            "logical_unit": "s1",
        },
    }
    assert pipeline._entry_call_checkpoint(entry, checkpoint) == (None, None)

    symlink.unlink()
    symlink.write_bytes(b"{" + b"x" * 128 + b"}")
    monkeypatch.setattr(pipeline, "_MAX_RECOVERY_CONTROL_BYTES", 32)
    assert pipeline._entry_call_checkpoint(entry, checkpoint) == (None, None)
