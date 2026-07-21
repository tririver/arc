from __future__ import annotations

import json
from pathlib import Path

import pytest

import arc_companion.pipeline as pipeline
from arc_companion.ledger import (
    advance_block,
    initialize_lane_ledger,
    invalidate_suffix,
    mark_needs_supervision,
)
from arc_companion.resume_transaction import begin_transaction
from arc_llm.sessions import LLMSessionManager


SESSION_KEY = "ch-0001:translation"


def _project(tmp_path: Path) -> tuple[Path, Path, Path, LLMSessionManager]:
    project = tmp_path / "run"
    checkpoint = project / "checkpoint"
    ledger_path = checkpoint / "chapters" / "ch-0001" / "translation-ledger.json"
    initialize_lane_ledger(
        ledger_path,
        chapter_id="ch-0001",
        lane="translation",
        segment_ids=["s1", "s2"],
    )
    manager = LLMSessionManager(checkpoint / "sessions")
    manager.get_or_create(
        key=SESSION_KEY,
        provider="codex-cli",
        model="test-model",
        runtime_fingerprint="runtime",
    )
    manager.update_native_session_id(SESSION_KEY, "native-1")
    project.mkdir(parents=True, exist_ok=True)
    (project / "state.json").write_text(json.dumps({
        "status": "needs_supervision",
        "checkpoint_dir": str(checkpoint),
        "recovery_options": {
            "paper_id": "local:auto-native-multi",
            "workers": 1,
            "recovery_policy": "auto",
        },
    }), encoding="utf-8")
    return project, checkpoint, ledger_path, manager


def _accept_remaining(path: Path) -> None:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    states = [
        "prepared", "submitted", "response_received", "schema_valid",
        "invariant_valid", "accepted",
    ]
    for block in ledger["blocks"]:
        segment_id = str(block["segment_id"])
        current = str(block["state"])
        for state in states[states.index(current) + 1:]:
            advance_block(
                path,
                segment_id=segment_id,
                state=state,
                input_sha256=f"input-{segment_id}",
                output_sha256=f"output-{segment_id}",
            )


def test_same_lane_resumable_blockers_are_all_native_reconciled_before_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _checkpoint, ledger_path, manager = _project(tmp_path)
    keys = {
        segment_id: f"{SESSION_KEY}:call-{segment_id}:generation-1"
        for segment_id in ("s1", "s2")
    }
    for segment_id in ("s1", "s2"):
        mark_needs_supervision(
            ledger_path,
            segment_id=segment_id,
            reason=f"submitted {segment_id}",
            recovery_context={
                "idempotency_key": keys[segment_id],
                "submission_state": "submitted",
                "resumable": True,
                "native_session_id": "native-1",
                "session_key": SESSION_KEY,
                "generation": 1,
            },
        )

    validated_segments: list[str] = []

    def validate_context(**kwargs):
        ledger = kwargs["ledger"]
        supervision = kwargs.get("supervision") or ledger["needs_supervision"]
        segment_id = str(supervision["segment_id"])
        validated_segments.append(segment_id)
        return {
            "session_key": SESSION_KEY,
            "segment_id": segment_id,
            "ledger_path": str(kwargs["ledger_path"]),
            "idempotency_key": keys[segment_id],
            "provider": "codex-cli",
            "model": "test-model",
            "runtime_fingerprint": "runtime",
            "generation": 1,
            "native_session_id_to_restore": None,
        }

    captured_keys: list[str] = []

    def continuation(options):
        captured_keys.extend(options.supervised_native_resume_keys)
        _accept_remaining(ledger_path)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "_validate_native_resume_context", validate_context)
    monkeypatch.setattr(pipeline, "build_companion", continuation)

    result = pipeline.resume_companion(project, action="auto")

    assert result["ok"] is True
    assert validated_segments == ["s1", "s2"]
    assert set(captured_keys) == set(keys.values())
    assert manager.get_existing(SESSION_KEY).generation == 1


def test_rotated_generation_does_not_reuse_stale_native_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _checkpoint, ledger_path, manager = _project(tmp_path)
    stale_key = f"{SESSION_KEY}:call-s1:generation-1"
    manager.rotate(SESSION_KEY, reason="crash after automatic rotate")
    invalidate_suffix(ledger_path, from_segment_id="s1", generation=2)
    mark_needs_supervision(
        ledger_path,
        segment_id="s1",
        reason="generation two continuation pending",
        recovery_context={
            "idempotency_key": f"{SESSION_KEY}:call-s1:generation-2",
            "submission_state": "submitted",
            "resumable": False,
            "generation": 2,
        },
    )
    begin_transaction(
        project,
        action="auto",
        policy="auto",
        recovery_options={
            "paper_id": "local:auto-native-multi",
            "workers": 1,
            "recovery_policy": "auto",
        },
        entries=[{
            "ledger_path": str(ledger_path),
            "session_key": SESSION_KEY,
            "segment_id": "s1",
            "idempotency_key": stale_key,
            "initial_generation": 1,
            "target_generation": 2,
            "recovery_action": "generation_restart_required",
        }],
        native_resume_contexts=[{
            "ledger_path": str(ledger_path),
            "session_key": SESSION_KEY,
            "segment_id": "s1",
            "idempotency_key": stale_key,
            "provider": "codex-cli",
            "model": "test-model",
            "runtime_fingerprint": "runtime",
            "generation": 1,
            "native_session_id_to_restore": "native-1",
        }],
    )
    captured_keys: list[str] = []

    def continuation(options):
        captured_keys.extend(options.supervised_native_resume_keys)
        _accept_remaining(ledger_path)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", continuation)

    result = pipeline.resume_companion(project, action="auto")

    assert result["ok"] is True
    assert stale_key not in captured_keys
    assert manager.get_existing(SESSION_KEY).generation == 2
    assert json.loads(ledger_path.read_text(encoding="utf-8"))["generation"] == 2
