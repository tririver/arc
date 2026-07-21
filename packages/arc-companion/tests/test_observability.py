from __future__ import annotations

from datetime import datetime, timezone
import json

from arc_companion.observability import append_state_event, enrich_status
from arc_companion.run_lock import ProjectBuildLock, inspect_lock


def test_lock_diagnostics_include_owner_start_and_live_identity(tmp_path) -> None:
    path = tmp_path / ".arc-companion-build.lock"
    lock = ProjectBuildLock(path)
    lock.acquire()
    try:
        owner = inspect_lock(path)
        assert owner is not None
        assert owner["active"] is True
        assert owner["started_at"]
        assert owner["process_identity_matches"] in {True, None}
    finally:
        lock.release()

    assert inspect_lock(path)["active"] is False


def test_status_reports_wait_reason_call_counts_and_phase_timings(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    call_path = checkpoint / "chapter" / "call-checkpoints" / "call.json"
    call_path.parent.mkdir(parents=True)
    call_path.write_text(json.dumps({"state": "submitted"}), encoding="utf-8")
    state_path = tmp_path / "state.json"
    state = {
        "status": "generating",
        "checkpoint_dir": str(checkpoint),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    append_state_event(state_path, state)

    result = enrich_status(tmp_path, state)

    assert result["current_phase"] == "generating"
    assert result["wait_reason"] == "provider_response"
    assert result["calls"] == {"active": 1, "queued": 0, "draining": 0}
    assert result["last_progress_at"]
    assert "generating" in result["phase_elapsed_seconds"]


def test_terminal_phase_elapsed_does_not_grow_after_completion(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    state = {
        "status": "complete",
        "updated_at": "2020-01-01T00:00:00+00:00",
    }
    append_state_event(state_path, state)

    result = enrich_status(tmp_path, state)

    assert result["phase_elapsed_seconds"]["complete"] == 0.0
