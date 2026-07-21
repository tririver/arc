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
    assert result["pending_call_count"] == 1
    assert result["pending_calls"] == [{
        "checkpoint": "chapter/call-checkpoints/call.json",
        "idempotency_key": None,
        "session_key": None,
        "generation": None,
        "state": "submitted",
        "submission_state": "unknown",
        "recovery_action": "operator-supervision",
        "blocking_reason": "submitted_call_without_an_accepted_response",
        "entry_status": None,
    }]
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


def test_status_merges_unresolved_resume_transaction_entries(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    ledger_dir = checkpoint / "production" / "lanes"
    ledger_dir.mkdir(parents=True)
    entries = []
    native_contexts = []
    for index, status in enumerate(
        ("reconciling", "pending", "authorized", "pending", "resolved"), 1,
    ):
        chapter_id = f"ch-{index:04d}"
        segment_id = f"{chapter_id}.seg-0001"
        key = f"{chapter_id}:translation:call-{segment_id}:generation-1"
        ledger_path = ledger_dir / f"{chapter_id}-translation-ledger.json"
        ledger_path.write_text(json.dumps({
            "chapter_id": chapter_id, "lane": "translation",
            "needs_supervision": ({
                "segment_id": segment_id,
                "reason": f"operator reason {index}",
                "recovery_context": {"submission_state": "unknown"},
            } if status != "resolved" else None),
            "blocks": [{
                "segment_id": segment_id,
                "state": "submitted" if index != 3 else "prepared",
                "submission_state": "submitted" if index != 3 else "not_submitted",
            }],
        }))
        entry = {
            "ledger_path": str(ledger_path),
            "session_key": f"{chapter_id}:translation",
            "segment_id": segment_id,
            "idempotency_key": key if index != 4 else "",
            "initial_generation": 1,
            "status": status,
        }
        if index == 2:
            entry["blocking_reason"] = "identity conflict requires operator"
        entries.append(entry)
        if index in {1, 3}:
            native_contexts.append({
                "idempotency_key": key,
                "session_key": f"{chapter_id}:translation",
            })
        if index == 1:
            call_path = checkpoint / "production" / "calls" / "call-checkpoints" / "call.json"
            call_path.parent.mkdir(parents=True)
            call_path.write_text(json.dumps({
                "state": "submitted", "submission_state": "submitted",
                "logical_identity": {
                    "idempotency_key": key,
                    "session_key": f"{chapter_id}:translation",
                    "generation": 1,
                },
            }))
    transaction_path = tmp_path / ".arc-companion" / "resume-transaction.json"
    transaction_path.parent.mkdir()
    transaction_path.write_text(json.dumps({
        "schema_version": "arc.companion.resume-transaction.v2",
        "action": "resume-native", "status": "continuation_failed",
        "entries": entries,
        "native_resume_contexts": native_contexts,
    }))
    state = {"status": "needs_supervision", "checkpoint_dir": str(checkpoint)}

    result = enrich_status(tmp_path, state)

    assert result["pending_call_count"] == 4
    by_session = {item["session_key"]: item for item in result["pending_calls"]}
    assert set(by_session) == {
        "ch-0001:translation", "ch-0002:translation",
        "ch-0003:translation", "ch-0004:translation",
    }
    assert by_session["ch-0001:translation"]["entry_status"] == "reconciling"
    assert by_session["ch-0001:translation"]["recovery_action"] == "resume-native"
    assert by_session["ch-0001:translation"]["submission_state"] == "submitted"
    assert by_session["ch-0002:translation"]["recovery_action"] == "operator-supervision"
    assert by_session["ch-0002:translation"]["blocking_reason"] == (
        "identity conflict requires operator"
    )
    assert by_session["ch-0003:translation"]["entry_status"] == "authorized"
    assert by_session["ch-0003:translation"]["state"] == "prepared"
    assert by_session["ch-0004:translation"]["recovery_action"] == "operator-supervision"


def test_status_reports_automatic_replacement_provenance(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    ledger = checkpoint / "translation-ledger.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({
        "chapter_id": "ch-0004", "lane": "translation",
        "blocks": [{
            "segment_id": "s8", "state": "prepared",
            "submission_state": "not_submitted", "generation": 2,
        }],
    }))
    journal = tmp_path / ".arc-companion" / "resume-transaction.json"
    journal.parent.mkdir()
    journal.write_text(json.dumps({
        "schema_version": "arc.companion.resume-transaction.v2",
        "action": "auto", "policy": "auto", "status": "continuing",
        "action_history": [{
            "action": "resume-native", "policy": "manual", "at": "before",
        }, {
            "action": "auto", "policy": "auto", "at": "after",
        }],
        "entries": [{
            "ledger_path": str(ledger), "session_key": "ch-0004:translation",
            "segment_id": "s8", "idempotency_key": "old-key", "status": "reconciling",
        }],
        "replacements": [{
            "replacement_id": "replacement-1", "session_key": "ch-0004:translation",
            "segment_id": "s8", "source_generation": 1, "target_generation": 2,
            "suffix_start_segment_id": "s8", "suffix_segment_ids": ["s8", "s9", "s10"],
            "attempt": 1, "trigger_code": "native_session_missing",
            "trigger_reason": "provider session no longer exists",
            "possible_duplicate_charge": True, "status": "suffix_invalidated",
            "abandoned_logical_key": "old-key",
        }],
    }))

    result = enrich_status(
        tmp_path, {"status": "needs_supervision", "checkpoint_dir": str(checkpoint)},
    )

    assert result["recovery_policy"] == "auto"
    assert [item["action"] for item in result["recovery_action_history"]] == [
        "resume-native", "auto",
    ]
    replacement = result["recovery_replacements"][0]
    assert replacement["source_generation"] == 1
    assert replacement["target_generation"] == 2
    assert replacement["suffix_segment_ids"] == ["s8", "s9", "s10"]
    assert replacement["restart_attempt"] == 1
    assert replacement["restart_trigger_code"] == "native_session_missing"
    assert replacement["possible_duplicate_charge"] is True
    pending = result["pending_calls"][0]
    assert pending["recovery_action"] == "restart-generation"
    assert pending["replacement_status"] == "suffix_invalidated"


def test_resolved_transaction_entry_hides_stale_failed_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    call_path = checkpoint / "llm" / "call-checkpoints" / "call.json"
    call_path.parent.mkdir(parents=True)
    call_path.write_text(json.dumps({
        "state": "failed", "submission_state": "submitted",
        "logical_identity": {
            "idempotency_key": "accepted-key",
            "session_key": "ch-0001:translation", "generation": 1,
        },
    }))
    journal = tmp_path / ".arc-companion" / "resume-transaction.json"
    journal.parent.mkdir()
    journal.write_text(json.dumps({
        "schema_version": "arc.companion.resume-transaction.v2",
        "action": "auto", "policy": "auto", "status": "complete",
        "entries": [{
            "ledger_path": str(checkpoint / "translation-ledger.json"),
            "session_key": "ch-0001:translation", "segment_id": "s1",
            "idempotency_key": "accepted-key", "status": "resolved",
        }],
    }))

    result = enrich_status(
        tmp_path, {"status": "complete", "checkpoint_dir": str(checkpoint)},
    )

    assert result["pending_call_count"] == 0
    assert result["pending_calls"] == []
