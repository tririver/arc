from __future__ import annotations

from datetime import datetime, timezone
import json

from arc_companion.observability import append_state_event, enrich_status
from arc_companion.artifact_ids import allocate_artifact_dir


OBS_FINGERPRINT = "a" * 64
from arc_companion.run_lock import ProjectBuildLock, inspect_lock


def _authorized_logical_identity(
    *, ledger, session: str, logical_unit: str, generation: int, key: str,
) -> dict[str, object]:
    control_address = str(ledger.resolve(strict=False))
    authorization = {
        "control_address": control_address,
        "session_key": session,
        "logical_unit": logical_unit,
        "generation": generation,
        "idempotency_key": key,
    }
    return {
        "provider": "fake",
        "model": "fake-model",
        "session_key": session,
        "generation": generation,
        "idempotency_key": key,
        "control_address": control_address,
        "logical_unit": logical_unit,
        "initial_native_authorization": authorization,
    }


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
    checkpoint = tmp_path / ".arc-companion" / "checkpoints" / OBS_FINGERPRINT
    call_path = checkpoint / "chapter" / "call-checkpoints" / "call.json"
    call_path.parent.mkdir(parents=True)
    call_path.write_text(json.dumps({"state": "submitted"}), encoding="utf-8")
    state_path = tmp_path / "state.json"
    state = {
        "status": "generating",
        "fingerprint": OBS_FINGERPRINT, "checkpoint_dir": str(checkpoint),
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
        "control_identity": None,
        "logical_identity": {
            "provider": None, "model": None, "session_key": None,
            "generation": None, "idempotency_key": None,
            "control_address": None, "logical_unit": None,
        },
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
    checkpoint = tmp_path / ".arc-companion" / "checkpoints" / OBS_FINGERPRINT
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
                "ledger_path": str(ledger_path),
                "segment_id": segment_id,
                "generation": 1,
            })
        if index == 1:
            call_path = checkpoint / "production" / "calls" / "call-checkpoints" / "call.json"
            call_path.parent.mkdir(parents=True)
            call_path.write_text(json.dumps({
                "state": "submitted", "submission_state": "submitted",
                "logical_identity": _authorized_logical_identity(
                    ledger=ledger_path,
                    session=f"{chapter_id}:translation",
                    logical_unit=segment_id, generation=1, key=key,
                ),
            }))
    transaction_path = tmp_path / ".arc-companion" / "resume-transaction.json"
    transaction_path.parent.mkdir(exist_ok=True)
    transaction_path.write_text(json.dumps({
        "schema_version": "arc.companion.resume-transaction.v2",
        "action": "resume-native", "status": "continuation_failed",
        "entries": entries,
        "native_resume_contexts": native_contexts,
    }))
    state = {"status": "needs_supervision", "fingerprint": OBS_FINGERPRINT, "checkpoint_dir": str(checkpoint)}

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
    checkpoint = tmp_path / ".arc-companion" / "checkpoints" / OBS_FINGERPRINT
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
    journal.parent.mkdir(exist_ok=True)
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
            "segment_id": "s8", "idempotency_key": "old-key",
            "initial_generation": 1, "status": "reconciling",
        }],
        "replacements": [{
            "replacement_id": "replacement-1", "session_key": "ch-0004:translation",
            "segment_id": "s8", "source_generation": 1, "target_generation": 2,
            "ledger_path": str(ledger),
            "suffix_start_segment_id": "s8", "suffix_segment_ids": ["s8", "s9", "s10"],
            "attempt": 1, "trigger_code": "native_session_missing",
            "trigger_reason": "provider session no longer exists",
            "possible_duplicate_charge": True, "status": "suffix_invalidated",
            "abandoned_logical_key": "old-key",
        }],
    }))

    result = enrich_status(
        tmp_path, {"status": "needs_supervision", "fingerprint": OBS_FINGERPRINT, "checkpoint_dir": str(checkpoint)},
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


def test_status_reports_typed_fresh_generation_without_response_or_prompt(tmp_path) -> None:
    checkpoint = tmp_path / ".arc-companion" / "checkpoints" / OBS_FINGERPRINT
    ledger = checkpoint / "translation-ledger.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({
        "chapter_id": "ch-0004", "lane": "translation",
        "needs_supervision": {
            "segment_id": "s2", "reason": "idle",
            "recovery_context": {"resumable": True},
        },
        "blocks": [
            {"segment_id": "s1", "state": "prepared",
             "submission_state": "not_submitted", "generation": 1},
            {"segment_id": "s2", "state": "submitted",
             "submission_state": "submitted", "generation": 1},
        ],
    }))
    journal = tmp_path / ".arc-companion" / "resume-transaction.json"
    journal.parent.mkdir(exist_ok=True)
    journal.write_text(json.dumps({
        "schema_version": "arc.companion.resume-transaction.v3",
        "action": "auto", "policy": "auto", "status": "continuing",
        "entries": [{
            "ledger_path": str(ledger), "session_key": "ch-0004:translation",
            "segment_id": "s2", "idempotency_key": "old-key", "status": "pending",
            "recovery_trigger": "idle_timeout",
            "automatic_native_resume_suppressed": True,
            "fresh_generation_required": True,
            "fresh_task_start_segment_id": "s1",
            "recovery_context": {
                "native_session_id": "must-not-be-exposed",
                "resumable": True,
                "raw_response": "must-not-be-exposed",
                "prompt": "must-not-be-exposed",
            },
        }],
    }))

    result = enrich_status(
        tmp_path, {"status": "needs_supervision", "fingerprint": OBS_FINGERPRINT, "checkpoint_dir": str(checkpoint)},
    )

    pending = result["pending_calls"][0]
    assert pending["recovery_action"] == "fresh-generation"
    assert pending["recovery_trigger"] == "idle_timeout"
    assert pending["fresh_task_start_segment_id"] == "s1"
    assert pending["automatic_native_resume_suppressed"] is True
    assert pending["abandoned_session"] is True
    rendered = json.dumps(result)
    assert "must-not-be-exposed" not in rendered


def test_resolved_transaction_entry_hides_stale_failed_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / ".arc-companion" / "checkpoints" / OBS_FINGERPRINT
    ledger = checkpoint / "translation-ledger.json"
    call_path = checkpoint / "llm" / "call-checkpoints" / "call.json"
    call_path.parent.mkdir(parents=True)
    call_path.write_text(json.dumps({
        "state": "failed", "submission_state": "submitted",
        "logical_identity": _authorized_logical_identity(
            ledger=ledger, session="ch-0001:translation", logical_unit="s1",
            generation=1, key="accepted-key",
        ),
    }))
    journal = tmp_path / ".arc-companion" / "resume-transaction.json"
    journal.parent.mkdir(exist_ok=True)
    journal.write_text(json.dumps({
        "schema_version": "arc.companion.resume-transaction.v2",
        "action": "auto", "policy": "auto", "status": "complete",
        "entries": [{
            "ledger_path": str(ledger),
            "session_key": "ch-0001:translation", "segment_id": "s1",
            "idempotency_key": "accepted-key", "initial_generation": 1,
            "status": "resolved",
        }],
    }))

    result = enrich_status(
        tmp_path, {"status": "complete", "fingerprint": OBS_FINGERPRINT, "checkpoint_dir": str(checkpoint)},
    )

    assert result["pending_call_count"] == 0
    assert result["pending_calls"] == []


def test_status_never_correlates_shared_key_across_session_or_generation(tmp_path) -> None:
    checkpoint = tmp_path / ".arc-companion" / "checkpoints" / OBS_FINGERPRINT
    calls = checkpoint / "llm" / "call-checkpoints"
    calls.mkdir(parents=True)
    for name, session, generation in (
        ("a", "chapter-a:translation", 1),
        ("b", "chapter-b:companion", 2),
    ):
        (calls / f"{name}.json").write_text(json.dumps({
            "state": "failed", "submission_state": "submitted",
            "logical_identity": _authorized_logical_identity(
                ledger=checkpoint / f"{name}-ledger.json", session=session,
                logical_unit=f"{name}-1", generation=generation,
                key="shared-key",
            ),
        }))
    journal = tmp_path / ".arc-companion" / "resume-transaction.json"
    journal.parent.mkdir(exist_ok=True)
    journal.write_text(json.dumps({
        "action": "auto", "status": "complete", "entries": [{
            "ledger_path": str(checkpoint / "a-ledger.json"),
            "session_key": "chapter-a:translation", "segment_id": "a-1",
            "initial_generation": 1, "idempotency_key": "shared-key",
            "status": "resolved",
        }],
    }))

    result = enrich_status(
        tmp_path, {"status": "complete", "fingerprint": OBS_FINGERPRINT, "checkpoint_dir": str(checkpoint)},
    )

    assert result["pending_call_count"] == 1
    assert result["pending_calls"][0]["session_key"] == "chapter-b:companion"
    assert result["pending_calls"][0]["generation"] == 2


def test_status_requires_matching_control_address_and_logical_unit(tmp_path) -> None:
    checkpoint = tmp_path / ".arc-companion" / "checkpoints" / OBS_FINGERPRINT
    calls = checkpoint / "llm" / "call-checkpoints"
    calls.mkdir(parents=True)
    key = "same-logical-key"
    session = "chapter-a:translation"
    for name, ledger, logical_unit in (
        ("different-control", checkpoint / "other-ledger.json", "s1"),
        ("different-unit", checkpoint / "owned-ledger.json", "s2"),
    ):
        (calls / f"{name}.json").write_text(json.dumps({
            "state": "failed", "submission_state": "submitted",
            "logical_identity": _authorized_logical_identity(
                ledger=ledger, session=session, logical_unit=logical_unit,
                generation=1, key=key,
            ),
        }))
    journal = tmp_path / ".arc-companion" / "resume-transaction.json"
    journal.parent.mkdir(exist_ok=True)
    journal.write_text(json.dumps({
        "action": "auto", "status": "complete", "entries": [{
            "ledger_path": str(checkpoint / "owned-ledger.json"),
            "session_key": session, "segment_id": "s1",
            "initial_generation": 1, "idempotency_key": key,
            "status": "resolved",
        }],
    }))

    result = enrich_status(
        tmp_path, {"status": "complete", "fingerprint": OBS_FINGERPRINT, "checkpoint_dir": str(checkpoint)},
    )

    assert result["pending_call_count"] == 2
    identities = {
        (
            item["control_identity"]["control_address"],
            item["control_identity"]["logical_unit"],
        )
        for item in result["pending_calls"]
    }
    assert identities == {
        (str((checkpoint / "other-ledger.json").resolve()), "s1"),
        (str((checkpoint / "owned-ledger.json").resolve()), "s2"),
    }


def test_action_history_projects_safe_fields_and_recursively_redacts_reasons(
    tmp_path,
) -> None:
    checkpoint = tmp_path / ".arc-companion" / "checkpoints" / OBS_FINGERPRINT
    checkpoint.mkdir(parents=True)
    secret = "sk-abcdefghijklmnop"
    journal = tmp_path / ".arc-companion" / "resume-transaction.json"
    journal.parent.mkdir(exist_ok=True)
    journal.write_text(json.dumps({
        "action": "auto", "action_history": [{
            "action": "auto", "policy": "auto",
            "reason": f"provider failed with {secret}",
            "details": {
                "error_message": f"nested {secret}",
                "prompt": "raw prompt must not escape",
                "credentials": secret,
            },
            "raw_response": "raw response must not escape",
        }],
    }))

    result = enrich_status(
        tmp_path, {"status": "needs_supervision", "fingerprint": OBS_FINGERPRINT, "checkpoint_dir": str(checkpoint)},
    )

    history = result["recovery_action_history"]
    assert history[0]["action"] == "auto"
    assert "[REDACTED" in history[0]["reason"]
    assert "[REDACTED" in history[0]["details"]["error_message"]
    rendered = json.dumps(history)
    assert secret not in rendered
    assert "raw prompt must not escape" not in rendered
    assert "raw response must not escape" not in rendered


def test_status_redacts_and_bounds_controller_and_provider_reasons(tmp_path) -> None:
    checkpoint = tmp_path / ".arc-companion" / "checkpoints" / OBS_FINGERPRINT
    call = checkpoint / "llm" / "call-checkpoints" / "call.json"
    call.parent.mkdir(parents=True)
    secret = "sk-abcdefghijklmnop"
    call.write_text(json.dumps({
        "state": "failed", "submission_state": "submitted",
        "error": f"authorization: {secret} " + ("x" * 5000),
    }))

    result = enrich_status(
        tmp_path, {"status": "needs_supervision", "fingerprint": OBS_FINGERPRINT, "checkpoint_dir": str(checkpoint)},
    )

    reason = result["pending_calls"][0]["blocking_reason"]
    assert secret not in reason
    assert "[REDACTED]" in reason
    assert len(reason) == 4096


def test_status_does_not_traverse_external_checkpoint_state(
    tmp_path: Path,
) -> None:
    external = tmp_path.parent / f"{tmp_path.name}-external-checkpoint"
    call = external / "call-checkpoints" / "call.json"
    call.parent.mkdir(parents=True)
    call.write_text(json.dumps({"state": "submitted"}), encoding="utf-8")
    result = enrich_status(
        tmp_path,
        {
            "status": "generating",
            "fingerprint": OBS_FINGERPRINT,
            "checkpoint_dir": str(external),
        },
    )
    assert result["calls"]["active"] == 0
    assert result["pending_call_count"] == 0


def test_status_resolves_short_checkpoint_and_rejects_identity_conflict(
    tmp_path: Path,
) -> None:
    allocation = allocate_artifact_dir(
        tmp_path / ".arc-companion" / "checkpoints",
        OBS_FINGERPRINT,
        kind="checkpoint",
    )
    call = (
        allocation.path / "call-checkpoints" / "call.json"
    )
    call.parent.mkdir()
    call.write_text(json.dumps({"state": "submitted"}), encoding="utf-8")
    state = {
        "status": "generating",
        "fingerprint": OBS_FINGERPRINT,
        "checkpoint_identity": OBS_FINGERPRINT,
        "checkpoint_dir": str(allocation.path),
        "checkpoint_identity_receipt_path": str(
            allocation.receipt_path
        ),
        "checkpoint_identity_receipt_sha256": (
            allocation.receipt_sha256
        ),
    }
    assert enrich_status(tmp_path, state)["calls"]["active"] == 1
    conflicted = {**state, "checkpoint_identity": "b" * 64}
    assert enrich_status(tmp_path, conflicted)["calls"]["active"] == 0


def test_status_accepts_checkpoint_identity_only_transition(
    tmp_path: Path,
) -> None:
    allocation = allocate_artifact_dir(
        tmp_path / ".arc-companion" / "checkpoints",
        OBS_FINGERPRINT,
        kind="checkpoint",
    )
    call = allocation.path / "call-checkpoints" / "call.json"
    call.parent.mkdir()
    call.write_text(json.dumps({"state": "submitted"}), encoding="utf-8")
    state = {
        "status": "generating",
        "checkpoint_identity": OBS_FINGERPRINT,
        "checkpoint_dir": str(allocation.path),
        "checkpoint_identity_receipt_path": str(
            allocation.receipt_path
        ),
        "checkpoint_identity_receipt_sha256": (
            allocation.receipt_sha256
        ),
    }
    assert enrich_status(tmp_path, state)["calls"]["active"] == 1


def test_status_accepts_durable_intent_guidance_recovery_root(
    tmp_path: Path,
) -> None:
    fingerprint = "c" * 64
    checkpoint = (
        tmp_path / ".arc-companion" / "intent-guidance" / ("d" * 64)
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "source-snapshot-receipt.json").write_text(
        json.dumps({
            "schema_version": (
                "arc.companion.source-snapshot-receipt.v2"
            ),
            "fingerprint": fingerprint,
            "checkpoint_identity": fingerprint,
        }),
        encoding="utf-8",
    )
    call = checkpoint / "call-checkpoints" / "call.json"
    call.parent.mkdir()
    call.write_text(json.dumps({"state": "submitted"}), encoding="utf-8")
    state = {
        "status": "failed",
        "checkpoint_dir": str(checkpoint),
        "recovery_root_kind": "intent-guidance",
        "recovery_root_fingerprint": fingerprint,
    }
    assert enrich_status(tmp_path, state)["calls"]["active"] == 1


def test_status_rejects_saved_receipt_symlink_even_when_target_is_valid(
    tmp_path: Path,
) -> None:
    allocation = allocate_artifact_dir(
        tmp_path / ".arc-companion" / "checkpoints",
        OBS_FINGERPRINT,
        kind="checkpoint",
    )
    call = allocation.path / "call-checkpoints" / "call.json"
    call.parent.mkdir()
    call.write_text(json.dumps({"state": "submitted"}), encoding="utf-8")
    saved_receipt = tmp_path / "outside-receipt-link"
    saved_receipt.symlink_to(allocation.receipt_path)
    state = {
        "status": "generating",
        "fingerprint": OBS_FINGERPRINT,
        "checkpoint_identity": OBS_FINGERPRINT,
        "checkpoint_dir": str(allocation.path),
        "checkpoint_identity_receipt_path": str(saved_receipt),
        "checkpoint_identity_receipt_sha256": (
            allocation.receipt_sha256
        ),
    }
    assert enrich_status(tmp_path, state)["calls"]["active"] == 0
