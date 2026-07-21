from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .run_lock import inspect_lock


EVENT_SCHEMA_VERSION = "arc.companion.progress-event.v1"


def append_state_event(state_path: Path, state: Mapping[str, Any]) -> None:
    """Persist an append-only phase event after the authoritative state write."""

    event = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event": "phase_state",
        "phase": str(state.get("status") or "unknown"),
        "updated_at": str(state.get("updated_at") or _utc_now()),
        "pid": os.getpid(),
    }
    for key in ("fingerprint", "checkpoint_dir", "paper_id"):
        if state.get(key) is not None:
            event[key] = state[key]
    path = state_path.parent / ".arc-companion" / "progress-events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def enrich_status(project_dir: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    """Add live lock, call, wait, and phase timing diagnostics to saved state."""

    root = project_dir.resolve()
    calls = _call_counts(root, state)
    pending_calls = _pending_calls(root, state)
    timings, last_progress = _phase_timings(root)
    provider_progress = _latest_provider_progress(root, state)
    if provider_progress and (last_progress is None or provider_progress > last_progress):
        last_progress = provider_progress
    lock_owner = inspect_lock(root / ".arc-companion-build.lock")
    phase = str(state.get("status") or "unknown")
    recovery = _recovery_status(root)
    return {
        **dict(state),
        "current_phase": phase,
        "wait_reason": _wait_reason(phase, calls, lock_owner),
        "lock_owner": lock_owner,
        "calls": calls,
        "pending_call_count": len(pending_calls),
        "pending_calls": pending_calls,
        "last_progress_at": last_progress or state.get("updated_at"),
        "phase_elapsed_seconds": timings,
        **recovery,
    }


def _call_counts(root: Path, state: Mapping[str, Any]) -> dict[str, int]:
    checkpoint = Path(str(state.get("checkpoint_dir") or ""))
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    counts = {"active": 0, "queued": 0, "draining": 0}
    if not checkpoint.is_dir():
        return counts
    for path in checkpoint.rglob("call-checkpoints/*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        call_state = value.get("state") if isinstance(value, dict) else None
        if call_state in {"submitted", "resuming"}:
            counts["active"] += 1
        elif call_state == "prepared":
            counts["queued"] += 1
        elif call_state == "draining":
            counts["draining"] += 1
    return counts


def _pending_calls(root: Path, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Merge durable call and resume-transaction supervision inventories."""

    checkpoint = Path(str(state.get("checkpoint_dir") or ""))
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    result_by_identity: dict[tuple[str, ...], dict[str, Any]] = {}
    checkpoints_by_key: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(checkpoint.rglob("call-checkpoints/*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(value, dict):
            continue
        submission_state = str(value.get("submission_state") or "unknown")
        call_state = str(value.get("state") or "unknown")
        if submission_state not in {"submitted", "unknown"}:
            continue
        if call_state not in {"submitted", "resuming", "failed"}:
            continue
        logical = value.get("logical_identity")
        logical = logical if isinstance(logical, dict) else {}
        key = str(logical.get("idempotency_key") or "")
        if key:
            checkpoints_by_key[key] = (path, value)
        resumable = bool(value.get("resumable"))
        identity = (("key", key) if key else ("checkpoint", str(path.resolve())))
        result_by_identity[identity] = {
            "checkpoint": str(path.relative_to(checkpoint)),
            "idempotency_key": key or None,
            "session_key": logical.get("session_key"),
            "generation": logical.get("generation"),
            "state": call_state,
            "submission_state": submission_state,
            "recovery_action": "resume-native" if resumable else "operator-supervision",
            "blocking_reason": (
                str(value.get("last_error") or value.get("error") or "")
                or "submitted_call_without_an_accepted_response"
            ),
            "entry_status": None,
        }

    transaction_path = root / ".arc-companion" / "resume-transaction.json"
    try:
        transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        transaction = None
    if isinstance(transaction, dict):
        action = str(transaction.get("action") or "resume-native")
        replacements = [
            dict(item) for item in transaction.get("replacements") or []
            if isinstance(item, dict)
        ]
        context_keys = {
            str(item.get("idempotency_key") or "")
            for item in transaction.get("native_resume_contexts") or []
            if isinstance(item, dict)
        }
        for raw in transaction.get("entries") or []:
            if not isinstance(raw, dict):
                continue
            entry = dict(raw)
            key = str(entry.get("idempotency_key") or "")
            if entry.get("status") == "resolved":
                if key:
                    result_by_identity.pop(("key", key), None)
                continue
            session_key = str(entry.get("session_key") or "")
            segment_id = str(entry.get("segment_id") or "")
            ledger_path = Path(str(entry.get("ledger_path") or ""))
            if not ledger_path.is_absolute():
                ledger_path = (root / ledger_path).resolve(strict=False)
            else:
                ledger_path = ledger_path.resolve(strict=False)
            identity = (
                ("key", key) if key else (
                    "entry", str(ledger_path), session_key, segment_id,
                )
            )
            checkpoint_item = checkpoints_by_key.get(key) if key else None
            checkpoint_path, checkpoint_value = (
                checkpoint_item if checkpoint_item is not None else (None, {})
            )
            recovery_context = entry.get("recovery_context")
            recovery_context = (
                recovery_context if isinstance(recovery_context, dict) else {}
            )
            ledger, ledger_block, supervision = _pending_ledger_context(
                ledger_path, segment_id,
            )
            supervision_context = supervision.get("recovery_context")
            supervision_context = (
                supervision_context if isinstance(supervision_context, dict) else {}
            )
            submission_state = _effective_submission_state(
                recovery_context.get("submission_state"),
                supervision_context.get("submission_state"),
                (ledger_block or {}).get("submission_state"),
                checkpoint_value.get("submission_state"),
            )
            blocking_reason = str(
                entry.get("blocking_reason")
                or supervision.get("reason")
                or checkpoint_value.get("last_error")
                or checkpoint_value.get("error")
                or "submitted_call_without_an_accepted_response"
            )
            can_resume_native = bool(
                key in context_keys
                or recovery_context.get("resumable")
                or recovery_context.get("native_session_id")
            )
            existing = result_by_identity.get(identity, {})
            replacement = next((
                item for item in replacements
                if str(item.get("session_key") or "") == session_key
                and str(item.get("segment_id") or "") == segment_id
            ), None)
            result_by_identity[identity] = {
                **existing,
                "checkpoint": (
                    str(checkpoint_path.relative_to(checkpoint))
                    if checkpoint_path is not None else existing.get("checkpoint")
                ),
                "idempotency_key": key or existing.get("idempotency_key"),
                "session_key": session_key or existing.get("session_key"),
                "generation": entry.get("generation") or entry.get("initial_generation")
                or existing.get("generation"),
                "state": str((ledger_block or {}).get("state") or existing.get("state") or "pending"),
                "submission_state": submission_state,
                "recovery_action": (
                    "restart-generation" if replacement is not None
                    else action if action == "resume-native" and can_resume_native
                    else "automatic-recovery" if action == "auto"
                    else "operator-supervision"
                ),
                "blocking_reason": blocking_reason,
                "entry_status": str(entry.get("status") or "pending"),
                "ledger_path": str(ledger_path),
                "segment_id": segment_id or None,
                **(_replacement_fields(replacement) if replacement is not None else {}),
            }
    return list(result_by_identity.values())


def _recovery_status(root: Path) -> dict[str, Any]:
    path = root / ".arc-companion" / "resume-transaction.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(value, dict):
        return {}
    replacements = [
        _replacement_fields(item)
        for item in value.get("replacements") or []
        if isinstance(item, dict)
    ]
    return {
        "recovery_policy": str(
            value.get("policy")
            or ("auto" if value.get("action") == "auto" else "manual")
        ),
        "recovery_action_history": [
            dict(item) for item in value.get("action_history") or []
            if isinstance(item, dict)
        ],
        "recovery_replacements": replacements,
    }


def _replacement_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "replacement_id": value.get("replacement_id"),
        "source_generation": value.get("source_generation"),
        "target_generation": value.get("target_generation"),
        "suffix_start_segment_id": value.get("suffix_start_segment_id"),
        "suffix_segment_ids": list(value.get("suffix_segment_ids") or []),
        "restart_attempt": value.get("attempt"),
        "restart_trigger_code": value.get("trigger_code"),
        "restart_trigger_reason": value.get("trigger_reason"),
        "possible_duplicate_charge": bool(value.get("possible_duplicate_charge")),
        "replacement_status": value.get("status"),
        "abandoned_logical_key": value.get("abandoned_logical_key") or None,
    }


def _pending_ledger_context(
    ledger_path: Path, segment_id: str,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    try:
        value = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}, None, {}
    if not isinstance(value, dict):
        return {}, None, {}
    block = next((
        item for item in value.get("blocks") or []
        if isinstance(item, dict) and str(item.get("segment_id") or "") == segment_id
    ), None)
    supervision = value.get("needs_supervision")
    return value, block, supervision if isinstance(supervision, dict) else {}


def _effective_submission_state(*values: Any) -> str:
    states = {str(value) for value in values if value is not None}
    if "submitted" in states:
        return "submitted"
    if "unknown" in states:
        return "unknown"
    return next(iter(states), "unknown")


def _phase_timings(root: Path) -> tuple[dict[str, float], str | None]:
    path = root / ".arc-companion" / "progress-events.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}, None
    events: list[tuple[str, datetime]] = []
    for line in lines:
        try:
            value = json.loads(line)
            stamp = datetime.fromisoformat(str(value["updated_at"]).replace("Z", "+00:00"))
            events.append((str(value["phase"]), stamp))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    totals: dict[str, float] = {}
    now = datetime.now(timezone.utc)
    for index, (phase, started) in enumerate(events):
        ended = events[index + 1][1] if index + 1 < len(events) else (
            started if phase in {"complete", "failed", "needs_supervision"} else now
        )
        totals[phase] = totals.get(phase, 0.0) + max(0.0, (ended - started).total_seconds())
    return ({key: round(value, 3) for key, value in totals.items()}, events[-1][1].isoformat() if events else None)


def _latest_provider_progress(root: Path, state: Mapping[str, Any]) -> str | None:
    checkpoint = Path(str(state.get("checkpoint_dir") or ""))
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if not checkpoint.is_dir():
        return None
    latest: datetime | None = None
    for path in checkpoint.rglob("progress.jsonl"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            try:
                value = json.loads(line)
                stamp = datetime.fromisoformat(str(value["updated_at"]).replace("Z", "+00:00"))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
            if latest is None or stamp > latest:
                latest = stamp
            break
    return latest.isoformat() if latest else None


def _wait_reason(phase: str, calls: Mapping[str, int], lock_owner: Mapping[str, Any] | None) -> str | None:
    if phase == "needs_supervision":
        return "operator_supervision"
    if calls.get("draining"):
        return "provider_calls_draining"
    if calls.get("active"):
        return "provider_response"
    if calls.get("queued"):
        return "provider_capacity"
    if lock_owner and lock_owner.get("active") and phase not in {"complete", "failed"}:
        return "active_build"
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
