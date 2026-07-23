from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping

from arc_llm.attempt_diagnostics import sanitize_diagnostic_text

from .recovery_responses import (
    RecoveryResponseError,
    discover_submission_receipts,
    resolve_recovery_path,
)
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
    checkpoint = _verified_checkpoint(root, state)
    counts = {"active": 0, "queued": 0, "draining": 0}
    if checkpoint is None or not checkpoint.is_dir():
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
    """Merge durable calls and transactions only through one exact control identity."""

    checkpoint = _verified_checkpoint(root, state)
    if checkpoint is None:
        return []
    result_by_identity: dict[tuple[object, ...], dict[str, Any]] = {}
    checkpoints_by_identity: dict[
        tuple[str, str, str, int, str], tuple[Path, dict[str, Any]]
    ] = {}
    receipt_identities = _receipt_control_identities(checkpoint)
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
        authorization_identity = _checkpoint_control_identity(logical)
        receipt_identity = receipt_identities.get(path.resolve(strict=False))
        control_identity = (
            None
            if authorization_identity is not None
            and receipt_identity is not None
            and authorization_identity != receipt_identity
            else authorization_identity or receipt_identity
        )
        if control_identity is not None:
            checkpoints_by_identity[control_identity] = (path, value)
        resumable = bool(value.get("resumable"))
        identity = (
            ("control", *control_identity)
            if control_identity is not None
            else ("checkpoint", str(path.resolve()))
        )
        result_by_identity[identity] = {
            "checkpoint": str(path.relative_to(checkpoint)),
            "idempotency_key": key or None,
            "session_key": logical.get("session_key"),
            "generation": logical.get("generation"),
            "control_identity": _identity_json(control_identity),
            "logical_identity": _logical_identity_json(logical),
            "state": call_state,
            "submission_state": submission_state,
            "recovery_action": "resume-native" if resumable else "operator-supervision",
            "blocking_reason": (
                _safe_reason(value.get("last_error") or value.get("error"))
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
        context_identities = {
            _transaction_recovery_identity(item)
            for item in transaction.get("native_resume_contexts") or []
            if isinstance(item, dict)
        }
        for raw in transaction.get("entries") or []:
            if not isinstance(raw, dict):
                continue
            entry = dict(raw)
            key = str(entry.get("idempotency_key") or "")
            session_key = str(entry.get("session_key") or "")
            segment_id = str(entry.get("segment_id") or "")
            try:
                generation = int(
                    entry.get("generation") or entry.get("initial_generation") or 0
                )
            except (TypeError, ValueError):
                generation = 0
            entry_control_identity = _transaction_recovery_identity(entry)
            complete_entry_identity = _recovery_identity_complete(
                entry_control_identity
            )
            if entry.get("status") == "resolved":
                if complete_entry_identity:
                    result_by_identity.pop(
                        ("control", *entry_control_identity), None,
                    )
                continue
            ledger_path = Path(str(entry.get("ledger_path") or ""))
            if not ledger_path.is_absolute():
                ledger_path = (root / ledger_path).resolve(strict=False)
            else:
                ledger_path = ledger_path.resolve(strict=False)
            identity = (
                ("control", *entry_control_identity)
                if complete_entry_identity else (
                    "entry", str(ledger_path), session_key, segment_id,
                )
            )
            checkpoint_item = (
                checkpoints_by_identity.get(entry_control_identity)
                if complete_entry_identity else None
            )
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
            blocking_reason = _safe_reason(
                entry.get("blocking_reason")
                or supervision.get("reason")
                or checkpoint_value.get("last_error")
                or checkpoint_value.get("error")
                or "submitted_call_without_an_accepted_response"
            )
            can_resume_native = bool(
                _transaction_recovery_identity(entry) in context_identities
            )
            existing = result_by_identity.get(identity, {})
            replacement = next((
                item for item in replacements
                if _replacement_recovery_identity(item) == entry_control_identity
                and complete_entry_identity
            ), None)
            result_by_identity[identity] = {
                **existing,
                "checkpoint": (
                    str(checkpoint_path.relative_to(checkpoint))
                    if checkpoint_path is not None else existing.get("checkpoint")
                ),
                "idempotency_key": key or existing.get("idempotency_key"),
                "session_key": session_key or existing.get("session_key"),
                "generation": generation or None
                or existing.get("generation"),
                "control_identity": (
                    _identity_json(entry_control_identity)
                    if complete_entry_identity else None
                ),
                "logical_identity": (
                    existing.get("logical_identity")
                    if checkpoint_item is not None else {
                        "session_key": session_key or None,
                        "generation": generation or None,
                        "idempotency_key": key or None,
                    }
                ),
                "state": str((ledger_block or {}).get("state") or existing.get("state") or "pending"),
                "submission_state": submission_state,
                "recovery_action": (
                    "fresh-generation" if (
                        replacement is not None
                        and replacement.get("trigger_code") == "idle_timeout"
                    )
                    else "restart-generation" if replacement is not None
                    else "fresh-generation" if entry.get("fresh_generation_required")
                    else action if action == "resume-native" and can_resume_native
                    else "automatic-recovery" if action == "auto"
                    else "operator-supervision"
                ),
                "recovery_trigger": entry.get("recovery_trigger"),
                "fresh_task_start_segment_id": entry.get(
                    "fresh_task_start_segment_id"
                ),
                "automatic_native_resume_suppressed": bool(
                    entry.get("automatic_native_resume_suppressed")
                ),
                "abandoned_session": bool(
                    recovery_context.get("native_session_id")
                    or recovery_context.get("resumable")
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
            _safe_action_history_item(item)
            for item in value.get("action_history") or []
            if isinstance(item, dict)
        ],
        "recovery_checkpoint_path": value.get("checkpoint_path"),
        "recovery_checkpoint_fingerprint": value.get("checkpoint_fingerprint"),
        "recovery_authorization_source": value.get("authorization_source"),
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
        "restart_max_attempts": value.get("max_auto_attempts"),
        "replacement_group_id": value.get("group_id"),
        "replacement_authorization_source": value.get("authorization_source"),
        "restart_trigger_code": value.get("trigger_code"),
        "restart_trigger_reason": _safe_reason(value.get("trigger_reason")),
        "recovery_mode": (
            "fresh-generation"
            if value.get("trigger_code") == "idle_timeout"
            else "generation-replacement"
        ),
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


def _receipt_control_identities(
    checkpoint: Path,
) -> dict[Path, tuple[str, str, str, int, str]]:
    """Index sealed stateless receipts by their exact call-checkpoint address."""

    if not checkpoint.is_dir():
        return {}
    try:
        receipts = discover_submission_receipts(checkpoint)
    except (RecoveryResponseError, OSError, ValueError):
        return {}
    output: dict[Path, tuple[str, str, str, int, str]] = {}
    ambiguous: set[Path] = set()
    for _receipt_path, receipt in receipts:
        if receipt.get("sealed") is not True:
            continue
        try:
            call_path = resolve_recovery_path(
                checkpoint, receipt.get("checkpoint_path"),
            ).resolve(strict=False)
            ledger_path = resolve_recovery_path(
                checkpoint, receipt.get("ledger_path"),
            ).resolve(strict=False)
            identity = (
                str(ledger_path),
                str(receipt.get("session_key") or ""),
                str(receipt.get("logical_unit") or ""),
                int(receipt.get("generation") or 0),
                str(receipt.get("idempotency_key") or ""),
            )
        except (RecoveryResponseError, TypeError, ValueError):
            continue
        if not _recovery_identity_complete(identity):
            continue
        previous = output.get(call_path)
        if previous is not None and previous != identity:
            ambiguous.add(call_path)
            output.pop(call_path, None)
        elif call_path not in ambiguous:
            output[call_path] = identity
    return output


def _transaction_recovery_identity(
    value: Mapping[str, Any],
) -> tuple[str, str, str, int, str]:
    context = value.get("recovery_context")
    context = context if isinstance(context, Mapping) else {}
    try:
        generation = int(
            value.get("generation")
            or value.get("initial_generation")
            or context.get("generation")
            or 0
        )
    except (TypeError, ValueError):
        generation = 0
    raw_path = str(value.get("ledger_path") or context.get("ledger_path") or "")
    ledger_path = (
        str(Path(raw_path).expanduser().resolve(strict=False)) if raw_path else ""
    )
    return (
        ledger_path,
        str(value.get("session_key") or context.get("session_key") or ""),
        str(
            value.get("segment_id")
            or value.get("logical_unit")
            or context.get("logical_unit")
            or ""
        ),
        generation,
        str(value.get("idempotency_key") or context.get("idempotency_key") or ""),
    )


def _checkpoint_control_identity(
    logical: Mapping[str, Any],
) -> tuple[str, str, str, int, str] | None:
    """Read the controller's five-field authorization without partial fallback."""

    authorization = logical.get("initial_native_authorization")
    authorization = authorization if isinstance(authorization, Mapping) else {}
    raw_address = str(
        logical.get("control_address") or authorization.get("control_address") or ""
    )
    session_key = str(logical.get("session_key") or "")
    logical_unit = str(
        logical.get("logical_unit") or authorization.get("logical_unit") or ""
    )
    key = str(logical.get("idempotency_key") or "")
    try:
        generation = int(logical.get("generation") or 0)
    except (TypeError, ValueError):
        return None
    if not raw_address:
        return None
    address = str(Path(raw_address).expanduser().resolve(strict=False))
    identity = (address, session_key, logical_unit, generation, key)
    expected_authorization = {
        "control_address": address,
        "session_key": session_key,
        "logical_unit": logical_unit,
        "generation": generation,
        "idempotency_key": key,
    }
    if (
        raw_address != address
        or not _recovery_identity_complete(identity)
        or dict(authorization) != expected_authorization
    ):
        return None
    return identity


def _replacement_recovery_identity(
    value: Mapping[str, Any],
) -> tuple[str, str, str, int, str]:
    """Return the abandoned generation identity represented by a replacement."""

    raw_path = str(value.get("ledger_path") or "")
    try:
        generation = int(value.get("source_generation") or 0)
    except (TypeError, ValueError):
        generation = 0
    return (
        str(Path(raw_path).expanduser().resolve(strict=False)) if raw_path else "",
        str(value.get("session_key") or ""),
        str(value.get("segment_id") or ""),
        generation,
        str(value.get("abandoned_logical_key") or ""),
    )


def _recovery_identity_complete(
    value: tuple[str, str, str, int, str],
) -> bool:
    return bool(
        value[0] and value[1] and value[2] and value[3] > 0 and value[4]
    )


def _identity_json(
    value: tuple[str, str, str, int, str] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "control_address": value[0],
        "session_key": value[1],
        "logical_unit": value[2],
        "generation": value[3],
        "idempotency_key": value[4],
    }


def _logical_identity_json(value: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only non-secret call identity fields, never the whole checkpoint."""

    return {
        "provider": value.get("provider"),
        "model": value.get("model"),
        "session_key": value.get("session_key"),
        "generation": value.get("generation"),
        "idempotency_key": value.get("idempotency_key"),
        "control_address": value.get("control_address"),
        "logical_unit": value.get("logical_unit"),
    }


_ACTION_HISTORY_SAFE_FIELDS = frozenset({
    "action", "policy", "at", "status", "trigger_code",
    "authorization_source", "replacement_id", "group_id", "attempt",
    "max_auto_attempts", "source_generation", "target_generation",
})


def _safe_action_history_item(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project action history to stable fields and redact all diagnostic prose."""

    output: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = str(raw_key)
        lowered = key.lower()
        if any(token in lowered for token in ("reason", "error", "message", "exception")):
            output[key] = _safe_reason(
                json.dumps(item, ensure_ascii=False, sort_keys=True)
                if isinstance(item, (Mapping, list, tuple)) else item
            )
            continue
        if key in _ACTION_HISTORY_SAFE_FIELDS and isinstance(
            item, (str, int, float, bool, type(None)),
        ):
            output[key] = item
            continue
        if isinstance(item, Mapping):
            nested = _safe_action_history_item(item)
            if nested:
                output[key] = nested
        elif isinstance(item, (list, tuple)):
            nested_items = [
                _safe_action_history_item(candidate)
                for candidate in item if isinstance(candidate, Mapping)
            ]
            if any(nested_items):
                output[key] = [candidate for candidate in nested_items if candidate]
    return output


def _safe_reason(value: Any) -> str:
    """Expose a bounded redacted diagnostic, never raw provider/controller text."""

    return sanitize_diagnostic_text(value or "")[:4096]


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


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
    checkpoint = _verified_checkpoint(root, state)
    if checkpoint is None or not checkpoint.is_dir():
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


def _verified_checkpoint(
    root: Path, state: Mapping[str, Any],
) -> Path | None:
    """Return only the same bounded recovery root accepted by the pipeline."""

    active_run = state.get("active_run")
    active_run = active_run if isinstance(active_run, Mapping) else {}
    raw_value = (
        state.get("checkpoint_dir")
        if "checkpoint_dir" in state
        else active_run.get("checkpoint_dir")
    )
    if not isinstance(raw_value, str) or not raw_value:
        return None
    raw = Path(raw_value)
    candidate = raw if raw.is_absolute() else root / raw
    try:
        # Runtime import avoids an import cycle while keeping status and
        # observability on one recovery-root authority implementation.
        from .pipeline import _resolve_recovery_state_root

        return _resolve_recovery_state_root(root, state, candidate)
    except (OSError, RuntimeError, ValueError):
        return None


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
