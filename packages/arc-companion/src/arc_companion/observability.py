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
    timings, last_progress = _phase_timings(root)
    provider_progress = _latest_provider_progress(root, state)
    if provider_progress and (last_progress is None or provider_progress > last_progress):
        last_progress = provider_progress
    lock_owner = inspect_lock(root / ".arc-companion-build.lock")
    phase = str(state.get("status") or "unknown")
    return {
        **dict(state),
        "current_phase": phase,
        "wait_reason": _wait_reason(phase, calls, lock_owner),
        "lock_owner": lock_owner,
        "calls": calls,
        "last_progress_at": last_progress or state.get("updated_at"),
        "phase_elapsed_seconds": timings,
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
