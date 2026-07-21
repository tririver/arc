from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .io import read_json, write_json


SCHEMA_VERSION = "arc.companion.resume-transaction.v2"
LEGACY_SCHEMA_VERSION = "arc.companion.resume-transaction.v1"
ENTRY_STATES = ("pending", "authorized", "reconciling", "resolved")
TRANSACTION_STATES = (
    "prepared", "continuing", "complete", "continuation_failed",
)


def transaction_path(project_dir: Path) -> Path:
    return project_dir / ".arc-companion" / "resume-transaction.json"


def load_transaction(project_dir: Path) -> dict[str, Any] | None:
    path = transaction_path(project_dir)
    if not path.is_file():
        return None
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid resume transaction journal: {path}")
    if value.get("schema_version") == LEGACY_SCHEMA_VERSION:
        value = _upgrade_v1(value)
        write_json(path, value)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Invalid resume transaction journal: {path}")
    compacted = _compact(value)
    if compacted != value:
        value = compacted
        write_json(path, value)
    return value


def begin_transaction(
    project_dir: Path,
    *,
    action: str,
    recovery_options: Mapping[str, Any],
    entries: list[Mapping[str, Any]],
    native_resume_contexts: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    existing = load_transaction(project_dir)
    if existing and existing.get("status") != "complete":
        if existing.get("action") != action:
            raise ValueError("An incomplete resume transaction uses a different action")
        return existing
    now = _now()
    value = {
        "schema_version": SCHEMA_VERSION,
        "action": action,
        "status": "prepared",
        "recovery_options": dict(recovery_options),
        "entries": [{**_canonical_entry(entry), "status": "pending"} for entry in entries],
        "native_resume_contexts": _dedupe_contexts(native_resume_contexts or []),
        "started_at": now,
        "updated_at": now,
    }
    write_json(transaction_path(project_dir), value)
    return value


def append_entries(
    project_dir: Path,
    entries: list[Mapping[str, Any]],
    *,
    native_resume_contexts: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Idempotently add calls discovered while continuation is in progress."""
    value = load_transaction(project_dir)
    if value is None:
        raise ValueError("Resume transaction journal is missing")
    current = list(value.get("entries") or [])
    identity_positions = {
        (_canonical_path(item.get("ledger_path")), str(item.get("session_key") or ""),
         str(item.get("segment_id") or "")): index
        for index, item in enumerate(current)
    }
    nonresumable_identities: set[tuple[str, str, str]] = set()
    nonresumable_keys: set[str] = set()
    for raw in entries:
        entry = _canonical_entry(raw)
        identity = (
            _canonical_path(entry.get("ledger_path")), str(entry.get("session_key") or ""),
            str(entry.get("segment_id") or ""),
        )
        existing_index = identity_positions.get(identity)
        if str(entry.get("recovery_action") or "") == "operator-supervision":
            nonresumable_identities.add(identity)
            if entry.get("idempotency_key"):
                nonresumable_keys.add(str(entry["idempotency_key"]))
        if existing_index is None:
            current.append({**entry, "status": "pending"})
            identity_positions[identity] = len(current) - 1
        else:
            # Later durable discovery may add a more precise local recovery
            # action or blocking context. Preserve transaction progress while
            # enriching the existing authorization entry.
            status = str(current[existing_index].get("status") or "pending")
            prior_key = str(current[existing_index].get("idempotency_key") or "")
            if identity in nonresumable_identities and prior_key:
                nonresumable_keys.add(prior_key)
            current[existing_index] = {
                **dict(current[existing_index]), **entry, "status": status,
            }
    contexts = _dedupe_contexts([
        *(value.get("native_resume_contexts") or []),
        *(native_resume_contexts or []),
    ])
    contexts = [
        context for context in contexts
        if str(context.get("idempotency_key") or "") not in nonresumable_keys
        and (
            _canonical_path(context.get("ledger_path")),
            str(context.get("session_key") or ""),
            str(context.get("segment_id") or ""),
        ) not in nonresumable_identities
    ]
    value.update({
        "entries": current,
        "native_resume_contexts": contexts,
        "updated_at": _now(),
    })
    write_json(transaction_path(project_dir), value)
    return value


def mark_entry(project_dir: Path, index: int, *, status: str, **receipt: Any) -> dict[str, Any]:
    if status not in ENTRY_STATES:
        raise ValueError(f"Invalid resume entry status: {status}")
    value = load_transaction(project_dir)
    if value is None:
        raise ValueError("Resume transaction journal is missing")
    entries = list(value.get("entries") or [])
    current = str(entries[index].get("status") or "pending")
    if ENTRY_STATES.index(status) < ENTRY_STATES.index(current):
        raise ValueError(f"Resume entry cannot move backward: {current} -> {status}")
    entries[index] = {**dict(entries[index]), **receipt, "status": status}
    value.update({"entries": entries, "updated_at": _now()})
    write_json(transaction_path(project_dir), value)
    return value


def mark_transaction(project_dir: Path, status: str) -> dict[str, Any]:
    if status not in TRANSACTION_STATES:
        raise ValueError(f"Invalid resume transaction status: {status}")
    value = load_transaction(project_dir)
    if value is None:
        raise ValueError("Resume transaction journal is missing")
    value.update({"status": status, "updated_at": _now()})
    write_json(transaction_path(project_dir), value)
    return value


def _upgrade_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate the crash-prone v1 journal without discarding authorization."""
    status_map = {
        "applying": "prepared",
        "continuation_ready": "continuing",
        "complete": "complete",
        "continuation_failed": "continuation_failed",
    }
    entries = []
    for raw in value.get("entries") or []:
        entry = dict(raw)
        entry["status"] = (
            "authorized" if entry.get("status") == "applied" else "pending"
        )
        entries.append(entry)
    migrated_status = status_map.get(str(value.get("status") or ""), "prepared")
    # v1 could report complete immediately after clearing supervision, before
    # continuation durably accepted the response. Conservatively retain those
    # authorizations for reconciliation.
    if migrated_status == "complete" and entries:
        migrated_status = "continuation_failed"
    return {
        **dict(value),
        "schema_version": SCHEMA_VERSION,
        "status": migrated_status,
        "entries": entries,
        "migrated_from": LEGACY_SCHEMA_VERSION,
        "updated_at": _now(),
    }


def _canonical_path(value: Any) -> str:
    text = str(value or "")
    return str(Path(text).expanduser().resolve(strict=False)) if text else ""


def _canonical_entry(value: Mapping[str, Any]) -> dict[str, Any]:
    entry = dict(value)
    if entry.get("ledger_path"):
        entry["ledger_path"] = _canonical_path(entry["ledger_path"])
    return entry


def _dedupe_contexts(values: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for raw in values:
        context = dict(raw)
        key = str(context.get("idempotency_key") or "")
        if key and key in positions:
            output[positions[key]] = {**output[positions[key]], **context}
            continue
        if key:
            positions[key] = len(output)
        output.append(context)
    return output


def _compact(value: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize paths and merge duplicate logical recovery entries."""
    rank = {state: index for index, state in enumerate(ENTRY_STATES)}
    entries: list[dict[str, Any]] = []
    identity_positions: dict[tuple[str, str, str], int] = {}
    key_positions: dict[str, int] = {}
    for raw in value.get("entries") or []:
        entry = _canonical_entry(raw)
        key = str(entry.get("idempotency_key") or "")
        identity = (
            str(entry.get("ledger_path") or ""),
            str(entry.get("session_key") or ""),
            str(entry.get("segment_id") or ""),
        )
        index = key_positions.get(key) if key else None
        if index is None:
            index = identity_positions.get(identity)
        if index is None:
            index = len(entries)
            entries.append(entry)
            identity_positions[identity] = index
            if key:
                key_positions[key] = index
            continue
        existing = entries[index]
        existing_rank = rank.get(str(existing.get("status") or "pending"), 0)
        entry_rank = rank.get(str(entry.get("status") or "pending"), 0)
        lower, higher = (
            (existing, entry) if entry_rank >= existing_rank else (entry, existing)
        )
        merged = {**lower, **higher}
        for field in ("blocking_code", "blocking_reason", "recovery_context"):
            if not merged.get(field):
                value_from_either = lower.get(field) or higher.get(field)
                if value_from_either:
                    merged[field] = value_from_either
                else:
                    merged.pop(field, None)
        entries[index] = merged
        identity_positions[identity] = index
        if key:
            key_positions[key] = index
    return {
        **dict(value),
        "entries": entries,
        "native_resume_contexts": _dedupe_contexts(
            list(value.get("native_resume_contexts") or [])
        ),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
