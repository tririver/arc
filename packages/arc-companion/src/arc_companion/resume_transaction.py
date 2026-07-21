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
RECOVERY_POLICIES = ("auto", "manual")
REPLACEMENT_STATES = (
    "claimed", "rotated", "suffix_invalidated", "response_persisted",
    "accepted", "failed",
)


class AutomaticRegenerationExhausted(RuntimeError):
    """The single automatic replacement budget for a logical segment is spent."""


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
    policy: str | None = None,
) -> dict[str, Any]:
    normalized_policy = _policy(action, policy)
    existing = load_transaction(project_dir)
    if existing and existing.get("status") != "complete":
        if existing.get("action") != action:
            if existing.get("action") == "resume-native" and action == "auto":
                return upgrade_transaction_action(
                    project_dir, action="auto", policy="auto",
                    reason="automatic recovery upgraded incomplete native reconciliation",
                )
            raise ValueError("An incomplete resume transaction uses a different action")
        return existing
    now = _now()
    value = {
        "schema_version": SCHEMA_VERSION,
        "action": action,
        "policy": normalized_policy,
        "action_history": [{
            "action": action, "policy": normalized_policy, "at": now,
            "reason": "transaction_started",
        }],
        "status": "prepared",
        "recovery_options": dict(recovery_options),
        "entries": [{**_canonical_entry(entry), "status": "pending"} for entry in entries],
        "native_resume_contexts": _dedupe_contexts(native_resume_contexts or []),
        "replacements": [],
        "restart_budgets": [],
        "started_at": now,
        "updated_at": now,
    }
    write_json(transaction_path(project_dir), value)
    return value


def upgrade_transaction_action(
    project_dir: Path, *, action: str, policy: str | None = None,
    reason: str = "recovery_action_changed",
) -> dict[str, Any]:
    """Idempotently upgrade an unfinished native transaction to automatic recovery."""

    value = load_transaction(project_dir)
    if value is None:
        raise ValueError("Resume transaction journal is missing")
    current_action = str(value.get("action") or "resume-native")
    normalized_policy = _policy(action, policy)
    if current_action == action and str(value.get("policy") or "") == normalized_policy:
        return value
    if not (current_action == "resume-native" and action == "auto"):
        raise ValueError("An incomplete resume transaction cannot change to that action")
    now = _now()
    history = list(value.get("action_history") or [])
    history.append({
        "action": action, "policy": normalized_policy, "at": now,
        "reason": reason,
    })
    value.update({
        "action": action, "policy": normalized_policy,
        "action_history": history, "updated_at": now,
    })
    write_json(transaction_path(project_dir), value)
    return value


def authorize_manual_restart(
    project_dir: Path,
    *,
    reason: str = "operator_confirmed_generation_restart",
) -> dict[str, Any]:
    """Record an explicit paid-restart override after automatic recovery stops."""

    value = load_transaction(project_dir)
    if value is None:
        raise ValueError("Resume transaction journal is missing")
    current_action = str(value.get("action") or "")
    if current_action == "restart-generation":
        return value
    if current_action != "auto":
        raise ValueError("Only an automatic transaction can be overridden manually")
    now = _now()
    history = list(value.get("action_history") or [])
    history.append({
        "action": "restart-generation", "policy": "manual", "at": now,
        "reason": reason,
    })
    entries = [dict(item) for item in value.get("entries") or []]
    for entry in entries:
        if entry.get("status") == "resolved":
            continue
        ledger_path = Path(str(entry.get("ledger_path") or ""))
        if not ledger_path.is_file():
            continue
        ledger = read_json(ledger_path)
        current_generation = int(ledger.get("generation") or 1)
        entry.update({
            "initial_generation": current_generation,
            "target_generation": current_generation + 1,
            "manual_restart_after_auto": True,
        })
    value.update({
        "action": "restart-generation", "policy": "manual",
        "action_history": history, "entries": entries, "updated_at": now,
    })
    write_json(transaction_path(project_dir), value)
    return value


def ensure_auto_transaction(
    project_dir: Path,
    *,
    recovery_options: Mapping[str, Any] | None = None,
    entries: list[Mapping[str, Any]] | None = None,
    native_resume_contexts: list[Mapping[str, Any]] | None = None,
    reason: str = "automatic_recovery_selected",
) -> dict[str, Any]:
    """Create, reuse, or upgrade the project's automatic recovery journal."""

    existing = load_transaction(project_dir)
    if existing is None or existing.get("status") == "complete":
        return begin_transaction(
            project_dir, action="auto", policy="auto",
            recovery_options=recovery_options or {}, entries=entries or [],
            native_resume_contexts=native_resume_contexts,
        )
    if str(existing.get("action") or "") == "resume-native":
        existing = upgrade_transaction_action(
            project_dir, action="auto", policy="auto", reason=reason,
        )
    elif str(existing.get("action") or "") != "auto":
        raise ValueError("An incomplete resume transaction uses a different action")
    if entries or native_resume_contexts:
        return append_entries(
            project_dir, entries or [],
            native_resume_contexts=native_resume_contexts,
        )
    return existing


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
            if status == "resolved":
                # Acceptance receipts are immutable audit facts.  Later
                # discovery may not rewrite them or reopen the entry.
                continue
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
    if current == "resolved" and status == "resolved":
        return value
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


def claim_automatic_restart(
    project_dir: Path,
    *,
    session_key: str,
    segment_id: str,
    ledger_path: str | Path,
    source_generation: int,
    target_generation: int,
    suffix_segment_ids: list[str],
    trigger_code: str,
    trigger_reason: str,
    abandoned_logical_key: str = "",
    possible_duplicate_charge: bool = False,
    suffix_start_segment_id: str | None = None,
) -> dict[str, Any]:
    """Claim the one automatic replacement, replaying the same claim idempotently."""

    value = load_transaction(project_dir)
    if value is None:
        raise ValueError("Resume transaction journal is missing")
    canonical_ledger = _canonical_path(ledger_path)
    budget_key = (str(session_key), str(segment_id))
    budgets = [dict(item) for item in value.get("restart_budgets") or []]
    replacements = [dict(item) for item in value.get("replacements") or []]
    existing = next((
        item for item in replacements
        if (str(item.get("session_key") or ""), str(item.get("segment_id") or ""))
        == budget_key
        and int(item.get("source_generation") or 0) == int(source_generation)
        and int(item.get("target_generation") or 0) == int(target_generation)
    ), None)
    if existing is not None:
        return existing
    budget = next((
        item for item in budgets
        if (str(item.get("session_key") or ""), str(item.get("segment_id") or ""))
        == budget_key
    ), None)
    if budget is not None and int(budget.get("attempts_used") or 0) >= 1:
        raise AutomaticRegenerationExhausted(
            f"automatic regeneration exhausted for {session_key}:{segment_id}"
        )
    now = _now()
    if budget is None:
        budget = {
            "session_key": session_key, "segment_id": segment_id,
            "attempts_used": 1, "max_auto_attempts": 1,
            "claimed_at": now,
        }
        budgets.append(budget)
    else:
        budget.update({"attempts_used": 1, "claimed_at": now})
    replacement_id = f"{session_key}:{segment_id}:replacement-1"
    replacement = {
        "replacement_id": replacement_id,
        "session_key": session_key,
        "segment_id": segment_id,
        "ledger_path": canonical_ledger,
        "source_generation": int(source_generation),
        "target_generation": int(target_generation),
        "attempt": 1,
        "abandoned_logical_key": abandoned_logical_key,
        "suffix_start_segment_id": suffix_start_segment_id or segment_id,
        "suffix_segment_ids": [str(value) for value in suffix_segment_ids],
        "trigger_code": trigger_code,
        "trigger_reason": trigger_reason,
        "possible_duplicate_charge": bool(possible_duplicate_charge),
        "status": "claimed",
        "claimed_at": now,
        "updated_at": now,
    }
    replacements.append(replacement)
    value.update({
        "restart_budgets": budgets, "replacements": replacements,
        "updated_at": now,
    })
    write_json(transaction_path(project_dir), value)
    return replacement


def plan_replacement(
    project_dir: Path,
    *,
    session_key: str,
    segment_id: str,
    ledger_path: str | Path,
    source_generation: int,
    target_generation: int,
    suffix_start_segment_id: str,
    suffix_segment_ids: list[str],
    trigger_code: str,
    trigger_reason: str,
    abandoned_logical_key: str = "",
    possible_duplicate_charge: bool = False,
) -> dict[str, Any]:
    """Plan one budgeted replacement and return its durable crash-replay record."""

    return claim_automatic_restart(
        project_dir, session_key=session_key, segment_id=segment_id,
        ledger_path=ledger_path, source_generation=source_generation,
        target_generation=target_generation,
        suffix_start_segment_id=suffix_start_segment_id,
        suffix_segment_ids=suffix_segment_ids, trigger_code=trigger_code,
        trigger_reason=trigger_reason,
        abandoned_logical_key=abandoned_logical_key,
        possible_duplicate_charge=possible_duplicate_charge,
    )


def mark_replacement(
    project_dir: Path, replacement_id: str, *, status: str, **receipt: Any,
) -> dict[str, Any]:
    """Advance a replacement record without allowing status regression."""

    if status not in REPLACEMENT_STATES:
        raise ValueError(f"Invalid replacement status: {status}")
    value = load_transaction(project_dir)
    if value is None:
        raise ValueError("Resume transaction journal is missing")
    replacements = [dict(item) for item in value.get("replacements") or []]
    index = next((
        index for index, item in enumerate(replacements)
        if str(item.get("replacement_id") or "") == replacement_id
    ), None)
    if index is None:
        raise ValueError(f"Unknown replacement record: {replacement_id}")
    current = str(replacements[index].get("status") or "claimed")
    if current in {"accepted", "failed"} and status == current:
        return replacements[index]
    if current in {"accepted", "failed"} and status != current:
        raise ValueError(f"Terminal replacement cannot move backward from {current}")
    if current != "failed" and status != "failed" and (
        REPLACEMENT_STATES.index(status) < REPLACEMENT_STATES.index(current)
    ):
        raise ValueError(f"Replacement cannot move backward: {current} -> {status}")
    now = _now()
    replacements[index] = {
        **replacements[index], **receipt, "status": status, "updated_at": now,
    }
    value.update({"replacements": replacements, "updated_at": now})
    write_json(transaction_path(project_dir), value)
    return replacements[index]


def update_replacement(
    project_dir: Path, replacement_id: str, *, phase: str, **receipt: Any,
) -> dict[str, Any]:
    """Persist a replacement phase boundary for idempotent continuation."""

    return mark_replacement(
        project_dir, replacement_id, status=phase, **receipt,
    )


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
        "policy": "manual",
        "action_history": [{
            "action": str(value.get("action") or "resume-native"),
            "policy": "manual", "at": str(value.get("started_at") or _now()),
            "reason": "migrated_v1",
        }],
        "entries": entries,
        "replacements": [],
        "restart_budgets": [],
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
        if str(existing.get("status") or "pending") == "resolved":
            merged = existing
        elif str(entry.get("status") or "pending") == "resolved":
            merged = entry
        else:
            merged = {**lower, **higher}
        # A legacy journal may contain the same logical entry twice: one
        # blocker record and one acceptance receipt.  Its one-time compaction
        # retains both audit halves.  Once compacted, append_entries() treats
        # the resolved result as immutable.
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
    action = str(value.get("action") or "resume-native")
    policy = _policy(action, value.get("policy"))
    history = list(value.get("action_history") or [])
    if not history:
        history = [{
            "action": action, "policy": policy,
            "at": str(value.get("started_at") or value.get("updated_at") or _now()),
            "reason": "legacy_v2_backfill",
        }]
    return {
        **dict(value),
        "policy": policy,
        "action_history": history,
        "entries": entries,
        "native_resume_contexts": _dedupe_contexts(
            list(value.get("native_resume_contexts") or [])
        ),
        "replacements": [dict(item) for item in value.get("replacements") or []],
        "restart_budgets": [dict(item) for item in value.get("restart_budgets") or []],
    }


def _policy(action: str, value: Any) -> str:
    policy = str(value or ("auto" if action == "auto" else "manual"))
    if policy not in RECOVERY_POLICIES:
        raise ValueError(f"Invalid recovery policy: {policy}")
    return policy


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
