from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any, Mapping


LANE_LEDGER_VERSION = "arc.companion.chapter-lane-ledger.v2"
BLOCK_STATES = (
    "prepared",
    "submitted",
    "response_received",
    "schema_valid",
    "invariant_valid",
    "accepted",
)


class LaneLedgerError(RuntimeError):
    pass


def initialize_lane_ledger(
    path: Path,
    *,
    chapter_id: str,
    lane: str,
    segment_ids: list[str],
    generation: int = 1,
) -> dict[str, Any]:
    if len(segment_ids) != len(set(segment_ids)) or not all(segment_ids):
        raise LaneLedgerError("segment ids must be non-empty and unique")
    if path.is_file():
        ledger = _read(path)
        if ledger.get("schema_version") == "arc.companion.chapter-lane-ledger.v1":
            ledger = _upgrade_v1(ledger)
            _write(path, ledger)
        _validate_identity(ledger, chapter_id=chapter_id, lane=lane, segment_ids=segment_ids)
        return ledger
    ledger = {
        "schema_version": LANE_LEDGER_VERSION,
        "chapter_id": chapter_id,
        "lane": lane,
        "generation": generation,
        "needs_supervision": None,
        "blocks": [
            {
                "segment_id": value,
                "state": "prepared",
                "submission_state": "not_submitted",
                "generation": generation,
            }
            for value in segment_ids
        ],
        "accepted_chain_sha256": _hash(""),
        "updated_at": time.time(),
    }
    _write(path, ledger)
    return ledger


def advance_block(
    path: Path,
    *,
    segment_id: str,
    state: str,
    receipt: Mapping[str, Any] | None = None,
    input_sha256: str | None = None,
    output_sha256: str | None = None,
    validation_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if state not in BLOCK_STATES:
        raise LaneLedgerError(f"unsupported block state: {state}")
    ledger = _read(path)
    index = _block_index(ledger, segment_id)
    block = dict(ledger["blocks"][index])
    current = str(block.get("state") or "prepared")
    if BLOCK_STATES.index(state) < BLOCK_STATES.index(current):
        raise LaneLedgerError(f"cannot move {segment_id} backward from {current} to {state}")
    if BLOCK_STATES.index(state) > BLOCK_STATES.index(current) + 1:
        raise LaneLedgerError(f"cannot skip validation state for {segment_id}: {current} -> {state}")
    if index and str(ledger["blocks"][index - 1].get("state")) != "accepted":
        raise LaneLedgerError(f"cannot advance {segment_id} before its predecessor is accepted")
    block["state"] = state
    if state == "prepared":
        block["submission_state"] = "not_submitted"
    elif state in {"submitted", "response_received", "schema_valid", "invariant_valid", "accepted"}:
        block["submission_state"] = "submitted"
    if receipt is not None:
        block["logical_receipt"] = dict(receipt)
    if input_sha256 is not None:
        block["input_sha256"] = input_sha256
    if output_sha256 is not None:
        block["output_sha256"] = output_sha256
    if validation_receipt is not None:
        block["validation_receipt"] = dict(validation_receipt)
    if state == "accepted" and current != "accepted":
        predecessor = str(ledger.get("accepted_chain_sha256") or _hash(""))
        block["predecessor_accepted_chain_sha256"] = predecessor
        block["accepted_chain_sha256"] = _hash(json.dumps({
            "predecessor": predecessor,
            "segment_id": segment_id,
            "input_sha256": block.get("input_sha256"),
            "output_sha256": block.get("output_sha256"),
            "generation": block.get("generation"),
        }, sort_keys=True, separators=(",", ":")))
        ledger["accepted_chain_sha256"] = block["accepted_chain_sha256"]
    ledger["blocks"][index] = block
    ledger["updated_at"] = time.time()
    _write(path, ledger)
    return ledger


def mark_needs_supervision(
    path: Path, *, segment_id: str, reason: str, recovery_context: Mapping[str, Any]
) -> dict[str, Any]:
    ledger = _read(path)
    index = _block_index(ledger, segment_id)
    block = dict(ledger["blocks"][index])
    submission_state = str(recovery_context.get("submission_state") or "unknown")
    if block.get("state") == "prepared" and submission_state in {"unknown", "submitted"}:
        block["state"] = "submitted"
        block["submission_state"] = submission_state
        ledger["blocks"][index] = block
    marker = {
        "segment_id": segment_id,
        "reason": reason,
        "recovery_context": dict(recovery_context),
        "created_at": time.time(),
    }
    raw_entries = list(ledger.get("supervision_entries") or [])
    if not raw_entries and isinstance(ledger.get("needs_supervision"), Mapping):
        raw_entries.append(dict(ledger["needs_supervision"]))
    entries = [
        dict(item) for item in raw_entries
        if isinstance(item, Mapping) and str(item.get("segment_id") or "") != segment_id
    ]
    entries.append(marker)
    ledger["supervision_entries"] = entries
    # Preserve the earliest unresolved lane stop for legacy readers while the
    # complete per-segment inventory remains available to recovery finalizers.
    if not ledger.get("needs_supervision"):
        ledger["needs_supervision"] = marker
    ledger["updated_at"] = time.time()
    _write(path, ledger)
    return ledger


def clear_needs_supervision(path: Path) -> dict[str, Any]:
    """Clear a supervised stop without changing the submitted block or generation."""
    ledger = _read(path)
    if not ledger.get("needs_supervision"):
        raise LaneLedgerError("lane ledger does not need supervision")
    current_segment = str(
        (ledger.get("needs_supervision") or {}).get("segment_id") or ""
    )
    entries = [
        dict(item) for item in ledger.get("supervision_entries") or []
        if isinstance(item, Mapping)
        and str(item.get("segment_id") or "") != current_segment
    ]
    ledger["supervision_entries"] = entries
    ledger["needs_supervision"] = entries[0] if entries else None
    ledger["updated_at"] = time.time()
    _write(path, ledger)
    return ledger


def invalidate_suffix(path: Path, *, from_segment_id: str, generation: int) -> dict[str, Any]:
    ledger = _read(path)
    index = _block_index(ledger, from_segment_id)
    source_generation = int(ledger.get("generation") or 1)
    prefix = list(ledger["blocks"][:index])
    if any(str(item.get("state")) != "accepted" for item in prefix):
        raise LaneLedgerError("only an accepted prefix may be preserved")
    suffix = []
    for item in ledger["blocks"][index:]:
        suffix.append({
            "segment_id": item["segment_id"],
            "state": "prepared",
            "submission_state": "not_submitted",
            "generation": generation,
        })
    ledger["blocks"] = prefix + suffix
    ledger["generation"] = generation
    suffix_ids = {str(item["segment_id"]) for item in suffix}
    active = [
        dict(item) for item in ledger.get("supervision_entries") or []
        if isinstance(item, Mapping)
    ]
    if not active and isinstance(ledger.get("needs_supervision"), Mapping):
        active = [dict(ledger["needs_supervision"])]
    archived_at = time.time()
    archived = []
    remaining = []
    for marker in active:
        if str(marker.get("segment_id") or "") in suffix_ids:
            archived.append({
                **marker,
                "archived_at": archived_at,
                "archive_reason": "generation_suffix_invalidated",
                "source_generation": source_generation,
                "target_generation": generation,
                "suffix_start_segment_id": from_segment_id,
            })
        else:
            remaining.append(marker)
    ledger["supervision_history"] = [
        *[
            dict(item) for item in ledger.get("supervision_history") or []
            if isinstance(item, Mapping)
        ],
        *archived,
    ]
    ledger["supervision_entries"] = remaining
    ledger["needs_supervision"] = remaining[0] if remaining else None
    ledger["accepted_chain_sha256"] = (
        str(prefix[-1].get("accepted_chain_sha256")) if prefix else _hash("")
    )
    ledger["updated_at"] = time.time()
    _write(path, ledger)
    return ledger


def next_pending(ledger: Mapping[str, Any]) -> str | None:
    if ledger.get("needs_supervision"):
        return None
    for item in ledger.get("blocks") or []:
        if str(item.get("state")) != "accepted":
            return str(item.get("segment_id") or "") or None
    return None


def accept_reused_block(
    path: Path,
    *,
    segment_id: str,
    input_sha256: str,
    output_sha256: str,
    artifact_id: str,
    validation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept a validated object-store hit without claiming provider submission."""

    ledger = _read(path)
    index = _block_index(ledger, segment_id)
    block = dict(ledger["blocks"][index])
    if block.get("state") != "prepared" or block.get("submission_state") != "not_submitted":
        raise LaneLedgerError(f"cannot reuse artifact for non-prepared block {segment_id}")
    if index and str(ledger["blocks"][index - 1].get("state")) != "accepted":
        raise LaneLedgerError(f"cannot reuse {segment_id} before its predecessor is accepted")
    predecessor = str(ledger.get("accepted_chain_sha256") or _hash(""))
    block.update({
        "state": "accepted",
        "submission_state": "not_submitted",
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "logical_receipt": {
            "kind": "accepted_artifact_reuse",
            "artifact_id": artifact_id,
            "provider_calls": 0,
        },
        "validation_receipt": dict(validation_receipt),
        "predecessor_accepted_chain_sha256": predecessor,
    })
    block["accepted_chain_sha256"] = _hash(json.dumps({
        "predecessor": predecessor,
        "segment_id": segment_id,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "generation": block.get("generation"),
    }, sort_keys=True, separators=(",", ":")))
    ledger["blocks"][index] = block
    ledger["accepted_chain_sha256"] = block["accepted_chain_sha256"]
    ledger["updated_at"] = time.time()
    _write(path, ledger)
    return ledger


def mark_submitted(path: Path, *, segment_id: str) -> dict[str, Any]:
    """Advance only when the provider confirms crossing its submission barrier."""

    ledger = _read(path)
    block = ledger["blocks"][_block_index(ledger, segment_id)]
    if block.get("state") == "prepared":
        return advance_block(path, segment_id=segment_id, state="submitted")
    return ledger


def mark_response_received(path: Path, *, segment_id: str) -> dict[str, Any]:
    """Record the first durable provider response for a logical lane item."""

    ledger = _read(path)
    block = ledger["blocks"][_block_index(ledger, segment_id)]
    if block.get("state") == "submitted":
        return advance_block(path, segment_id=segment_id, state="response_received")
    return ledger


def _validate_identity(ledger: Mapping[str, Any], *, chapter_id: str, lane: str, segment_ids: list[str]) -> None:
    actual = [str(item.get("segment_id") or "") for item in ledger.get("blocks") or []]
    if ledger.get("schema_version") != LANE_LEDGER_VERSION or ledger.get("chapter_id") != chapter_id or ledger.get("lane") != lane or actual != segment_ids:
        raise LaneLedgerError("lane ledger identity changed")


def _upgrade_v1(ledger: Mapping[str, Any]) -> dict[str, Any]:
    upgraded = dict(ledger)
    blocks = []
    for raw in upgraded.get("blocks") or []:
        block = dict(raw)
        state = str(block.get("state") or "pending")
        if state == "pending":
            block["state"] = "prepared"
            block["submission_state"] = "not_submitted"
        else:
            block["submission_state"] = "submitted"
        blocks.append(block)
    upgraded["blocks"] = blocks
    upgraded["schema_version"] = LANE_LEDGER_VERSION
    return upgraded


def _block_index(ledger: Mapping[str, Any], segment_id: str) -> int:
    for index, item in enumerate(ledger.get("blocks") or []):
        if str(item.get("segment_id") or "") == segment_id:
            return index
    raise LaneLedgerError(f"unknown segment id: {segment_id}")


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaneLedgerError(f"could not read lane ledger {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LaneLedgerError("lane ledger is not an object")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
