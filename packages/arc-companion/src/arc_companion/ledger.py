from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any, Mapping


LANE_LEDGER_VERSION = "arc.companion.chapter-lane-ledger.v1"
BLOCK_STATES = ("pending", "submitted", "schema_valid", "invariant_valid", "accepted")


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
        _validate_identity(ledger, chapter_id=chapter_id, lane=lane, segment_ids=segment_ids)
        return ledger
    ledger = {
        "schema_version": LANE_LEDGER_VERSION,
        "chapter_id": chapter_id,
        "lane": lane,
        "generation": generation,
        "needs_supervision": None,
        "blocks": [
            {"segment_id": value, "state": "pending", "generation": generation}
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
    current = str(block.get("state") or "pending")
    if BLOCK_STATES.index(state) < BLOCK_STATES.index(current):
        raise LaneLedgerError(f"cannot move {segment_id} backward from {current} to {state}")
    if BLOCK_STATES.index(state) > BLOCK_STATES.index(current) + 1:
        raise LaneLedgerError(f"cannot skip validation state for {segment_id}: {current} -> {state}")
    if index and str(ledger["blocks"][index - 1].get("state")) != "accepted":
        raise LaneLedgerError(f"cannot advance {segment_id} before its predecessor is accepted")
    block["state"] = state
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
    _block_index(ledger, segment_id)
    ledger["needs_supervision"] = {
        "segment_id": segment_id,
        "reason": reason,
        "recovery_context": dict(recovery_context),
        "created_at": time.time(),
    }
    ledger["updated_at"] = time.time()
    _write(path, ledger)
    return ledger


def clear_needs_supervision(path: Path) -> dict[str, Any]:
    """Clear a supervised stop without changing the submitted block or generation."""
    ledger = _read(path)
    if not ledger.get("needs_supervision"):
        raise LaneLedgerError("lane ledger does not need supervision")
    ledger["needs_supervision"] = None
    ledger["updated_at"] = time.time()
    _write(path, ledger)
    return ledger


def invalidate_suffix(path: Path, *, from_segment_id: str, generation: int) -> dict[str, Any]:
    ledger = _read(path)
    index = _block_index(ledger, from_segment_id)
    prefix = list(ledger["blocks"][:index])
    if any(str(item.get("state")) != "accepted" for item in prefix):
        raise LaneLedgerError("only an accepted prefix may be preserved")
    suffix = []
    for item in ledger["blocks"][index:]:
        suffix.append({"segment_id": item["segment_id"], "state": "pending", "generation": generation})
    ledger["blocks"] = prefix + suffix
    ledger["generation"] = generation
    ledger["needs_supervision"] = None
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


def _validate_identity(ledger: Mapping[str, Any], *, chapter_id: str, lane: str, segment_ids: list[str]) -> None:
    actual = [str(item.get("segment_id") or "") for item in ledger.get("blocks") or []]
    if ledger.get("schema_version") != LANE_LEDGER_VERSION or ledger.get("chapter_id") != chapter_id or ledger.get("lane") != lane or actual != segment_ids:
        raise LaneLedgerError("lane ledger identity changed")


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
