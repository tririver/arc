from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping


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


@dataclass(frozen=True)
class LaneTransitionGuard:
    expected_generation: int
    expected_ledger_sha256: str
    authorization: tuple[str, str, str, int, str]


_CONTROL_LEDGER_LOCK = threading.RLock()


def initialize_lane_ledger(
    path: Path,
    *,
    chapter_id: str,
    lane: str,
    segment_ids: list[str],
    generation: int = 1,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    if len(segment_ids) != len(set(segment_ids)) or not all(segment_ids):
        raise LaneLedgerError("segment ids must be non-empty and unique")
    def validate_and_upgrade(current: dict[str, Any]) -> dict[str, Any]:
        if current.get("schema_version") == "arc.companion.chapter-lane-ledger.v1":
            current = _upgrade_v1(current)
        _validate_identity(
            current,
            chapter_id=chapter_id,
            lane=lane,
            segment_ids=segment_ids,
        )
        return current

    if path.is_file():
        ledger = _read(path)
        _register(path, ledger, checkpoint_dir=checkpoint_dir)
        return _mutate(
            path, validate_and_upgrade, checkpoint_dir=checkpoint_dir,
        )
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
    from .ledger_registry import LaneLedgerRegistryError

    try:
        _write(path, ledger, checkpoint_dir=checkpoint_dir)
        return ledger
    except (LaneLedgerError, LaneLedgerRegistryError):
        # A concurrent create-only winner may have published after the absent
        # check. Adopt its exact bytes and validate identity; never overwrite it.
        if not path.is_file():
            raise
        current = _read(path)
        _register(path, current, checkpoint_dir=checkpoint_dir)
        return _mutate(
            path, validate_and_upgrade, checkpoint_dir=checkpoint_dir,
        )


def initialize_control_ledger(
    path: Path,
    *,
    chapter_id: str,
    lane: str,
    segment_ids: list[str],
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Create or atomically reconcile one ordered synthetic control ledger.

    An unchanged accepted prefix survives a changed chunk topology. The first
    changed or nonaccepted block and its suffix move together to one new shared
    generation, so no per-sibling ledger can pretend to own prefix semantics.
    """

    if len(segment_ids) != len(set(segment_ids)) or not all(segment_ids):
        raise LaneLedgerError("control segment ids must be non-empty and unique")
    with _CONTROL_LEDGER_LOCK:
        if not path.is_file():
            return initialize_lane_ledger(
                path,
                chapter_id=chapter_id,
                lane=lane,
                segment_ids=segment_ids,
                checkpoint_dir=checkpoint_dir,
            )
        ledger = _read(path)
        _register(path, ledger, checkpoint_dir=checkpoint_dir)
        return _mutate(
            path,
            lambda current: _apply_control_reconciliation(
                current,
                chapter_id=chapter_id,
                lane=lane,
                segment_ids=segment_ids,
            ),
            checkpoint_dir=checkpoint_dir,
        )


def _apply_control_reconciliation(
    ledger: dict[str, Any],
    *,
    chapter_id: str,
    lane: str,
    segment_ids: list[str],
) -> dict[str, Any]:
    if ledger.get("schema_version") == "arc.companion.chapter-lane-ledger.v1":
        ledger = _upgrade_v1(ledger)
    if (
        ledger.get("schema_version") != LANE_LEDGER_VERSION
        or ledger.get("chapter_id") != chapter_id
        or ledger.get("lane") != lane
    ):
        raise LaneLedgerError("control ledger identity changed")
    blocks = [dict(item) for item in ledger.get("blocks") or []]
    old_ids = [str(item.get("segment_id") or "") for item in blocks]
    if old_ids == segment_ids:
        return ledger
    prefix_length = 0
    for old, new in zip(blocks, segment_ids):
        if (
            str(old.get("segment_id") or "") != new
            or old.get("state") != "accepted"
        ):
            break
        prefix_length += 1
    if any(item.get("state") == "accepted" for item in blocks[prefix_length:]):
        raise LaneLedgerError("control ledger contains a non-prefix acceptance")
    generation = int(ledger.get("generation") or 1) + 1
    predecessor = _hash("")
    prefix: list[dict[str, Any]] = []
    for item in blocks[:prefix_length]:
        preserved = dict(item)
        preserved["generation"] = generation
        preserved["reconciled_from_generation"] = item.get("generation")
        preserved["predecessor_accepted_chain_sha256"] = predecessor
        preserved["accepted_chain_sha256"] = _hash(json.dumps({
            "predecessor": predecessor,
            "segment_id": preserved.get("segment_id"),
            "input_sha256": preserved.get("input_sha256"),
            "output_sha256": preserved.get("output_sha256"),
            "generation": generation,
        }, sort_keys=True, separators=(",", ":")))
        predecessor = str(preserved["accepted_chain_sha256"])
        prefix.append(preserved)
    suffix = [{
        "segment_id": value,
        "state": "prepared",
        "submission_state": "not_submitted",
        "generation": generation,
    } for value in segment_ids[prefix_length:]]
    active = [
        dict(item) for item in ledger.get("supervision_entries") or []
        if isinstance(item, Mapping)
    ]
    if not active and isinstance(ledger.get("needs_supervision"), Mapping):
        active = [dict(ledger["needs_supervision"])]
    prefix_ids = set(segment_ids[:prefix_length])
    now = time.time()
    retained = [
        item for item in active
        if str(item.get("segment_id") or "") in prefix_ids
    ]
    archived = [{
        **item,
        "archived_at": now,
        "archive_reason": "control_topology_suffix_invalidated",
        "target_generation": generation,
    } for item in active if item not in retained]
    ledger.update({
        "generation": generation,
        "blocks": [*prefix, *suffix],
        "accepted_chain_sha256": predecessor,
        "supervision_entries": retained,
        "needs_supervision": retained[0] if retained else None,
        "supervision_history": [
            *[
                dict(item) for item in ledger.get("supervision_history") or []
                if isinstance(item, Mapping)
            ],
            *archived,
        ],
        "updated_at": now,
    })
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
    expected_ledger_sha256: str | None = None,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    if expected_ledger_sha256 is not None and checkpoint_dir is None:
        raise LaneLedgerError("registered ledger mutation requires checkpoint_dir")
    return _mutate(
        path,
        lambda ledger: _apply_block_advance(
            ledger,
            segment_id=segment_id,
            state=state,
            receipt=receipt,
            input_sha256=input_sha256,
            output_sha256=output_sha256,
            validation_receipt=validation_receipt,
        ),
        checkpoint_dir=checkpoint_dir,
        expected_ledger_sha256=expected_ledger_sha256,
    )


def _advance_block_unlocked(
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
    return _mutate(
        path,
        lambda ledger: _apply_block_advance(
            ledger,
            segment_id=segment_id,
            state=state,
            receipt=receipt,
            input_sha256=input_sha256,
            output_sha256=output_sha256,
            validation_receipt=validation_receipt,
        ),
    )


def _apply_block_advance(
    ledger: dict[str, Any],
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
    index = _block_index(ledger, segment_id)
    block = dict(ledger["blocks"][index])
    current = str(block.get("state") or "prepared")
    if BLOCK_STATES.index(state) < BLOCK_STATES.index(current):
        raise LaneLedgerError(f"cannot move {segment_id} backward from {current} to {state}")
    if BLOCK_STATES.index(state) > BLOCK_STATES.index(current) + 1:
        raise LaneLedgerError(f"cannot skip validation state for {segment_id}: {current} -> {state}")
    if (
        state == "accepted"
        and index
        and str(ledger["blocks"][index - 1].get("state")) != "accepted"
    ):
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
        # A normal generation after a staged artifact failed local validation
        # must not leave the rejected deferred payload attached to the newly
        # accepted block.
        for key in list(block):
            if key.startswith("deferred_"):
                block.pop(key, None)
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
    return ledger


def mark_needs_supervision(
    path: Path, *, segment_id: str, reason: str, recovery_context: Mapping[str, Any],
    expected_ledger_sha256: str | None = None,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    if expected_ledger_sha256 is not None and checkpoint_dir is None:
        raise LaneLedgerError("registered ledger mutation requires checkpoint_dir")
    return _mutate(
        path,
        lambda ledger: _apply_supervision_marker(
            ledger,
            segment_id=segment_id,
            reason=reason,
            recovery_context=recovery_context,
        ),
        checkpoint_dir=checkpoint_dir,
        expected_ledger_sha256=expected_ledger_sha256,
    )


def _apply_supervision_marker(
    ledger: dict[str, Any],
    *,
    segment_id: str,
    reason: str,
    recovery_context: Mapping[str, Any],
) -> dict[str, Any]:
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
        if isinstance(item, Mapping)
        and str(item.get("segment_id") or "") != segment_id
    ]
    entries.append(marker)
    ledger["supervision_entries"] = entries
    primary = ledger.get("needs_supervision")
    if (
        not isinstance(primary, Mapping)
        or str(primary.get("segment_id") or "") == segment_id
    ):
        ledger["needs_supervision"] = marker
    ledger["updated_at"] = time.time()
    return ledger


def clear_needs_supervision(
    path: Path,
    *,
    expected_ledger_sha256: str | None = None,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """Clear a supervised stop without changing the submitted block or generation."""
    if expected_ledger_sha256 is not None and checkpoint_dir is None:
        raise LaneLedgerError("registered ledger mutation requires checkpoint_dir")
    return _mutate(
        path,
        _apply_clear_supervision,
        checkpoint_dir=checkpoint_dir,
        expected_ledger_sha256=expected_ledger_sha256,
    )


def _apply_clear_supervision(ledger: dict[str, Any]) -> dict[str, Any]:
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
    return ledger


def invalidate_suffix(
    path: Path,
    *,
    from_segment_id: str,
    generation: int,
    staged_outputs: Mapping[str, Mapping[str, Any]] | None = None,
    expected_ledger_sha256: str | None = None,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    if expected_ledger_sha256 is not None and checkpoint_dir is None:
        raise LaneLedgerError("registered ledger mutation requires checkpoint_dir")
    return _mutate(
        path,
        lambda ledger: _apply_suffix_invalidation(
            ledger,
            from_segment_id=from_segment_id,
            generation=generation,
            staged_outputs=staged_outputs,
        ),
        checkpoint_dir=checkpoint_dir,
        expected_ledger_sha256=expected_ledger_sha256,
    )


def _apply_suffix_invalidation(
    ledger: dict[str, Any],
    *,
    from_segment_id: str,
    generation: int,
    staged_outputs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    index = _block_index(ledger, from_segment_id)
    source_generation = int(ledger.get("generation") or 1)
    prefix = list(ledger["blocks"][:index])
    if any(str(item.get("state")) != "accepted" for item in prefix):
        raise LaneLedgerError("only an accepted prefix may be preserved")
    suffix = []
    for item in ledger["blocks"][index:]:
        replacement = {
            "segment_id": item["segment_id"],
            "state": "prepared",
            "submission_state": "not_submitted",
            "generation": generation,
        }
        staged = (staged_outputs or {}).get(str(item["segment_id"]))
        if not isinstance(staged, Mapping):
            prior_receipt = item.get("deferred_logical_receipt")
            # Targeted regeneration may crash after invalidating the suffix.
            # Preserve only its explicitly staged suffix payloads on re-entry;
            # other deferred candidates belong to migration/reuse policy and
            # must not silently survive an unrelated generation restart.
            if (
                isinstance(prior_receipt, Mapping)
                and prior_receipt.get("kind")
                == "targeted_regeneration_suffix_stage"
                and isinstance(item.get("deferred_output"), Mapping)
            ):
                staged = {
                    "output": item.get("deferred_output"),
                    "output_sha256": item.get("deferred_output_sha256"),
                    "logical_receipt": prior_receipt,
                    "validation_receipt": item.get(
                        "deferred_validation_receipt"
                    ),
                }
        if isinstance(staged, Mapping):
            replacement.update({
                "deferred_output": staged.get("output"),
                "deferred_output_sha256": staged.get("output_sha256"),
                "deferred_logical_receipt": dict(
                    staged.get("logical_receipt") or {}
                ),
                "deferred_validation_receipt": dict(
                    staged.get("validation_receipt") or {}
                ),
            })
        suffix.append(replacement)
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
    return _mutate(
        path,
        lambda ledger: _apply_reused_block(
            ledger,
            segment_id=segment_id,
            input_sha256=input_sha256,
            output_sha256=output_sha256,
            artifact_id=artifact_id,
            validation_receipt=validation_receipt,
        ),
    )


def _apply_reused_block(
    ledger: dict[str, Any],
    *,
    segment_id: str,
    input_sha256: str,
    output_sha256: str,
    artifact_id: str,
    validation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
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
    return ledger


def accept_deferred_block(
    path: Path,
    *,
    segment_id: str,
    input_sha256: str,
    output_sha256: str,
    logical_receipt: Mapping[str, Any],
    validation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept a staged, locally revalidated hit after its predecessor arrives."""
    return _mutate(
        path,
        lambda ledger: _apply_deferred_block(
            ledger,
            segment_id=segment_id,
            input_sha256=input_sha256,
            output_sha256=output_sha256,
            logical_receipt=logical_receipt,
            validation_receipt=validation_receipt,
        ),
    )


def _apply_deferred_block(
    ledger: dict[str, Any],
    *,
    segment_id: str,
    input_sha256: str,
    output_sha256: str,
    logical_receipt: Mapping[str, Any],
    validation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    index = _block_index(ledger, segment_id)
    block = dict(ledger["blocks"][index])
    if block.get("state") != "prepared" or block.get("submission_state") != "not_submitted":
        raise LaneLedgerError(f"cannot accept deferred non-prepared block {segment_id}")
    if index and str(ledger["blocks"][index - 1].get("state")) != "accepted":
        raise LaneLedgerError(f"cannot accept deferred {segment_id} before its predecessor")
    staged_output_sha256 = block.get("deferred_output_sha256")
    if (
        staged_output_sha256 is not None
        and str(staged_output_sha256) != output_sha256
    ):
        raise LaneLedgerError(f"deferred output hash changed for {segment_id}")
    predecessor = str(ledger.get("accepted_chain_sha256") or _hash(""))
    for key in list(block):
        if key.startswith("deferred_"):
            block.pop(key, None)
    block.update({
        "state": "accepted",
        "submission_state": "not_submitted",
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "logical_receipt": dict(logical_receipt),
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
    return ledger


def accept_controller_skipped_block(
    path: Path,
    *,
    segment_id: str,
    input_sha256: str,
    output_sha256: str,
    reason: str,
) -> dict[str, Any]:
    """Accept a deterministic no-provider lane item while retaining audit lineage."""
    return _mutate(
        path,
        lambda ledger: _apply_controller_skipped_block(
            ledger,
            segment_id=segment_id,
            input_sha256=input_sha256,
            output_sha256=output_sha256,
            reason=reason,
        ),
    )


def _apply_controller_skipped_block(
    ledger: dict[str, Any],
    *,
    segment_id: str,
    input_sha256: str,
    output_sha256: str,
    reason: str,
) -> dict[str, Any]:
    index = _block_index(ledger, segment_id)
    block = dict(ledger["blocks"][index])
    if block.get("state") != "prepared" or block.get("submission_state") != "not_submitted":
        raise LaneLedgerError(f"cannot locally skip non-prepared block {segment_id}")
    if index and str(ledger["blocks"][index - 1].get("state")) != "accepted":
        raise LaneLedgerError(f"cannot locally skip {segment_id} before its predecessor is accepted")
    predecessor = str(ledger.get("accepted_chain_sha256") or _hash(""))
    block.update({
        "state": "accepted",
        "submission_state": "not_submitted",
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "logical_receipt": {
            "kind": "controller_skipped_structural_heading",
            "reason": reason,
            "provider_calls": 0,
        },
        "validation_receipt": {
            "local_validation": True,
            "structural_only": True,
        },
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
    return ledger


def lane_transition_guard(
    path: Path,
    *,
    segment_id: str,
    session_key: str,
    idempotency_key: str,
    checkpoint_dir: Path | None = None,
) -> LaneTransitionGuard:
    """Snapshot exact inputs for one guarded production state transition."""

    from .ledger_registry import read_registered_lane_ledger

    ledger, digest = read_registered_lane_ledger(
        checkpoint_dir or _checkpoint_root_for_owned_ledger(path), path,
    )
    block = ledger["blocks"][_block_index(ledger, segment_id)]
    generation = int(block.get("generation") or 0)
    if generation < 1 or generation != int(ledger.get("generation") or 0):
        raise LaneLedgerError("lane transition generation is invalid")
    authorization = _normalize_transition_authorization((
        str(path.expanduser().resolve(strict=False)),
        session_key,
        segment_id,
        generation,
        idempotency_key,
    ))
    return LaneTransitionGuard(generation, digest, authorization)


def mark_submitted(
    path: Path,
    *,
    segment_id: str,
    expected_generation: int | None = None,
    expected_ledger_sha256: str | None = None,
    authorization: tuple[Any, ...] | None = None,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """Advance only when the provider confirms crossing its submission barrier."""

    guard = _transition_guard_values(
        path,
        segment_id=segment_id,
        expected_generation=expected_generation,
        expected_ledger_sha256=expected_ledger_sha256,
        authorization=authorization,
        checkpoint_dir=checkpoint_dir,
    )

    def apply(ledger: dict[str, Any]) -> dict[str, Any]:
        index = _block_index(ledger, segment_id)
        block = ledger["blocks"][index]
        _validate_transition_guard(
            ledger, block, path=path, segment_id=segment_id, guard=guard,
        )
        if block.get("state") == "prepared":
            updated = _apply_block_advance(
                ledger, segment_id=segment_id, state="submitted",
            )
            if guard is not None:
                submitted = dict(updated["blocks"][index])
                submitted["submission_authorization"] = _authorization_json(
                    guard.authorization
                )
                updated["blocks"][index] = submitted
            return updated
        if guard is not None and block.get("state") == "submitted":
            _validate_stored_submission_authorization(block, guard.authorization)
        return ledger

    return _mutate(
        path,
        apply,
        checkpoint_dir=checkpoint_dir,
        expected_ledger_sha256=(
            guard.expected_ledger_sha256 if guard is not None else None
        ),
    )


def mark_response_received(
    path: Path,
    *,
    segment_id: str,
    expected_generation: int | None = None,
    expected_ledger_sha256: str | None = None,
    authorization: tuple[Any, ...] | None = None,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """Record the first durable provider response for a logical lane item."""

    guard = _transition_guard_values(
        path,
        segment_id=segment_id,
        expected_generation=expected_generation,
        expected_ledger_sha256=expected_ledger_sha256,
        authorization=authorization,
        checkpoint_dir=checkpoint_dir,
    )

    def apply(ledger: dict[str, Any]) -> dict[str, Any]:
        block = ledger["blocks"][_block_index(ledger, segment_id)]
        _validate_transition_guard(
            ledger, block, path=path, segment_id=segment_id, guard=guard,
        )
        if guard is not None:
            _validate_stored_submission_authorization(block, guard.authorization)
        if block.get("state") == "submitted":
            return _apply_block_advance(
                ledger, segment_id=segment_id, state="response_received",
            )
        return ledger

    return _mutate(
        path,
        apply,
        checkpoint_dir=checkpoint_dir,
        expected_ledger_sha256=(
            guard.expected_ledger_sha256 if guard is not None else None
        ),
    )


def _checkpoint_root_for_owned_ledger(path: Path) -> Path:
    from .ledger_registry import owned_lane_ledger_root

    root = owned_lane_ledger_root(path)
    if root is None:
        raise LaneLedgerError("lane transition guard requires a production-owned ledger")
    return root


def _normalize_transition_authorization(
    value: tuple[Any, ...] | None,
) -> tuple[str, str, str, int, str]:
    if isinstance(value, bool) or not isinstance(value, tuple) or len(value) != 5:
        raise LaneLedgerError(
            "lane transition authorization must be one complete five-field tuple"
        )
    control_address, session_key, logical_unit, generation, idempotency_key = value
    if (
        not isinstance(control_address, str)
        or not isinstance(session_key, str)
        or not isinstance(logical_unit, str)
        or type(generation) is not int
        or not isinstance(idempotency_key, str)
        or not control_address
        or not session_key
        or not logical_unit
        or generation < 1
        or not idempotency_key
    ):
        raise LaneLedgerError("lane transition authorization fields are invalid")
    canonical = str(Path(control_address).expanduser().resolve(strict=False))
    if control_address != canonical:
        raise LaneLedgerError("lane transition control address must be canonical")
    return canonical, session_key, logical_unit, generation, idempotency_key


def _transition_guard_values(
    path: Path,
    *,
    segment_id: str,
    expected_generation: int | None,
    expected_ledger_sha256: str | None,
    authorization: tuple[Any, ...] | None,
    checkpoint_dir: Path | None,
) -> LaneTransitionGuard | None:
    from .ledger_registry import owned_lane_ledger_root

    root = owned_lane_ledger_root(path, checkpoint_dir=checkpoint_dir)
    supplied = (
        expected_generation is not None
        or expected_ledger_sha256 is not None
        or authorization is not None
    )
    if root is None and not supplied:
        return None
    if (
        expected_generation is None
        or type(expected_generation) is not int
        or expected_generation < 1
        or not isinstance(expected_ledger_sha256, str)
        or len(expected_ledger_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_ledger_sha256)
        or authorization is None
    ):
        raise LaneLedgerError(
            "production lane transition requires generation, digest, and complete authorization"
        )
    normalized = _normalize_transition_authorization(authorization)
    if normalized[0] != str(path.expanduser().resolve(strict=False)):
        raise LaneLedgerError("lane transition authorization names another control ledger")
    if normalized[2] != segment_id or normalized[3] != expected_generation:
        raise LaneLedgerError(
            "lane transition authorization does not match logical unit/generation"
        )
    return LaneTransitionGuard(
        expected_generation, expected_ledger_sha256, normalized,
    )


def _validate_transition_guard(
    ledger: Mapping[str, Any],
    block: Mapping[str, Any],
    *,
    path: Path,
    segment_id: str,
    guard: LaneTransitionGuard | None,
) -> None:
    del path, segment_id
    if guard is None:
        return
    if (
        ledger.get("generation") != guard.expected_generation
        or block.get("generation") != guard.expected_generation
    ):
        raise LaneLedgerError("stale lane transition generation")


def _authorization_json(
    authorization: tuple[str, str, str, int, str],
) -> dict[str, Any]:
    return {
        "control_address": authorization[0],
        "session_key": authorization[1],
        "logical_unit": authorization[2],
        "generation": authorization[3],
        "idempotency_key": authorization[4],
    }


def _validate_stored_submission_authorization(
    block: Mapping[str, Any],
    authorization: tuple[str, str, str, int, str],
) -> None:
    if block.get("submission_authorization") != _authorization_json(authorization):
        raise LaneLedgerError("lane response authorization differs from submission")


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


def _mutate(
    path: Path,
    apply: Callable[[dict[str, Any]], Mapping[str, Any]],
    *,
    checkpoint_dir: Path | None = None,
    expected_ledger_sha256: str | None = None,
) -> dict[str, Any]:
    """Apply one RMW operation to the latest cross-process-locked snapshot."""

    from .ledger_registry import LaneLedgerRegistryError, mutate_lane_ledger

    try:
        updated = mutate_lane_ledger(
            path,
            checkpoint_dir=checkpoint_dir,
            expected_sha256=expected_ledger_sha256,
            mutate=apply,
        )
    except LaneLedgerRegistryError as exc:
        raise LaneLedgerError("registered lane ledger changed before mutation") from exc
    if updated is not None:
        return updated
    # Non-production fixtures retain their lightweight process-local path.
    with _CONTROL_LEDGER_LOCK:
        value = dict(apply(_read(path)))
        _write(path, value)
        return value


def _write(
    path: Path,
    value: Mapping[str, Any],
    *,
    checkpoint_dir: Path | None = None,
) -> None:
    from .ledger_registry import create_lane_ledger

    if create_lane_ledger(path, value, checkpoint_dir=checkpoint_dir):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _register(
    path: Path,
    value: Mapping[str, Any],
    *,
    checkpoint_dir: Path | None = None,
) -> None:
    # Local fixtures outside ARC's owned checkpoint layouts intentionally do
    # not acquire automatic-recovery ownership.
    from .ledger_registry import register_lane_ledger

    register_lane_ledger(path, value, checkpoint_dir=checkpoint_dir)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
