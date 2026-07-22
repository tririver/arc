from __future__ import annotations

from dataclasses import dataclass, replace
import ctypes
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import threading
from contextlib import contextmanager
from typing import Iterator
from typing import Any, Mapping

from arc_llm.call_checkpoint import (
    checkpoint_recomputation_binding,
    promote_recovered_response,
)
from arc_llm.call_record import ARC_LLM_CALL_RECORD_FIELD
from arc_llm.response_candidates import (
    persist_selection_receipt,
    select_response_candidate,
)
from arc_llm.raw_completion import RawCompletionError, validate_raw_completion
from arc_llm.schema_cache import canonical_json
from arc_llm.usage import LLMProviderResponse, LLMUsage, ResponseCandidateMaterial

from .io import sha256_json
from .ledger_registry import (
    LaneLedgerRegistryError,
    read_registered_lane_ledger,
)
from .recovery_units import RECOVERY_UNIT_REGISTRY
from .secure_io import (
    SecureReadError,
    read_bounded_file,
    read_bounded_json,
    safe_relative_path,
)


SUBMISSION_SCHEMA_VERSION = "arc.companion.recovery-submission.v1"
SUBMISSION_SEAL_SCHEMA_VERSION = "arc.companion.recovery-submission-seal.v1"
SUBMISSION_INDEX_SCHEMA_VERSION = "arc.companion.recovery-submission-index.v2"
VALIDATOR_VERSION = "arc.companion.business-replay.v1"
LEDGER_SNAPSHOT_SCHEMA_VERSION = "arc.companion.recovery-ledger-snapshot.v1"
MAX_RECOVERY_STREAM_BYTES = 16 * 1024 * 1024
MAX_CONTROL_JSON_BYTES = 16 * 1024 * 1024
MAX_SUBMISSION_SEAL_BYTES = 16 * 1024 * 1024
MAX_SUBMISSION_INDEX_BYTES = 16 * 1024 * 1024
MAX_SUBMISSION_INDEX_ENTRIES = 10_000
MAX_ATTEMPT_RECORDS = 32
_SUBMISSION_INDEX_LOCK = threading.Lock()
_HELD_ROOT = threading.local()
_ACTIVE_SUBMISSION_LOCK = threading.local()
_LEDGER_HANDLERS = {
    unit: (spec.validator, spec.application)
    for unit, spec in RECOVERY_UNIT_REGISTRY.items()
}


class RecoveryResponseError(RuntimeError):
    pass


@dataclass(frozen=True)
class _LeafSnapshot:
    raw: bytes
    identity: tuple[int, int, int, int, int, int]


def _recovery_write_fault(_cutpoint: str) -> None:
    """Test-only crash hook at named durable submission boundaries."""


def write_ledger_submission_receipt(
    *,
    checkpoint_dir: Path,
    artifact_dir: Path,
    ledger_path: Path,
    session_key: str,
    logical_unit: str,
    generation: int,
    idempotency_key: str,
    schema: Mapping[str, Any],
    prompt: str,
    recovery_unit: str,
    input_sha256: str,
    ordered_siblings: list[str],
    suffix: list[str],
    validator: str,
    application: str,
    group_sha256: str | None = None,
    acceptance_checkpoint: Path | None = None,
    stateful_checkpoint_identity: bool = True,
) -> Path:
    with _checkpoint_root_scope(checkpoint_dir) as root:
        return _write_ledger_submission_receipt_held(
            root=root,
            checkpoint_dir=checkpoint_dir,
            artifact_dir=artifact_dir,
            ledger_path=ledger_path,
            session_key=session_key,
            logical_unit=logical_unit,
            generation=generation,
            idempotency_key=idempotency_key,
            schema=schema,
            prompt=prompt,
            recovery_unit=recovery_unit,
            input_sha256=input_sha256,
            ordered_siblings=ordered_siblings,
            suffix=suffix,
            validator=validator,
            application=application,
            group_sha256=group_sha256,
            acceptance_checkpoint=acceptance_checkpoint,
            stateful_checkpoint_identity=stateful_checkpoint_identity,
        )


def _write_ledger_submission_receipt_held(
    *,
    root: Path,
    checkpoint_dir: Path,
    artifact_dir: Path,
    ledger_path: Path,
    session_key: str,
    logical_unit: str,
    generation: int,
    idempotency_key: str,
    schema: Mapping[str, Any],
    prompt: str,
    recovery_unit: str,
    input_sha256: str,
    ordered_siblings: list[str],
    suffix: list[str],
    validator: str,
    application: str,
    group_sha256: str | None,
    acceptance_checkpoint: Path | None,
    stateful_checkpoint_identity: bool,
) -> Path:
    _require_no_symlink_components(checkpoint_dir, root)
    _require_no_symlink_components(artifact_dir, root)
    _require_no_symlink_components(ledger_path, root)
    if acceptance_checkpoint is not None:
        _require_no_symlink_components(acceptance_checkpoint, root)
    artifact = _secure_ensure_directory(root, artifact_dir)
    ledger_address = ledger_path.resolve(strict=False)
    _require_contained(ledger_address, root)
    if (
        not idempotency_key
        or isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
        or not logical_unit
        or not session_key
    ):
        raise RecoveryResponseError("recovery submission identity is incomplete")
    if _LEDGER_HANDLERS.get(recovery_unit) != (validator, application):
        raise RecoveryResponseError("recovery submission handler is not registered")
    try:
        registered_ledger, registered_ledger_sha256 = read_registered_lane_ledger(
            checkpoint_dir, ledger_address,
        )
    except LaneLedgerRegistryError as exc:
        raise RecoveryResponseError(
            "recovery submission ledger is not one exact registered ledger"
        ) from exc
    ledger_snapshot = _ledger_semantic_snapshot(
        root,
        ledger_address,
        registered_ledger,
        session_key=session_key,
        logical_unit=logical_unit,
        generation=generation,
        receipt_input_sha256=input_sha256,
        ordered_siblings=ordered_siblings,
        suffix=suffix,
    )
    schema_value = dict(schema)
    payload = {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "session_key": session_key,
        "logical_unit": logical_unit,
        "generation": generation,
        "idempotency_key": idempotency_key,
        "schema": schema_value,
        "schema_sha256": hashlib.sha256(
            canonical_json(schema_value).encode("utf-8")
        ).hexdigest(),
        "validator": validator,
        "application": application,
        "recovery_unit": recovery_unit,
        "input_sha256": input_sha256,
        "group_sha256": str(group_sha256 or input_sha256),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "ordered_siblings": list(ordered_siblings),
        "suffix": list(suffix),
        "side_effect_policy": "no_unproven_external_side_effects",
        "external_side_effects": False,
        "artifact_dir": artifact.relative_to(root).as_posix(),
        "ledger_path": ledger_address.relative_to(root).as_posix(),
        "ledger_snapshot": ledger_snapshot,
        "ledger_snapshot_sha256": sha256_json(ledger_snapshot),
        "registered_ledger_sha256_at_creation": registered_ledger_sha256,
        "acceptance_checkpoint": (
            (acceptance_checkpoint or ledger_address).resolve(strict=False)
            .relative_to(root).as_posix()
        ),
        "checkpoint_session_key": session_key if stateful_checkpoint_identity else None,
        "checkpoint_generation": generation if stateful_checkpoint_identity else None,
        "attempt_records": [],
        "sealed": False,
    }
    payload["identity_sha256"] = sha256_json({
        key: payload[key] for key in (
            "session_key", "logical_unit", "generation", "idempotency_key",
            "schema_sha256", "validator", "application", "artifact_dir",
            "ledger_path", "side_effect_policy", "recovery_unit",
            "input_sha256", "prompt_sha256", "ordered_siblings", "suffix",
            "group_sha256",
            "acceptance_checkpoint", "checkpoint_session_key",
            "checkpoint_generation", "ledger_snapshot",
            "ledger_snapshot_sha256",
        )
    })
    path = artifact / f"recovery-submission-{payload['identity_sha256'][:16]}.json"
    receipt_raw = _json_bytes(payload)
    if len(receipt_raw) > MAX_CONTROL_JSON_BYTES:
        raise RecoveryResponseError("recovery submission receipt exceeds its byte limit")
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    existing_raw = _try_read_immutable_path(root, path)
    if existing_raw is not None:
        try:
            existing = json.loads(existing_raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RecoveryResponseError(
                "recovery submission receipt is invalid"
            ) from exc
        if not isinstance(existing, dict):
            raise RecoveryResponseError("recovery submission receipt is invalid")
        existing_stable = {
            key: value for key, value in existing.items()
            if key != "registered_ledger_sha256_at_creation"
        }
        payload_stable = {
            key: value for key, value in payload.items()
            if key != "registered_ledger_sha256_at_creation"
        }
        if existing_stable != payload_stable or not _valid_sha256(
            existing.get("registered_ledger_sha256_at_creation")
        ):
            raise RecoveryResponseError("recovery submission receipt identity changed")
        # Repair the bounded discovery index after a crash between the durable
        # receipt replace and index publication.
        _index_submission(
            root,
            path,
            existing,
            state="prepared",
            receipt_sha256=hashlib.sha256(existing_raw).hexdigest(),
        )
        return path
    # Reserve the bounded address before publishing the receipt.  Discovery
    # ignores a reserved-but-missing file, while the opposite crash cut can no
    # longer leave a durable receipt permanently undiscoverable.
    _index_submission(root, path, payload, state="reserved", receipt_sha256=receipt_sha256)
    _recovery_write_fault("reservation:durable")
    _exclusive_write(root, path, receipt_raw, kind="prepared_receipt")
    _index_submission(root, path, payload, state="prepared", receipt_sha256=receipt_sha256)
    _recovery_write_fault("prepared_index:durable")
    return path


def seal_submission_attempts(
    path: Path,
    *,
    checkpoint_dir: Path,
    attempt_references: list[Mapping[str, Any]],
) -> dict[str, Any]:
    with _checkpoint_root_scope(checkpoint_dir) as root:
        return _seal_submission_attempts_held(
            path,
            root=root,
            checkpoint_dir=checkpoint_dir,
            attempt_references=attempt_references,
        )


def _seal_submission_attempts_held(
    path: Path,
    *,
    root: Path,
    checkpoint_dir: Path,
    attempt_references: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(attempt_references) > MAX_ATTEMPT_RECORDS:
        raise RecoveryResponseError("too many explicit attempt records")
    receipt_path = _safe_supplied_path(root, path)
    receipt_raw = _try_read_immutable_path(root, receipt_path)
    if receipt_raw is None:
        raise RecoveryResponseError("recovery submission receipt is missing")
    try:
        receipt = json.loads(receipt_raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryResponseError("recovery submission receipt is invalid") from exc
    if not isinstance(receipt, dict) or receipt.get("schema_version") != SUBMISSION_SCHEMA_VERSION:
        raise RecoveryResponseError("recovery submission receipt is invalid")
    seal_path = _seal_path(receipt_path)
    artifact = _safe_relative_path(root, receipt.get("artifact_dir"))
    checkpoint_path = artifact / "call-checkpoints" / (
        "idempotency-"
        + hashlib.sha256(str(receipt.get("idempotency_key") or "").encode("utf-8")).hexdigest()
        + ".json"
    )
    _require_contained(checkpoint_path.resolve(strict=False), root)
    checkpoint = _read_json_path(root, checkpoint_path, suffixes=(".json",))
    if not isinstance(checkpoint, dict):
        raise RecoveryResponseError("submitted call has no call checkpoint")
    logical = checkpoint.get("logical_identity")
    recipe = checkpoint.get("request_recipe")
    if not isinstance(logical, Mapping) or not isinstance(recipe, Mapping):
        raise RecoveryResponseError("submitted call checkpoint identity is incomplete")
    if (
        logical.get("session_key") != receipt.get("checkpoint_session_key")
        or logical.get("generation") != receipt.get("checkpoint_generation")
        or logical.get("idempotency_key") != receipt.get("idempotency_key")
        or recipe.get("prompt_sha256") != receipt.get("prompt_sha256")
        or recipe.get("schema_sha256") != receipt.get("schema_sha256")
    ):
        raise RecoveryResponseError("submitted call checkpoint identity changed")
    refs = []
    records: list[tuple[Path, dict[str, Any]]] = []
    seen_paths: set[str] = set()
    for supplied in attempt_references:
        if not isinstance(supplied, Mapping):
            raise RecoveryResponseError("explicit attempt reference is invalid")
        record_path = _safe_relative_path(root, supplied.get("path"))
        try:
            relative = record_path.relative_to(artifact / "attempts")
        except ValueError as exc:
            raise RecoveryResponseError(
                "explicit attempt reference is outside submitted artifact"
            ) from exc
        if len(relative.parts) != 2 or relative.parts[-1] != "record.json":
            raise RecoveryResponseError("attempt record is not a regular file")
        raw = _read_path_bytes(root, record_path, suffixes=("record.json",))
        digest = hashlib.sha256(raw).hexdigest()
        if digest != supplied.get("sha256"):
            raise RecoveryResponseError("explicit attempt record hash changed")
        reference = {
            "path": record_path.relative_to(root).as_posix(),
            "sha256": digest,
        }
        if reference["path"] in seen_paths:
            raise RecoveryResponseError("explicit attempt reference is duplicated")
        seen_paths.add(reference["path"])
        try:
            record_value = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RecoveryResponseError("explicit attempt record is invalid") from exc
        if not isinstance(record_value, dict):
            raise RecoveryResponseError("explicit attempt record is invalid")
        refs.append(reference)
        records.append((record_path, record_value))
        if len(refs) > MAX_ATTEMPT_RECORDS:
            raise RecoveryResponseError("too many explicit attempt records")
    if not refs:
        raise RecoveryResponseError("submitted call has no immutable attempt record")
    _validate_attempt_record_lifecycle(records)
    attempt_evidence = _build_attempt_evidence(root, artifact, refs)
    sidecar = {
        "seal_schema_version": SUBMISSION_SEAL_SCHEMA_VERSION,
        "receipt_path": receipt_path.relative_to(root).as_posix(),
        "receipt_sha256": hashlib.sha256(
            _read_path_bytes(root, receipt_path, suffixes=(".json",))
        ).hexdigest(),
        "identity_sha256": receipt["identity_sha256"],
        "attempt_records": refs,
        "attempt_evidence": attempt_evidence,
        "attempt_evidence_sha256": sha256_json(attempt_evidence),
        "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
        "checkpoint_identity": str(checkpoint.get("identity") or ""),
        "provider": str(logical.get("provider") or ""),
        "model": logical.get("model"),
        "sealed": True,
    }
    sidecar_raw = _json_bytes(sidecar)
    if len(sidecar_raw) > MAX_SUBMISSION_SEAL_BYTES:
        raise RecoveryResponseError("recovery submission sidecar exceeds its byte limit")
    _exclusive_write(root, seal_path, sidecar_raw, kind="sealed_sidecar")
    _index_submission(
        root, receipt_path, receipt, state="sealed",
        receipt_sha256=sidecar["receipt_sha256"],
        sidecar_path=seal_path,
        sidecar_sha256=hashlib.sha256(sidecar_raw).hexdigest(),
    )
    _recovery_write_fault("sealed_index:durable")
    return {**receipt, **sidecar}


def explicit_attempt_references(
    value: Any,
    *,
    checkpoint_dir: Path,
    artifact_dir: Path,
) -> list[dict[str, str]]:
    """Resolve only attempt refs explicitly returned by this exact call."""

    record: Mapping[str, Any] | None = None
    if isinstance(value, Mapping):
        candidate = value.get(ARC_LLM_CALL_RECORD_FIELD)
        record = candidate if isinstance(candidate, Mapping) else None
    else:
        candidate = getattr(value, "call_record", None)
        record = candidate if isinstance(candidate, Mapping) else None
    supplied: list[Mapping[str, Any]] = []
    if record is not None:
        attempts = record.get("attempts")
        if not isinstance(attempts, list):
            raise RecoveryResponseError("returned attempt aggregate is invalid")
        for item in attempts:
            if not isinstance(item, Mapping):
                raise RecoveryResponseError("returned attempt aggregate is invalid")
            path_present = item.get("diagnostic_path") is not None
            hash_present = item.get("diagnostic_sha256") is not None
            if path_present != hash_present:
                raise RecoveryResponseError("returned attempt aggregate is partial")
            if not path_present:
                raise RecoveryResponseError(
                    "returned attempt aggregate lacks immutable diagnostics"
                )
            supplied.append(item)
    if isinstance(value, BaseException):
        terminal_refs = getattr(value, "attempt_diagnostic_refs", None)
        if terminal_refs is not None:
            if not isinstance(terminal_refs, tuple) or not all(
                isinstance(item, Mapping) for item in terminal_refs
            ):
                raise RecoveryResponseError("returned attempt aggregate is invalid")
            supplied.extend(terminal_refs)
        else:
            cursor = value.__cause__
            while cursor is not None:
                cause_refs = getattr(cursor, "attempt_diagnostic_refs", None)
                if cause_refs is not None:
                    if not isinstance(cause_refs, tuple) or not all(
                        isinstance(item, Mapping) for item in cause_refs
                    ):
                        raise RecoveryResponseError(
                            "returned attempt aggregate is invalid"
                        )
                    supplied.extend(cause_refs)
                    break
                cursor = cursor.__cause__
    root = checkpoint_dir.resolve(strict=False)
    artifact = artifact_dir.resolve(strict=False)
    output: list[dict[str, str]] = []
    records: list[tuple[Path, dict[str, Any]]] = []
    seen_paths: set[str] = set()
    for item in supplied:
        relative = item.get("diagnostic_path", item.get("path"))
        relative_path = Path(str(relative or ""))
        if (
            not relative_path.parts
            or relative_path.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise RecoveryResponseError("returned attempt reference path is invalid")
        candidate = artifact / relative_path
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RecoveryResponseError("attempt reference escapes checkpoint") from exc
        path = _safe_relative_path(root, candidate.relative_to(root).as_posix())
        try:
            path.relative_to(artifact)
        except ValueError as exc:
            raise RecoveryResponseError("attempt reference escapes call artifact") from exc
        raw = _read_path_bytes(root, path, suffixes=("record.json",))
        digest = hashlib.sha256(raw).hexdigest()
        expected = item.get("diagnostic_sha256", item.get("sha256"))
        if digest != expected:
            raise RecoveryResponseError("returned attempt reference hash changed")
        reference = {
            "path": path.relative_to(root).as_posix(),
            "sha256": digest,
        }
        if reference["path"] in seen_paths:
            raise RecoveryResponseError("returned attempt aggregate is duplicated")
        seen_paths.add(reference["path"])
        try:
            attempt_record = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RecoveryResponseError("returned attempt record is invalid") from exc
        if not isinstance(attempt_record, dict):
            raise RecoveryResponseError("returned attempt record is invalid")
        output.append(reference)
        records.append((path, attempt_record))
    if records:
        _validate_attempt_record_lifecycle(records)
    return output


def submission_receipt_reference(path: Path, *, checkpoint_dir: Path) -> dict[str, str]:
    root = checkpoint_dir.resolve(strict=False)
    resolved = _safe_supplied_path(root, path)
    raw = _read_path_bytes(root, resolved, suffixes=(".json",))
    try:
        receipt = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryResponseError("recovery submission receipt is invalid") from exc
    identity_sha256 = receipt.get("identity_sha256") if isinstance(receipt, Mapping) else None
    if not _valid_sha256(identity_sha256):
        raise RecoveryResponseError("recovery submission receipt identity is invalid")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "identity_sha256": str(identity_sha256),
    }


def validate_ledger_submission_reference(
    reference: Mapping[str, Any],
    *,
    checkpoint_dir: Path,
    expected_recovery_identity: tuple[str, str, str, int, str],
    expected_receipt_identity_sha256: str,
) -> dict[str, Any]:
    if (
        not isinstance(expected_recovery_identity, tuple)
        or len(expected_recovery_identity) != 5
    ):
        raise RecoveryResponseError("expected recovery identity is invalid")
    (
        expected_ledger_value,
        session_key,
        logical_unit,
        generation,
        idempotency_key,
    ) = expected_recovery_identity
    if (
        not isinstance(expected_ledger_value, str)
        or not session_key
        or not logical_unit
        or isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
        or not idempotency_key
        or not _valid_sha256(expected_receipt_identity_sha256)
    ):
        raise RecoveryResponseError("expected recovery identity is invalid")
    root = checkpoint_dir.resolve(strict=False)
    expected_ledger = _canonical_expected_ledger_path(root, expected_ledger_value)
    receipt_path = _safe_relative_path(root, reference.get("path"))
    raw = _read_path_bytes(root, receipt_path, suffixes=(".json",))
    if hashlib.sha256(raw).hexdigest() != reference.get("sha256"):
        raise RecoveryResponseError("recovery submission receipt hash changed")
    receipt = _load_indexed_submission(root, receipt_path)
    if not isinstance(receipt, dict):
        raise RecoveryResponseError("recovery submission receipt is invalid")
    _validate_receipt_identity(receipt)
    if (
        receipt.get("identity_sha256") != expected_receipt_identity_sha256
        or reference.get("identity_sha256") != expected_receipt_identity_sha256
    ):
        raise RecoveryResponseError("recovery submission receipt identity changed")
    if (
        _safe_relative_path(root, receipt.get("ledger_path")) != expected_ledger
        or receipt.get("session_key") != session_key
        or receipt.get("logical_unit") != logical_unit
        or receipt.get("generation") != generation
        or receipt.get("idempotency_key") != idempotency_key
    ):
        raise RecoveryResponseError("recovery submission does not match lane entry")
    current_ledger, current_digest, current_snapshot = _current_ledger_snapshot(
        checkpoint_dir, root, expected_ledger, receipt,
    )
    if receipt.get("recovery_unit") != current_ledger.get("lane"):
        raise RecoveryResponseError("recovery submission does not match lane entry")
    if not receipt.get("sealed"):
        raise RecoveryResponseError("recovery submission attempts were not sealed")
    _validate_checkpoint_binding(receipt, root)
    _validate_attempt_refs(receipt, root)
    return {
        **receipt,
        "validated_ledger_snapshot": current_snapshot,
        "current_registered_ledger_sha256": current_digest,
    }


def recover_complete_ledger_response(
    reference: Mapping[str, Any], *, checkpoint_dir: Path,
    ledger_path: Path,
    session_key: str,
    logical_unit: str,
    generation: int,
    idempotency_key: str,
    expected_receipt_identity_sha256: str,
) -> dict[str, Any]:
    """Promote one complete raw candidate, then let normal business code replay it."""

    root = checkpoint_dir.resolve(strict=False)
    receipt = validate_ledger_submission_reference(
        reference,
        checkpoint_dir=checkpoint_dir,
        expected_recovery_identity=(
            str(ledger_path), session_key, logical_unit, generation, idempotency_key,
        ),
        expected_receipt_identity_sha256=expected_receipt_identity_sha256,
    )
    if not receipt.get("sealed"):
        raise RecoveryResponseError("recovery submission attempts were not sealed")
    records = _validate_attempt_refs(receipt, root)
    artifact = _safe_relative_path(root, receipt.get("artifact_dir"))
    key = str(receipt["idempotency_key"])
    checkpoint_path = _safe_relative_path(root, receipt.get("checkpoint_path"))
    expected_checkpoint_path = artifact / "call-checkpoints" / (
        "idempotency-" + hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json"
    )
    if checkpoint_path != expected_checkpoint_path:
        raise RecoveryResponseError("call checkpoint address does not match recovery submission")
    checkpoint = _read_json_path(root, checkpoint_path, suffixes=(".json",))
    expected_logical = _validated_checkpoint_logical_identity(
        checkpoint, receipt, root,
    )
    recipe = checkpoint.get("request_recipe")
    if (
        str(checkpoint.get("identity") or "") != receipt.get("checkpoint_identity")
        or not isinstance(recipe, Mapping)
        or recipe.get("prompt_sha256") != receipt.get("prompt_sha256")
        or recipe.get("schema_sha256") != receipt.get("schema_sha256")
    ):
        raise RecoveryResponseError("call checkpoint recipe does not match recovery submission")
    material = _material_from_attempts(
        records,
        expected_provider=str(receipt.get("provider") or ""),
        expected_model=receipt.get("model"),
        expected_call_label=str(recipe.get("call_label") or ""),
        expected_checkpoint_identity=str(checkpoint.get("identity") or ""),
    )
    if not material:
        return {"complete": False, "source": "no_complete_raw_candidate"}
    response = LLMProviderResponse(
        {}, usage=LLMUsage(), candidate_material=tuple(material),
    )
    selection = select_response_candidate(
        response,
        schema=receipt["schema"],
        checkpoint_identity=str(checkpoint.get("identity") or ""),
        replayed=True,
    )
    selection_receipt = {
        **selection.receipt,
        "recovery_evidence": receipt["attempt_evidence"],
        "recovery_evidence_sha256": receipt["attempt_evidence_sha256"],
    }
    if selection.conflict is not None:
        selection_path, selection_sha = persist_selection_receipt(
            checkpoint_path, selection_receipt, replayed=True,
        )
        raise selection.conflict
    if selection.receipt.get("decision") == "no_schema_valid_candidate":
        return {"complete": False, "source": "no_schema_valid_raw_candidate"}
    selection_path, selection_sha = persist_selection_receipt(
        checkpoint_path, selection_receipt, replayed=True,
    )
    selected = replace(
        selection.response,
        candidate_selection=selection_receipt,
    )
    _current_ledger, final_ledger_digest, final_ledger_snapshot = (
        _current_ledger_snapshot(
            checkpoint_dir,
            root,
            _canonical_expected_ledger_path(root, str(ledger_path)),
            receipt,
        )
    )
    recomputation_binding = checkpoint_recomputation_binding(checkpoint_path)
    promote_recovered_response(
        checkpoint_path,
        selected,
        expected_logical_identity=expected_logical,
        expected_schema_sha256=str(receipt["schema_sha256"]),
        selection_receipt_path=selection_path,
        selection_receipt_sha256=selection_sha,
        expected_recomputation_binding=recomputation_binding,
    )
    return {
        "complete": True,
        "source": "promoted_response_pending_business",
        "business_status": "pending_validation_and_application",
        "validated_ledger_snapshot": final_ledger_snapshot,
        "current_registered_ledger_sha256": final_ledger_digest,
        "candidate_sha256": selection.receipt.get("selected_sha256"),
        "selection_receipt": selection_path,
        "selection_receipt_sha256": selection_sha,
    }


def _material_from_attempts(
    records: list[tuple[Path, dict[str, Any]]],
    *,
    expected_provider: str,
    expected_model: Any,
    expected_call_label: str,
    expected_checkpoint_identity: str,
) -> list[ResponseCandidateMaterial]:
    output: list[ResponseCandidateMaterial] = []
    position = 0
    incomplete_attempts = 0
    for record_path, record in records:
        provider = str(record.get("provider") or "")
        # The attempt record is immutable evidence for this exact prepared
        # provider call, never a global pool of response-looking objects.
        if (
            provider != expected_provider
            or record.get("model") != expected_model
            or str(record.get("call_label") or "") != expected_call_label
            or record.get("checkpoint_identity") != expected_checkpoint_identity
            or record.get("submission_state") not in {"submitted", "unknown"}
        ):
            raise RecoveryResponseError("attempt record does not match submitted call")
        streams = record.get("streams")
        streams = streams if isinstance(streams, Mapping) else {}
        parsed_material = _validated_parsed_candidate_material(
            record_path, record, streams,
        )
        raw_text = _verified_stream(record_path.parent, streams.get("raw_events"))
        events = _json_objects(raw_text)
        stdout_text = _verified_stream(record_path.parent, streams.get("stdout"))
        stdout_missing = not stdout_text.strip()
        if not stdout_missing:
            stdout_events = _json_objects(stdout_text)
            inbound_events = [
                event for event in events
                if event.get("direction") not in {"request", "reverse_response"}
            ]
            if stdout_events != inbound_events:
                raise RecoveryResponseError(
                    "attempt stdout and raw-event inventory differ"
                )
            if provider in {"codex-cli", "claude-cli"}:
                events = stdout_events
        try:
            completion = validate_raw_completion(
                provider, events,
                native_session_id=(str(record.get("native_session_id")) if record.get("native_session_id") else None),
            )
        except RawCompletionError as exc:
            if _incomplete_raw_completion(exc):
                if parsed_material:
                    raise RecoveryResponseError(
                        "incomplete raw attempt conflicts with parsed candidates"
                    ) from exc
                incomplete_attempts += 1
                continue
            raise RecoveryResponseError(str(exc)) from exc
        if stdout_missing:
            raise RecoveryResponseError(
                "complete raw attempt has an empty stdout inventory"
            )
        if not completion.material:
            if parsed_material:
                raise RecoveryResponseError(
                    "empty raw attempt conflicts with parsed candidates"
                )
            incomplete_attempts += 1
            continue
        for item in completion.material:
            if item.value is None and not str(item.text or "").strip():
                continue
            output.append(replace(item, protocol_position=position))
            position += 1
        for item in parsed_material:
            output.append(replace(item, protocol_position=position))
            position += 1
    if incomplete_attempts and output:
        raise RecoveryResponseError(
            "incomplete or empty attempt conflicts with complete sealed material"
        )
    return output


def _incomplete_raw_completion(error: RawCompletionError) -> bool:
    message = str(error)
    return (
        "event stream is empty" in message
        or "lacks one final" in message
    )


def _validated_parsed_candidate_material(
    record_path: Path,
    record: Mapping[str, Any],
    streams: Mapping[str, Any],
) -> list[ResponseCandidateMaterial]:
    candidate_text = _verified_stream(
        record_path.parent, streams.get("response_candidates"),
    )
    candidate_lines = candidate_text.splitlines()
    metadata = record.get("parsed_response_candidates")
    metadata = metadata if isinstance(metadata, list) else []
    if (
        int(record.get("parsed_response_candidates_dropped") or 0) != 0
        or int(record.get("parsed_response_candidate_count") or 0) != len(metadata)
        or len(candidate_lines) != len(metadata)
    ):
        raise RecoveryResponseError("attempt candidate inventory is incomplete")
    previous_sequence = 0
    parsed_material: list[ResponseCandidateMaterial] = []
    for index, line in enumerate(candidate_lines):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecoveryResponseError("attempt candidate stream is malformed") from exc
        value = item.get("value") if isinstance(item, Mapping) else None
        meta = metadata[index] if isinstance(metadata[index], Mapping) else {}
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ) if isinstance(value, Mapping) else ""
        sequence = item.get("sequence") if isinstance(item, Mapping) else None
        if (
            not isinstance(item, Mapping)
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence <= previous_sequence
            or sequence != meta.get("sequence")
            or item.get("source") != meta.get("source")
            or not isinstance(value, Mapping)
            or hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
            != meta.get("sha256")
            or len(encoded.encode("utf-8")) != meta.get("bytes")
            or meta.get("value_type") != "dict"
        ):
            raise RecoveryResponseError("attempt candidate metadata changed")
        previous_sequence = sequence
        parsed_material.append(
            ResponseCandidateMaterial(
                source="recovery.response_candidate_stream",
                protocol_position=0,
                value=dict(value),
                event_id=(
                    f"attempt-{record.get('attempt_id')}-candidate-{sequence}"
                ),
            )
        )
    return parsed_material


def _validate_attempt_record_lifecycle(
    records: list[tuple[Path, dict[str, Any]]],
) -> None:
    if not records or len(records) > MAX_ATTEMPT_RECORDS:
        raise RecoveryResponseError("attempt record lifecycle is empty or oversized")
    seen_paths: set[str] = set()
    seen_digests: set[str] = set()
    seen_ids: set[str] = set()
    seen_positions: set[tuple[int, int]] = set()
    call_label: Any = records[0][1].get("call_label")
    previous_position: tuple[int, int] | None = None
    terminal_outcomes = {"success", "replayed"}
    allowed_outcomes = {"error", "timeout", "cancelled", "failed", *terminal_outcomes}
    for index, (path, record) in enumerate(records):
        if record.get("schema_version") != "arc.llm.attempt_diagnostic.v1":
            raise RecoveryResponseError("attempt record schema is unsupported")
        rendered_path = str(path)
        digest = hashlib.sha256(_json_bytes(record)).hexdigest()
        attempt_id = record.get("attempt_id")
        fallback_index = record.get("fallback_index")
        attempt = record.get("attempt")
        outcome = record.get("outcome")
        if (
            rendered_path in seen_paths
            or digest in seen_digests
            or not isinstance(attempt_id, str)
            or not attempt_id
            or attempt_id in seen_ids
        ):
            raise RecoveryResponseError("attempt record lifecycle is duplicated")
        if (
            isinstance(fallback_index, bool)
            or not isinstance(fallback_index, int)
            or fallback_index < 0
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt < 1
            or record.get("call_label") != call_label
            or outcome not in allowed_outcomes
        ):
            raise RecoveryResponseError("attempt record lifecycle is invalid")
        position = (fallback_index, attempt)
        if position in seen_positions:
            raise RecoveryResponseError("attempt record lifecycle is duplicated")
        if previous_position is None:
            if position != (0, 1):
                raise RecoveryResponseError("attempt record lifecycle does not start at one")
        elif not (
            position == (previous_position[0], previous_position[1] + 1)
            or position == (previous_position[0] + 1, 1)
        ):
            raise RecoveryResponseError("attempt record lifecycle is not contiguous")
        if index < len(records) - 1 and outcome in terminal_outcomes:
            raise RecoveryResponseError("attempt record follows a successful terminal")
        seen_paths.add(rendered_path)
        seen_digests.add(digest)
        seen_ids.add(attempt_id)
        seen_positions.add(position)
        previous_position = position


def _validate_attempt_refs(
    receipt: Mapping[str, Any], root: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    output = []
    refs = receipt.get("attempt_records")
    if not isinstance(refs, list) or not refs or len(refs) > MAX_ATTEMPT_RECORDS:
        raise RecoveryResponseError("recovery submission has no attempt records")
    artifact = _safe_relative_path(root, receipt.get("artifact_dir"))
    attempts_root = artifact / "attempts"
    seen_paths: set[str] = set()
    for ref in refs:
        if not isinstance(ref, Mapping):
            raise RecoveryResponseError("attempt reference is invalid")
        path = _safe_relative_path(root, ref.get("path"))
        try:
            relative = path.relative_to(attempts_root)
        except ValueError as exc:
            raise RecoveryResponseError("attempt record is outside submitted artifact") from exc
        if len(relative.parts) != 2 or relative.parts[-1] != "record.json":
            raise RecoveryResponseError("attempt record address is invalid")
        rendered_path = path.relative_to(root).as_posix()
        if rendered_path in seen_paths:
            raise RecoveryResponseError("attempt record reference is duplicated")
        seen_paths.add(rendered_path)
        raw = _read_path_bytes(root, path, suffixes=("record.json",))
        if hashlib.sha256(raw).hexdigest() != ref.get("sha256"):
            raise RecoveryResponseError("attempt record hash changed")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RecoveryResponseError("attempt record is invalid")
        if value.get("schema_version") != "arc.llm.attempt_diagnostic.v1":
            raise RecoveryResponseError("attempt record schema is unsupported")
        output.append((path, value))
    _validate_attempt_record_lifecycle(output)
    evidence = _build_attempt_evidence(root, artifact, refs)
    if (
        receipt.get("attempt_evidence") != evidence
        or receipt.get("attempt_evidence_sha256") != sha256_json(evidence)
    ):
        raise RecoveryResponseError("attempt evidence manifest changed")
    return output


def _build_attempt_evidence(
    root: Path,
    artifact: Path,
    refs: list[Mapping[str, Any]] | list[dict[str, str]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    stream_order = ("raw_events", "stdout", "response_candidates", "stderr")
    for ordinal, ref in enumerate(refs, 1):
        path = _safe_relative_path(root, ref.get("path"))
        raw = _read_path_bytes(root, path, suffixes=("record.json",))
        if hashlib.sha256(raw).hexdigest() != ref.get("sha256"):
            raise RecoveryResponseError("attempt record hash changed")
        try:
            record = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RecoveryResponseError("attempt record is invalid") from exc
        if (
            not isinstance(record, Mapping)
            or record.get("schema_version") != "arc.llm.attempt_diagnostic.v1"
        ):
            raise RecoveryResponseError("attempt record schema is unsupported")
        streams = record.get("streams")
        if not isinstance(streams, Mapping) or set(streams) - set(stream_order):
            raise RecoveryResponseError("attempt stream inventory is invalid")
        stream_evidence: list[dict[str, Any]] = []
        for name in stream_order:
            source = streams.get(name)
            if source is None:
                continue
            if not isinstance(source, Mapping):
                raise RecoveryResponseError("attempt stream receipt is invalid")
            relative = str(source.get("path") or "")
            if not relative or Path(relative).name != relative:
                raise RecoveryResponseError("attempt stream address is invalid")
            stream_path = path.parent / relative
            try:
                stream_path.relative_to(artifact)
            except ValueError as exc:
                raise RecoveryResponseError(
                    "attempt stream escapes submitted artifact"
                ) from exc
            stream_evidence.append({
                "name": name,
                "path": stream_path.relative_to(root).as_posix(),
                "sha256": source.get("sha256"),
                "stored_bytes": source.get("stored_bytes"),
                "compression": source.get("compression"),
                "truncated": source.get("truncated"),
                "lossless": source.get("lossless"),
            })
        evidence.append({
            "ordinal": ordinal,
            "record_path": path.relative_to(root).as_posix(),
            "record_sha256": ref.get("sha256"),
            "attempt_id": record.get("attempt_id"),
            "streams": stream_evidence,
        })
    return evidence


def _verified_stream(attempt_dir: Path, value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    if value.get("truncated") is not False:
        raise RecoveryResponseError("truncated attempt streams are not replay-safe")
    relative = str(value.get("path") or "")
    if not relative or Path(relative).name != relative:
        raise RecoveryResponseError("attempt stream address is invalid")
    path = attempt_dir / relative
    try:
        raw = read_bounded_file(
            attempt_dir, relative, max_bytes=MAX_RECOVERY_STREAM_BYTES,
            suffixes=(".jsonl", ".jsonl.gz", ".txt", ".txt.gz"),
        )
    except SecureReadError as exc:
        raise RecoveryResponseError("attempt stream is unsafe or missing") from exc
    if hashlib.sha256(raw).hexdigest() != value.get("sha256"):
        raise RecoveryResponseError("attempt stream hash changed")
    if (
        not isinstance(value.get("observed_bytes"), int)
        or not isinstance(value.get("sanitized_bytes"), int)
        or value["observed_bytes"] < 0
        or value["sanitized_bytes"] < 0
        or value.get("stored_bytes") != len(raw)
        or value.get("lossless") is not True
    ):
        raise RecoveryResponseError("attempt stream byte receipt changed")
    compression = value.get("compression")
    if compression == "gzip":
        if path.suffix != ".gz":
            raise RecoveryResponseError("attempt stream compression suffix changed")
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as handle:
                raw = handle.read(MAX_RECOVERY_STREAM_BYTES + 1)
        except (OSError, EOFError) as exc:
            raise RecoveryResponseError("attempt stream gzip is invalid") from exc
    elif compression == "none":
        if path.suffix == ".gz":
            raise RecoveryResponseError("attempt stream compression suffix changed")
    else:
        raise RecoveryResponseError("attempt stream compression is invalid")
    if len(raw) > MAX_RECOVERY_STREAM_BYTES:
        raise RecoveryResponseError("expanded attempt stream exceeds recovery byte limit")
    return raw.decode("utf-8", errors="strict")


def _json_objects(text: str) -> list[dict[str, Any]]:
    output = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecoveryResponseError("attempt event stream is malformed") from exc
        if not isinstance(value, dict):
            raise RecoveryResponseError("attempt event stream contains a non-object")
        output.append(value)
    return output


def _require_no_symlink_components(path: Path, root: Path) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise RecoveryResponseError("recovery artifact escapes the active checkpoint") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RecoveryResponseError("recovery artifact address contains a symlink")


def _require_contained(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RecoveryResponseError("recovery artifact escapes the active checkpoint") from exc


def _safe_relative_path(root: Path, value: Any) -> Path:
    try:
        relative = safe_relative_path(value)
    except SecureReadError as exc:
        raise RecoveryResponseError(
            "recovery artifact address is not a safe relative path"
        ) from exc
    return root / relative


def _safe_supplied_path(root: Path, path: Path) -> Path:
    lexical = path.absolute()
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise RecoveryResponseError("recovery artifact escapes the active checkpoint") from exc
    return _safe_relative_path(root, relative.as_posix())


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_expected_ledger_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return _safe_supplied_path(root, candidate)
    return _safe_relative_path(root, value)


def _ledger_semantic_snapshot(
    root: Path,
    ledger_path: Path,
    ledger: Mapping[str, Any],
    *,
    session_key: str,
    logical_unit: str,
    generation: int,
    receipt_input_sha256: str,
    ordered_siblings: list[str],
    suffix: list[str],
) -> dict[str, Any]:
    relative = ledger_path.relative_to(root)
    owner = {
        "chapters": "arc-companion.chapter-lane",
        "recovery-controls": "arc-companion.pipeline-recovery-control",
    }.get(relative.parts[0] if relative.parts else "")
    blocks = ledger.get("blocks")
    if owner is None or not isinstance(blocks, list):
        raise RecoveryResponseError("registered ledger semantic identity is invalid")
    if not all(isinstance(item, Mapping) for item in blocks):
        raise RecoveryResponseError("registered ledger block topology is invalid")
    topology = [str(item.get("segment_id") or "") for item in blocks]
    if (
        not topology
        or not all(topology)
        or len(topology) != len(set(topology))
        or topology != list(ordered_siblings)
        or logical_unit not in topology
        or list(suffix) != topology[topology.index(logical_unit):]
    ):
        raise RecoveryResponseError("registered ledger ordered ownership changed")
    targets = [item for item in blocks if item.get("segment_id") == logical_unit]
    if len(targets) != 1:
        raise RecoveryResponseError("registered ledger logical unit is absent or ambiguous")
    target = targets[0]
    ledger_generation = ledger.get("generation")
    block_generation = target.get("generation")
    if (
        isinstance(ledger_generation, bool)
        or not isinstance(ledger_generation, int)
        or ledger_generation < 1
        or isinstance(block_generation, bool)
        or not isinstance(block_generation, int)
        or block_generation != generation
    ):
        raise RecoveryResponseError("registered ledger generation changed")
    block_input = target.get("input_sha256")
    if block_input is not None and not isinstance(block_input, str):
        raise RecoveryResponseError("registered ledger block input is invalid")
    chapter_id = str(ledger.get("chapter_id") or "")
    lane = str(ledger.get("lane") or "")
    if not chapter_id or not lane or not session_key or not receipt_input_sha256:
        raise RecoveryResponseError("registered ledger semantic identity is incomplete")
    return {
        "schema_version": LEDGER_SNAPSHOT_SCHEMA_VERSION,
        "owner": owner,
        "ledger_path": relative.as_posix(),
        "chapter_id": chapter_id,
        "lane": lane,
        "session_key": session_key,
        "logical_unit": logical_unit,
        "ledger_generation": ledger_generation,
        "block_generation": block_generation,
        "block_input_sha256": block_input,
        "receipt_input_sha256": receipt_input_sha256,
        "ordered_segment_ids": topology,
        "ordered_siblings": list(ordered_siblings),
        "suffix": list(suffix),
    }


def _current_ledger_snapshot(
    checkpoint_dir: Path,
    root: Path,
    ledger_path: Path,
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    try:
        ledger, digest = read_registered_lane_ledger(checkpoint_dir, ledger_path)
    except LaneLedgerRegistryError as exc:
        raise RecoveryResponseError(
            "recovery submission ledger is not one exact registered ledger"
        ) from exc
    snapshot = _ledger_semantic_snapshot(
        root,
        ledger_path,
        ledger,
        session_key=str(receipt.get("session_key") or ""),
        logical_unit=str(receipt.get("logical_unit") or ""),
        generation=int(receipt.get("generation") or 0),
        receipt_input_sha256=str(receipt.get("input_sha256") or ""),
        ordered_siblings=list(receipt.get("ordered_siblings") or []),
        suffix=list(receipt.get("suffix") or []),
    )
    if (
        snapshot != receipt.get("ledger_snapshot")
        or sha256_json(snapshot) != receipt.get("ledger_snapshot_sha256")
    ):
        raise RecoveryResponseError("registered ledger semantic snapshot changed")
    return ledger, digest, snapshot


def _validate_checkpoint_binding(
    receipt: Mapping[str, Any], root: Path,
) -> dict[str, Any]:
    artifact = _safe_relative_path(root, receipt.get("artifact_dir"))
    key = str(receipt.get("idempotency_key") or "")
    checkpoint_path = _safe_relative_path(root, receipt.get("checkpoint_path"))
    expected_path = artifact / "call-checkpoints" / (
        "idempotency-" + hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json"
    )
    if checkpoint_path != expected_path:
        raise RecoveryResponseError("call checkpoint address does not match recovery submission")
    checkpoint = _read_json_path(root, checkpoint_path, suffixes=(".json",))
    _validated_checkpoint_logical_identity(checkpoint, receipt, root)
    recipe = checkpoint.get("request_recipe") if isinstance(checkpoint, Mapping) else None
    if (
        not isinstance(recipe, Mapping)
        or str(checkpoint.get("identity") or "") != receipt.get("checkpoint_identity")
        or recipe.get("prompt_sha256") != receipt.get("prompt_sha256")
        or recipe.get("schema_sha256") != receipt.get("schema_sha256")
    ):
        raise RecoveryResponseError("call checkpoint does not match recovery submission")
    return dict(checkpoint)


def _validated_checkpoint_logical_identity(
    checkpoint: Mapping[str, Any],
    receipt: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Validate checkpoint identity fields without rejecting sealed v5 metadata."""

    logical = checkpoint.get("logical_identity")
    key = str(receipt.get("idempotency_key") or "")
    if (
        not isinstance(logical, Mapping)
        or logical.get("provider") != receipt.get("provider")
        or logical.get("model") != receipt.get("model")
        or logical.get("session_key") != receipt.get("checkpoint_session_key")
        or logical.get("generation") != receipt.get("checkpoint_generation")
        or logical.get("idempotency_key") != key
    ):
        raise RecoveryResponseError("call checkpoint does not match recovery submission")

    # Stateful v5 identities bind native reconciliation to the controller's
    # complete five-field authorization.  Verify both the logical-identity
    # copy and the independently persisted checkpoint copy.  Stateless calls
    # intentionally have no native authorization and retain null session and
    # generation fields in their checkpoint identity.
    if receipt.get("checkpoint_session_key") is not None:
        ledger_path = _safe_relative_path(root, receipt.get("ledger_path"))
        expected_authorization = {
            "control_address": str(ledger_path.resolve(strict=False)),
            "session_key": receipt.get("session_key"),
            "logical_unit": receipt.get("logical_unit"),
            "generation": receipt.get("generation"),
            "idempotency_key": key,
        }
        if (
            logical.get("control_address")
            != expected_authorization["control_address"]
            or logical.get("logical_unit")
            != expected_authorization["logical_unit"]
            or logical.get("initial_native_authorization")
            != expected_authorization
            or checkpoint.get("initial_native_authorization")
            != expected_authorization
        ):
            raise RecoveryResponseError(
                "call checkpoint does not match recovery submission"
            )
    elif any(
        value is not None
        for value in (
            logical.get("control_address"),
            logical.get("logical_unit"),
            logical.get("initial_native_authorization"),
            checkpoint.get("initial_native_authorization"),
        )
    ):
        raise RecoveryResponseError("call checkpoint does not match recovery submission")

    return dict(logical)


def _validate_receipt_identity(receipt: Mapping[str, Any]) -> None:
    keys = (
        "session_key", "logical_unit", "generation", "idempotency_key",
        "schema_sha256", "validator", "application", "artifact_dir",
        "ledger_path", "side_effect_policy", "recovery_unit",
        "input_sha256", "prompt_sha256", "ordered_siblings", "suffix",
        "group_sha256",
        "acceptance_checkpoint", "checkpoint_session_key",
        "checkpoint_generation", "ledger_snapshot",
        "ledger_snapshot_sha256",
    )
    if (
        receipt.get("schema_version") != SUBMISSION_SCHEMA_VERSION
        or receipt.get("validator_version") != VALIDATOR_VERSION
        or receipt.get("side_effect_policy") != "no_unproven_external_side_effects"
        or receipt.get("external_side_effects") is not False
        or isinstance(receipt.get("generation"), bool)
        or not isinstance(receipt.get("generation"), int)
        or receipt["generation"] < 1
        or _LEDGER_HANDLERS.get(str(receipt.get("recovery_unit") or ""))
        != (receipt.get("validator"), receipt.get("application"))
        or not all(key in receipt for key in keys)
        or "registered_ledger_sha256_at_creation" not in receipt
        or not isinstance(receipt.get("ledger_snapshot"), Mapping)
        or receipt["ledger_snapshot"].get("schema_version")
        != LEDGER_SNAPSHOT_SCHEMA_VERSION
        or sha256_json(receipt["ledger_snapshot"])
        != receipt.get("ledger_snapshot_sha256")
        or not _valid_sha256(receipt.get("registered_ledger_sha256_at_creation"))
        or sha256_json({key: receipt[key] for key in keys})
        != receipt.get("identity_sha256")
        or not (
            (receipt.get("sealed") is False and receipt.get("attempt_records") == [])
            or (
                receipt.get("sealed") is True
                and receipt.get("seal_schema_version") == SUBMISSION_SEAL_SCHEMA_VERSION
                and isinstance(receipt.get("attempt_records"), list)
                and bool(receipt.get("attempt_records"))
            )
        )
    ):
        raise RecoveryResponseError("recovery submission identity is invalid")


def _index_submission(
    root: Path,
    path: Path,
    receipt: Mapping[str, Any],
    *,
    state: str,
    receipt_sha256: str,
    sidecar_path: Path | None = None,
    sidecar_sha256: str | None = None,
) -> None:
    if state not in {"reserved", "prepared", "sealed"}:
        raise RecoveryResponseError("recovery submission index state is invalid")
    index_path = root / "recovery-submissions" / "index.json"
    entry = {
        "path": path.relative_to(root).as_posix(),
        "identity_sha256": receipt["identity_sha256"],
        "receipt_sha256": receipt_sha256,
        "state": state,
        "sidecar_path": (
            sidecar_path.relative_to(root).as_posix() if sidecar_path else None
        ),
        "sidecar_sha256": sidecar_sha256,
    }
    with _submission_index_lock(root) as directory_fd:
        current: dict[str, Any] = {
            "schema_version": SUBMISSION_INDEX_SCHEMA_VERSION,
            "entries": [],
        }
        index_snapshot = _try_read_regular_snapshot_at(
            directory_fd, index_path.name, max_bytes=MAX_SUBMISSION_INDEX_BYTES,
        )
        if index_snapshot is not None:
            try:
                loaded = json.loads(index_snapshot.raw)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise RecoveryResponseError(
                    "recovery submission index is invalid"
                ) from exc
            if (
                not isinstance(loaded, dict)
                or set(loaded) != {"schema_version", "entries"}
                or loaded.get("schema_version") != current["schema_version"]
            ):
                raise RecoveryResponseError("recovery submission index is invalid")
            current = loaded
        entries = _validated_index_entries(current.get("entries"))
        matches = [item for item in entries if item.get("path") == entry["path"]]
        if matches and any(
            item.get("identity_sha256") != entry["identity_sha256"]
            or item.get("receipt_sha256") != entry["receipt_sha256"]
            for item in matches
        ):
            raise RecoveryResponseError("recovery submission index collision")
        if not matches and len(entries) >= MAX_SUBMISSION_INDEX_ENTRIES:
            raise RecoveryResponseError("recovery submission index entry limit exceeded")
        previous = matches[0] if matches else None
        ranks = {"reserved": 0, "prepared": 1, "sealed": 2}
        if previous and ranks.get(str(previous.get("state")), -1) > ranks[state]:
            entry = previous
        elif previous and previous.get("state") == "sealed" and entry != previous:
            raise RecoveryResponseError("sealed recovery submission index changed")
        entries = [item for item in entries if item.get("path") != entry["path"]]
        entries.append(dict(entry))
        if entries != current.get("entries"):
            entries.sort(key=lambda item: (str(item.get("path") or ""), str(item.get("identity_sha256") or "")))
            current["entries"] = entries
            _durable_write_json_at(
                directory_fd, index_path.name, current,
                expected=index_snapshot,
            )


def discover_submission_receipts(checkpoint_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    with _checkpoint_root_scope(checkpoint_dir) as root:
        return _discover_submission_receipts_held(root)


def _discover_submission_receipts_held(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Read the bounded active control index; never discover receipts by rglob."""
    index_path = root / "recovery-submissions" / "index.json"
    # Index publication uses atomic replacement.  Take its exact snapshot
    # under the same cross-process lock as writers so a legitimate concurrent
    # publication cannot trip the fail-closed leaf TOCTOU check.  The lock is
    # released before loading immutable receipts, which remain bound to this
    # snapshot by their recorded content hashes.
    try:
        with _open_directory_dirfd(
            root, Path("recovery-submissions"), create=False,
        ):
            pass
    except FileNotFoundError:
        return []
    with _submission_index_lock(root) as directory_fd:
        index_raw = _try_read_regular_at(
            directory_fd, index_path.name,
            max_bytes=MAX_SUBMISSION_INDEX_BYTES,
        )
    if index_raw is None:
        return []
    try:
        index = json.loads(index_raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryResponseError("recovery submission index is invalid") from exc
    if (
        not isinstance(index, dict)
        or index.get("schema_version") != SUBMISSION_INDEX_SCHEMA_VERSION
        or set(index) != {"schema_version", "entries"}
    ):
        raise RecoveryResponseError("recovery submission index is invalid")
    entries = _validated_index_entries(index.get("entries"))
    output: list[tuple[Path, dict[str, Any]]] = []
    for item in entries:
        path = _safe_relative_path(root, item.get("path"))
        receipt_raw = _try_read_immutable_path(root, path)
        if receipt_raw is None:
            if item.get("state") == "reserved":
                continue
            raise RecoveryResponseError("indexed recovery submission is missing")
        receipt = _load_indexed_submission(
            root, path, expected_entry=item, receipt_raw=receipt_raw,
        )
        if not isinstance(receipt, dict):
            raise RecoveryResponseError("recovery submission receipt is invalid")
        # Prepared receipts remain discoverable, but only the submitting call
        # may seal one by supplying its explicit returned attempt refs. An
        # index reader must never glob a shared artifact directory and guess.
        _validate_receipt_identity(receipt)
        if receipt.get("identity_sha256") != item.get("identity_sha256"):
            raise RecoveryResponseError("recovery submission index identity changed")
        output.append((path, receipt))
    return output


def _validated_index_entries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_SUBMISSION_INDEX_ENTRIES:
        raise RecoveryResponseError("recovery submission index is invalid")
    entries: list[dict[str, Any]] = []
    paths: set[str] = set()
    fields = {
        "path", "identity_sha256", "receipt_sha256", "state",
        "sidecar_path", "sidecar_sha256",
    }
    for supplied in value:
        if not isinstance(supplied, Mapping) or set(supplied) != fields:
            raise RecoveryResponseError("recovery submission index entry is invalid")
        item = dict(supplied)
        try:
            receipt_path = safe_relative_path(item["path"], suffixes=(".json",))
        except SecureReadError as exc:
            raise RecoveryResponseError(
                "recovery submission index entry is invalid"
            ) from exc
        if receipt_path.name.endswith(".sealed.json"):
            raise RecoveryResponseError("recovery submission index entry is invalid")
        if item["path"] in paths:
            raise RecoveryResponseError("recovery submission index has duplicate paths")
        paths.add(item["path"])
        for key in ("identity_sha256", "receipt_sha256"):
            digest = item.get(key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise RecoveryResponseError(
                    "recovery submission index entry is invalid"
                )
        state = item.get("state")
        if state in {"reserved", "prepared"}:
            if item.get("sidecar_path") is not None or item.get("sidecar_sha256") is not None:
                raise RecoveryResponseError(
                    "recovery submission index entry is invalid"
                )
        elif state == "sealed":
            try:
                sidecar_path = safe_relative_path(
                    item.get("sidecar_path"), suffixes=(".sealed.json",),
                )
            except SecureReadError as exc:
                raise RecoveryResponseError(
                    "recovery submission index entry is invalid"
                ) from exc
            digest = item.get("sidecar_sha256")
            if (
                sidecar_path.name != f"{receipt_path.stem}.sealed.json"
                or sidecar_path.parent != receipt_path.parent
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise RecoveryResponseError(
                    "recovery submission index entry is invalid"
                )
        else:
            raise RecoveryResponseError("recovery submission index entry is invalid")
        entries.append(item)
    return entries


def resolve_recovery_path(checkpoint_dir: Path, value: Any) -> Path:
    """Resolve one receipt-owned path while rejecting every symlink component."""

    return _safe_relative_path(checkpoint_dir.resolve(strict=False), value)


def _durable_write_json_at(
    directory_fd: int,
    name: str,
    value: Mapping[str, Any],
    *,
    expected: _LeafSnapshot | None,
) -> None:
    _verify_active_submission_lock(directory_fd)
    _verify_held_root_binding()
    encoded = _json_bytes(value)
    if len(encoded) > MAX_SUBMISSION_INDEX_BYTES:
        raise RecoveryResponseError("recovery submission index exceeds its byte limit")
    temporary = _temporary_name(name)
    _cleanup_exact_index_stage(directory_fd, temporary)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
    exchanged = False
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            _recovery_write_fault("index:after_file_write")
            os.fsync(handle.fileno())
            _recovery_write_fault("index:after_file_fsync")
            written_stat = os.fstat(handle.fileno())
        # The locked read and this replace are one compare-and-swap.  A regular
        # attacker-controlled replacement must not be overwritten merely
        # because it still has a safe file type.
        _require_leaf_snapshot_at(
            directory_fd, name, expected=expected,
            max_bytes=MAX_SUBMISSION_INDEX_BYTES,
        )
        _verify_active_submission_lock(directory_fd)
        _verify_held_root_binding()
        if expected is None:
            os.link(
                temporary,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            linked = True
        else:
            _exchange_entries_at(directory_fd, temporary, name)
            exchanged = True
        _recovery_write_fault("index:after_replace")
        if linked:
            _verify_leaf_inode_at(directory_fd, name, written_stat, nlink=2)
        else:
            _verify_leaf_binding_at(directory_fd, name, written_stat)
        os.fsync(directory_fd)
        _recovery_write_fault("index:after_directory_fsync")
        if linked:
            _verify_leaf_inode_at(directory_fd, name, written_stat, nlink=2)
        else:
            _verify_leaf_binding_at(directory_fd, name, written_stat)
        os.unlink(temporary, dir_fd=directory_fd)
        os.fsync(directory_fd)
        _verify_leaf_binding_at(directory_fd, name, written_stat)
        _verify_active_submission_lock(directory_fd)
        _verify_held_root_binding()
    except BaseException:
        if exchanged:
            try:
                _verify_active_submission_lock(directory_fd)
                _exchange_entries_at(directory_fd, temporary, name)
                os.fsync(directory_fd)
                _require_leaf_snapshot_at(
                    directory_fd,
                    name,
                    expected=expected,
                    max_bytes=MAX_SUBMISSION_INDEX_BYTES,
                )
            except BaseException as rollback_error:
                raise RecoveryResponseError(
                    "recovery submission index exchange rollback failed"
                ) from rollback_error
        if linked:
            try:
                _verify_active_submission_lock(directory_fd)
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                staged = os.stat(
                    temporary, dir_fd=directory_fd, follow_symlinks=False,
                )
                if (current.st_dev, current.st_ino) == (
                    staged.st_dev, staged.st_ino,
                ):
                    os.unlink(name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
            except FileNotFoundError:
                pass
        try:
            _verify_active_submission_lock(directory_fd)
            _cleanup_exact_index_stage(directory_fd, temporary)
        except FileNotFoundError:
            pass
        raise


def _exchange_entries_at(directory_fd: int, left: str, right: str) -> None:
    """Atomically exchange two directory entries using Linux renameat2."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:  # pragma: no cover - non-Linux safety fallback
        raise RecoveryResponseError(
            "atomic directory-entry exchange is unavailable"
        )
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        directory_fd,
        os.fsencode(left),
        directory_fd,
        os.fsencode(right),
        2,  # RENAME_EXCHANGE
    ) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), f"{left}<->{right}")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    ).encode("utf-8")


def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


@contextmanager
def _checkpoint_root_scope(checkpoint_dir: Path) -> Iterator[Path]:
    """Hold one checkpoint-root inode for a complete receipt transition."""

    root = checkpoint_dir.resolve(strict=False)
    flags = _directory_open_flags()
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise RecoveryResponseError("recovery checkpoint root is unsafe") from exc
    previous = getattr(_HELD_ROOT, "value", None)
    root_stat = os.fstat(descriptor)
    held = (root, descriptor, (root_stat.st_dev, root_stat.st_ino))
    _HELD_ROOT.value = held
    try:
        _verify_held_root_binding()
        yield root
        _verify_held_root_binding()
    finally:
        try:
            _verify_held_root_binding()
        finally:
            _HELD_ROOT.value = previous
            os.close(descriptor)


def _verify_held_root_binding() -> None:
    held = getattr(_HELD_ROOT, "value", None)
    if held is None:
        return
    root, _descriptor, identity = held
    try:
        current_fd = os.open(root, _directory_open_flags())
    except OSError as exc:
        raise RecoveryResponseError("recovery checkpoint root binding changed") from exc
    try:
        current = os.fstat(current_fd)
        if (current.st_dev, current.st_ino) != identity:
            raise RecoveryResponseError("recovery checkpoint root binding changed")
    finally:
        os.close(current_fd)


def _relative_control_path(root: Path, path: Path) -> Path:
    lexical = path if path.is_absolute() else Path.cwd() / path
    try:
        relative = lexical.relative_to(root)
        return safe_relative_path(relative.as_posix())
    except (ValueError, SecureReadError) as exc:
        raise RecoveryResponseError(
            "recovery control path is outside the active checkpoint"
        ) from exc


@contextmanager
def _open_directory_dirfd(
    root: Path, relative: Path, *, create: bool,
) -> Iterator[int]:
    flags = _directory_open_flags()
    held = getattr(_HELD_ROOT, "value", None)
    if held is not None and held[0] == root:
        _verify_held_root_binding()
        descriptor = os.dup(held[1])
    else:
        descriptor = os.open(root, flags)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise RecoveryResponseError("recovery control root is not a directory")
        for part in relative.parts:
            if part in {"", ".", ".."}:
                raise RecoveryResponseError("recovery control directory is unsafe")
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    _verify_held_root_binding()
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    # A concurrent writer may have created the same parent.
                    # The no-follow open below still validates its type.
                    pass
                os.fsync(descriptor)
                _verify_held_root_binding()
                child = os.open(part, flags, dir_fd=descriptor)
            child_stat = os.fstat(child)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child)
                raise RecoveryResponseError(
                    "recovery control path component is not a directory"
                )
            os.close(descriptor)
            descriptor = child
        yield descriptor
    except (RecoveryResponseError, FileNotFoundError):
        raise
    except OSError as exc:
        raise RecoveryResponseError(
            "recovery control directory is unsafe or unavailable"
        ) from exc
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


@contextmanager
def _open_parent_dirfd(
    root: Path, path: Path, *, create: bool,
) -> Iterator[tuple[int, str]]:
    relative = _relative_control_path(root, path)
    parent = Path(*relative.parts[:-1])
    with _open_directory_dirfd(root, parent, create=create) as directory_fd:
        yield directory_fd, relative.parts[-1]


def _secure_ensure_directory(root: Path, path: Path) -> Path:
    relative = _relative_control_path(root, path / ".control-placeholder").parent
    with _open_directory_dirfd(root, relative, create=True):
        pass
    return root / relative


def _try_stat_regular_at(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        result = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(result.st_mode):
        raise RecoveryResponseError("recovery control leaf is not a regular file")
    if result.st_nlink != 1:
        raise RecoveryResponseError("recovery control leaf must have exactly one link")
    return result


def _verify_leaf_binding_at(
    directory_fd: int, name: str, expected: os.stat_result,
) -> None:
    current = _try_stat_regular_at(directory_fd, name)
    if current is None or (
        current.st_dev, current.st_ino, current.st_mode, current.st_nlink,
    ) != (
        expected.st_dev, expected.st_ino, expected.st_mode, expected.st_nlink,
    ):
        raise RecoveryResponseError("recovery control leaf binding changed")


def _verify_leaf_inode_at(
    directory_fd: int,
    name: str,
    expected: os.stat_result,
    *,
    nlink: int,
) -> None:
    try:
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise RecoveryResponseError("recovery control leaf binding changed") from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_nlink != nlink
        or (current.st_dev, current.st_ino, current.st_mode)
        != (expected.st_dev, expected.st_ino, expected.st_mode)
    ):
        raise RecoveryResponseError("recovery control leaf binding changed")


def _verify_parent_binding(root: Path, path: Path, expected_fd: int) -> None:
    expected = os.fstat(expected_fd)
    with _open_parent_dirfd(root, path, create=False) as (current_fd, _name):
        current = os.fstat(current_fd)
    if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
        raise RecoveryResponseError("recovery control parent binding changed")


def _leaf_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
        value.st_size, value.st_mtime_ns,
    )


def _read_regular_snapshot_at(
    directory_fd: int, name: str, *, max_bytes: int,
) -> _LeafSnapshot:
    flags = os.O_RDONLY
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise RecoveryResponseError(
            "recovery control leaf is unsafe or unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RecoveryResponseError(
                "recovery control leaf is not a singly-linked regular file"
            )
        _recovery_write_fault("immutable_read:after_open")
        named_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(named_before.st_mode)
            or named_before.st_nlink != 1
            or (named_before.st_dev, named_before.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise RecoveryResponseError("recovery control named identity changed")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            raise RecoveryResponseError("recovery control exceeds its byte limit")
        after = os.fstat(descriptor)
        _recovery_write_fault("immutable_read:after_read")
        named_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        before_identity = _leaf_identity(before)
        after_identity = _leaf_identity(after)
        if (
            before_identity != after_identity
            or after.st_size != len(raw)
            or not stat.S_ISREG(named_after.st_mode)
            or named_after.st_nlink != 1
            or (named_after.st_dev, named_after.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise RecoveryResponseError("recovery control changed while reading")
        return _LeafSnapshot(raw=raw, identity=after_identity)
    finally:
        os.close(descriptor)


def _read_regular_at(directory_fd: int, name: str, *, max_bytes: int) -> bytes:
    return _read_regular_snapshot_at(
        directory_fd, name, max_bytes=max_bytes,
    ).raw


def _try_read_regular_snapshot_at(
    directory_fd: int, name: str, *, max_bytes: int,
) -> _LeafSnapshot | None:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return _read_regular_snapshot_at(directory_fd, name, max_bytes=max_bytes)


def _try_read_regular_at(
    directory_fd: int, name: str, *, max_bytes: int,
) -> bytes | None:
    snapshot = _try_read_regular_snapshot_at(
        directory_fd, name, max_bytes=max_bytes,
    )
    return snapshot.raw if snapshot is not None else None


def _require_leaf_snapshot_at(
    directory_fd: int,
    name: str,
    *,
    expected: _LeafSnapshot | None,
    max_bytes: int,
) -> None:
    current = _try_read_regular_snapshot_at(
        directory_fd, name, max_bytes=max_bytes,
    )
    if expected is None:
        if current is not None:
            raise RecoveryResponseError(
                "recovery submission index compare-and-swap failed"
            )
        return
    if (
        current is None
        or current.identity != expected.identity
        or current.raw != expected.raw
    ):
        raise RecoveryResponseError(
            "recovery submission index compare-and-swap failed"
        )


def _try_read_path_bytes(
    root: Path,
    path: Path,
    *,
    suffixes: tuple[str, ...],
    max_bytes: int = MAX_CONTROL_JSON_BYTES,
) -> bytes | None:
    relative = _relative_control_path(root, path)
    if suffixes and not any(relative.name.endswith(item) for item in suffixes):
        raise RecoveryResponseError("recovery control path has an unexpected suffix")
    try:
        with _open_parent_dirfd(root, path, create=False) as (directory_fd, name):
            return _try_read_regular_at(
                directory_fd, name, max_bytes=max_bytes,
            )
    except FileNotFoundError:
        return None


def _temporary_name(name: str) -> str:
    return f".{name}.arc-stage"


def _cleanup_exact_index_stage(directory_fd: int, name: str) -> None:
    _verify_active_submission_lock(directory_fd)
    try:
        value = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(value.st_mode):
        os.unlink(name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        return
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise RecoveryResponseError("recovery submission index stage is unsafe")
    os.unlink(name, dir_fd=directory_fd)
    os.fsync(directory_fd)


def _exclusive_write(
    root: Path, path: Path, payload: bytes, *, kind: str,
) -> None:
    # Serialize immutable publication with index transitions across processes.
    # This keeps an equal concurrent writer from observing a partially written
    # O_EXCL leaf while retaining immutable collision semantics.
    with _submission_index_lock(root):
        try:
            _exclusive_write_locked(root, path, payload, kind=kind)
        except BaseException:
            stage = f".{path.name}.{hashlib.sha256(payload).hexdigest()[:24]}.staged"
            try:
                _verify_active_submission_lock()
                with _open_parent_dirfd(root, path, create=False) as (directory_fd, _name):
                    value = os.stat(stage, dir_fd=directory_fd, follow_symlinks=False)
                    if stat.S_ISLNK(value.st_mode) or (
                        stat.S_ISREG(value.st_mode) and value.st_nlink == 1
                    ):
                        os.unlink(stage, dir_fd=directory_fd)
                        os.fsync(directory_fd)
            except (FileNotFoundError, OSError, RecoveryResponseError):
                pass
            raise


def _exclusive_write_locked(
    root: Path, path: Path, payload: bytes, *, kind: str,
) -> None:
    with _open_parent_dirfd(root, path, create=True) as (directory_fd, name):
        _verify_active_submission_lock()
        _verify_held_root_binding()
        stage = f".{name}.{hashlib.sha256(payload).hexdigest()[:24]}.staged"
        _reconcile_immutable_stage(directory_fd, name, stage, payload)
        existing = _try_read_regular_allow_link_at(
            directory_fd, name, max_bytes=MAX_CONTROL_JSON_BYTES,
        )
        if existing is not None:
            if existing != payload:
                raise RecoveryResponseError("immutable recovery control collision")
            return
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(stage, flags, 0o400, dir_fd=directory_fd)
        except FileExistsError:
            _reconcile_immutable_stage(directory_fd, name, stage, payload)
            descriptor = -1
        if descriptor >= 0:
            created_stat = os.fstat(descriptor)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    _recovery_write_fault(f"{kind}:after_file_write")
                    os.fsync(handle.fileno())
                    _recovery_write_fault(f"{kind}:after_file_fsync")
                os.fsync(directory_fd)
            except BaseException:
                try:
                    _verify_active_submission_lock()
                    current = os.stat(stage, dir_fd=directory_fd, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) == (
                        created_stat.st_dev, created_stat.st_ino,
                    ):
                        os.unlink(stage, dir_fd=directory_fd)
                        os.fsync(directory_fd)
                except FileNotFoundError:
                    pass
                raise
            _verify_stage_binding_at(directory_fd, stage, created_stat)
        _verify_parent_binding(root, path, directory_fd)
        _verify_active_submission_lock()
        _verify_held_root_binding()
        if _try_stat_regular_at(directory_fd, name) is not None:
            existing = _read_regular_at(directory_fd, name, max_bytes=MAX_CONTROL_JSON_BYTES)
            if existing != payload:
                raise RecoveryResponseError("immutable recovery control collision")
            _reconcile_immutable_stage(directory_fd, name, stage, payload)
            return
        try:
            os.link(
                stage,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            existing = _try_read_regular_allow_link_at(
                directory_fd, name, max_bytes=MAX_CONTROL_JSON_BYTES,
            )
            if existing != payload:
                raise RecoveryResponseError("immutable recovery control collision")
        _recovery_write_fault(f"{kind}:after_publish")
        os.fsync(directory_fd)
        _reconcile_immutable_stage(directory_fd, name, stage, payload)
        _recovery_write_fault(f"{kind}:after_directory_fsync")
        _verify_active_submission_lock()
        _verify_held_root_binding()
        _verify_parent_binding(root, path, directory_fd)


def _verify_stage_binding_at(
    directory_fd: int, name: str, expected: os.stat_result,
) -> None:
    try:
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise RecoveryResponseError("immutable recovery stage binding changed") from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino)
    ):
        raise RecoveryResponseError("immutable recovery stage leaf binding changed")


def _try_read_regular_allow_link_at(
    directory_fd: int, name: str, *, max_bytes: int,
) -> bytes | None:
    try:
        result = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(result.st_mode) or result.st_nlink not in {1, 2}:
        raise RecoveryResponseError("recovery control leaf has unsafe link count")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        _recovery_write_fault("immutable_read:after_open")
        named_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(named_before.st_mode)
            or named_before.st_nlink not in {1, 2}
            or (named_before.st_dev, named_before.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise RecoveryResponseError("recovery control named identity changed")
        raw = b""
        while len(raw) <= max_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - len(raw)))
            if not chunk:
                break
            raw += chunk
        if len(raw) > max_bytes:
            raise RecoveryResponseError("recovery control exceeds its byte limit")
        after = os.fstat(descriptor)
        _recovery_write_fault("immutable_read:after_read")
        named_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or after.st_size != len(raw)
            or not stat.S_ISREG(named_after.st_mode)
            or named_after.st_nlink not in {1, 2}
            or (named_after.st_dev, named_after.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise RecoveryResponseError("recovery control changed while reading")
        return raw
    finally:
        os.close(descriptor)


def _reconcile_immutable_stage(
    directory_fd: int, name: str, stage: str, payload: bytes,
) -> None:
    """Repair only the exact content-addressed stage owned by this leaf."""

    try:
        staged_stat = os.stat(stage, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(staged_stat.st_mode) or staged_stat.st_nlink not in {1, 2}:
        raise RecoveryResponseError("immutable recovery stage is unsafe")
    staged = _try_read_regular_allow_link_at(
        directory_fd, stage, max_bytes=MAX_CONTROL_JSON_BYTES,
    )
    final = _try_read_regular_allow_link_at(
        directory_fd, name, max_bytes=MAX_CONTROL_JSON_BYTES,
    )
    if staged != payload:
        # A killed short write is not a published control and is safe to remove.
        if final is None and staged_stat.st_nlink == 1:
            _verify_active_submission_lock()
            os.unlink(stage, dir_fd=directory_fd)
            os.fsync(directory_fd)
            return
        raise RecoveryResponseError("immutable recovery stage collision")
    if final is not None:
        if final != payload:
            raise RecoveryResponseError("immutable recovery control collision")
        final_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (final_stat.st_dev, final_stat.st_ino) != (staged_stat.st_dev, staged_stat.st_ino):
            raise RecoveryResponseError("immutable recovery stage binding changed")
        _verify_active_submission_lock()
        os.unlink(stage, dir_fd=directory_fd)
        os.fsync(directory_fd)


def _read_path_bytes(
    root: Path,
    path: Path,
    *,
    suffixes: tuple[str, ...],
    max_bytes: int = MAX_CONTROL_JSON_BYTES,
) -> bytes:
    try:
        relative = path.relative_to(root)
        return read_bounded_file(
            root, relative, max_bytes=max_bytes, suffixes=suffixes,
        )
    except (ValueError, SecureReadError) as exc:
        raise RecoveryResponseError("recovery control path is unsafe or invalid") from exc


def _try_read_immutable_path(
    root: Path, path: Path, *, max_bytes: int = MAX_CONTROL_JSON_BYTES,
) -> bytes | None:
    """Read and reconcile only this leaf's exact content-addressed stage."""

    with _submission_index_lock(root):
        try:
            with _open_parent_dirfd(root, path, create=False) as (directory_fd, name):
                raw = _try_read_regular_allow_link_at(
                    directory_fd, name, max_bytes=max_bytes,
                )
                if raw is None:
                    return None
                stage = f".{name}.{hashlib.sha256(raw).hexdigest()[:24]}.staged"
                _reconcile_immutable_stage(directory_fd, name, stage, raw)
                final = _read_regular_at(
                    directory_fd, name, max_bytes=max_bytes,
                )
                _verify_parent_binding(root, path, directory_fd)
                return final
        except FileNotFoundError:
            return None


def _read_json_path(
    root: Path,
    path: Path,
    *,
    suffixes: tuple[str, ...],
    max_bytes: int = MAX_CONTROL_JSON_BYTES,
) -> Any:
    try:
        relative = path.relative_to(root)
        return read_bounded_json(
            root, relative, max_bytes=max_bytes, suffixes=suffixes,
        )
    except (ValueError, SecureReadError) as exc:
        raise RecoveryResponseError("recovery control JSON is unsafe or invalid") from exc


def _seal_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(f"{receipt_path.stem}.sealed.json")


def _load_indexed_submission(
    root: Path,
    receipt_path: Path,
    *,
    expected_entry: Mapping[str, Any] | None = None,
    receipt_raw: bytes | None = None,
) -> dict[str, Any]:
    held = getattr(_HELD_ROOT, "value", None)
    if held is None:
        with _checkpoint_root_scope(root) as held_root:
            return _load_indexed_submission(
                held_root,
                receipt_path,
                expected_entry=expected_entry,
                receipt_raw=receipt_raw,
            )
    if held[0] != root:
        raise RecoveryResponseError("recovery checkpoint root binding changed")
    if expected_entry is None:
        index_path = root / "recovery-submissions" / "index.json"
        index = _read_json_path(
            root, index_path, suffixes=(".json",), max_bytes=MAX_SUBMISSION_INDEX_BYTES,
        )
        matches = [
            item for item in index.get("entries", [])
            if isinstance(item, Mapping)
            and item.get("path") == receipt_path.relative_to(root).as_posix()
        ] if isinstance(index, Mapping) else []
        if len(matches) != 1:
            raise RecoveryResponseError("recovery submission is not exactly indexed")
        expected_entry = matches[0]
    raw = receipt_raw
    if raw is None:
        raw = _read_path_bytes(root, receipt_path, suffixes=(".json",))
    if hashlib.sha256(raw).hexdigest() != expected_entry.get("receipt_sha256"):
        raise RecoveryResponseError("recovery submission receipt hash changed")
    try:
        receipt = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryResponseError("recovery submission receipt is invalid") from exc
    if not isinstance(receipt, dict):
        raise RecoveryResponseError("recovery submission receipt is invalid")
    _validate_receipt_identity(receipt)
    if expected_entry.get("identity_sha256") != receipt.get("identity_sha256"):
        raise RecoveryResponseError("recovery submission index identity changed")
    state = expected_entry.get("state")
    if state == "prepared":
        # A crash may leave an immutable, fully fsynced sidecar after the
        # prepared index but before the sealed index transition. Reconcile only
        # this receipt's exact deterministic sidecar; never scan the artifact.
        sidecar_path = _seal_path(receipt_path)
        sidecar_raw = _try_read_immutable_path(
            root, sidecar_path, max_bytes=MAX_SUBMISSION_SEAL_BYTES,
        )
        if sidecar_raw is not None:
            sidecar = _validated_seal_sidecar(
                root, receipt_path, receipt, sidecar_path, sidecar_raw,
            )
            _index_submission(
                root,
                receipt_path,
                receipt,
                state="sealed",
                receipt_sha256=expected_entry["receipt_sha256"],
                sidecar_path=sidecar_path,
                sidecar_sha256=hashlib.sha256(sidecar_raw).hexdigest(),
            )
            return {**receipt, **sidecar}
        return receipt
    if state == "reserved":
        _index_submission(
            root,
            receipt_path,
            receipt,
            state="prepared",
            receipt_sha256=expected_entry["receipt_sha256"],
        )
        return receipt
    if state != "sealed":
        raise RecoveryResponseError("recovery submission index state is invalid")
    sidecar_path = _safe_relative_path(root, expected_entry.get("sidecar_path"))
    if sidecar_path != _seal_path(receipt_path):
        raise RecoveryResponseError("recovery submission sidecar address changed")
    sidecar_raw = _try_read_immutable_path(
        root, sidecar_path, max_bytes=MAX_SUBMISSION_SEAL_BYTES,
    )
    if sidecar_raw is None:
        raise RecoveryResponseError("recovery submission sidecar is missing")
    if hashlib.sha256(sidecar_raw).hexdigest() != expected_entry.get("sidecar_sha256"):
        raise RecoveryResponseError("recovery submission sidecar hash changed")
    sidecar = _validated_seal_sidecar(
        root, receipt_path, receipt, sidecar_path, sidecar_raw,
    )
    return {**receipt, **sidecar}


def _validated_seal_sidecar(
    root: Path,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    sidecar_path: Path,
    sidecar_raw: bytes,
) -> dict[str, Any]:
    try:
        sidecar = json.loads(sidecar_raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryResponseError("recovery submission sidecar is invalid") from exc
    attempts = sidecar.get("attempt_records") if isinstance(sidecar, Mapping) else None
    evidence = sidecar.get("attempt_evidence") if isinstance(sidecar, Mapping) else None
    if (
        not isinstance(sidecar, dict)
        or sidecar.get("seal_schema_version") != SUBMISSION_SEAL_SCHEMA_VERSION
        or sidecar.get("receipt_path") != receipt_path.relative_to(root).as_posix()
        or sidecar.get("receipt_sha256")
        != hashlib.sha256(_json_bytes(receipt)).hexdigest()
        or sidecar.get("identity_sha256") != receipt.get("identity_sha256")
        or sidecar.get("sealed") is not True
        or not isinstance(attempts, list)
        or not attempts
        or len(attempts) > MAX_ATTEMPT_RECORDS
        or not isinstance(evidence, list)
        or len(evidence) != len(attempts)
        or sidecar.get("attempt_evidence_sha256") != sha256_json(evidence)
        or sidecar_path != _seal_path(receipt_path)
    ):
        raise RecoveryResponseError("recovery submission sidecar binding is invalid")
    return sidecar


def _verify_active_submission_lock(directory_fd: int | None = None) -> None:
    active = getattr(_ACTIVE_SUBMISSION_LOCK, "value", None)
    if active is None:
        raise RecoveryResponseError("recovery submission mutation is not locked")
    active_directory_fd, descriptor, identity = active
    if directory_fd is not None:
        left = os.fstat(directory_fd)
        right = os.fstat(active_directory_fd)
        if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
            raise RecoveryResponseError("recovery submission lock directory changed")
    try:
        named = os.stat(
            ".index.lock", dir_fd=active_directory_fd, follow_symlinks=False,
        )
    except OSError as exc:
        raise RecoveryResponseError("recovery submission index lock changed") from exc
    if (
        not stat.S_ISREG(named.st_mode)
        or named.st_nlink != 1
        or (named.st_dev, named.st_ino) != identity
        or (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino) != identity
    ):
        raise RecoveryResponseError("recovery submission index lock changed")


@contextmanager
def _submission_index_lock(root: Path) -> Iterator[int]:
    directory = _secure_ensure_directory(root, root / "recovery-submissions")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    with _SUBMISSION_INDEX_LOCK:
        with _open_directory_dirfd(
            root, directory.relative_to(root), create=False,
        ) as directory_fd:
            descriptor = os.open(".index.lock", flags, 0o600, dir_fd=directory_fd)
            try:
                lock_stat = os.fstat(descriptor)
                if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
                    raise RecoveryResponseError(
                        "recovery submission index lock is unsafe"
                    )
                os.fsync(directory_fd)
                if os.name == "nt":  # pragma: no cover
                    import msvcrt
                    msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
                else:
                    import fcntl
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                named = os.stat(
                    ".index.lock", dir_fd=directory_fd, follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(named.st_mode)
                    or named.st_nlink != 1
                    or (named.st_dev, named.st_ino)
                    != (lock_stat.st_dev, lock_stat.st_ino)
                ):
                    raise RecoveryResponseError(
                        "recovery submission index lock changed"
                    )
                previous = getattr(_ACTIVE_SUBMISSION_LOCK, "value", None)
                _ACTIVE_SUBMISSION_LOCK.value = (
                    directory_fd,
                    descriptor,
                    (lock_stat.st_dev, lock_stat.st_ino),
                )
                _verify_active_submission_lock(directory_fd)
                try:
                    yield directory_fd
                    _verify_active_submission_lock(directory_fd)
                finally:
                    try:
                        _verify_active_submission_lock(directory_fd)
                    finally:
                        _ACTIVE_SUBMISSION_LOCK.value = previous
                _verify_parent_binding(
                    root, directory / "index.json", directory_fd,
                )
            finally:
                if os.name == "nt":  # pragma: no cover
                    import msvcrt
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
