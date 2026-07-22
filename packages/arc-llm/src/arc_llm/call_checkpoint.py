from __future__ import annotations

import json
import hashlib
import ctypes
import errno
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, NamedTuple

from .schema_cache import canonical_json, sha256_text
from .secure_io import SecureReadError, read_bounded_file
from .usage import LLMProviderResponse, LLMUsage, ResponseCandidateMaterial
from .providers.base import (
    LLMAbortScope,
    LLMFailureCategory,
    LLMSubmissionState,
    LLMWorkerError,
    LLMWorkerCancelled,
    LLMWorkerTimeout,
    failure_disposition,
)


SCHEMA_VERSION = "arc.llm.call_checkpoint.v5"
MAX_CALL_CHECKPOINT_BYTES = 16 * 1024 * 1024


def _before_checkpoint_replace(_path: Path) -> None:
    """Fault-injection cutpoint after staging and before address revalidation."""


def _before_checkpoint_exchange(_path: Path) -> None:
    """Fault-injection cutpoint in the final check-to-publication window."""


class SupervisedNativeResumeAuthorization(NamedTuple):
    """Exact controller ownership required to reconcile one native turn."""

    control_address: str
    session_key: str
    logical_unit: str
    generation: int
    idempotency_key: str

    def to_json(self) -> dict[str, Any]:
        return {
            "control_address": self.control_address,
            "session_key": self.session_key,
            "logical_unit": self.logical_unit,
            "generation": self.generation,
            "idempotency_key": self.idempotency_key,
        }


def normalize_supervised_native_resume_authorization(
    value: object,
) -> SupervisedNativeResumeAuthorization | None:
    """Reject boolean/partial authorization and return one canonical tuple."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, tuple) or len(value) != 5:
        raise ValueError(
            "supervised_native_resume must be a complete five-field authorization tuple or None"
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
        raise ValueError(
            "supervised_native_resume authorization fields must be complete and typed"
        )
    canonical_address = str(Path(control_address).expanduser().resolve(strict=False))
    if control_address != canonical_address:
        raise ValueError(
            "supervised_native_resume control address must be canonical and absolute"
        )
    return SupervisedNativeResumeAuthorization(
        control_address,
        session_key,
        logical_unit,
        generation,
        idempotency_key,
    )


class CallIdentity(str):
    """String-compatible checkpoint identity with structured components."""

    logical_identity: dict[str, Any]
    request_digest: str
    request_recipe: dict[str, Any]

    def __new__(
        cls,
        *,
        logical_identity: Mapping[str, Any],
        request_digest: str,
        request_recipe: Mapping[str, Any],
    ) -> "CallIdentity":
        logical = dict(logical_identity)
        recipe = dict(request_recipe)
        value = sha256_text(canonical_json({
            "logical_identity": logical,
            "request_digest": request_digest,
        }))
        instance = str.__new__(cls, value)
        instance.logical_identity = logical
        instance.request_digest = request_digest
        instance.request_recipe = recipe
        return instance


class LLMCallCheckpointError(LLMWorkerError):
    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            retryable=False,
            category=LLMFailureCategory.LOCAL_IO,
            abort_scope=LLMAbortScope.BATCH,
            submission_state=LLMSubmissionState.UNKNOWN,
        )


class LLMCallNeedsSupervision(LLMCallCheckpointError):
    def __init__(self, *, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        LLMWorkerError.__init__(
            self,
            "Submitted LLM call has no recorded response and needs explicit supervision",
            retryable=False,
            category=LLMFailureCategory.TIMEOUT,
            abort_scope=LLMAbortScope.BATCH,
            submission_state=LLMSubmissionState.UNKNOWN,
        )


# Backward-compatible import name. The call is no longer retried after a delay.
LLMCallRetryDeferred = LLMCallNeedsSupervision


class LLMCallRetryExhausted(LLMCallCheckpointError):
    def __init__(self, message: str, *, checkpoint_path: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        super().__init__(message)


@dataclass
class _CheckpointLock:
    handle: BinaryIO
    parent_fd: int
    parent_path: Path
    parent_identity: tuple[int, int, int]
    checkpoint_name: str
    lock_name: str
    lock_identity: tuple[int, int, int, int]


@dataclass
class PreparedCall:
    path: Path
    identity: str
    attempt: int
    recomputation_binding: dict[str, Any]
    replay_response: LLMProviderResponse[Any] | None = None
    replayed: bool = False
    _lock_handle: _CheckpointLock | None = None

    @property
    def owns_lock(self) -> bool:
        return self._lock_handle is not None

    def release_lock(self) -> None:
        handle, self._lock_handle = self._lock_handle, None
        if handle is None:
            return
        _release_handle(handle)


def checkpoint_path(
    artifact_dir: Path,
    *,
    prompt: str,
    schema: Mapping[str, Any] | None,
    provider: str,
    model: str | None,
    call_label: str | None,
    session_policy: str = "stateless",
    session_key: str | None = None,
    session_turn: int | None = None,
    runtime_fingerprint: str | None = None,
    idempotency_key: str | None = None,
    generation: int | None = None,
    progress_contract_scope: str | None = None,
    initial_native_authorization: (
        SupervisedNativeResumeAuthorization | tuple[Any, ...] | None
    ) = None,
) -> tuple[Path, str]:
    initial_authorization = normalize_supervised_native_resume_authorization(
        initial_native_authorization
    )
    if initial_authorization is not None and (
        initial_authorization.session_key != session_key
        or initial_authorization.generation != generation
        or initial_authorization.idempotency_key != idempotency_key
    ):
        raise ValueError(
            "initial_native_authorization does not match session/generation/idempotency identity"
        )
    logical_identity = {
        "provider": provider,
        "model": model,
        "session_key": session_key,
        "generation": generation,
        "idempotency_key": idempotency_key,
    }
    if initial_authorization is not None:
        logical_identity.update({
            "control_address": initial_authorization.control_address,
            "logical_unit": initial_authorization.logical_unit,
            "initial_native_authorization": initial_authorization.to_json(),
        })
    request_recipe = {
        "prompt_sha256": sha256_text(prompt),
        "schema_sha256": sha256_text(canonical_json(dict(schema))) if schema is not None else None,
        "call_label": call_label,
        "session_policy": session_policy,
        # A stable logical key must survive a crash after the session receipt
        # advanced the observable turn count.
        "session_turn": None if idempotency_key else session_turn,
        "runtime_fingerprint": runtime_fingerprint,
        "progress_contract_scope": progress_contract_scope,
    }
    request_digest = sha256_text(canonical_json(request_recipe))
    identity = CallIdentity(
        logical_identity=logical_identity,
        request_digest=request_digest,
        request_recipe=request_recipe,
    )
    safe_label = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(call_label or "call")
    )[:80]
    if idempotency_key:
        key_hash = sha256_text(idempotency_key)
        filename = f"idempotency-{key_hash}.json"
    else:
        filename = f"{safe_label}-{identity[:16]}.json"
    return artifact_dir / "call-checkpoints" / filename, identity


def prepare_call(
    path: Path,
    *,
    identity: str,
    now: float | None = None,
    retry_delay_seconds: int | None = None,
    deadline: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
    supervised_native_resume: (
        SupervisedNativeResumeAuthorization | tuple[Any, ...] | None
    ) = None,
    native_session_available: bool = False,
    runtime_capabilities: Mapping[str, Any] | None = None,
    validated_legacy_logical_identity: Mapping[str, Any] | None = None,
) -> PreparedCall:
    del retry_delay_seconds
    resume_authorization = normalize_supervised_native_resume_authorization(
        supervised_native_resume
    )
    current_time = time.time() if now is None else now
    lock_handle = _acquire_lock(path, deadline=deadline, cancel_check=cancel_check)
    try:
        existing = _read(path, lock=lock_handle)
        if existing is not None:
            if existing.get("schema_version") in {
                "arc.llm.call_checkpoint.v2",
                "arc.llm.call_checkpoint.v3",
                "arc.llm.call_checkpoint.v4",
            }:
                existing = _upgrade_legacy_checkpoint(existing)
                if validated_legacy_logical_identity is not None:
                    existing["logical_identity"] = dict(validated_legacy_logical_identity)
                    existing["initial_native_authorization"] = (
                        _initial_authorization_from_logical_identity(
                            validated_legacy_logical_identity
                        )
                    )
                    existing["legacy_logical_identity_validated"] = True
                _write(path, existing, lock=lock_handle)
            elif (
                validated_legacy_logical_identity is not None
                and existing.get("legacy_checkpoint_upgraded") is True
                and existing.get("logical_identity") is None
            ):
                existing["logical_identity"] = dict(validated_legacy_logical_identity)
                existing["initial_native_authorization"] = (
                    _initial_authorization_from_logical_identity(
                        validated_legacy_logical_identity
                    )
                )
                existing["legacy_logical_identity_validated"] = True
                _write(path, existing, lock=lock_handle)
            if existing.get("schema_version") != SCHEMA_VERSION:
                raise LLMCallCheckpointError(f"LLM call checkpoint identity mismatch: {path}")
            incoming = _identity_parts(identity)
            _validate_resume_authorization_binding(resume_authorization, incoming)
            _validate_persisted_resume_authorization(existing, resume_authorization)
            reconciliation_identity: str | None = None
            if existing.get("identity") != str(identity):
                if _can_rebuild_prepared(existing, incoming):
                    existing = _new_checkpoint_payload(
                        identity=identity, current_time=current_time,
                        runtime_capabilities=runtime_capabilities,
                    )
                    _write(path, existing, lock=lock_handle)
                elif not _can_resume_changed_request(
                    existing,
                    incoming=incoming,
                    supervised_native_resume=resume_authorization,
                    native_session_available=native_session_available,
                ):
                    raise LLMCallCheckpointError(
                        f"LLM call checkpoint identity mismatch: {path}"
                    )
                # Preserve the submitted request digest.  The current prompt is
                # used only to ask the original native session for reconciliation.
                reconciliation_identity = str(identity)
                identity = str(existing["identity"])
            state = existing.get("state")
            if state == "prepared" and resume_authorization is not None:
                raise LLMCallCheckpointError(
                    "supervised native resume cannot authorize an unsubmitted checkpoint"
                )
            if state in {"response_received", "validated"}:
                response = _response_from_json(existing.get("response"))
                _release_handle(lock_handle)
                return PreparedCall(
                    path=path,
                    identity=identity,
                    attempt=int(existing.get("attempt") or 1),
                    recomputation_binding=_checkpoint_recomputation_binding(
                        path, existing
                    ),
                    replay_response=response,
                    replayed=True,
                )
            if state == "prepared" and existing.get("submission_state") == "not_submitted":
                existing.update({"started_at": current_time, "updated_at": current_time})
                _write(path, existing, lock=lock_handle)
                return PreparedCall(
                    path=path,
                    identity=identity,
                    attempt=int(existing.get("attempt") or 1),
                    recomputation_binding=_checkpoint_recomputation_binding(
                        path, existing
                    ),
                    _lock_handle=lock_handle,
                )
            if state in {"submitted", "resuming"}:
                if resume_authorization is not None:
                    if not native_session_available:
                        raise LLMCallCheckpointError(
                            "supervised native resume requires an existing provider session id"
                        )
                    _authorize_native_resume(
                        existing, current_time=current_time,
                        reconciliation_identity=reconciliation_identity,
                        authorization=resume_authorization,
                    )
                    _write(path, existing, lock=lock_handle)
                    return PreparedCall(
                        path=path,
                        identity=identity,
                        attempt=int(existing.get("attempt") or 1),
                        recomputation_binding=_checkpoint_recomputation_binding(
                            path, existing
                        ),
                        _lock_handle=lock_handle,
                    )
                raise LLMCallNeedsSupervision(checkpoint_path=path)
            elif state == "failed":
                if resume_authorization is not None and bool(existing.get("resumable")):
                    if not native_session_available:
                        raise LLMCallCheckpointError(
                            "supervised native resume requires an existing provider session id"
                        )
                    _authorize_native_resume(
                        existing, current_time=current_time,
                        reconciliation_identity=reconciliation_identity,
                        authorization=resume_authorization,
                    )
                    _write(path, existing, lock=lock_handle)
                    return PreparedCall(
                        path=path,
                        identity=identity,
                        attempt=int(existing.get("attempt") or 1),
                        recomputation_binding=_checkpoint_recomputation_binding(
                            path, existing
                        ),
                        _lock_handle=lock_handle,
                    )
                raise LLMCallRetryExhausted(
                    f"LLM call reached a known terminal failure and will not be submitted again: {path}",
                    checkpoint_path=path,
                )
            else:
                raise LLMCallCheckpointError(f"Invalid LLM call checkpoint state {state!r}: {path}")
        else:
            if resume_authorization is not None:
                raise LLMCallCheckpointError(
                    "supervised native resume requires an existing submitted checkpoint"
                )
            next_attempt = 1
        payload = _new_checkpoint_payload(
            identity=identity, current_time=current_time,
            runtime_capabilities=runtime_capabilities,
        )
        _write(path, payload, lock=lock_handle)
        return PreparedCall(
            path=path,
            identity=str(identity),
            attempt=next_attempt,
            recomputation_binding=_checkpoint_recomputation_binding(path, payload),
            _lock_handle=lock_handle,
        )
    except BaseException:
        _release_handle(lock_handle)
        raise


def record_response(
    prepared: PreparedCall,
    response: LLMProviderResponse[Any],
    *,
    after_write: Callable[[], None] | None = None,
) -> None:
    try:
        current = _require_current(prepared)
        current.update(
            {
                "state": "response_received",
                "submission_state": "submitted",
                "response": _response_to_json(response),
                "updated_at": time.time(),
            }
        )
        _write(prepared.path, current, lock=prepared._lock_handle)
        if after_write is not None:
            after_write()
    finally:
        prepared.release_lock()


def promote_recovered_response(
    path: Path,
    response: LLMProviderResponse[Any],
    *,
    expected_logical_identity: Mapping[str, Any],
    expected_schema_sha256: str,
    selection_receipt_path: str,
    selection_receipt_sha256: str,
    expected_recomputation_binding: Mapping[str, Any] | None = None,
) -> None:
    """Atomically promote a complete immutable-attempt result for normal replay."""

    handle = _acquire_lock(path)
    try:
        current = _read(path, lock=handle)
        if not isinstance(current, dict):
            raise LLMCallCheckpointError(
                f"LLM call checkpoint is missing for recovered response: {path}"
            )
        current_binding = _checkpoint_recomputation_binding(path, current)
        if (
            expected_recomputation_binding is not None
            and current_binding != _canonical_mapping(expected_recomputation_binding)
        ):
            raise LLMCallCheckpointError(
                f"Recovered response recomputation binding mismatch: {path}"
            )
        if current.get("logical_identity") != dict(expected_logical_identity):
            raise LLMCallCheckpointError(
                f"Recovered response logical identity mismatch: {path}"
            )
        recipe = current.get("request_recipe")
        if not isinstance(recipe, Mapping) or recipe.get("schema_sha256") != expected_schema_sha256:
            raise LLMCallCheckpointError(
                f"Recovered response schema identity mismatch: {path}"
            )
        receipt_path = path.with_name(selection_receipt_path)
        if (
            Path(selection_receipt_path).name != selection_receipt_path
            or receipt_path != path.with_name(f"{path.stem}.candidate-selection.json")
        ):
            raise LLMCallCheckpointError(
                f"Recovered response selection receipt address is invalid: {path}"
            )
        try:
            receipt_raw = read_bounded_file(
                receipt_path.parent, receipt_path.name,
                max_bytes=1024 * 1024, suffixes=(".candidate-selection.json",),
            )
        except SecureReadError as exc:
            raise LLMCallCheckpointError(
                f"Recovered response selection receipt is unsafe: {path}"
            ) from exc
        if hashlib.sha256(receipt_raw).hexdigest() != selection_receipt_sha256:
            raise LLMCallCheckpointError(
                f"Recovered response selection receipt hash mismatch: {path}"
            )
        try:
            selection = json.loads(receipt_raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise LLMCallCheckpointError(
                f"Recovered response selection receipt is invalid: {path}"
            ) from exc
        material_payload = [item.to_json() for item in response.candidate_material]
        response_sha256 = sha256_text(canonical_json(response.value))
        candidates = selection.get("candidates") if isinstance(selection, Mapping) else None
        selected_candidates = [
            item for item in candidates or []
            if isinstance(item, Mapping)
            and item.get("ordinal") == selection.get("selected_ordinal")
            and item.get("sha256") == response_sha256
            and item.get("schema_valid") is True
        ]
        if (
            not isinstance(selection, Mapping)
            or selection.get("schema_version") != "arc.llm.response_candidate_selection.v1"
            or selection.get("checkpoint_identity") != current.get("identity")
            or selection.get("business_schema_sha256") != expected_schema_sha256
            or selection.get("selected_sha256") != response_sha256
            or selection.get("decision") not in {
                "last_substantive", "protocol_supersession", "last_valid_empty",
            }
            or len(selected_candidates) != 1
            or selection.get("material_sha256")
            != sha256_text(canonical_json(material_payload))
            or response.candidate_selection != dict(selection)
        ):
            raise LLMCallCheckpointError(
                f"Recovered response selection receipt does not match response: {path}"
            )
        encoded_response = _response_to_json(response)
        if current.get("state") in {"response_received", "validated"}:
            if current.get("response") != encoded_response:
                raise LLMCallCheckpointError(
                    f"Recovered response conflicts with persisted response: {path}"
                )
            return
        if current.get("state") not in {"submitted", "resuming", "failed"}:
            raise LLMCallCheckpointError(
                f"Cannot promote recovered response from state {current.get('state')!r}: {path}"
            )
        if current.get("submission_state") not in {"submitted", "unknown"}:
            raise LLMCallCheckpointError(
                f"Recovered response was not submitted: {path}"
            )
        if current.get("response") is not None:
            raise LLMCallCheckpointError(
                f"Recovered response checkpoint already contains an incompatible payload: {path}"
            )
        prior_failure = {
            key: current.get(key)
            for key in ("state", "submission_state", "failure_category", "resumable")
            if key in current
        }
        current.update({
            "state": "response_received",
            "submission_state": "submitted",
            "response": encoded_response,
            "recovered_response": {
                "selection_receipt_path": selection_receipt_path,
                "selection_receipt_sha256": selection_receipt_sha256,
                "recomputation_binding_sha256": sha256_text(
                    canonical_json(current_binding)
                ),
                "prior_failure": prior_failure,
            },
            "updated_at": time.time(),
        })
        current.pop("failure_category", None)
        current.pop("resumable", None)
        _write(path, current, lock=handle)
    finally:
        _release_handle(handle)


def checkpoint_recomputation_binding(path: Path) -> dict[str, Any]:
    """Read the immutable recipe/address binding used to recompute one call."""

    handle = _acquire_lock(path)
    try:
        current = _read(path, lock=handle)
        if not isinstance(current, Mapping):
            raise LLMCallCheckpointError(
                f"LLM call checkpoint is missing for recomputation: {path}"
            )
        return _checkpoint_recomputation_binding(path, current)
    finally:
        _release_handle(handle)


def record_submitted(prepared: PreparedCall) -> None:
    """Cross the provider submission barrier while retaining call ownership."""

    current = _require_current(prepared)
    if current.get("state") in {"submitted", "response_received", "validated"}:
        return
    if current.get("state") not in {"prepared", "resuming"}:
        raise LLMCallCheckpointError(
            f"Cannot submit checkpoint from state {current.get('state')!r}: {prepared.path}"
        )
    current.update(
        {
            "state": "submitted",
            "submission_state": "submitted",
            "submitted_at": time.time(),
            "updated_at": time.time(),
        }
    )
    _write(prepared.path, current, lock=prepared._lock_handle)


def record_failure(prepared: PreparedCall, exc: BaseException) -> None:
    """Release ownership without charging a non-submitted failure as a paid attempt."""
    try:
        disposition = failure_disposition(exc)
        if disposition is None:
            current = _require_current(prepared)
            if current.get("state") == "prepared":
                _unlink_checkpoint(prepared.path, lock=prepared._lock_handle)
            elif current.get("state") in {"submitted", "resuming"}:
                current.update(
                    {
                        "state": "submitted",
                        "submission_state": "unknown",
                        "failure_category": LLMFailureCategory.UNKNOWN.value,
                        "updated_at": time.time(),
                    }
                )
                _write(prepared.path, current, lock=prepared._lock_handle)
        if disposition is not None and disposition.submission_state == LLMSubmissionState.NOT_SUBMITTED:
            current = _require_current(prepared)
            if current.get("state") == "prepared" and int(current.get("attempt") or 1) == prepared.attempt:
                _unlink_checkpoint(prepared.path, lock=prepared._lock_handle)
        elif disposition is not None and disposition.submission_state in {
            LLMSubmissionState.SUBMITTED,
            LLMSubmissionState.UNKNOWN,
        }:
            current = _require_current(prepared)
            if current.get("state") in {"prepared", "submitted", "resuming"}:
                current.update(
                    {
                        "state": (
                            "submitted"
                            if disposition.submission_state == LLMSubmissionState.UNKNOWN
                            else "failed"
                        ),
                        "submission_state": disposition.submission_state.value,
                        "failure_category": disposition.category.value,
                        "resumable": disposition.submission_state
                        in {LLMSubmissionState.SUBMITTED, LLMSubmissionState.UNKNOWN}
                        and disposition.category
                        in {LLMFailureCategory.TIMEOUT, LLMFailureCategory.CANCELLED},
                        "progress_journal": str(
                            prepared.path.parent.parent / "progress.jsonl"
                        ),
                        "updated_at": time.time(),
                    }
                )
                _write(prepared.path, current, lock=prepared._lock_handle)
    finally:
        prepared.release_lock()


def _authorize_native_resume(
    checkpoint: dict[str, Any], *, current_time: float,
    reconciliation_identity: str | None = None,
    authorization: SupervisedNativeResumeAuthorization,
) -> None:
    """Record the explicit, single invocation that may reconcile a submitted turn."""

    serialized = authorization.to_json()
    existing = checkpoint.get("native_resume_authorization")
    if existing is not None and existing != serialized:
        raise LLMCallCheckpointError(
            "supervised native resume authorization changed for this checkpoint"
        )
    checkpoint.update(
        {
            "state": "resuming",
            "submission_state": "submitted",
            "resume_count": int(checkpoint.get("resume_count") or 0) + 1,
            "resume_authorized_at": current_time,
            "native_resume_authorization": serialized,
            "updated_at": current_time,
        }
    )
    if reconciliation_identity is not None:
        checkpoint["reconciliation_identity"] = reconciliation_identity


def _can_resume_changed_request(
    checkpoint: Mapping[str, Any],
    *,
    incoming: Mapping[str, Any],
    supervised_native_resume: SupervisedNativeResumeAuthorization | None,
    native_session_available: bool,
) -> bool:
    stored_logical = checkpoint.get("logical_identity")
    if stored_logical is None:
        # Legacy checkpoints did not persist enough structure to prove that a
        # changed request still names the same provider turn.  Exact identity
        # remains resumable; changed identity must stay supervised.
        return False
    if stored_logical != incoming.get("logical_identity"):
        return False
    state = checkpoint.get("state")
    submission_state = checkpoint.get("submission_state")
    if state in {"response_received", "validated"}:
        # A paid stateful logical turn may have advanced its durable session
        # receipt before its caller ledger was accepted.  Replay that response
        # by logical identity even if restart reconstruction changes bootstrap
        # versus delta prompt shape.  Stateless keys retain digest strictness.
        logical = checkpoint.get("logical_identity")
        return bool(
            isinstance(logical, Mapping)
            and logical.get("session_key")
        ) or checkpoint.get("reconciliation_identity") == incoming.get("identity")
    if supervised_native_resume is None or not native_session_available:
        return False
    if state in {"submitted", "resuming"}:
        return submission_state in {"submitted", "unknown"}
    return bool(
        state == "failed"
        and checkpoint.get("resumable")
        and submission_state in {"submitted", "unknown"}
    )


def _validate_resume_authorization_binding(
    authorization: SupervisedNativeResumeAuthorization | None,
    incoming: Mapping[str, Any],
) -> None:
    if authorization is None:
        return
    logical = incoming.get("logical_identity")
    if not isinstance(logical, Mapping):
        raise LLMCallCheckpointError(
            "supervised native resume requires a structured call identity"
        )
    if logical.get("initial_native_authorization") != authorization.to_json():
        raise LLMCallCheckpointError(
            "supervised native resume authorization does not match the complete initial call identity"
        )


def _validate_persisted_resume_authorization(
    checkpoint: Mapping[str, Any],
    authorization: SupervisedNativeResumeAuthorization | None,
) -> None:
    if authorization is None:
        return
    initial = checkpoint.get("initial_native_authorization")
    if initial != authorization.to_json():
        raise LLMCallCheckpointError(
            "supervised native resume authorization does not match its initial authorization"
        )
    persisted = checkpoint.get("native_resume_authorization")
    if persisted is None:
        return
    if persisted != authorization.to_json():
        raise LLMCallCheckpointError(
            "supervised native resume authorization does not match its persisted identity"
        )


def _can_rebuild_prepared(
    checkpoint: Mapping[str, Any], incoming: Mapping[str, Any]
) -> bool:
    return bool(
        checkpoint.get("state") == "prepared"
        and checkpoint.get("submission_state") == "not_submitted"
        and checkpoint.get("logical_identity") == incoming.get("logical_identity")
    )


def _identity_parts(identity: str) -> dict[str, Any]:
    if isinstance(identity, CallIdentity):
        return {
            "identity": str(identity),
            "logical_identity": dict(identity.logical_identity),
            "request_digest": identity.request_digest,
            "request_recipe": dict(identity.request_recipe),
        }
    return {
        "identity": str(identity), "logical_identity": None,
        "request_digest": str(identity), "request_recipe": {},
    }


def _initial_authorization_from_logical_identity(
    logical_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    value = logical_identity.get("initial_native_authorization")
    return dict(value) if isinstance(value, Mapping) else None


def _canonical_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Detach a JSON-shaped mapping from caller-owned mutable objects."""

    try:
        decoded = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LLMCallCheckpointError(
            "LLM call recomputation binding is not canonical JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise LLMCallCheckpointError(
            "LLM call recomputation binding is not an object"
        )
    return decoded


def _checkpoint_recomputation_binding(
    path: Path, checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the exact immutable inputs that authorize recomputation."""

    recipe_value = checkpoint.get("request_recipe")
    recipe = dict(recipe_value) if isinstance(recipe_value, Mapping) else {}
    logical_value = checkpoint.get("logical_identity")
    logical = dict(logical_value) if isinstance(logical_value, Mapping) else None
    label_hash = recipe.get("call_label_sha256")
    if not isinstance(label_hash, str):
        label_hash = sha256_text(str(recipe.get("call_label") or ""))
    authorization_value = checkpoint.get("native_resume_authorization")
    authorization = (
        dict(authorization_value)
        if isinstance(authorization_value, Mapping)
        else None
    )
    binding = {
        "schema_version": "arc.llm.checkpoint_recomputation_binding.v1",
        "checkpoint_path": str(path.expanduser().resolve(strict=False)),
        "checkpoint_identity": str(checkpoint.get("identity") or ""),
        "logical_identity": logical,
        "request_digest": str(checkpoint.get("request_digest") or ""),
        "request_recipe_sha256": sha256_text(canonical_json(recipe)),
        "idempotency_key": logical.get("idempotency_key") if logical else None,
        "session_key": logical.get("session_key") if logical else None,
        "generation": logical.get("generation") if logical else None,
        "prompt_sha256": recipe.get("prompt_sha256"),
        "schema_sha256": recipe.get("schema_sha256"),
        "call_label_sha256": label_hash,
        "native_resume_authorization": authorization,
        "initial_native_authorization": (
            dict(checkpoint["initial_native_authorization"])
            if isinstance(checkpoint.get("initial_native_authorization"), Mapping)
            else None
        ),
    }
    return _canonical_mapping(binding)


def _new_checkpoint_payload(
    *, identity: str, current_time: float,
    runtime_capabilities: Mapping[str, Any] | None,
) -> dict[str, Any]:
    parts = _identity_parts(identity)
    logical = parts["logical_identity"]
    initial_authorization = (
        dict(logical["initial_native_authorization"])
        if isinstance(logical, Mapping)
        and isinstance(logical.get("initial_native_authorization"), Mapping)
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": str(identity),
        "logical_identity": parts["logical_identity"],
        "request_digest": parts["request_digest"],
        "request_recipe": parts["request_recipe"],
        "initial_native_authorization": initial_authorization,
        "state": "prepared",
        "submission_state": "not_submitted",
        "attempt": 1,
        "started_at": current_time,
        "updated_at": current_time,
        "runtime_capabilities": dict(runtime_capabilities or {}),
    }


def _upgrade_legacy_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Upgrade v2/v3 without making an uncertain provider call replayable."""

    upgraded = dict(checkpoint)
    upgraded["schema_version"] = SCHEMA_VERSION
    upgraded.setdefault("logical_identity", None)
    upgraded.setdefault("request_digest", str(upgraded.get("identity") or ""))
    upgraded.setdefault("request_recipe", {})
    upgraded.setdefault("initial_native_authorization", None)
    upgraded["legacy_checkpoint_upgraded"] = True
    if upgraded.get("state") == "started":
        upgraded["state"] = "submitted"
        upgraded["submission_state"] = "unknown"
    return upgraded


def record_validated(prepared: PreparedCall) -> None:
    acquired = prepared._lock_handle is None
    lock = prepared._lock_handle or _acquire_lock(prepared.path)
    try:
        current = _read(prepared.path, lock=lock)
        if current is None or current.get("identity") != prepared.identity:
            raise LLMCallCheckpointError(
                f"LLM call checkpoint disappeared or changed: {prepared.path}"
            )
        if current.get("state") == "validated":
            return
        if current.get("state") != "response_received":
            raise LLMCallCheckpointError(
                f"Cannot validate checkpoint before response: {prepared.path}"
            )
        current.update({
            "state": "validated", "validated_at": time.time(),
            "updated_at": time.time(),
        })
        _write(prepared.path, current, lock=lock)
    finally:
        if acquired:
            _release_handle(lock)


def _require_current(prepared: PreparedCall) -> dict[str, Any]:
    current = _read(prepared.path, lock=prepared._lock_handle)
    if current is None or current.get("identity") != prepared.identity:
        raise LLMCallCheckpointError(f"LLM call checkpoint disappeared or changed: {prepared.path}")
    return current


def _response_to_json(response: LLMProviderResponse[Any]) -> dict[str, Any]:
    return {
        "value": response.value,
        "usage": response.usage.to_json(),
        "native_session_id": response.native_session_id,
        "raw_events": list(response.raw_events),
        "raw_output": response.raw_output,
        "raw_model_output": response.raw_model_output,
        "prompt_sent_sha256": response.prompt_sent_sha256,
        "prompt_sent_bytes": response.prompt_sent_bytes,
        "structured_output": response.structured_output,
        "candidate_material": [item.to_json() for item in response.candidate_material],
        "candidate_selection": response.candidate_selection,
    }


def _response_from_json(value: Any) -> LLMProviderResponse[Any]:
    if not isinstance(value, Mapping):
        raise LLMCallCheckpointError("LLM response checkpoint is missing its response payload")
    usage = value.get("usage") if isinstance(value.get("usage"), Mapping) else {}
    try:
        candidate_material = tuple(
            ResponseCandidateMaterial.from_json(item)
            for item in value.get("candidate_material") or ()
        )
    except (TypeError, ValueError) as exc:
        raise LLMCallCheckpointError(
            f"LLM response checkpoint has invalid candidate material: {exc}"
        ) from exc
    return LLMProviderResponse(
        value.get("value"),
        usage=LLMUsage(
            input_tokens=usage.get("input_tokens"),
            cached_input_tokens=usage.get("cached_input_tokens"),
            output_tokens=usage.get("output_tokens"),
            reasoning_output_tokens=usage.get("reasoning_output_tokens"),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
            cache_read_input_tokens=usage.get("cache_read_input_tokens"),
            raw=dict(usage.get("raw") or {}),
        ),
        native_session_id=value.get("native_session_id"),
        raw_events=tuple(value.get("raw_events") or ()),
        raw_output=str(value.get("raw_output") or ""),
        raw_model_output=str(value.get("raw_model_output") or ""),
        prompt_sent_sha256=value.get("prompt_sent_sha256"),
        prompt_sent_bytes=value.get("prompt_sent_bytes"),
        structured_output=(
            dict(value["structured_output"])
            if isinstance(value.get("structured_output"), Mapping)
            else None
        ),
        candidate_material=candidate_material,
        candidate_selection=(
            dict(value["candidate_selection"])
            if isinstance(value.get("candidate_selection"), Mapping)
            else None
        ),
    )


def _directory_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _open_directory(path: Path, *, create: bool) -> tuple[int, Path]:
    """Open a directory chain component-by-component without following links."""

    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    anchor = Path(absolute.anchor or os.sep)
    descriptor = os.open(anchor, _directory_flags())
    try:
        for part in absolute.parts[1:]:
            while True:
                try:
                    child = os.open(part, _directory_flags(), dir_fd=descriptor)
                    break
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        os.mkdir(part, 0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
            value = os.fstat(child)
            if not stat.S_ISDIR(value.st_mode):
                os.close(child)
                raise OSError("checkpoint path component is not a directory")
            os.close(descriptor)
            descriptor = child
        return descriptor, absolute
    except BaseException:
        os.close(descriptor)
        raise


def _revalidate_lock_address(lock: _CheckpointLock, path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    if absolute.parent != lock.parent_path or absolute.name != lock.checkpoint_name:
        raise LLMCallCheckpointError(
            f"Checkpoint lock does not own this address: {path}"
        )
    descriptor, reopened = _open_directory(lock.parent_path, create=False)
    try:
        value = os.fstat(descriptor)
        identity = value.st_dev, value.st_ino, value.st_mode
        if reopened != lock.parent_path or identity != lock.parent_identity:
            raise LLMCallCheckpointError(
                f"LLM call checkpoint parent changed while locked: {path}"
            )
    finally:
        os.close(descriptor)
    try:
        lock_value = os.stat(
            lock.lock_name, dir_fd=lock.parent_fd, follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise LLMCallCheckpointError(
            f"LLM call checkpoint lock disappeared: {path}"
        ) from exc
    lock_identity = (
        lock_value.st_dev, lock_value.st_ino,
        lock_value.st_mode, lock_value.st_nlink,
    )
    if lock_identity != lock.lock_identity or not stat.S_ISREG(lock_value.st_mode):
        raise LLMCallCheckpointError(
            f"LLM call checkpoint lock address changed: {path}"
        )


def _checkpoint_leaf_identity(
    lock: _CheckpointLock, name: str,
) -> tuple[int, int, int, int, int, int] | None:
    try:
        value = os.stat(name, dir_fd=lock.parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise LLMCallCheckpointError(
            f"LLM call checkpoint leaf is unsafe: {lock.parent_path / name}"
        )
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
        value.st_size, value.st_mtime_ns,
    )


def _read_descriptor_bounded(
    descriptor: int, limit: int, *, allowed_links: tuple[int, ...] = (1,),
) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink not in allowed_links:
        raise SecureReadError("checkpoint is not a singly-linked regular file")
    chunks: list[bytes] = []
    remaining = limit + 1
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    if len(raw) > limit:
        raise SecureReadError("checkpoint exceeds its byte limit")
    after = os.fstat(descriptor)
    before_identity = (
        before.st_dev, before.st_ino, before.st_mode, before.st_nlink,
        before.st_size, before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev, after.st_ino, after.st_mode, after.st_nlink,
        after.st_size, after.st_mtime_ns,
    )
    if before_identity != after_identity or after.st_size != len(raw):
        raise SecureReadError("checkpoint changed while reading")
    return raw


def _write_all(descriptor: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(descriptor, value[offset:])
        if written <= 0:
            raise OSError("short checkpoint write")
        offset += written


def _checkpoint_stage_name(path: Path, encoded: bytes) -> str:
    return f".{path.name}.arc-stage-{hashlib.sha256(encoded).hexdigest()}"


def _read_stage_snapshot(
    lock: _CheckpointLock, name: str, *, allowed_links: tuple[int, ...] = (1,),
) -> tuple[tuple[int, int, int, int, int, int], bytes] | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=lock.parent_fd)
    except FileNotFoundError:
        return None
    try:
        value = os.fstat(descriptor)
        raw = _read_descriptor_bounded(
            descriptor,
            MAX_CALL_CHECKPOINT_BYTES,
            allowed_links=allowed_links,
        )
        identity = (
            value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns,
        )
        return identity, raw
    finally:
        os.close(descriptor)


def _cleanup_checkpoint_stages(lock: _CheckpointLock, path: Path) -> None:
    prefix = f".{path.name}.arc-stage-"
    for name in os.listdir(lock.parent_fd):
        if not name.startswith(prefix):
            continue
        digest = name[len(prefix):]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            continue
        snapshot = _read_stage_snapshot(lock, name, allowed_links=(1, 2))
        if snapshot is None:
            continue
        target = _read_stage_snapshot(
            lock, path.name, allowed_links=(1, 2),
        )
        stage_matches = hashlib.sha256(snapshot[1]).hexdigest() == digest
        target_matches = (
            target is not None and hashlib.sha256(target[1]).hexdigest() == digest
        )
        hardlink_publish = (
            stage_matches and snapshot[0][3] == 2
            and target is not None and target[0] == snapshot[0]
        )
        exchange_publish = (
            not stage_matches and snapshot[0][3] == 1 and target_matches
        )
        unpublished = stage_matches and snapshot[0][3] == 1
        if not (hardlink_publish or exchange_publish or unpublished):
            raise LLMCallCheckpointError(
                f"LLM call checkpoint staged leaf is unsafe: {path}"
            )
        _revalidate_lock_address(lock, path)
        if _read_stage_snapshot(lock, name, allowed_links=(1, 2)) != snapshot:
            raise LLMCallCheckpointError(
                f"LLM call checkpoint staged leaf changed: {path}"
            )
        os.unlink(name, dir_fd=lock.parent_fd)
        os.fsync(lock.parent_fd)


def _remove_checkpoint_stage(
    lock: _CheckpointLock, name: str, encoded: bytes,
) -> None:
    try:
        snapshot = _read_stage_snapshot(lock, name)
    except (OSError, SecureReadError, LLMCallCheckpointError):
        return
    if snapshot is None or snapshot[1] != encoded:
        return
    try:
        _revalidate_lock_address(lock, lock.parent_path / lock.checkpoint_name)
        if _read_stage_snapshot(lock, name) != snapshot:
            return
        os.unlink(name, dir_fd=lock.parent_fd)
        os.fsync(lock.parent_fd)
    except (OSError, LLMCallCheckpointError):
        return


def _publish_checkpoint_stage(
    lock: _CheckpointLock,
    path: Path,
    staged_name: str,
    encoded: bytes,
    *,
    expected: tuple[tuple[int, int, int, int, int, int], bytes] | None,
) -> None:
    """Atomically CAS a checkpoint leaf or fail closed."""

    _before_checkpoint_exchange(path)
    _revalidate_lock_address(lock, path)
    if expected is None:
        try:
            os.link(
                staged_name,
                path.name,
                src_dir_fd=lock.parent_fd,
                dst_dir_fd=lock.parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise LLMCallCheckpointError(
                f"LLM call checkpoint appeared in final publication window: {path}"
            ) from exc
        staged = os.stat(staged_name, dir_fd=lock.parent_fd, follow_symlinks=False)
        target = os.stat(path.name, dir_fd=lock.parent_fd, follow_symlinks=False)
        if (
            staged.st_dev, staged.st_ino, staged.st_nlink
        ) != (target.st_dev, target.st_ino, 2):
            raise LLMCallCheckpointError(
                f"LLM call checkpoint hardlink publication changed: {path}"
            )
        os.unlink(staged_name, dir_fd=lock.parent_fd)
    elif _checkpoint_rename_exchange(lock.parent_fd, staged_name, path.name):
        displaced = _read_stage_snapshot(lock, staged_name)
        if displaced != expected:
            if not _checkpoint_rename_exchange(
                lock.parent_fd, staged_name, path.name,
            ):
                raise LLMCallCheckpointError(
                    f"Late checkpoint replacement could not be restored: {path}"
                )
            raise LLMCallCheckpointError(
                f"LLM call checkpoint changed in final publication window: {path}"
            )
        os.unlink(staged_name, dir_fd=lock.parent_fd)
    else:  # pragma: no cover - fail closed when exact exchange is unavailable
        raise LLMCallCheckpointError(
            f"Atomic checkpoint exchange is unsupported: {path}"
        )
    os.fsync(lock.parent_fd)


def _checkpoint_rename_exchange(parent_fd: int, first: str, second: str) -> bool:
    if os.name != "posix":
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return False
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd, os.fsencode(first), parent_fd, os.fsencode(second), 2,
    )
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error in {22, 38, 95}:
        return False
    raise OSError(error, os.strerror(error))


def _unlink_checkpoint(path: Path, *, lock: _CheckpointLock | None) -> None:
    if lock is None:
        raise LLMCallCheckpointError(
            f"Checkpoint removal requires locked directory ownership: {path}"
        )
    try:
        _revalidate_lock_address(lock, path)
        _checkpoint_leaf_identity(lock, path.name)
        try:
            os.unlink(path.name, dir_fd=lock.parent_fd)
        except FileNotFoundError:
            return
        os.fsync(lock.parent_fd)
    except OSError as exc:
        raise LLMCallCheckpointError(
            f"Could not remove LLM call checkpoint {path}: {exc}"
        ) from exc


def _read(
    path: Path, *, lock: _CheckpointLock | None = None,
) -> dict[str, Any] | None:
    if lock is None:
        raise LLMCallCheckpointError(
            f"Checkpoint read requires locked directory ownership: {path}"
        )
    try:
        _revalidate_lock_address(lock, path)
        flags = os.O_RDONLY
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path.name, flags, dir_fd=lock.parent_fd)
        except FileNotFoundError:
            return None
        try:
            raw = _read_descriptor_bounded(descriptor, MAX_CALL_CHECKPOINT_BYTES)
        finally:
            os.close(descriptor)
        value = json.loads(raw)
    except (OSError, SecureReadError, UnicodeError, json.JSONDecodeError) as exc:
        raise LLMCallCheckpointError(f"Could not read LLM call checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMCallCheckpointError(f"LLM call checkpoint is not an object: {path}")
    return value


def _write(
    path: Path,
    payload: Mapping[str, Any],
    *,
    lock: _CheckpointLock | None = None,
) -> None:
    if lock is None:
        raise LLMCallCheckpointError(
            f"Checkpoint write requires locked directory ownership: {path}"
        )
    encoded = (
        json.dumps(
            dict(payload), ensure_ascii=False, sort_keys=True, default=str,
        ) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_CALL_CHECKPOINT_BYTES:
        raise LLMCallCheckpointError(f"LLM call checkpoint is too large: {path}")
    temporary_name = _checkpoint_stage_name(path, encoded)
    try:
        _revalidate_lock_address(lock, path)
        before = _read_stage_snapshot(lock, path.name)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(
                temporary_name, flags, 0o600, dir_fd=lock.parent_fd,
            )
        except FileExistsError:
            snapshot = _read_stage_snapshot(lock, temporary_name)
            if snapshot is None or snapshot[1] != encoded:
                raise LLMCallCheckpointError(
                    f"LLM call checkpoint staged leaf was replaced: {path}"
                )
        else:
            try:
                _write_all(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(lock.parent_fd)
        staged = _read_stage_snapshot(lock, temporary_name)
        _before_checkpoint_replace(path)
        _revalidate_lock_address(lock, path)
        if staged is None or _read_stage_snapshot(lock, temporary_name) != staged:
            raise LLMCallCheckpointError(
                f"LLM call checkpoint staged leaf changed before persistence: {path}"
            )
        if _read_stage_snapshot(lock, path.name) != before:
            raise LLMCallCheckpointError(
                f"LLM call checkpoint address changed before persistence: {path}"
            )
        _publish_checkpoint_stage(
            lock,
            path,
            temporary_name,
            encoded,
            expected=before,
        )
    except OSError as exc:
        raise LLMCallCheckpointError(f"Could not persist LLM call checkpoint {path}: {exc}") from exc
    finally:
        _remove_checkpoint_stage(lock, temporary_name, encoded)


def _acquire_lock(
    path: Path,
    *,
    deadline: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> _CheckpointLock:
    lock: _CheckpointLock | None = None
    parent_fd = -1
    descriptor = -1
    handle: BinaryIO | None = None
    try:
        parent_fd, parent_path = _open_directory(path.parent, create=True)
        parent_value = os.fstat(parent_fd)
        parent_identity = (
            parent_value.st_dev, parent_value.st_ino, parent_value.st_mode,
        )
        lock_name = path.name + ".lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_name, flags, 0o600, dir_fd=parent_fd)
        handle = os.fdopen(descriptor, "r+b")
        descriptor = -1
        os.fchmod(handle.fileno(), 0o600)
        lock_value = os.fstat(handle.fileno())
        if not stat.S_ISREG(lock_value.st_mode) or lock_value.st_nlink != 1:
            raise LLMCallCheckpointError(
                f"LLM call checkpoint lock is not a singly-linked regular file: {path}"
            )
        lock = _CheckpointLock(
            handle=handle,
            parent_fd=parent_fd,
            parent_path=parent_path,
            parent_identity=parent_identity,
            checkpoint_name=path.name,
            lock_name=lock_name,
            lock_identity=(
                lock_value.st_dev, lock_value.st_ino,
                lock_value.st_mode, lock_value.st_nlink,
            ),
        )
        if os.name == "nt":
            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
        while True:
            if cancel_check is not None and cancel_check():
                _close_unlocked_lock(lock)
                raise LLMWorkerCancelled("LLM call cancelled while waiting for checkpoint ownership")
            if deadline is not None and time.monotonic() >= deadline:
                _close_unlocked_lock(lock)
                raise LLMWorkerTimeout("LLM call timed out while waiting for checkpoint ownership")
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                time.sleep(0.05)
                continue
            _revalidate_lock_address(lock, path)
            _cleanup_checkpoint_stages(lock, path)
            _revalidate_lock_address(lock, path)
            return lock
    except OSError as exc:
        if lock is not None:
            _close_unlocked_lock(lock)
        else:
            _close_raw_lock_resources(handle, descriptor, parent_fd)
        raise LLMCallCheckpointError(f"Could not lock LLM call checkpoint {path}: {exc}") from exc
    except BaseException:
        if lock is not None:
            _close_unlocked_lock(lock)
        else:
            _close_raw_lock_resources(handle, descriptor, parent_fd)
        raise


def _unlock(lock: _CheckpointLock) -> None:
    handle = lock.handle
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _close_unlocked_lock(lock: _CheckpointLock) -> None:
    if not lock.handle.closed:
        lock.handle.close()
    if lock.parent_fd >= 0:
        os.close(lock.parent_fd)
        lock.parent_fd = -1


def _close_raw_lock_resources(
    handle: BinaryIO | None, descriptor: int, parent_fd: int,
) -> None:
    if handle is not None and not handle.closed:
        handle.close()
    elif descriptor >= 0:
        os.close(descriptor)
    if parent_fd >= 0:
        os.close(parent_fd)


def _release_handle(lock: _CheckpointLock) -> None:
    validation_error: BaseException | None = None
    try:
        try:
            _revalidate_lock_address(
                lock, lock.parent_path / lock.checkpoint_name,
            )
        except BaseException as exc:
            validation_error = (
                exc
                if isinstance(exc, LLMCallCheckpointError)
                else LLMCallCheckpointError(
                    f"LLM call checkpoint lock address changed before unlock: {exc}"
                )
            )
        _unlock(lock)
    finally:
        lock.handle.close()
        if lock.parent_fd >= 0:
            os.close(lock.parent_fd)
            lock.parent_fd = -1
    if validation_error is not None:
        raise validation_error
