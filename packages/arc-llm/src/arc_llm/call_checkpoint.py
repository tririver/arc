from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping

from .schema_cache import canonical_json, sha256_text
from .usage import LLMProviderResponse, LLMUsage
from .providers.base import (
    LLMAbortScope,
    LLMFailureCategory,
    LLMSubmissionState,
    LLMWorkerError,
    LLMWorkerCancelled,
    LLMWorkerTimeout,
    failure_disposition,
)


SCHEMA_VERSION = "arc.llm.call_checkpoint.v2"


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
class PreparedCall:
    path: Path
    identity: str
    attempt: int
    replay_response: LLMProviderResponse[Any] | None = None
    replayed: bool = False
    _lock_handle: BinaryIO | None = None

    def release_lock(self) -> None:
        handle, self._lock_handle = self._lock_handle, None
        if handle is None:
            return
        _unlock(handle)
        handle.close()


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
) -> tuple[Path, str]:
    identity_payload = {
        "prompt_sha256": sha256_text(prompt),
        "schema": dict(schema) if schema is not None else None,
        "provider": provider,
        "model": model,
        "call_label": call_label,
        "session_policy": session_policy,
        "session_key": session_key,
        # A stable logical key must survive a crash after the session receipt
        # advanced the observable turn count.
        "session_turn": None if idempotency_key else session_turn,
        "runtime_fingerprint": runtime_fingerprint,
        "idempotency_key": idempotency_key,
        "generation": generation,
        "progress_contract_scope": progress_contract_scope,
    }
    identity = sha256_text(canonical_json(identity_payload))
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
    supervised_native_resume: bool = False,
    native_session_available: bool = False,
) -> PreparedCall:
    del retry_delay_seconds
    current_time = time.time() if now is None else now
    lock_handle = _acquire_lock(path, deadline=deadline, cancel_check=cancel_check)
    try:
        existing = _read(path)
        if existing is not None:
            if existing.get("schema_version") != SCHEMA_VERSION or existing.get("identity") != identity:
                raise LLMCallCheckpointError(f"LLM call checkpoint identity mismatch: {path}")
            state = existing.get("state")
            if state in {"response_received", "validated"}:
                response = _response_from_json(existing.get("response"))
                _release_handle(lock_handle)
                return PreparedCall(
                    path=path,
                    identity=identity,
                    attempt=int(existing.get("attempt") or 1),
                    replay_response=response,
                    replayed=True,
                )
            if state in {"started", "resuming"}:
                if supervised_native_resume:
                    if not native_session_available:
                        raise LLMCallCheckpointError(
                            "supervised native resume requires an existing provider session id"
                        )
                    _authorize_native_resume(existing, current_time=current_time)
                    _write(path, existing)
                    return PreparedCall(
                        path=path,
                        identity=identity,
                        attempt=int(existing.get("attempt") or 1),
                        _lock_handle=lock_handle,
                    )
                raise LLMCallNeedsSupervision(checkpoint_path=path)
            elif state == "failed":
                if supervised_native_resume and bool(existing.get("resumable")):
                    if not native_session_available:
                        raise LLMCallCheckpointError(
                            "supervised native resume requires an existing provider session id"
                        )
                    _authorize_native_resume(existing, current_time=current_time)
                    _write(path, existing)
                    return PreparedCall(
                        path=path,
                        identity=identity,
                        attempt=int(existing.get("attempt") or 1),
                        _lock_handle=lock_handle,
                    )
                raise LLMCallRetryExhausted(
                    f"LLM call reached a known terminal failure and will not be submitted again: {path}",
                    checkpoint_path=path,
                )
            else:
                raise LLMCallCheckpointError(f"Invalid LLM call checkpoint state {state!r}: {path}")
        else:
            next_attempt = 1
        _write(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "identity": identity,
                "state": "started",
                "submission_state": "unknown",
                "attempt": next_attempt,
                "started_at": current_time,
                "updated_at": current_time,
            },
        )
        return PreparedCall(path=path, identity=identity, attempt=next_attempt, _lock_handle=lock_handle)
    except BaseException:
        _release_handle(lock_handle)
        raise


def record_response(prepared: PreparedCall, response: LLMProviderResponse[Any]) -> None:
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
        _write(prepared.path, current)
    finally:
        prepared.release_lock()


def record_failure(prepared: PreparedCall, exc: BaseException) -> None:
    """Release ownership without charging a non-submitted failure as a paid attempt."""
    try:
        disposition = failure_disposition(exc)
        if disposition is not None and disposition.submission_state == LLMSubmissionState.NOT_SUBMITTED:
            current = _require_current(prepared)
            if current.get("state") == "started" and int(current.get("attempt") or 1) == prepared.attempt:
                prepared.path.unlink(missing_ok=True)
        elif disposition is not None and disposition.submission_state == LLMSubmissionState.SUBMITTED:
            current = _require_current(prepared)
            if current.get("state") in {"started", "resuming"}:
                current.update(
                    {
                        "state": "failed",
                        "submission_state": "submitted",
                        "failure_category": disposition.category.value,
                        "resumable": disposition.category
                        in {LLMFailureCategory.TIMEOUT, LLMFailureCategory.CANCELLED},
                        "progress_journal": str(
                            prepared.path.parent.parent / "progress.jsonl"
                        ),
                        "updated_at": time.time(),
                    }
                )
                _write(prepared.path, current)
    finally:
        prepared.release_lock()


def _authorize_native_resume(checkpoint: dict[str, Any], *, current_time: float) -> None:
    """Record the explicit, single invocation that may reconcile a submitted turn."""

    checkpoint.update(
        {
            "state": "resuming",
            "submission_state": "submitted",
            "resume_count": int(checkpoint.get("resume_count") or 0) + 1,
            "resume_authorized_at": current_time,
            "updated_at": current_time,
        }
    )


def record_validated(prepared: PreparedCall) -> None:
    current = _require_current(prepared)
    if current.get("state") == "validated":
        return
    if current.get("state") != "response_received":
        raise LLMCallCheckpointError(f"Cannot validate checkpoint before response: {prepared.path}")
    current.update({"state": "validated", "validated_at": time.time(), "updated_at": time.time()})
    _write(prepared.path, current)


def _require_current(prepared: PreparedCall) -> dict[str, Any]:
    current = _read(prepared.path)
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
    }


def _response_from_json(value: Any) -> LLMProviderResponse[Any]:
    if not isinstance(value, Mapping):
        raise LLMCallCheckpointError("LLM response checkpoint is missing its response payload")
    usage = value.get("usage") if isinstance(value.get("usage"), Mapping) else {}
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
    )


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMCallCheckpointError(f"Could not read LLM call checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMCallCheckpointError(f"LLM call checkpoint is not an object: {path}")
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise LLMCallCheckpointError(f"Could not persist LLM call checkpoint {path}: {exc}") from exc


def _acquire_lock(
    path: Path,
    *,
    deadline: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> BinaryIO:
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = path.with_suffix(path.suffix + ".lock")
        handle = lock_path.open("a+b")
        os.chmod(lock_path, 0o600)
        if os.name == "nt":
            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
        while True:
            if cancel_check is not None and cancel_check():
                handle.close()
                raise LLMWorkerCancelled("LLM call cancelled while waiting for checkpoint ownership")
            if deadline is not None and time.monotonic() >= deadline:
                handle.close()
                raise LLMWorkerTimeout("LLM call timed out while waiting for checkpoint ownership")
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return handle
            except (BlockingIOError, OSError):
                time.sleep(0.05)
    except OSError as exc:
        raise LLMCallCheckpointError(f"Could not lock LLM call checkpoint {path}: {exc}") from exc


def _unlock(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _release_handle(handle: BinaryIO) -> None:
    try:
        _unlock(handle)
    finally:
        handle.close()
