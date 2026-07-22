"""Durable first-real-task canaries for unproven structured-output contracts."""

from __future__ import annotations

import errno
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, TypeVar

from .providers.base import (
    LLMAbortScope,
    LLMFailureCategory,
    LLMSubmissionState,
    LLMWorkerCancelled,
    LLMWorkerError,
    failure_disposition,
)
from .schema_cache import canonical_json, sha256_text


SCHEMA_CANARY_RECEIPT_VERSION = "arc.llm.schema_canary_receipt.v1"
SCHEMA_CANARY_RECEIPT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "identity",
        "identity_sha256",
        "status",
        "recorded_at",
        "failure",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_CANARY_RECEIPT_VERSION},
        "identity": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "provider_id",
                "runtime_fingerprint",
                "effective_model",
                "effective_schema_sha256",
                "transport_mode",
            ],
            "properties": {
                "provider_id": {"type": "string", "minLength": 1},
                "runtime_fingerprint": {"type": "string", "minLength": 1},
                "effective_model": {"type": ["string", "null"]},
                "effective_schema_sha256": {"type": "string", "minLength": 1},
                "transport_mode": {"enum": ["strict", "prompt"]},
            },
        },
        "identity_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "status": {"enum": ["proven", "rejected"]},
        "recorded_at": {"type": "number"},
        "failure": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["category", "submission_state"],
                    "properties": {
                        "category": {"type": "string"},
                        "submission_state": {"type": "string"},
                    },
                },
            ]
        },
    },
}


T = TypeVar("T")


@dataclass(frozen=True)
class SchemaCanaryIdentity:
    """The complete provider contract whose successful use may be reused."""

    provider_id: str
    runtime_fingerprint: str
    effective_model: str | None
    effective_schema_sha256: str
    transport_mode: str

    def __post_init__(self) -> None:
        if not self.provider_id or not self.runtime_fingerprint or not self.effective_schema_sha256:
            raise ValueError("schema canary identity fields must be non-empty")
        if self.transport_mode not in {"strict", "prompt"}:
            raise ValueError("schema canary transport_mode must be strict or prompt")

    @property
    def sha256(self) -> str:
        return sha256_text(canonical_json(asdict(self)))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class SchemaCanaryStorageError(LLMWorkerError):
    """A local receipt failure after submission may require supervision."""

    def __init__(self, message: str, *, submission_state: LLMSubmissionState) -> None:
        super().__init__(
            message,
            retryable=False,
            category=LLMFailureCategory.LOCAL_IO,
            abort_scope=LLMAbortScope.BATCH,
            submission_state=submission_state,
        )


class SchemaCanaryBlocked(LLMWorkerError):
    """A follower was withheld after the first real task was rejected."""

    def __init__(self, identity: SchemaCanaryIdentity, *, category: str) -> None:
        self.identity_sha256 = identity.sha256
        self.rejection_category = category
        super().__init__(
            f"LLM call blocked by schema canary rejection ({category}, {identity.sha256})",
            retryable=False,
            category=(
                category
                if category in {item.value for item in LLMFailureCategory}
                else LLMFailureCategory.SCHEMA
            ),
            abort_scope=LLMAbortScope.BATCH,
            submission_state=LLMSubmissionState.NOT_SUBMITTED,
        )


def schema_canary_receipt_path(root: Path, identity: SchemaCanaryIdentity) -> Path:
    """Return the receipt path below the caller-owned artifact root."""

    return root / "schema-canaries" / f"{identity.sha256}.json"


def run_schema_canary(
    *,
    root: Path,
    identity: SchemaCanaryIdentity,
    invoke: Callable[[], T],
    cancel_check: Callable[[], bool] | None = None,
) -> T:
    """Run or follow the first real task for ``identity``.

    The identity lock is held through the first provider result and atomic
    receipt commit. Followers therefore consume no provider capacity until the
    contract is proven. A stable deterministic rejection is persisted only
    within this caller-owned run tree and fans out as a non-submitted error.
    """

    path = schema_canary_receipt_path(root, identity)
    lock: BinaryIO | None = _acquire_lock(path, cancel_check=cancel_check)
    try:
        receipt = _read_receipt(path, identity)
        if receipt is not None:
            if receipt["status"] == "proven":
                # The receipt is immutable and atomic. Once observed, holding
                # the proof lock across ordinary calls would accidentally turn
                # the canary into a permanent per-identity semaphore.
                _release_lock(lock)
                lock = None
                return invoke()
            failure = receipt.get("failure") or {}
            raise SchemaCanaryBlocked(
                identity, category=str(failure.get("category") or LLMFailureCategory.SCHEMA.value)
            )

        try:
            result = invoke()
        except BaseException as exc:
            disposition = failure_disposition(exc)
            if _is_deterministic_rejection(disposition):
                _write_receipt(
                    path,
                    _receipt_payload(
                        identity,
                        status="rejected",
                        failure={
                            "category": disposition.category.value,
                            "submission_state": disposition.submission_state.value,
                        },
                    ),
                    submission_state=disposition.submission_state,
                )
            raise
        _write_receipt(
            path,
            _receipt_payload(identity, status="proven", failure=None),
            submission_state=LLMSubmissionState.SUBMITTED,
        )
        return result
    finally:
        if lock is not None:
            _release_lock(lock)


def _is_deterministic_rejection(disposition: Any) -> bool:
    return bool(
        disposition is not None
        and not disposition.retryable
        and disposition.category == LLMFailureCategory.SCHEMA
    )


def _receipt_payload(
    identity: SchemaCanaryIdentity,
    *,
    status: str,
    failure: Mapping[str, str] | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_CANARY_RECEIPT_VERSION,
        "identity": identity.to_json(),
        "identity_sha256": identity.sha256,
        "status": status,
        "recorded_at": time.time(),
        "failure": dict(failure) if failure is not None else None,
    }


def _read_receipt(
    path: Path, identity: SchemaCanaryIdentity
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaCanaryStorageError(
            f"Could not read schema canary receipt {path}: {exc}",
            submission_state=LLMSubmissionState.NOT_SUBMITTED,
        ) from exc
    if not isinstance(payload, dict):
        raise SchemaCanaryStorageError(
            f"Schema canary receipt is not an object: {path}",
            submission_state=LLMSubmissionState.NOT_SUBMITTED,
        )
    required = {
        "schema_version",
        "identity",
        "identity_sha256",
        "status",
        "recorded_at",
        "failure",
    }
    failure = payload.get("failure")
    valid_failure = (
        payload.get("status") == "proven"
        and failure is None
    ) or (
        payload.get("status") == "rejected"
        and isinstance(failure, dict)
        and set(failure) == {"category", "submission_state"}
        and failure.get("category") in {item.value for item in LLMFailureCategory}
        and failure.get("submission_state") in {item.value for item in LLMSubmissionState}
    )
    if (
        set(payload) != required
        or payload.get("schema_version") != SCHEMA_CANARY_RECEIPT_VERSION
        or payload.get("identity") != identity.to_json()
        or payload.get("identity_sha256") != identity.sha256
        or payload.get("status") not in {"proven", "rejected"}
        or not isinstance(payload.get("recorded_at"), (int, float))
        or isinstance(payload.get("recorded_at"), bool)
        or not valid_failure
    ):
        raise SchemaCanaryStorageError(
            f"Schema canary receipt is invalid or mismatched: {path}",
            submission_state=LLMSubmissionState.NOT_SUBMITTED,
        )
    return payload


def _write_receipt(
    path: Path,
    payload: Mapping[str, Any],
    *,
    submission_state: LLMSubmissionState,
) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise SchemaCanaryStorageError(
            f"Could not persist schema canary receipt {path}: {exc}",
            submission_state=submission_state,
        ) from exc


def _acquire_lock(
    path: Path, *, cancel_check: Callable[[], bool] | None
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
                raise LLMWorkerCancelled(
                    "LLM call cancelled while waiting for schema canary",
                    submission_state=LLMSubmissionState.NOT_SUBMITTED,
                )
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return handle
            except BlockingIOError:
                time.sleep(0.05)
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                time.sleep(0.05)
    except OSError as exc:
        raise SchemaCanaryStorageError(
            f"Could not lock schema canary receipt {path}: {exc}",
            submission_state=LLMSubmissionState.NOT_SUBMITTED,
        ) from exc


def _release_lock(handle: BinaryIO) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _fsync_parent(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
