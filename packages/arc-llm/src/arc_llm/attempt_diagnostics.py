from __future__ import annotations

import contextlib
import contextvars
import gzip
import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .providers.base import (
    LLMAbortScope,
    LLMFailureCategory,
    LLMSubmissionState,
    LLMWorkerError,
    failure_disposition,
)


ATTEMPT_DIAGNOSTIC_SCHEMA_VERSION = "arc.llm.attempt_diagnostic.v1"
MAX_ATTEMPT_STREAM_BYTES = 1024 * 1024
COMPRESS_ATTEMPT_STREAM_BYTES = 64 * 1024
MAX_TIMELINE_EVENTS = 4096
MAX_TIMELINE_EVENT_BYTES = 16 * 1024
MAX_TIMELINE_BYTES = 2 * 1024 * 1024
MAX_ERROR_MESSAGE_CHARS = 4096
MAX_PROVIDER_BYTES = 128
MAX_MODEL_BYTES = 512
MAX_CALL_LABEL_BYTES = 512
MAX_NATIVE_SESSION_ID_BYTES = 512
MAX_CANDIDATE_SOURCE_BYTES = 128
MAX_RESPONSE_CANDIDATES = 256

_SECRET_KEY_PARTS = (
    "api_key",
    "api-key",
    "apikey",
    "authorization",
    "auth_token",
    "credential",
    "cookie",
    "password",
    "private_key",
    "secret",
    "session_token",
    "token",
)
_CONFIG_ENV_PARTS = ("_CONFIG", "_PROFILE", "_ENV_JSON", "_HEADERS")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:api[-_]?key|authorization|credential|cookie|password|private[-_]?key|"
    r"secret|session[-_]?token|token)\b\s*[:=]\s*)([^\s,;\"']+|\"[^\"]*\"|'[^']*')"
)
_BEARER_RE = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{6,}")
_URI_CREDENTIAL_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^/@\s:]+:[^/@\s]+@")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_PROVIDER_KEY_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|AKIA[A-Z0-9]{12,})\b")

_CURRENT_ATTEMPT: contextvars.ContextVar[AttemptDiagnostics | None] = contextvars.ContextVar(
    "arc_llm_attempt_diagnostics", default=None
)


@dataclass(frozen=True)
class AttemptDiagnosticRef:
    path: str
    sha256: str


class AttemptDiagnosticsError(LLMWorkerError):
    def __init__(self, message: str, *, submission_state: LLMSubmissionState) -> None:
        super().__init__(
            message,
            retryable=False,
            category=LLMFailureCategory.LOCAL_IO,
            abort_scope=LLMAbortScope.BATCH,
            submission_state=submission_state,
        )


class DiagnosticRedactor:
    """Redact secret-shaped fields and values before durable serialization."""

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        material = dict(env or {})
        sensitive: set[str] = set()
        for key, raw_value in material.items():
            value = str(raw_value or "")
            lowered = str(key).lower()
            if not value:
                continue
            is_config = any(part in str(key).upper() for part in _CONFIG_ENV_PARTS)
            if _is_secret_key(lowered) or is_config:
                sensitive.add(value)
                if is_config:
                    with contextlib.suppress(json.JSONDecodeError):
                        decoded = json.loads(value)
                        sensitive.update(
                            item for item in _string_leaves(decoded) if item
                        )
            # Long environment values include PATH-like and provider-specific
            # opaque configuration. Redacting an echoed value is safer than
            # trying to enumerate every host's credential naming convention.
            elif len(value) >= 12:
                sensitive.add(value)
        self._sensitive_values = tuple(sorted(sensitive, key=len, reverse=True))

    def text(self, value: object) -> str:
        rendered = str(value)
        for secret in self._sensitive_values:
            rendered = rendered.replace(secret, "[REDACTED_ENV]")
        rendered = _URI_CREDENTIAL_RE.sub(r"\1[REDACTED]@", rendered)
        rendered = _BEARER_RE.sub(lambda match: f"{match.group(1)} [REDACTED]", rendered)
        rendered = _SECRET_ASSIGNMENT_RE.sub(r"\1[REDACTED]", rendered)
        rendered = _JWT_RE.sub("[REDACTED_TOKEN]", rendered)
        rendered = _PROVIDER_KEY_RE.sub("[REDACTED_KEY]", rendered)
        return rendered

    def value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, item in value.items():
                raw_key = str(key)
                safe_key = self.text(raw_key)
                candidate = safe_key
                collision = 1
                while candidate in result:
                    collision += 1
                    candidate = f"{safe_key}#{collision}"
                result[candidate] = (
                    "[REDACTED]"
                    if _is_secret_key(raw_key.lower())
                    else self.value(item)
                )
            return result
        if isinstance(value, (list, tuple)):
            return [self.value(item) for item in value]
        if isinstance(value, str):
            return self.text(value)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return self.text(value)


class _BoundedStream:
    def __init__(self, *, redactor: DiagnosticRedactor, max_bytes: int) -> None:
        self.redactor = redactor
        self.max_bytes = max(1024, int(max_bytes))
        self._head_limit = self.max_bytes // 2
        self._tail_limit = self.max_bytes - self._head_limit
        self._head = bytearray()
        self._tail = bytearray()
        self.observed_bytes = 0
        self.redacted_bytes = 0
        self.lossless = True

    def append(self, text: object) -> None:
        self.append_sanitized(text, self.redactor.text(text))

    def append_sanitized(self, raw_text: object, sanitized_text: object) -> None:
        raw = str(raw_text).encode("utf-8", errors="replace")
        self.observed_bytes += len(raw)
        sanitized = str(sanitized_text).encode("utf-8", errors="replace")
        self.lossless = self.lossless and raw == sanitized
        self.redacted_bytes += len(sanitized)
        remaining = self._head_limit - len(self._head)
        if remaining > 0:
            self._head.extend(sanitized[:remaining])
            sanitized = sanitized[remaining:]
        if sanitized:
            self._tail.extend(sanitized)
            if len(self._tail) > self._tail_limit:
                del self._tail[: len(self._tail) - self._tail_limit]

    @property
    def truncated(self) -> bool:
        return self.redacted_bytes > len(self._head) + len(self._tail)

    def bytes(self) -> bytes:
        if not self.truncated:
            return bytes(self._head + self._tail)
        omitted = self.redacted_bytes - len(self._head) - len(self._tail)
        marker = f"\n[... {omitted} sanitized byte(s) truncated ...]\n".encode("utf-8")
        content_budget = self.max_bytes - len(marker)
        head_budget = content_budget // 2
        tail_budget = content_budget - head_budget
        return bytes(self._head[:head_budget]) + marker + bytes(self._tail[-tail_budget:])


class AttemptDiagnostics:
    """Append during one provider attempt, then finalize exactly once."""

    def __init__(
        self,
        artifact_dir: Path,
        *,
        provider: str,
        model: str | None,
        fallback_index: int,
        attempt: int,
        call_label: str | None,
        env: Mapping[str, str] | None,
        max_stream_bytes: int = MAX_ATTEMPT_STREAM_BYTES,
    ) -> None:
        self.artifact_dir = artifact_dir.resolve(strict=False)
        self.redactor = DiagnosticRedactor(env)
        self.provider = _bounded_text(self.redactor.text(provider), MAX_PROVIDER_BYTES)
        self.model = (
            _bounded_text(self.redactor.text(model), MAX_MODEL_BYTES)
            if model is not None
            else None
        )
        self.fallback_index = fallback_index
        self.attempt = attempt
        self.call_label = (
            _bounded_text(self.redactor.text(call_label), MAX_CALL_LABEL_BYTES)
            if call_label is not None
            else None
        )
        self.attempt_id = uuid.uuid4().hex
        safe_provider = _safe_component(self.provider, "provider")
        safe_label = _safe_component(self.call_label or "call", "call")[:48]
        directory_name = (
            f"{safe_label}-{safe_provider}-f{fallback_index:02d}-a{attempt:02d}-"
            f"{self.attempt_id[:12]}"
        )
        attempts_root = self.artifact_dir / "attempts"
        try:
            attempts_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.path = attempts_root / directory_name
            self.path.mkdir(mode=0o700)
        except OSError as exc:
            raise AttemptDiagnosticsError(
                f"Could not create LLM attempt diagnostics: {exc}",
                submission_state=LLMSubmissionState.NOT_SUBMITTED,
            ) from exc
        self._timeline_path = self.path / "timeline.jsonl"
        self._lock = threading.RLock()
        self._timeline_sequence = 0
        self._timeline_events = 0
        self._timeline_dropped = 0
        self._timeline_bytes = 0
        self._streams = {
            "stdout": _BoundedStream(redactor=self.redactor, max_bytes=max_stream_bytes),
            "raw_events": _BoundedStream(redactor=self.redactor, max_bytes=max_stream_bytes),
            "stderr": _BoundedStream(redactor=self.redactor, max_bytes=max_stream_bytes),
            "response_candidates": _BoundedStream(
                redactor=self.redactor, max_bytes=max_stream_bytes
            ),
        }
        self._candidate_metadata: list[dict[str, Any]] = []
        self._candidate_count = 0
        self._candidate_dropped = 0
        self._native_session_id: str | None = None
        self._checkpoint_identity: str | None = None
        self._checkpoint_binding: dict[str, Any] | None = None
        self._submission_state = LLMSubmissionState.NOT_SUBMITTED
        self._started_at = _utc_now()
        self._started_monotonic = time.monotonic()
        self._finalized_ref: AttemptDiagnosticRef | None = None
        self._persistence_error: AttemptDiagnosticsError | None = None
        self.event("attempt_started")
        if self._persistence_error is not None:
            raise self._persistence_error

    def event(self, event: str, **details: Any) -> None:
        with self._lock:
            if self._finalized_ref is not None:
                return
            if (
                self._timeline_events >= MAX_TIMELINE_EVENTS
                or self._timeline_bytes >= MAX_TIMELINE_BYTES
            ):
                self._timeline_dropped += 1
                return
            self._timeline_sequence += 1
            payload = {
                "sequence": self._timeline_sequence,
                "event": _safe_event_name(event),
                "at": _utc_now(),
                "elapsed_seconds": max(0.0, time.monotonic() - self._started_monotonic),
                "details": self.redactor.value(details),
            }
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
            encoded_bytes = encoded.encode("utf-8")
            if len(encoded_bytes) > MAX_TIMELINE_EVENT_BYTES:
                details_json = json.dumps(
                    self.redactor.value(details),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                payload = {
                    key: value for key, value in payload.items() if key in {
                        "sequence", "event", "at", "elapsed_seconds"
                    }
                }
                payload.update(
                    {
                        "details_truncated": True,
                        "details_sha256": hashlib.sha256(
                            details_json.encode("utf-8")
                        ).hexdigest(),
                        "details_bytes": len(details_json.encode("utf-8")),
                    }
                )
                encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
                encoded_bytes = encoded.encode("utf-8")
            if len(encoded_bytes) > MAX_TIMELINE_EVENT_BYTES:
                raise AssertionError("bounded attempt timeline event exceeded its byte cap")
            if self._timeline_bytes + len(encoded_bytes) > MAX_TIMELINE_BYTES:
                self._timeline_dropped += 1
                return
            try:
                with self._timeline_path.open("a", encoding="utf-8") as handle:
                    os.chmod(self._timeline_path, 0o600)
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                self._persistence_error = AttemptDiagnosticsError(
                    f"Could not append LLM attempt timeline: {exc}",
                    submission_state=self._submission_state,
                )
                return
            self._timeline_events += 1
            self._timeline_bytes += len(encoded_bytes)

    def mark_submitted(self) -> None:
        with self._lock:
            if self._submission_state == LLMSubmissionState.SUBMITTED:
                return
            self._submission_state = LLMSubmissionState.SUBMITTED
        self.event("submission_barrier_crossed", submission_state="submitted")

    def capture_stdout(self, text: str) -> None:
        self._capture("stdout", text)
        try:
            event = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(event, Mapping):
            self.capture_raw_event(event)
            native_id = _native_session_id(event)
            if native_id:
                self.record_native_session_id(native_id)

    def capture_stderr(self, text: str) -> None:
        self._capture("stderr", text)

    def capture_raw_event(self, event: Mapping[str, Any]) -> None:
        raw = json.dumps(dict(event), ensure_ascii=False, sort_keys=True, default=str) + "\n"
        sanitized = self.redactor.value(dict(event))
        rendered = json.dumps(
            sanitized, ensure_ascii=False, sort_keys=True, default=str,
        ) + "\n"
        with self._lock:
            if self._finalized_ref is None:
                self._streams["raw_events"].append_sanitized(raw, rendered)

    def record_native_session_id(self, native_session_id: str | None) -> None:
        if not native_session_id:
            return
        sanitized = _bounded_text(
            self.redactor.text(native_session_id), MAX_NATIVE_SESSION_ID_BYTES
        )
        with self._lock:
            if sanitized == self._native_session_id:
                return
            self._native_session_id = sanitized
        self.event("native_session_observed", native_session_id=sanitized)

    def bind_checkpoint_identity(self, checkpoint_identity: str) -> None:
        """Legacy identity-only fixture binding; production must use bind_checkpoint."""

        value = str(checkpoint_identity or "")
        if not value:
            raise AttemptDiagnosticsError(
                "Call checkpoint identity is empty",
                submission_state=self._submission_state,
            )
        with self._lock:
            if self._finalized_ref is not None:
                raise AttemptDiagnosticsError(
                    "Attempt diagnostics already finalized",
                    submission_state=self._submission_state,
                )
            if self._checkpoint_identity not in {None, value}:
                raise AttemptDiagnosticsError(
                    "Attempt diagnostics checkpoint identity changed",
                    submission_state=self._submission_state,
                )
            self._checkpoint_identity = value

    def bind_checkpoint(self, binding: Mapping[str, Any]) -> None:
        """Bind this attempt to one exact persisted checkpoint recipe and address."""

        try:
            detached = _canonical_object(binding, description="checkpoint binding")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AttemptDiagnosticsError(
                "Call checkpoint binding is not canonical JSON",
                submission_state=self._submission_state,
            ) from exc
        if detached.get("schema_version") != "arc.llm.checkpoint_recomputation_binding.v1":
            raise AttemptDiagnosticsError(
                "Call checkpoint binding schema is invalid",
                submission_state=self._submission_state,
            )
        checkpoint_identity = detached.get("checkpoint_identity")
        checkpoint_path_value = detached.get("checkpoint_path")
        if not isinstance(checkpoint_identity, str) or not checkpoint_identity:
            raise AttemptDiagnosticsError(
                "Call checkpoint binding identity is empty",
                submission_state=self._submission_state,
            )
        if not isinstance(checkpoint_path_value, str) or not checkpoint_path_value:
            raise AttemptDiagnosticsError(
                "Call checkpoint binding path is empty",
                submission_state=self._submission_state,
            )
        checkpoint_path = Path(checkpoint_path_value).expanduser().resolve(strict=False)
        try:
            relative_path = checkpoint_path.relative_to(self.artifact_dir).as_posix()
        except ValueError as exc:
            raise AttemptDiagnosticsError(
                "Call checkpoint binding is outside the attempt artifact directory",
                submission_state=self._submission_state,
            ) from exc
        if not relative_path or relative_path.startswith("../"):
            raise AttemptDiagnosticsError(
                "Call checkpoint binding path is invalid",
                submission_state=self._submission_state,
            )
        for field in (
            "request_digest",
            "request_recipe_sha256",
            "prompt_sha256",
            "call_label_sha256",
        ):
            if not _is_sha256(detached.get(field)):
                raise AttemptDiagnosticsError(
                    f"Call checkpoint binding {field} is invalid",
                    submission_state=self._submission_state,
                )
        schema_sha256 = detached.get("schema_sha256")
        if schema_sha256 is not None and not _is_sha256(schema_sha256):
            raise AttemptDiagnosticsError(
                "Call checkpoint binding schema_sha256 is invalid",
                submission_state=self._submission_state,
            )
        generation = detached.get("generation")
        if generation is not None and (type(generation) is not int or generation < 1):
            raise AttemptDiagnosticsError(
                "Call checkpoint binding generation is invalid",
                submission_state=self._submission_state,
            )
        for authorization_field in (
            "initial_native_authorization", "native_resume_authorization",
        ):
            authorization = detached.get(authorization_field)
            if authorization is None:
                continue
            if not isinstance(authorization, Mapping) or set(authorization) != {
                "control_address", "session_key", "logical_unit", "generation",
                "idempotency_key",
            }:
                raise AttemptDiagnosticsError(
                    f"Call checkpoint binding {authorization_field} is incomplete",
                    submission_state=self._submission_state,
                )
            if (
                any(
                    not isinstance(authorization.get(field), str)
                    or not authorization.get(field)
                    for field in (
                        "control_address", "session_key", "logical_unit",
                        "idempotency_key",
                    )
                )
                or type(authorization.get("generation")) is not int
                or int(authorization["generation"]) < 1
            ):
                raise AttemptDiagnosticsError(
                    f"Call checkpoint binding {authorization_field} is invalid",
                    submission_state=self._submission_state,
                )
        detached["checkpoint_path"] = relative_path
        with self._lock:
            if self._finalized_ref is not None:
                raise AttemptDiagnosticsError(
                    "Attempt diagnostics already finalized",
                    submission_state=self._submission_state,
                )
            if self._checkpoint_binding is not None and self._checkpoint_binding != detached:
                raise AttemptDiagnosticsError(
                    "Attempt diagnostics checkpoint binding changed",
                    submission_state=self._submission_state,
                )
            if self._checkpoint_identity not in {None, checkpoint_identity}:
                raise AttemptDiagnosticsError(
                    "Attempt diagnostics checkpoint identity changed",
                    submission_state=self._submission_state,
                )
            self._checkpoint_identity = checkpoint_identity
            self._checkpoint_binding = detached

    def record_candidate(
        self,
        value: Any,
        *,
        source: str,
        ordinal: int | None = None,
        canonical_sha256: str | None = None,
    ) -> None:
        with self._lock:
            if ordinal is None:
                self._candidate_count += 1
                sequence = self._candidate_count
            else:
                sequence = int(ordinal)
                if sequence <= self._candidate_count:
                    raise ValueError("diagnostic candidate ordinals must increase")
                self._candidate_count = sequence
            if len(self._candidate_metadata) >= MAX_RESPONSE_CANDIDATES:
                self._candidate_dropped += 1
                if self._candidate_dropped == 1:
                    self.event(
                        "parsed_response_candidates_truncated",
                        retained=MAX_RESPONSE_CANDIDATES,
                    )
                return
            # Keep reservation, serialization, metadata, and bounded stream
            # append under one lock so concurrent captures cannot overrun the
            # retained-candidate cap or reorder their sequence numbers.
            sanitized = self.redactor.value(value)
            encoded = json.dumps(
                sanitized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            digest = (
                str(canonical_sha256)
                if canonical_sha256 is not None
                else hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            )
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("diagnostic candidate SHA-256 must be lowercase hex")
            safe_source = _bounded_text(
                self.redactor.text(source), MAX_CANDIDATE_SOURCE_BYTES
            )
            self._candidate_metadata.append(
                {
                    "sequence": sequence,
                    "source": _safe_component(safe_source, "parsed_response"),
                    "sha256": digest,
                    "bytes": len(encoded.encode("utf-8")),
                    "value_type": _bounded_text(type(value).__name__, 128),
                }
            )
            self._streams["response_candidates"].append(
                json.dumps(
                    {"sequence": sequence, "source": safe_source, "value": sanitized},
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
            self.event(
                "parsed_response_candidate",
                sequence=sequence,
                sha256=digest,
                source=safe_source,
            )

    def finalize(
        self,
        *,
        outcome: str,
        error: BaseException | None = None,
        native_session_id: str | None = None,
    ) -> AttemptDiagnosticRef:
        with self._lock:
            if self._finalized_ref is not None:
                return self._finalized_ref
            if self._persistence_error is not None:
                raise self._persistence_error
            if native_session_id:
                self.record_native_session_id(native_session_id)
            disposition = failure_disposition(error) if error is not None else None
            if disposition is not None:
                self._submission_state = disposition.submission_state
            if error is not None:
                if disposition and disposition.category == LLMFailureCategory.CANCELLED:
                    self.event("cancellation_observed")
                elif disposition and disposition.category == LLMFailureCategory.TIMEOUT:
                    self.event("timeout_observed")
            self.event("attempt_outcome", outcome=outcome)
            try:
                streams = {
                    name: self._write_stream(name, stream)
                    for name, stream in self._streams.items()
                    if stream.observed_bytes or name in {"stdout", "stderr", "raw_events"}
                }
                timeline = _file_receipt(self._timeline_path)
            except OSError as exc:
                raise AttemptDiagnosticsError(
                    f"Could not persist LLM attempt streams: {exc}",
                    submission_state=self._submission_state,
                ) from exc
            error_payload: dict[str, Any] | None = None
            if error is not None:
                error_payload = {
                    "type": _bounded_text(type(error).__name__, 128),
                    "message": self.redactor.text(error)[:MAX_ERROR_MESSAGE_CHARS],
                    "category": (
                        disposition.category.value
                        if disposition is not None
                        else LLMFailureCategory.UNKNOWN.value
                    ),
                    "abort_scope": disposition.abort_scope.value if disposition is not None else None,
                    "retryable": disposition.retryable if disposition is not None else None,
                }
            payload = {
                "schema_version": ATTEMPT_DIAGNOSTIC_SCHEMA_VERSION,
                "attempt_id": self.attempt_id,
                "provider": self.provider,
                "model": self.model,
                "fallback_index": self.fallback_index,
                "attempt": self.attempt,
                "call_label": self.call_label,
                "started_at": self._started_at,
                "finished_at": _utc_now(),
                "submission_state": self._submission_state.value,
                "native_session_id": self._native_session_id,
                "checkpoint_identity": self._checkpoint_identity,
                "checkpoint_path": (
                    self._checkpoint_binding.get("checkpoint_path")
                    if self._checkpoint_binding is not None else None
                ),
                "checkpoint_binding": self._checkpoint_binding,
                "idempotency_key": (
                    self._checkpoint_binding.get("idempotency_key")
                    if self._checkpoint_binding is not None else None
                ),
                "session_key": (
                    self._checkpoint_binding.get("session_key")
                    if self._checkpoint_binding is not None else None
                ),
                "generation": (
                    self._checkpoint_binding.get("generation")
                    if self._checkpoint_binding is not None else None
                ),
                "prompt_sha256": (
                    self._checkpoint_binding.get("prompt_sha256")
                    if self._checkpoint_binding is not None else None
                ),
                "schema_sha256": (
                    self._checkpoint_binding.get("schema_sha256")
                    if self._checkpoint_binding is not None else None
                ),
                "call_label_sha256": (
                    self._checkpoint_binding.get("call_label_sha256")
                    if self._checkpoint_binding is not None else None
                ),
                "outcome": _bounded_text(self.redactor.text(outcome), 64),
                "error": error_payload,
                "timeline": {
                    **timeline,
                    "events": self._timeline_events,
                    "dropped_events": self._timeline_dropped,
                },
                "streams": streams,
                "parsed_response_candidates": list(self._candidate_metadata),
                "parsed_response_candidate_count": self._candidate_count,
                "parsed_response_candidates_dropped": self._candidate_dropped,
            }
            source_manifest = {
                "schema_version": "arc.llm.attempt_immutable_source.v1",
                "checkpoint_binding_sha256": (
                    _canonical_sha256(self._checkpoint_binding)
                    if self._checkpoint_binding is not None else None
                ),
                "timeline": timeline,
                "streams": streams,
                "parsed_response_candidates": list(self._candidate_metadata),
                "parsed_response_candidate_count": self._candidate_count,
                "parsed_response_candidates_dropped": self._candidate_dropped,
            }
            payload["immutable_source"] = {
                "manifest": source_manifest,
                "manifest_sha256": _canonical_sha256(source_manifest),
            }
            record_path = self.path / "record.json"
            try:
                _exclusive_json_write(record_path, payload)
            except OSError as exc:
                raise AttemptDiagnosticsError(
                    f"Could not finalize LLM attempt diagnostics: {exc}",
                    submission_state=self._submission_state,
                ) from exc
            reference = AttemptDiagnosticRef(
                path=record_path.relative_to(self.artifact_dir).as_posix(),
                sha256=_sha256_file(record_path),
            )
            self._make_read_only()
            self._finalized_ref = reference
            return reference

    def _capture(self, name: str, text: str) -> None:
        with self._lock:
            if self._finalized_ref is not None:
                return
            self._streams[name].append(text)

    def _write_stream(self, name: str, stream: _BoundedStream) -> dict[str, Any]:
        raw = stream.bytes()
        compressed = (
            stream.observed_bytes >= COMPRESS_ATTEMPT_STREAM_BYTES
            or len(raw) >= COMPRESS_ATTEMPT_STREAM_BYTES
        )
        compressed_payload = gzip.compress(raw, compresslevel=6, mtime=0) if compressed else raw
        if compressed and len(compressed_payload) > self._streams[name].max_bytes:
            compressed = False
            payload = raw
        else:
            payload = compressed_payload
        if len(payload) > stream.max_bytes:
            raise AssertionError("bounded attempt stream exceeded its byte cap")
        suffix = ".jsonl" if name in {"raw_events", "response_candidates"} else ".txt"
        filename = f"{name}{suffix}{'.gz' if compressed else ''}"
        path = self.path / filename
        _exclusive_bytes_write(path, payload)
        return {
            "path": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "observed_bytes": stream.observed_bytes,
            "sanitized_bytes": stream.redacted_bytes,
            "stored_bytes": len(payload),
            "truncated": stream.truncated,
            "compression": "gzip" if compressed else "none",
            "lossless": stream.lossless,
        }

    def _make_read_only(self) -> None:
        try:
            for child in self.path.iterdir():
                child.chmod(0o400)
            self.path.chmod(0o500)
        except OSError as exc:
            raise AttemptDiagnosticsError(
                f"Could not make LLM attempt diagnostics immutable: {exc}",
                submission_state=self._submission_state,
            ) from exc


@contextlib.contextmanager
def bind_attempt_diagnostics(
    diagnostics: AttemptDiagnostics | None,
) -> Iterator[AttemptDiagnostics | None]:
    token = _CURRENT_ATTEMPT.set(diagnostics)
    try:
        yield diagnostics
    finally:
        _CURRENT_ATTEMPT.reset(token)


def current_attempt_diagnostics() -> AttemptDiagnostics | None:
    return _CURRENT_ATTEMPT.get()


def sanitize_diagnostic_text(value: object, env: Mapping[str, str] | None = None) -> str:
    return DiagnosticRedactor(env).text(value)


def _is_secret_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part.replace("-", "_") in normalized for part in _SECRET_KEY_PARTS)


def _string_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _string_leaves(child)]
    if isinstance(value, list):
        return [item for child in value for item in _string_leaves(child)]
    return []


def _native_session_id(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("native_session_id", "session_id", "sessionId", "thread_id", "threadId"):
            candidate = value.get(key)
            if isinstance(candidate, (str, int)) and str(candidate):
                return str(candidate)
        for item in value.values():
            candidate = _native_session_id(item)
            if candidate:
                return candidate
    elif isinstance(value, list):
        for item in value:
            candidate = _native_session_id(item)
            if candidate:
                return candidate
    return None


def _safe_component(value: object, default: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._-")
    return rendered or default


def _safe_event_name(value: object) -> str:
    rendered = _safe_component(value, "provider_event")
    if len(rendered.encode("utf-8")) <= 128:
        return rendered
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]
    return f"{rendered[:96]}-{digest}"


def _bounded_text(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return value
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    suffix = f"...[sha256:{digest}]"
    budget = max(0, max_bytes - len(suffix.encode("utf-8")))
    prefix = encoded[:budget].decode("utf-8", errors="ignore")
    return prefix + suffix


def _canonical_object(value: Mapping[str, Any], *, description: str) -> dict[str, Any]:
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError(f"{description} is not an object")
    return decoded


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _exclusive_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str) + "\n"
    _exclusive_bytes_write(path, encoded.encode("utf-8"))


def _exclusive_bytes_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise


def _file_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
