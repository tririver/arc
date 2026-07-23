from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from .providers.base import (
    LLMAbortScope,
    LLMFailureCategory,
    LLMSubmissionState,
    LLMWorkerError,
)


PROGRESS_SCHEMA_VERSION = "arc.llm.progress.v1"
MAX_PROGRESS_SUMMARY_CHARS = 4096
MAX_ARTIFACT_PATHS = 32


class ProgressJournalError(LLMWorkerError):
    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            retryable=False,
            category=LLMFailureCategory.LOCAL_IO,
            abort_scope=LLMAbortScope.BATCH,
            submission_state=LLMSubmissionState.UNKNOWN,
        )


class ProgressJournal:
    """Persist bounded provider progress and optionally forward it to a host."""

    def __init__(
        self,
        *,
        artifact_dir: Path | None,
        call_label: str | None,
        provider: str,
        callback: Callable[[dict[str, Any]], None] | None,
        identity: Mapping[str, Any] | None = None,
        submission_callback: Callable[[], None] | None = None,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.call_label = call_label
        self.provider = provider
        self.callback = callback
        self.identity = {key: value for key, value in dict(identity or {}).items() if value is not None}
        self.submission_callback = submission_callback
        self._lock = threading.Lock()
        self._sequence = 0
        self._native_session_id: str | None = None
        self._submission_recorded = False

    def __call__(self, raw_event: Mapping[str, Any]) -> None:
        with self._lock:
            if (
                not self._submission_recorded
                and self.submission_callback is not None
            ):
                self.submission_callback()
                self._submission_recorded = True
            self._sequence += 1
            event = _normalize_event(
                raw_event,
                sequence=self._sequence,
                call_label=self.call_label,
                provider=self.provider,
            )
            event.update(self.identity)
            if event.get("native_session_id"):
                self._native_session_id = str(event["native_session_id"])
            elif self._native_session_id:
                event["native_session_id"] = self._native_session_id
                event["resumable"] = True
            if self.artifact_dir is not None:
                self._persist(event)
            if self.callback is not None:
                self.callback(dict(event))

    def bind_submission_callback(self, callback: Callable[[], None]) -> None:
        """Bind the checkpoint barrier before invoking the provider."""

        with self._lock:
            self.submission_callback = callback
            self._submission_recorded = False

    def record_submission_barrier(self) -> None:
        """Persist the bound submission barriers exactly once."""

        with self._lock:
            if self._submission_recorded:
                return
            if self.submission_callback is not None:
                self.submission_callback()
            self._submission_recorded = True

    def update_identity(self, **values: Any) -> None:
        """Attach identity fields learned only after the session lock is held."""

        with self._lock:
            self.identity.update(
                {key: value for key, value in values.items() if value is not None}
            )

    def _persist(self, event: Mapping[str, Any]) -> None:
        assert self.artifact_dir is not None
        try:
            self.artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            journal = self.artifact_dir / "progress.jsonl"
            encoded = json.dumps(dict(event), ensure_ascii=False, separators=(",", ":")) + "\n"
            with journal.open("a", encoding="utf-8") as handle:
                os.chmod(journal, 0o600)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            _atomic_write_json(self.artifact_dir / "latest_progress.json", event)
        except OSError as exc:
            raise ProgressJournalError(
                f"Could not persist LLM progress journal in {self.artifact_dir}: {exc}"
            ) from exc


def _normalize_event(
    raw_event: Mapping[str, Any],
    *,
    sequence: int,
    call_label: str | None,
    provider: str,
) -> dict[str, Any]:
    event_name = str(raw_event.get("event") or raw_event.get("type") or "provider_progress")
    if event_name == "progress":
        event_name = "provider_progress"
    summary = str(
        raw_event.get("summary")
        or raw_event.get("excerpt")
        or raw_event.get("message")
        or ""
    ).strip()
    paths = raw_event.get("artifact_paths")
    if not isinstance(paths, list):
        single = raw_event.get("artifact_path")
        paths = [single] if isinstance(single, str) and single else []
    normalized: dict[str, Any] = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "event": event_name,
        "provider": str(raw_event.get("provider") or provider),
        "sequence": sequence,
        "call_label": call_label,
        "activity_kind": str(
            raw_event.get("activity_kind") or raw_event.get("activity_type") or "provider"
        ),
        "substantive": bool(raw_event.get("substantive", event_name == "provider_progress")),
        "updated_at": str(raw_event.get("updated_at") or _utc_now()),
    }
    if summary:
        normalized["summary"] = summary[:MAX_PROGRESS_SUMMARY_CHARS]
    if paths:
        normalized["artifact_paths"] = [str(item) for item in paths[:MAX_ARTIFACT_PATHS]]
    for key in (
        "review_sequence",
        "elapsed_seconds",
        "idle_seconds",
        "native_session_id",
        "phase",
        "resumable",
    ):
        if key in raw_event:
            normalized[key] = raw_event[key]
    return normalized


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
