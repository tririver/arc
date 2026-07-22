from __future__ import annotations

import hashlib
import math
import re
import time
import os
from pathlib import Path
from collections.abc import Callable
from collections.abc import Mapping
from typing import Any

from .base import LLMFailureCategory, LLMSubmissionState, LLMWorkerError, LLMWorkerTimeout


DEFAULT_IDLE_TIMEOUT_SECONDS = 30 * 60
DEFAULT_REVIEW_INTERVAL_SECONDS = 30 * 60

ProgressCallback = Callable[[dict[str, Any]], None]

_PROVIDER_IDLE_TIMEOUT_KEYS = {
    "codex-cli": "ARC_CODEX_IDLE_TIMEOUT_SECONDS",
    "claude-cli": "ARC_CLAUDE_IDLE_TIMEOUT_SECONDS",
    "kimi-code-cli": "ARC_KIMI_IDLE_TIMEOUT_SECONDS",
}

_REMOVED_TOTAL_TIMEOUT_KEYS = (
    "ARC_LLM_TIMEOUT_SECONDS",
    "ARC_CODEX_TIMEOUT_SECONDS",
    "ARC_CLAUDE_TIMEOUT_SECONDS",
    "ARC_KIMI_TIMEOUT_SECONDS",
)

_EMPTY_HEARTBEAT_RE = re.compile(
    r"^(?:(?:i(?:'m|\s+am)\s+)?still(?:\s+(?:alive|working|running))?|alive|working|"
    r"processing|in\s+progress|please\s+wait|continuing|thinking)(?:[.!…\s]*)$",
    re.IGNORECASE,
)
_EMPTY_HEARTBEAT_WORDS = {
    "i",
    "im",
    "am",
    "still",
    "alive",
    "working",
    "work",
    "running",
    "processing",
    "in",
    "on",
    "it",
    "task",
    "progress",
    "please",
    "wait",
    "continuing",
    "thinking",
}


def resolve_idle_timeout_seconds(
    explicit: float | int | None,
    *,
    env: Mapping[str, str] | None,
    provider: str,
) -> float:
    material = os.environ if env is None else env
    removed = [key for key in _REMOVED_TOTAL_TIMEOUT_KEYS if str(material.get(key) or "").strip()]
    if removed:
        replacements = ", ".join(
            key.replace("_TIMEOUT_SECONDS", "_IDLE_TIMEOUT_SECONDS") for key in removed
        )
        raise ValueError(
            "LLM total-timeout environment variables were removed; use idle timeout "
            f"variables instead: {replacements}"
        )
    value: object = explicit
    name = "idle_timeout_seconds"
    provider_key = _PROVIDER_IDLE_TIMEOUT_KEYS.get(provider)
    if value is None and provider_key and material.get(provider_key) not in {None, ""}:
        value = material[provider_key]
        name = provider_key
    if value is None and material.get("ARC_LLM_IDLE_TIMEOUT_SECONDS") not in {None, ""}:
        value = material["ARC_LLM_IDLE_TIMEOUT_SECONDS"]
        name = "ARC_LLM_IDLE_TIMEOUT_SECONDS"
    if value is None:
        return float(DEFAULT_IDLE_TIMEOUT_SECONDS)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive number")
    return result


class ActivityTracker:
    """Track meaningful provider activity without treating heartbeats as work."""

    def __init__(
        self,
        *,
        provider: str,
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
        review_interval_seconds: float = DEFAULT_REVIEW_INTERVAL_SECONDS,
        progress_callback: ProgressCallback | None = None,
        progress_emit_interval_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if idle_timeout_seconds <= 0:
            raise ValueError("idle_timeout_seconds must be positive")
        if review_interval_seconds <= 0:
            raise ValueError("review_interval_seconds must be positive")
        self.provider = provider
        self.idle_timeout_seconds = float(idle_timeout_seconds)
        self.review_interval_seconds = float(review_interval_seconds)
        self.progress_callback = progress_callback
        self.progress_emit_interval_seconds = max(0.0, float(progress_emit_interval_seconds))
        self._clock = clock
        self._submitted_at: float | None = None
        self._last_activity_at: float | None = None
        self._next_review_at: float | None = None
        self._sequence = 0
        self._review_sequence = 0
        self._seen: set[str] = set()
        self._callback_error: Exception | None = None
        self._last_progress_emit_at: float | None = None
        self._tool_states: dict[str, str] = {}

    def submitted(self) -> None:
        if self._submitted_at is not None:
            return
        now = self._clock()
        self._submitted_at = now
        self._last_activity_at = now
        self._next_review_at = now + self.review_interval_seconds
        # Imported lazily to keep the provider activity layer lightweight and
        # avoid a module cycle during package initialization.
        from arc_llm.attempt_diagnostics import current_attempt_diagnostics

        diagnostics = current_attempt_diagnostics()
        if diagnostics is not None:
            diagnostics.mark_submitted()
        self._emit("submitted", activity_type="provider_submission")

    @property
    def is_submitted(self) -> bool:
        return self._submitted_at is not None

    def record(
        self,
        activity_type: str,
        *,
        text: str | None = None,
        artifact_path: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> bool:
        """Record unique visible progress and return whether idle time was refreshed."""
        if self._submitted_at is None:
            return False
        normalized = " ".join((text or "").split())
        if activity_type in {"heartbeat", "handshake", "reasoning", "stderr"}:
            return False
        if activity_type == "assistant" and _is_empty_heartbeat(normalized):
            return False
        if not normalized and activity_type not in {"tool", "artifact"}:
            return False
        signature_source = f"{activity_type}\0{normalized}\0{artifact_path or ''}\0{detail or {}}"
        signature = hashlib.sha256(signature_source.encode("utf-8")).hexdigest()
        if signature in self._seen:
            return False
        self._seen.add(signature)
        now = self._clock()
        self._last_activity_at = now
        should_emit = (
            activity_type in {"tool", "artifact"}
            or self._last_progress_emit_at is None
            or now - self._last_progress_emit_at >= self.progress_emit_interval_seconds
        )
        if should_emit:
            self._last_progress_emit_at = now
            self._emit(
                "progress",
                activity_type=activity_type,
                text=normalized[:2000] or None,
                artifact_path=artifact_path,
                detail=detail,
            )
        return True

    def record_tool_state(
        self,
        *,
        tool_type: str,
        status: str,
        tool_id: str | None = None,
    ) -> bool:
        """Record only a changed, sanitized tool state; never tool arguments."""
        safe_type = _safe_token(tool_type, default="tool")
        safe_status = _safe_tool_status(status)
        state_key = f"{safe_type}:{_safe_token(tool_id or 'anonymous', default='anonymous')}"
        if self._tool_states.get(state_key) == safe_status:
            return False
        self._tool_states[state_key] = safe_status
        return self.record(
            "tool",
            text=f"{safe_type} {safe_status}",
            detail={"tool_type": safe_type, "status": safe_status},
        )

    def record_artifact(self, path: str | os.PathLike[str]) -> bool:
        """Record an artifact only after verifying that the exact path exists."""
        candidate = Path(path)
        try:
            verified = candidate.resolve(strict=True)
        except OSError:
            return False
        return self.record("artifact", text="artifact saved", artifact_path=str(verified))

    def record_metadata(
        self,
        activity_kind: str,
        *,
        text: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Publish resumability metadata without extending the idle deadline."""
        if self._submitted_at is None:
            return
        self._emit(
            "provider_progress",
            activity_type=activity_kind,
            text=" ".join((text or "").split())[:2000] or None,
            detail=detail,
        )

    def check(self) -> None:
        if self._callback_error is not None:
            raise LLMWorkerError(
                f"Could not persist provider progress: {self._callback_error}",
                retryable=False,
                category=LLMFailureCategory.LOCAL_IO,
                submission_state=LLMSubmissionState.SUBMITTED,
            ) from self._callback_error
        if self._submitted_at is None:
            return
        now = self._clock()
        assert self._last_activity_at is not None
        assert self._next_review_at is not None
        while now >= self._next_review_at:
            self._review_sequence += 1
            self._emit(
                "review_due",
                activity_type="review",
                detail={
                    "review_sequence": self._review_sequence,
                    "idle_seconds": max(0.0, now - self._last_activity_at),
                },
            )
            self._next_review_at += self.review_interval_seconds
        if now - self._last_activity_at >= self.idle_timeout_seconds:
            self._emit(
                "idle_timeout",
                activity_type="timeout",
                detail={"idle_seconds": max(0.0, now - self._last_activity_at)},
            )
            raise LLMWorkerTimeout(
                f"{self.provider} produced no meaningful output for "
                f"{self.idle_timeout_seconds:g} seconds",
                submission_state=LLMSubmissionState.SUBMITTED,
            )

    def _emit(
        self,
        event_type: str,
        *,
        activity_type: str,
        text: str | None = None,
        artifact_path: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        if self.progress_callback is None:
            return
        self._sequence += 1
        event: dict[str, Any] = {
            "schema_version": "arc.llm.progress.v1",
            "event": "provider_progress" if event_type in {"progress", "provider_progress"} else event_type,
            "provider": self.provider,
            "sequence": self._sequence,
            "activity_kind": activity_type,
            "substantive": event_type == "progress",
            "monotonic_at": self._clock(),
        }
        if text:
            event["summary"] = text
        if artifact_path:
            event["artifact_paths"] = [artifact_path]
        if detail:
            event.update(detail)
        try:
            self.progress_callback(event)
        except Exception as exc:
            # Losing the recovery journal must stop a paid long-running call.
            self._callback_error = exc


def _safe_token(value: str, *, default: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value))[:80].strip("_")
    return token or default


def _is_empty_heartbeat(value: str) -> bool:
    if _EMPTY_HEARTBEAT_RE.fullmatch(value):
        return True
    words = re.findall(r"[A-Za-z]+", value.lower().replace("'", ""))
    return bool(words) and set(words) <= _EMPTY_HEARTBEAT_WORDS


def _safe_tool_status(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "in_progress": "running",
        "started": "running",
        "starting": "running",
        "success": "completed",
        "succeeded": "completed",
        "done": "completed",
        "complete": "completed",
        "failed": "failed",
        "error": "failed",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "pending": "pending",
        "queued": "pending",
    }
    return aliases.get(normalized, normalized if normalized in {"running", "completed"} else "updated")
