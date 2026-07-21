from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from threading import BoundedSemaphore, Lock
from typing import Any, Iterator, Mapping


STATEFUL_TURN_VERSION = "arc.companion.stateful-turn.v2"
DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000
ROLLOVER_FRACTION = 0.70


class StatefulSessionError(RuntimeError):
    """A paid stateful turn completed but its session invariant was not accepted."""


class LLMSubmissionLimiter:
    """One build-wide permit pool around provider submissions themselves."""

    def __init__(self, workers: int) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self._semaphore = BoundedSemaphore(workers)

    @contextmanager
    def permit(self) -> Iterator[None]:
        with self._semaphore:
            yield


@dataclass
class StatefulPromptStream:
    """Construct one static bootstrap followed by segment-local delta turns."""

    chapter_id: str
    lane: str
    fixed_rules: Mapping[str, Any]
    static_context: Mapping[str, Any]
    generation: int = 1
    continuity_capsule: Mapping[str, Any] | None = None
    _turns: int = 0

    def request(
        self,
        request: str,
        *,
        cursor: str,
        source_sha256: str,
        current_payload: Mapping[str, Any],
    ) -> str:
        segment_payload = {"request": request, **dict(current_payload)}
        if self._turns == 0:
            payload = {
                "schema_version": STATEFUL_TURN_VERSION,
                "turn_kind": "generation_bootstrap",
                "generation": self.generation,
                "chapter_id": self.chapter_id,
                "lane": self.lane,
                "fixed_rules": dict(self.fixed_rules),
                "static_context": dict(self.static_context),
                "continuity_capsule": dict(self.continuity_capsule or {}),
                "cursor": cursor,
                "source_sha256": source_sha256,
                "current_payload": segment_payload,
            }
        else:
            payload = {
                "schema_version": STATEFUL_TURN_VERSION,
                "turn_kind": "delta",
                "cursor": cursor,
                "source_sha256": source_sha256,
                "current_payload": segment_payload,
            }
        self._turns += 1
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass
class ContextRolloverBudget:
    """Track one native generation and request rollover only at accepted boundaries."""

    context_window_tokens: int = field(default_factory=lambda: _configured_context_window())
    input_tokens: int = 0
    output_tokens: int = 0
    _has_known_input: bool = False
    _max_known_footprint: int = 0
    _unmeasured_tokens: int = 0

    @property
    def threshold(self) -> int:
        return int(self.context_window_tokens * ROLLOVER_FRACTION)

    def record(self, usage: Any, *, prompt_bytes: int | None = None) -> None:
        payload = usage.to_json() if hasattr(usage, "to_json") else usage
        payload = payload if isinstance(payload, Mapping) else {}
        known_input = payload.get("total_input_tokens", payload.get("input_tokens"))
        known_output = payload.get("output_tokens")
        if isinstance(known_input, int):
            # Stateful providers commonly report the complete turn input,
            # including native history.  The maximum is therefore the best
            # context-size observation and avoids double-counting history.
            self.input_tokens = max(self.input_tokens, max(0, known_input))
            self._has_known_input = True
            turn_output = max(0, known_output) if isinstance(known_output, int) else 0
            self._max_known_footprint = max(
                self._max_known_footprint, max(0, known_input) + turn_output
            )
            # A later complete-input observation already contains earlier turns.
            self._unmeasured_tokens = 0
        elif prompt_bytes:
            estimate = max(1, prompt_bytes // 4)
            self.input_tokens += estimate
            if self._has_known_input:
                self._unmeasured_tokens += estimate
        if isinstance(known_output, int):
            if isinstance(known_input, int):
                self.output_tokens = max(self.output_tokens, max(0, known_output))
            else:
                self.output_tokens += max(0, known_output)
                if self._has_known_input:
                    self._unmeasured_tokens += max(0, known_output)

    def rollover_due(self) -> bool:
        if self._has_known_input:
            return self._max_known_footprint + self._unmeasured_tokens >= self.threshold
        return self.input_tokens + self.output_tokens >= self.threshold


class CorrectionBudget:
    """Allow at most one model correction turn for a logical source block."""

    def __init__(self) -> None:
        self._used: set[str] = set()
        self._lock = Lock()

    def consume(self, segment_id: str) -> None:
        with self._lock:
            if segment_id in self._used:
                raise RuntimeError(
                    f"stateful correction turn already consumed for {segment_id}"
                )
            self._used.add(segment_id)


def continuity_capsule(
    *, accepted_chain_sha256: str, segment_id: str, input_sha256: str, output_sha256: str
) -> dict[str, str]:
    return {
        "accepted_chain_sha256": accepted_chain_sha256,
        "last_accepted_segment_id": segment_id,
        "last_input_sha256": input_sha256,
        "last_output_sha256": output_sha256,
    }


def _configured_context_window() -> int:
    raw = os.environ.get("ARC_LLM_CONTEXT_WINDOW_TOKENS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            pass
        else:
            if value > 0:
                return value
    return DEFAULT_CONTEXT_WINDOW_TOKENS
