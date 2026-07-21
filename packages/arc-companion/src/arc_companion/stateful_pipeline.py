from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from threading import BoundedSemaphore, Lock
from typing import Any, Iterator, Mapping


GLOSSARY_SETUP_MAX_BYTES = 60 * 1024
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
    """Construct auditable generation bootstraps and compact follow-up turns."""

    chapter_id: str
    lane: str
    fixed_rules: Mapping[str, Any]
    chapter: Mapping[str, Any]
    guide: Mapping[str, Any]
    compact_glossary: list[Mapping[str, Any]]
    generation: int = 1
    continuity_capsule: Mapping[str, Any] | None = None
    _turns: int = 0
    _setup_emitted: bool = False

    def request(
        self,
        request: str,
        *,
        cursor: str,
        source_sha256: str,
        block_glossary: list[Mapping[str, Any]] | None = None,
        evidence: Mapping[str, Any] | None = None,
        preserve_delta_instructions: bool = False,
    ) -> str:
        if self._turns == 0:
            payload = {
                "turn_kind": "generation_bootstrap",
                "generation": self.generation,
                "chapter_id": self.chapter_id,
                "lane": self.lane,
                "fixed_rules": dict(self.fixed_rules),
                "chapter": dict(self.chapter),
                "chapter_guide": dict(self.guide),
                "chapter_glossary_mapping": [dict(item) for item in self.compact_glossary],
                "continuity_capsule": dict(self.continuity_capsule or {}),
                "cursor": cursor,
                "source_sha256": source_sha256,
                "current_request": request,
            }
        else:
            delta_request = (
                request if preserve_delta_instructions else compact_primary_request(request)
            )
            payload = {
                "turn_kind": "delta",
                "generation": self.generation,
                "chapter_id": self.chapter_id,
                "lane": self.lane,
                "cursor": cursor,
                "source_sha256": source_sha256,
                "glossary_mapping": [dict(item) for item in self.compact_glossary],
                "block_glossary": [dict(item) for item in (block_glossary or [])],
                "evidence": dict(evidence or {}),
                "current_request": delta_request,
            }
        self._turns += 1
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def setup_turns(self) -> list[str]:
        """Return lossless glossary setup chunks only when the mapping exceeds 60 KiB."""
        if self._setup_emitted:
            return []
        self._setup_emitted = True
        encoded = json.dumps(self.compact_glossary, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) <= GLOSSARY_SETUP_MAX_BYTES:
            return []
        chunks: list[list[Mapping[str, Any]]] = []
        current: list[Mapping[str, Any]] = []
        for item in self.compact_glossary:
            candidate = [*current, item]
            size = len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            if current and size > GLOSSARY_SETUP_MAX_BYTES:
                chunks.append(current)
                current = [item]
            else:
                current = candidate
        if current:
            chunks.append(current)
        turns = []
        for index, chunk in enumerate(chunks, 1):
            payload = {
                "turn_kind": "generation_bootstrap" if index == 1 else "glossary_setup_delta",
                "chapter_id": self.chapter_id,
                "lane": self.lane,
                "generation": self.generation,
                "chunk": index,
                "chunk_count": len(chunks),
                "entries": [dict(item) for item in chunk],
            }
            if index == 1:
                payload.update({
                    "fixed_rules": dict(self.fixed_rules),
                    "chapter": dict(self.chapter),
                    "chapter_guide": dict(self.guide),
                    "continuity_capsule": dict(self.continuity_capsule or {}),
                })
            turns.append(json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ))
        self._turns = 1
        return turns


@dataclass
class ContextRolloverBudget:
    """Track one native generation and request rollover only at accepted boundaries."""

    context_window_tokens: int = field(default_factory=lambda: _configured_context_window())
    input_tokens: int = 0
    output_tokens: int = 0
    _has_known_input: bool = False

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
        elif prompt_bytes:
            estimate = max(1, prompt_bytes // 4)
            if self._has_known_input:
                self.input_tokens += estimate
            else:
                self.input_tokens += estimate
        if isinstance(known_output, int):
            self.output_tokens += max(0, known_output)

    def rollover_due(self) -> bool:
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


def compact_primary_request(prompt: str) -> str:
    """Remove repeated fixed instructions while retaining the current source payload."""
    markers = ("\n\nSEGMENT:\n", "\n\nSEGMENT ID:\n")
    offsets = [prompt.find(marker) for marker in markers if prompt.find(marker) >= 0]
    if not offsets:
        return prompt
    return prompt[min(offsets) + 2 :]


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
