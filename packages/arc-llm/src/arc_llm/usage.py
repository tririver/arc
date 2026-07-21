from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_claude_cache_fields(self) -> bool:
        return self.cache_creation_input_tokens is not None or self.cache_read_input_tokens is not None

    @property
    def total_input_tokens(self) -> int | None:
        if self.has_claude_cache_fields:
            return (
                (self.input_tokens or 0)
                + (self.cache_creation_input_tokens or 0)
                + (self.cache_read_input_tokens or 0)
            )
        return self.input_tokens

    @property
    def effective_cached_input_tokens(self) -> int | None:
        if self.has_claude_cache_fields:
            return self.cache_read_input_tokens or 0
        return self.cached_input_tokens

    @property
    def cached_input_ratio(self) -> float | None:
        total = self.total_input_tokens
        cached = self.effective_cached_input_tokens
        if total is None or cached is None or total <= 0:
            return None
        return cached / max(1, total)

    def to_json(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "total_input_tokens": self.total_input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "effective_cached_input_tokens": self.effective_cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cached_input_ratio": self.cached_input_ratio,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class LLMProviderResponse(Generic[T]):
    value: T
    usage: LLMUsage = field(default_factory=LLMUsage)
    native_session_id: str | None = None
    raw_events: tuple[dict[str, Any], ...] = ()
    raw_output: str = ""
    # The agent's actual message, distinct from a CLI JSON envelope or stderr.
    # Recovery must never prefer wrapper diagnostics over this value.
    raw_model_output: str = ""
    prompt_sent_sha256: str | None = None
    structured_output: dict[str, Any] | None = None
