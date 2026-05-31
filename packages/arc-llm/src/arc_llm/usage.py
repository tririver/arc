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
    def cached_input_ratio(self) -> float | None:
        if self.input_tokens and self.cached_input_tokens is not None:
            return self.cached_input_tokens / max(1, self.input_tokens)
        total = None
        if self.cache_creation_input_tokens is not None or self.cache_read_input_tokens is not None:
            total = (self.cache_creation_input_tokens or 0) + (self.cache_read_input_tokens or 0)
        if total:
            return (self.cache_read_input_tokens or 0) / max(1, total)
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
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
