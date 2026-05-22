from __future__ import annotations

from typing import Protocol


class LLMProviderError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str

    def generate_summary(self, task: dict, *, model: str | None = None) -> dict:
        ...
