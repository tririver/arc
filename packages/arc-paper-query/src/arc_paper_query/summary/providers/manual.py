from __future__ import annotations

from .base import LLMProviderError


class ManualProvider:
    name = "manual"

    def generate_summary(self, task: dict, *, model: str | None = None) -> dict:
        raise LLMProviderError("manual provider cannot generate summaries")
