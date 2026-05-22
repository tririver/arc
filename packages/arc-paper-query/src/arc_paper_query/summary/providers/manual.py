from __future__ import annotations

from typing import Any, Callable

from .base import LLMProviderError


class ManualProvider:
    name = "manual"

    def generate_summary(
        self,
        task: dict,
        *,
        model: str | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict:
        raise LLMProviderError("manual provider cannot generate summaries")
