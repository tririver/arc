from __future__ import annotations

from typing import Any, Callable, Protocol

from arc_llm_worker.providers.base import LLMWorkerError as LLMProviderError


class LLMProvider(Protocol):
    name: str

    def generate_summary(
        self,
        task: dict,
        *,
        model: str | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict:
        ...
