from __future__ import annotations

from typing import Any, Protocol


class LLMWorkerError(RuntimeError):
    pass


class PromptProvider(Protocol):
    name: str

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        ...

    def generate_text(self, prompt: str, *, model: str | None = None) -> str:
        ...
