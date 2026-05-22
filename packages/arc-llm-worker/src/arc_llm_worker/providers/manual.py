from __future__ import annotations

from typing import Any

from .base import LLMWorkerError


class ManualProvider:
    name = "manual"

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        raise LLMWorkerError("manual provider cannot generate JSON")

    def generate_text(self, prompt: str, *, model: str | None = None) -> str:
        raise LLMWorkerError("manual provider cannot generate text")
