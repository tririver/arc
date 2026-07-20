from __future__ import annotations

from typing import Any, Protocol


class LLMWorkerError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMWorkerTimeout(LLMWorkerError):
    """The total worker-call deadline expired."""


class LLMWorkerCancelled(LLMWorkerError):
    """The caller requested worker-call cancellation."""


class LLMSchemaError(LLMWorkerError):
    """The provider-facing output schema is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


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
