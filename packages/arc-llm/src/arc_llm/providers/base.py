from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class LLMFailureCategory(StrEnum):
    UNKNOWN = "unknown"
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    INVALID_REQUEST = "invalid_request"
    SCHEMA = "schema"
    TRANSPORT = "transport"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    OUTPUT_INVALID = "output_invalid"
    LOCAL_IO = "local_io"
    PROVIDER_INTERNAL = "provider_internal"


class LLMAbortScope(StrEnum):
    CALL = "call"
    BATCH = "batch"
    PROVIDER = "provider"


class LLMSubmissionState(StrEnum):
    NOT_SUBMITTED = "not_submitted"
    SUBMITTED = "submitted"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LLMFailureDisposition:
    category: LLMFailureCategory
    abort_scope: LLMAbortScope
    submission_state: LLMSubmissionState
    retryable: bool
    retry_after_seconds: float | None = None


class LLMWorkerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        abort_batch: bool = False,
        category: LLMFailureCategory | str = LLMFailureCategory.UNKNOWN,
        abort_scope: LLMAbortScope | str | None = None,
        submission_state: LLMSubmissionState | str = LLMSubmissionState.UNKNOWN,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)
        self.category = LLMFailureCategory(category)
        self.abort_scope = LLMAbortScope(
            abort_scope if abort_scope is not None else (LLMAbortScope.BATCH if abort_batch else LLMAbortScope.CALL)
        )
        # Keep the historical public flag synchronized with the richer scope.
        self.submission_state = LLMSubmissionState(submission_state)
        self.retry_after_seconds = retry_after_seconds

    @property
    def abort_batch(self) -> bool:
        return self.abort_scope in {LLMAbortScope.BATCH, LLMAbortScope.PROVIDER}

    @abort_batch.setter
    def abort_batch(self, value: bool) -> None:
        # Historical providers mutate this flag after constructing an error.
        # Keep that behavior while preventing the typed scope from going stale.
        if value and self.abort_scope == LLMAbortScope.CALL:
            self.abort_scope = LLMAbortScope.BATCH
        elif not value:
            self.abort_scope = LLMAbortScope.CALL

    @property
    def disposition(self) -> LLMFailureDisposition:
        return LLMFailureDisposition(
            category=self.category,
            abort_scope=self.abort_scope,
            submission_state=self.submission_state,
            retryable=self.retryable,
            retry_after_seconds=self.retry_after_seconds,
        )


class LLMWorkerTimeout(LLMWorkerError):
    """The provider produced no meaningful activity before its idle deadline."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("category", LLMFailureCategory.TIMEOUT)
        kwargs.setdefault("submission_state", LLMSubmissionState.UNKNOWN)
        super().__init__(message, **kwargs)


class LLMWorkerCancelled(LLMWorkerError):
    """The caller requested worker-call cancellation."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("category", LLMFailureCategory.CANCELLED)
        super().__init__(message, **kwargs)


class LLMConfigurationError(LLMWorkerError):
    """Local LLM configuration is invalid before provider submission."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            retryable=False,
            category=LLMFailureCategory.INVALID_REQUEST,
            abort_scope=LLMAbortScope.BATCH,
            submission_state=LLMSubmissionState.NOT_SUBMITTED,
        )


class LLMSchemaError(LLMWorkerError):
    """The provider-facing output schema is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            retryable=False,
            category=LLMFailureCategory.SCHEMA,
            abort_scope=LLMAbortScope.BATCH,
            submission_state=LLMSubmissionState.NOT_SUBMITTED,
        )


def failure_disposition(exc: BaseException) -> LLMFailureDisposition | None:
    """Return the strongest typed LLM failure in an exception chain.

    Wrapping layers should use this instead of inspecting only the outer
    exception, so provider-wide failures cannot accidentally become retries or
    deterministic fallbacks.
    """
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    dispositions: list[LLMFailureDisposition] = []
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, LLMWorkerError):
            dispositions.append(current.disposition)
        cause = current.__cause__
        context = current.__context__
        if cause is not None:
            pending.append(cause)
        if context is not None and context is not cause:
            pending.append(context)
    if not dispositions:
        return None
    scope_weight = {LLMAbortScope.CALL: 0, LLMAbortScope.BATCH: 1, LLMAbortScope.PROVIDER: 2}
    return max(
        dispositions,
        key=lambda item: (
            scope_weight[item.abort_scope],
            item.category != LLMFailureCategory.UNKNOWN,
        ),
    )


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
