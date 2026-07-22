from __future__ import annotations

import re

from .providers.base import (
    LLMAbortScope,
    LLMFailureCategory,
    LLMFailureDisposition,
    LLMSubmissionState,
)


_QUOTA_PATTERNS = (
    r"\binsufficient[_ ]quota\b",
    r"\bquota (?:exceeded|exhausted|depleted)\b",
    r"\busage limit\b",
    r"\breached (?:your )?(?:usage|spending) limit\b",
    r"\byou(?:'ve| have) reached your (?:usage|spending) limit\b",
    r"\bbilling hard limit\b",
    r"\b(?:credit|account) balance (?:is )?(?:too low|exhausted|depleted)\b",
    r"\bout of credits\b",
)
_AUTH_PATTERNS = (
    r"\binvalid (?:api[ _-]?)?key\b",
    r"\bauthentication (?:failed|required|error)\b",
    r"\bunauthori[sz]ed\b",
    r"\blogin required\b",
    r"\bnot logged in\b",
    r"\binvalid credentials?\b",
)
_PERMISSION_PATTERNS = (
    r"\bforbidden\b",
    r"\bpermission denied\b",
    r"\baccess denied\b",
)
_RATE_LIMIT_PATTERNS = (
    r"\brate[ _-]?limit(?:ed|ing)?\b",
    r"\btoo many requests\b",
    r"\boverload(?:ed|ing)?\b",
    r"\bserver is busy\b",
    r"\bcapacity temporarily unavailable\b",
)
_SCHEMA_REJECTION_PATTERNS = (
    r"\binvalid (?:json|output|response) schema\b",
    r"\bjson_schema\b.*\b(?:invalid|unsupported|required)\b",
    r"\bresponse_format\b.*\b(?:invalid|unsupported|required)\b",
    r"\badditionalproperties\b.*\b(?:required|must)\b",
)
_RETRY_AFTER_PATTERNS = (
    r"retry-after\s*[:=]\s*(\d+(?:\.\d+)?)",
    r"retry after\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b",
    r"try again in\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b",
)


def classify_provider_diagnostic(
    text: str,
    *,
    http_status: int | None = None,
    submission_state: LLMSubmissionState | str = LLMSubmissionState.UNKNOWN,
    retry_after_seconds: float | None = None,
) -> LLMFailureDisposition | None:
    """Classify a provider error diagnostic without treating it as model output.

    The function is intentionally conservative: callers should pass only a
    provider error envelope/stderr, never arbitrary scientific response text.
    It does not make post-submission failures immediately retryable.
    """

    normalized = " ".join(str(text or "").lower().split())
    if http_status is None:
        status_match = re.search(r"\b(?:http|status|code)\s*[:=]?\s*(4\d\d)\b", normalized)
        if status_match is not None:
            http_status = int(status_match.group(1))
    state = LLMSubmissionState(submission_state)
    parsed_retry_after = _retry_after(normalized, explicit=retry_after_seconds)

    if _matches(normalized, _QUOTA_PATTERNS):
        return _disposition(LLMFailureCategory.QUOTA, state)
    if http_status == 401 or _matches(normalized, _AUTH_PATTERNS):
        return _disposition(LLMFailureCategory.AUTHENTICATION, state)
    if http_status == 403 or _matches(normalized, _PERMISSION_PATTERNS):
        return _disposition(LLMFailureCategory.PERMISSION, state)
    if http_status == 429 or _matches(normalized, _RATE_LIMIT_PATTERNS):
        return _disposition(
            LLMFailureCategory.RATE_LIMIT,
            state,
            retry_after_seconds=parsed_retry_after,
        )
    if http_status == 400 and _matches(normalized, _SCHEMA_REJECTION_PATTERNS):
        return LLMFailureDisposition(
            category=LLMFailureCategory.SCHEMA,
            abort_scope=LLMAbortScope.BATCH,
            submission_state=state,
            retryable=False,
        )
    if http_status is not None and 400 <= http_status < 500:
        return LLMFailureDisposition(
            category=LLMFailureCategory.INVALID_REQUEST,
            abort_scope=LLMAbortScope.BATCH,
            submission_state=state,
            retryable=False,
        )
    return None


def disposition_error_kwargs(disposition: LLMFailureDisposition) -> dict[str, object]:
    """Convert a classification into keyword arguments for ``LLMWorkerError``."""

    return {
        "retryable": disposition.retryable,
        "category": disposition.category,
        "abort_scope": disposition.abort_scope,
        "submission_state": disposition.submission_state,
        "retry_after_seconds": disposition.retry_after_seconds,
    }


def _disposition(
    category: LLMFailureCategory,
    submission_state: LLMSubmissionState,
    *,
    retry_after_seconds: float | None = None,
) -> LLMFailureDisposition:
    return LLMFailureDisposition(
        category=category,
        abort_scope=LLMAbortScope.PROVIDER,
        submission_state=submission_state,
        retryable=False,
        retry_after_seconds=retry_after_seconds,
    )


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) is not None for pattern in patterns)


def _retry_after(text: str, *, explicit: float | None) -> float | None:
    if explicit is not None:
        try:
            value = float(explicit)
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None
    for pattern in _RETRY_AFTER_PATTERNS:
        matched = re.search(pattern, text, flags=re.IGNORECASE)
        if matched is not None:
            return float(matched.group(1))
    return None
