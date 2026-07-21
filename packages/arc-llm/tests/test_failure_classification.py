from __future__ import annotations

import pytest

from arc_llm.failure_classification import classify_provider_diagnostic, disposition_error_kwargs
from arc_llm.providers.base import LLMAbortScope, LLMFailureCategory, LLMSubmissionState, LLMWorkerError


@pytest.mark.parametrize(
    ("text", "status", "category"),
    [
        ("You've reached your usage limit", 403, LLMFailureCategory.QUOTA),
        ("insufficient_quota", 429, LLMFailureCategory.QUOTA),
        ("invalid API key", None, LLMFailureCategory.AUTHENTICATION),
        ("request forbidden", 403, LLMFailureCategory.PERMISSION),
        ("Too many requests", None, LLMFailureCategory.RATE_LIMIT),
        ("service overloaded", 503, LLMFailureCategory.RATE_LIMIT),
    ],
)
def test_provider_fatal_diagnostics_are_typed(text, status, category):
    result = classify_provider_diagnostic(
        text,
        http_status=status,
        submission_state=LLMSubmissionState.SUBMITTED,
    )

    assert result is not None
    assert result.category == category
    assert result.abort_scope == LLMAbortScope.PROVIDER
    assert result.submission_state == LLMSubmissionState.SUBMITTED
    assert result.retryable is False


def test_rate_limit_parses_retry_after_but_quota_remains_persistent():
    rate = classify_provider_diagnostic("HTTP 429; Retry-After: 123")
    quota = classify_provider_diagnostic("HTTP 429: quota exhausted; retry after 5 seconds")

    assert rate is not None and rate.retry_after_seconds == 123
    assert quota is not None and quota.category == LLMFailureCategory.QUOTA
    assert quota.retry_after_seconds is None


def test_http_status_can_be_inferred_from_diagnostic_envelope_text():
    result = classify_provider_diagnostic("request failed: HTTP 401")
    assert result is not None and result.category == LLMFailureCategory.AUTHENTICATION


def test_explicit_retry_after_takes_priority():
    result = classify_provider_diagnostic(
        "server overloaded; retry after 5 seconds",
        retry_after_seconds=42,
    )
    assert result is not None and result.retry_after_seconds == 42


def test_other_client_error_aborts_batch_without_poisoning_provider():
    result = classify_provider_diagnostic("bad schema", http_status=400, submission_state="not_submitted")

    assert result is not None
    assert result.category == LLMFailureCategory.INVALID_REQUEST
    assert result.abort_scope == LLMAbortScope.BATCH
    assert result.submission_state == LLMSubmissionState.NOT_SUBMITTED


def test_unrelated_text_is_not_classified_and_kwargs_construct_error():
    assert classify_provider_diagnostic("the paper discusses quotas in gauge theory") is None
    result = classify_provider_diagnostic("HTTP 429 too many requests")
    assert result is not None

    error = LLMWorkerError("limited", **disposition_error_kwargs(result))
    assert error.disposition == result
