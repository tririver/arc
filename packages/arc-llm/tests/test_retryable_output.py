from __future__ import annotations

from arc_llm.retryable_output import ProviderOutputClass, classify_provider_output_text


def test_retryable_classifier_flags_short_service_unavailable_reply():
    result = classify_provider_output_text("Service unavailable")

    assert result.classification == ProviderOutputClass.RETRYABLE_PROVIDER_FAILURE
    assert "service unavailable" in result.reason


def test_retryable_classifier_flags_error_json_status_code():
    result = classify_provider_output_text('{"error":{"code":503,"message":"upstream overloaded"}}')

    assert result.classification == ProviderOutputClass.RETRYABLE_PROVIDER_FAILURE
    assert "503" in result.reason


def test_retryable_classifier_ignores_incidental_400_and_overloaded_in_science_text():
    text = (
        "The expansion has 400 boundary modes. The overloaded notation in Eq. (3) "
        "is harmless because the two limits commute after renormalization."
    )

    result = classify_provider_output_text(text)

    assert result.classification == ProviderOutputClass.OK


def test_retryable_classifier_marks_http_400_as_fatal_provider_failure():
    result = classify_provider_output_text("HTTP 400 Bad Request: invalid schema")

    assert result.classification == ProviderOutputClass.FATAL_PROVIDER_FAILURE
    assert "400" in result.reason
