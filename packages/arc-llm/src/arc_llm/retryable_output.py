from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class ProviderOutputClass(str, Enum):
    OK = "ok"
    RETRYABLE_PROVIDER_FAILURE = "retryable_provider_failure"
    FATAL_PROVIDER_FAILURE = "fatal_provider_failure"


@dataclass(frozen=True)
class ProviderOutputClassification:
    classification: ProviderOutputClass
    reason: str = ""


RETRYABLE_CODES = {"429", "500", "502", "503", "504"}
FATAL_CODES = {"400", "401", "403", "404"}


def classify_provider_output_text(text: str | None, metadata: Mapping[str, Any] | None = None) -> ProviderOutputClassification:
    raw = str(text or "").strip()
    if not raw:
        return ProviderOutputClassification(ProviderOutputClass.OK)
    metadata_classification = _classify_mapping(metadata or {})
    if metadata_classification.classification != ProviderOutputClass.OK:
        return metadata_classification
    parsed = _parse_json_object(raw)
    if parsed is not None:
        parsed_classification = _classify_mapping(parsed)
        if parsed_classification.classification != ProviderOutputClass.OK:
            return parsed_classification
    html_classification = _classify_html_error(raw)
    if html_classification.classification != ProviderOutputClass.OK:
        return html_classification
    status_classification = _classify_status_line(raw)
    if status_classification.classification != ProviderOutputClass.OK:
        return status_classification
    short_message = _classify_short_error_message(raw)
    if short_message.classification != ProviderOutputClass.OK:
        return short_message
    return ProviderOutputClassification(ProviderOutputClass.OK)


def _classify_mapping(value: Mapping[str, Any]) -> ProviderOutputClassification:
    if not value:
        return ProviderOutputClassification(ProviderOutputClass.OK)
    haystack = _mapping_text(value)
    code = _first_error_code(haystack)
    if code in RETRYABLE_CODES:
        return ProviderOutputClassification(ProviderOutputClass.RETRYABLE_PROVIDER_FAILURE, f"provider error code {code}")
    if code in FATAL_CODES:
        return ProviderOutputClassification(ProviderOutputClass.FATAL_PROVIDER_FAILURE, f"provider error code {code}")
    if _has_retryable_error_phrase(haystack):
        return ProviderOutputClassification(ProviderOutputClass.RETRYABLE_PROVIDER_FAILURE, _matched_phrase(haystack))
    if _has_fatal_error_phrase(haystack):
        return ProviderOutputClassification(ProviderOutputClass.FATAL_PROVIDER_FAILURE, _matched_phrase(haystack))
    return ProviderOutputClassification(ProviderOutputClass.OK)


def _parse_json_object(text: str) -> Mapping[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, Mapping) else None


def _mapping_text(value: Mapping[str, Any]) -> str:
    interesting: list[str] = []

    def visit(item: Any, *, key: str = "") -> None:
        key_text = str(key).lower()
        if isinstance(item, Mapping):
            for child_key, child_value in item.items():
                visit(child_value, key=str(child_key))
            return
        if isinstance(item, list):
            for child in item:
                visit(child, key=key)
            return
        if key_text in {"error", "errors", "message", "status", "status_code", "code", "type", "subtype", "detail"}:
            interesting.append(f"{key_text}: {item}")

    visit(value)
    if value.get("is_error") is True:
        interesting.append("is_error: true")
    if value.get("error") is not None:
        interesting.append("error present")
    return " ".join(interesting).lower()


def _first_error_code(text: str) -> str | None:
    match = re.search(r"\b(?:http|status|status_code|code|error|bad request)\D{0,24}(400|401|403|404|429|500|502|503|504)\b", text)
    return match.group(1) if match else None


def _classify_html_error(text: str) -> ProviderOutputClassification:
    lower = text[:4000].lower()
    if "<html" not in lower and "<title" not in lower:
        return ProviderOutputClassification(ProviderOutputClass.OK)
    title_match = re.search(r"<title[^>]*>(.*?)</title>", lower, flags=re.DOTALL)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else lower[:500]
    code = _first_error_code(title)
    if code in RETRYABLE_CODES:
        return ProviderOutputClassification(ProviderOutputClass.RETRYABLE_PROVIDER_FAILURE, f"html provider error {code}")
    if code in FATAL_CODES:
        return ProviderOutputClassification(ProviderOutputClass.FATAL_PROVIDER_FAILURE, f"html provider error {code}")
    if _has_retryable_error_phrase(title):
        return ProviderOutputClassification(ProviderOutputClass.RETRYABLE_PROVIDER_FAILURE, _matched_phrase(title))
    if _has_fatal_error_phrase(title):
        return ProviderOutputClassification(ProviderOutputClass.FATAL_PROVIDER_FAILURE, _matched_phrase(title))
    return ProviderOutputClassification(ProviderOutputClass.OK)


def _classify_status_line(text: str) -> ProviderOutputClassification:
    first = text.splitlines()[0].strip().lower()[:300]
    code = _first_error_code(first)
    if code in RETRYABLE_CODES:
        return ProviderOutputClassification(ProviderOutputClass.RETRYABLE_PROVIDER_FAILURE, f"provider status line {code}")
    if code in FATAL_CODES:
        return ProviderOutputClassification(ProviderOutputClass.FATAL_PROVIDER_FAILURE, f"provider status line {code}")
    return ProviderOutputClassification(ProviderOutputClass.OK)


def _classify_short_error_message(text: str) -> ProviderOutputClassification:
    compact = re.sub(r"\s+", " ", text.strip().lower())
    if len(compact) > 320:
        return ProviderOutputClassification(ProviderOutputClass.OK)
    if _has_retryable_error_phrase(compact):
        return ProviderOutputClassification(ProviderOutputClass.RETRYABLE_PROVIDER_FAILURE, _matched_phrase(compact))
    if _has_fatal_error_phrase(compact):
        return ProviderOutputClassification(ProviderOutputClass.FATAL_PROVIDER_FAILURE, _matched_phrase(compact))
    return ProviderOutputClassification(ProviderOutputClass.OK)


def _has_retryable_error_phrase(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "service unavailable",
            "temporarily unavailable",
            "too many requests",
            "rate limit",
            "quota exceeded",
            "gateway timeout",
            "upstream error",
            "connection reset",
            "request timed out",
            "server overloaded",
            "model overloaded",
            "provider overloaded",
            "overloaded, retry",
            "overloaded. retry",
        )
    )


def _has_fatal_error_phrase(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "bad request",
            "invalid request",
            "invalid api key",
            "authentication failed",
            "permission denied",
            "forbidden",
            "unauthorized",
        )
    )


def _matched_phrase(text: str) -> str:
    for phrase in (
        "service unavailable",
        "temporarily unavailable",
        "too many requests",
        "rate limit",
        "quota exceeded",
        "gateway timeout",
        "upstream error",
        "connection reset",
        "request timed out",
        "server overloaded",
        "model overloaded",
        "provider overloaded",
        "bad request",
        "invalid request",
        "invalid api key",
        "authentication failed",
        "permission denied",
        "forbidden",
        "unauthorized",
    ):
        if phrase in text:
            return phrase
    return "provider error"
