from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from jsonschema.exceptions import SchemaError as JsonSchemaError

from arc_llm.call_record import strip_arc_llm_call_records
from arc_llm.structured_recovery import structured_metadata


JsonRunner = Callable[..., Any]


class SchemaFormatError(RuntimeError):
    pass


@dataclass(frozen=True)
class SchemaFormatResult:
    value: dict[str, Any]
    structured_output: dict[str, Any]


@dataclass(frozen=True)
class SchemaFormatDecision:
    action: str
    value: dict[str, Any] | None
    reason: str
    structured_output: dict[str, Any]


FORMATTER_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["action", "reason", "formatted_output"],
    "properties": {
        "action": {"type": "string", "enum": ["format", "retry"]},
        "reason": {"type": "string"},
        "formatted_output": {
            "anyOf": [
                {"type": "object", "additionalProperties": True},
                {"type": "null"},
            ]
        },
    },
    "additionalProperties": False,
}


def format_to_schema(
    *,
    raw_text: str,
    schema: Mapping[str, Any],
    role_hint: str | None = None,
    json_runner: JsonRunner,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> SchemaFormatResult:
    source = str(raw_text or "").strip()
    if not source:
        raise SchemaFormatError("schema_formatter_empty_source")
    prompt = _formatter_prompt(source, schema=schema, role_hint=role_hint)
    kwargs = {
        "schema": dict(schema),
        "provider": provider,
        "model": model,
        "model_tier": model_tier,
        "env": dict(env or {}),
    }
    for key, value in {
        "validate_schema": True,
        "output_recovery": "warn",
        "role_hint": "schema_formatter",
        "timeout_seconds": timeout_seconds,
        "cancel_check": cancel_check,
    }.items():
        if _accepts_keyword(json_runner, key):
            kwargs[key] = value
    value = json_runner(prompt, **kwargs)
    if not isinstance(value, dict):
        raise SchemaFormatError("schema_formatter_non_object_output")
    value = strip_arc_llm_call_records(value)
    if not isinstance(value, dict):
        raise SchemaFormatError("schema_formatter_non_object_output")
    _validate(value, schema)
    missing = _numeric_values_not_in_source(value, raw_text=source)
    if missing:
        raise SchemaFormatError("missing_required_numeric_fields: " + ", ".join(missing))
    return SchemaFormatResult(
        value=value,
        structured_output=structured_metadata(
            severity="minor",
            warnings=["Formatted content-rich output to match schema without retrying original worker."],
            raw_text=source,
            strategy="schema_formatter",
            provider_error_type=None,
        ),
    )


def format_to_schema_or_retry(
    *,
    raw_text: str,
    schema: Mapping[str, Any],
    role_hint: str | None = None,
    json_runner: JsonRunner,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: list[str] | None = None,
    timeout_seconds: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> SchemaFormatDecision:
    source = str(raw_text or "").strip()
    if not source:
        raise SchemaFormatError("schema_formatter_empty_source")
    prompt = _formatter_decision_prompt(source, schema=schema, role_hint=role_hint)
    kwargs = {
        "schema": dict(FORMATTER_DECISION_SCHEMA),
        "provider": provider,
        "model": model,
        "model_tier": model_tier,
        "env": dict(env or {}),
    }
    for key, value in {
        "validate_schema": True,
        "output_recovery": "strict",
        "role_hint": "schema_formatter",
        "process_chain": process_chain,
        "timeout_seconds": timeout_seconds,
        "cancel_check": cancel_check,
    }.items():
        if _accepts_keyword(json_runner, key):
            kwargs[key] = value
    decision = json_runner(prompt, **kwargs)
    if not isinstance(decision, dict):
        raise SchemaFormatError("schema_formatter_non_object_output")
    decision = strip_arc_llm_call_records(decision)
    if not isinstance(decision, dict):
        raise SchemaFormatError("schema_formatter_non_object_output")
    _validate(decision, FORMATTER_DECISION_SCHEMA)
    action = str(decision.get("action") or "")
    reason = str(decision.get("reason") or "")
    if action == "retry":
        return SchemaFormatDecision(
            action="retry",
            value=None,
            reason=reason,
            structured_output=structured_metadata(
                severity="major",
                warnings=["Schema formatter requested original worker retry.", reason],
                raw_text=source,
                strategy="schema_formatter_retry",
                provider_error_type=None,
            ),
        )
    formatted = decision.get("formatted_output")
    if not isinstance(formatted, dict):
        raise SchemaFormatError("schema_formatter_missing_formatted_output")
    formatted = strip_arc_llm_call_records(formatted)
    if not isinstance(formatted, dict):
        raise SchemaFormatError("schema_formatter_non_object_output")
    _validate(formatted, schema)
    missing = _numeric_values_not_in_source(formatted, raw_text=source)
    if missing:
        raise SchemaFormatError("missing_required_numeric_fields: " + ", ".join(missing))
    return SchemaFormatDecision(
        action="format",
        value=formatted,
        reason=reason,
        structured_output=structured_metadata(
            severity="minor",
            warnings=["Formatted content-rich output to match schema without retrying original worker.", reason],
            raw_text=source,
            strategy="schema_formatter",
            provider_error_type=None,
        ),
    )


def _formatter_prompt(raw_text: str, *, schema: Mapping[str, Any], role_hint: str | None) -> str:
    return (
        "## Schema Formatter\n"
        "Reformat source text into exactly one JSON object matching the schema.\n"
        "Do not add new scientific claims, scores, evidence, or judgments.\n"
        "Only copy or reorganize information explicitly present in source text.\n"
        "For numeric fields, use only numbers explicitly present in source text. "
        "If no explicit number exists, do not invent one.\n"
        "If a required numeric field has no explicit source number, return the closest schema-valid object you can; "
        "ARC will reject fabricated numbers after validation.\n"
        "Use null or \"N.A.\" for missing non-numeric fields only when schema allows it.\n"
        f"Role hint: {role_hint or 'unknown'}\n\n"
        "## Source Text\n"
        f"{raw_text[:12000]}\n\n"
        "## JSON Schema\n"
        f"{json.dumps(dict(schema), ensure_ascii=False, sort_keys=True)}\n"
    )


def _formatter_decision_prompt(raw_text: str, *, schema: Mapping[str, Any], role_hint: str | None) -> str:
    return (
        "## Schema Formatter\n"
        "Decide whether source text contains enough information to reformat into the target schema.\n"
        "Return exactly one JSON object matching the formatter decision schema.\n"
        "Use action=\"format\" only when source text contains enough information to fill required fields without inventing scientific claims, scores, evidence, or judgments.\n"
        "Use action=\"retry\" when source text is empty, mostly provider/CLI metadata, too incomplete, or would require invented content.\n"
        "For numeric fields, use only numbers explicitly present in source text. If a required numeric field has no explicit source number, choose action=\"retry\".\n"
        "Do not wrap the object in Markdown.\n"
        f"Role hint: {role_hint or 'unknown'}\n\n"
        "## Source Text\n"
        f"{raw_text[:12000]}\n\n"
        "## Target JSON Schema\n"
        f"{json.dumps(dict(schema), ensure_ascii=False, sort_keys=True)}\n\n"
        "## Formatter Decision Schema\n"
        f"{json.dumps(FORMATTER_DECISION_SCHEMA, ensure_ascii=False, sort_keys=True)}\n"
    )


def _validate(value: dict[str, Any], schema: Mapping[str, Any]) -> None:
    try:
        validate_json_schema(instance=value, schema=dict(schema))
    except JsonSchemaValidationError as exc:
        raise SchemaFormatError(f"schema_formatter_validation_failed: {exc.message}") from exc
    except JsonSchemaError as exc:
        raise SchemaFormatError(f"schema_formatter_schema_invalid: {exc.message}") from exc


def _numeric_values_not_in_source(value: Any, *, raw_text: str) -> list[str]:
    source_numbers = _source_numbers(raw_text)
    missing: list[str] = []
    _collect_missing_numeric(value, source_numbers=source_numbers, path="", missing=missing)
    return missing


def _collect_missing_numeric(
    value: Any,
    *,
    source_numbers: set[str],
    path: str,
    missing: list[str],
) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not _number_in_source(value, source_numbers):
            missing.append(path or "$")
        return
    if isinstance(value, Mapping):
        for key, child_value in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            _collect_missing_numeric(
                child_value,
                source_numbers=source_numbers,
                path=child_path,
                missing=missing,
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _collect_missing_numeric(
                item,
                source_numbers=source_numbers,
                path=f"{path}[{index}]",
                missing=missing,
            )


def _accepts_keyword(callable_obj: JsonRunner, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or (
            parameter.kind in {inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
            and parameter.name == name
        )
        for parameter in signature.parameters.values()
    )


def _source_numbers(text: str) -> set[str]:
    numbers = set()
    for match in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?", text):
        raw = match.group(0)
        numbers.add(raw)
        try:
            number = float(raw)
        except ValueError:
            continue
        numbers.add(str(int(number)) if number.is_integer() else str(number))
    return numbers


def _number_in_source(value: Any, source_numbers: set[str]) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return str(value) in source_numbers
    if isinstance(value, float):
        if value.is_integer() and str(int(value)) in source_numbers:
            return True
        return str(value) in source_numbers
    return False
