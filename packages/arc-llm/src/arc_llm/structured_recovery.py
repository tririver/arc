from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from jsonschema.exceptions import SchemaError as JsonSchemaError


SCHEMA_VERSION = "arc.llm.structured_output.v1"


@dataclass(frozen=True)
class StructuredRecoveryResult:
    value: dict[str, Any]
    structured_output: dict[str, Any] | None


def parse_json_object_relaxed(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    raw = text.strip()
    if not raw:
        return None, ["No text was available for JSON extraction."]
    parsed = _loads_object(raw)
    if parsed is not None:
        if isinstance(parsed.get("result"), str):
            nested = _loads_object(parsed["result"])
            if nested is not None:
                warnings.append("Parsed JSON object from result string.")
                return nested, warnings
        return parsed, warnings
    fenced = _first_fenced_block(raw)
    if fenced:
        parsed = _loads_object(_clean_json_text(fenced))
        if parsed is not None:
            warnings.append("Extracted JSON object from Markdown fence.")
            return parsed, warnings
    balanced = _first_balanced_object(raw)
    if balanced:
        parsed = _loads_object(_clean_json_text(balanced))
        if parsed is not None:
            warnings.append("Extracted first balanced JSON object from surrounding text.")
            return parsed, warnings
    cleaned = _clean_json_text(raw)
    if cleaned != raw:
        parsed = _loads_object(cleaned)
        if parsed is not None:
            warnings.append("Recovered JSON object after cleanup.")
            return parsed, warnings
    return None, ["No JSON object could be extracted."]


def recover_json_output(
    *,
    value: Any,
    schema: Mapping[str, Any] | None,
    raw_text: str | None,
    error: Exception | None = None,
    role_hint: str | None = None,
    strict_first: bool = True,
    provider_metadata: Mapping[str, Any] | None = None,
) -> StructuredRecoveryResult:
    provider_meta = dict(provider_metadata or {})
    warnings = [str(item) for item in provider_meta.get("warnings", []) if item]
    raw = provider_meta.get("raw_text_excerpt") or raw_text or ""
    provider_severity = str(provider_meta.get("severity") or "none")
    provider_strategy = provider_meta.get("recovery_strategy")
    provider_error_type = provider_meta.get("provider_error_type")

    if strict_first and isinstance(value, dict):
        if schema is None or _validates(value, schema):
            return StructuredRecoveryResult(value=value, structured_output=None)

    extracted_from_text = False
    if not isinstance(value, dict):
        extracted, extraction_warnings = parse_json_object_relaxed(raw or str(value or ""))
        warnings.extend(extraction_warnings)
        if isinstance(extracted, dict):
            value = extracted
            extracted_from_text = True
        else:
            value = {}
    elif provider_meta.get("mode") == "recovered" and not value:
        extracted, extraction_warnings = parse_json_object_relaxed(raw)
        warnings.extend(extraction_warnings)
        if isinstance(extracted, dict):
            value = extracted
            extracted_from_text = True

    if schema is None:
        return StructuredRecoveryResult(
            value=dict(value),
            structured_output=structured_metadata(
                severity=_max_severity(provider_severity, "major" if warnings else "none"),
                warnings=warnings,
                raw_text=raw,
                strategy=str(provider_strategy or "extract_json"),
                provider_error_type=provider_error_type,
            ),
        )

    normalized, normalize_warnings = _normalize_existing(value, schema)
    if _validates(normalized, schema):
        severity = "minor"
        if extracted_from_text or provider_severity in {"major", "fatal"}:
            severity = "major"
        return StructuredRecoveryResult(
            value=normalized,
            structured_output=structured_metadata(
                severity=_max_severity(provider_severity, severity),
                warnings=[*warnings, *normalize_warnings],
                raw_text=raw,
                strategy="extract_json" if extracted_from_text else "schema_coerce",
                provider_error_type=provider_error_type,
            ),
        )

    fallback = _known_schema_fallback(schema, raw_text=raw, role_hint=role_hint)
    if fallback is not None and _validates(fallback, schema):
        return StructuredRecoveryResult(
            value=fallback,
            structured_output=structured_metadata(
                severity=_max_severity(provider_severity, "major"),
                warnings=[
                    *warnings,
                    "Recovered output did not validate; used schema-shaped fallback.",
                    *( [str(error)] if error is not None else [] ),
                ],
                raw_text=raw,
                strategy=str(provider_strategy or "natural_language_fallback"),
                provider_error_type=provider_error_type,
            ),
        )

    raise ValueError("Structured output recovery could not produce a schema-valid object.")


def structured_metadata(
    *,
    severity: str,
    warnings: list[str],
    raw_text: str | None,
    strategy: str,
    provider_error_type: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "recovered",
        "severity": severity,
        "warnings": warnings,
        "raw_text_excerpt": _excerpt(raw_text or "", limit=4000),
        "provider_error_type": provider_error_type,
        "recovery_strategy": strategy,
    }


def _loads_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _first_fenced_block(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        fenced = _first_fenced_block(cleaned)
        if fenced is not None:
            cleaned = fenced
    balanced = _first_balanced_object(cleaned)
    if balanced is not None:
        cleaned = balanced
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned.strip()


def _normalize_existing(value: Any, schema: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    normalized = _normalize_value(value, schema, warnings=warnings, fill_required=False)
    if isinstance(normalized, dict):
        return normalized, warnings
    return {}, [*warnings, "Recovered value was not an object."]


def _normalize_value(value: Any, schema: Mapping[str, Any], *, warnings: list[str], fill_required: bool) -> Any:
    if "const" in schema:
        if value != schema["const"]:
            warnings.append("Replaced value with schema const.")
        return schema["const"]
    if "anyOf" in schema:
        for branch in schema.get("anyOf", []):
            if isinstance(branch, Mapping):
                candidate = _normalize_value(value, branch, warnings=warnings, fill_required=fill_required)
                if _validates(candidate, branch):
                    return candidate
        for branch in schema.get("anyOf", []):
            if isinstance(branch, Mapping) and _schema_allows_null(branch):
                return None
    if "oneOf" in schema:
        for branch in schema.get("oneOf", []):
            if isinstance(branch, Mapping):
                candidate = _normalize_value(value, branch, warnings=warnings, fill_required=fill_required)
                if _validates(candidate, branch):
                    return candidate
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if value is None and "null" in schema_type:
            return None
        for item_type in schema_type:
            if item_type == "null":
                continue
            candidate = _normalize_value(value, {**dict(schema), "type": item_type}, warnings=warnings, fill_required=fill_required)
            if _validates(candidate, {**dict(schema), "type": item_type}):
                return candidate
        if "null" in schema_type:
            return None
    if "enum" in schema:
        enum_values = list(schema.get("enum") or [])
        if value in enum_values:
            return value
        chosen = _safe_enum(enum_values)
        warnings.append("Replaced value with safe enum default.")
        return chosen
    if schema_type == "object" or "properties" in schema:
        props = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        required = [str(item) for item in schema.get("required", [])]
        source = value if isinstance(value, Mapping) else {}
        result: dict[str, Any] = {}
        for key, child_schema in props.items():
            if key in source:
                result[str(key)] = _normalize_value(source[key], child_schema, warnings=warnings, fill_required=fill_required)
            elif fill_required and key in required:
                result[str(key)] = _schema_default(child_schema, warnings=warnings)
        extra_schema = schema.get("additionalProperties", True)
        extra_keys = [key for key in source if key not in props]
        if extra_schema is False:
            dropped = sorted(str(key) for key in extra_keys)
            if dropped:
                warnings.append("Dropped extra properties: " + ", ".join(dropped))
        elif isinstance(extra_schema, Mapping):
            for key in extra_keys:
                result[str(key)] = _normalize_value(
                    source[key],
                    extra_schema,
                    warnings=warnings,
                    fill_required=fill_required,
                )
        else:
            for key in extra_keys:
                result[str(key)] = source[key]
        return result
    if schema_type == "array":
        item_schema = schema.get("items") if isinstance(schema.get("items"), Mapping) else {}
        if isinstance(value, list):
            return [_normalize_value(item, item_schema, warnings=warnings, fill_required=fill_required) for item in value]
        if fill_required and value not in (None, ""):
            return [_normalize_value(value, item_schema, warnings=warnings, fill_required=fill_required)]
        return []
    if schema_type == "string":
        return value if isinstance(value, str) else ("" if value is None else str(value))
    if schema_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        return False
    if schema_type in {"number", "integer"}:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        if "minimum" in schema:
            number = max(number, float(schema["minimum"]))
        if "maximum" in schema:
            number = min(number, float(schema["maximum"]))
        return int(number) if schema_type == "integer" else number
    return value


def _known_schema_fallback(schema: Mapping[str, Any], *, raw_text: str, role_hint: str | None) -> dict[str, Any] | None:
    required = {str(item) for item in schema.get("required", [])}
    if {"title", "idea_summary", "motivation", "novelty_checks", "calculation_plan", "validation_checks", "risks"} <= required:
        excerpt = _excerpt(raw_text, limit=4000)
        title = _first_nonempty_line(raw_text)[:120] or "Recovered unstructured idea"
        return {
            "title": title,
            "idea_summary": excerpt,
            "motivation": excerpt,
            "novelty_checks": [],
            "calculation_plan": excerpt,
            "validation_checks": [],
            "risks": ["This output was recovered from non-schema text; review carefully."],
        }
    if {"result_summary", "derivation", "assumptions", "validity_scope", "final_result", "work_note_assessment"} <= required:
        excerpt = _excerpt(raw_text, limit=4000)
        return {
            "result_summary": _first_nonempty_line(raw_text)[:500] or "Recovered unstructured calculation output",
            "derivation": excerpt,
            "assumptions": "Recovered from non-schema output; assumptions were not separately structured.",
            "validity_scope": "Uncertain: output required structured recovery. Reviewer must verify before acceptance.",
            "final_result": excerpt,
            "work_note_assessment": {
                "needs_revision": True,
                "issue_type": "other",
                "proposed_revision": None,
                "rationale": (
                    "The proposer output did not fully comply with the required JSON schema; "
                    "do not accept without reviewer/human inspection."
                ),
                "can_continue_without_revision": False,
            },
        }
    if {"schema_version", "controller", "proposer_messages", "review_payload"} <= required:
        return _reviewer_envelope_fallback(schema, raw_text=raw_text)
    return None


def _reviewer_envelope_fallback(schema: Mapping[str, Any], *, raw_text: str) -> dict[str, Any]:
    props = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
    proposer_schema = props.get("proposer_messages", {}) if isinstance(props.get("proposer_messages"), Mapping) else {}
    proposer_ids = [str(item) for item in proposer_schema.get("required", [])]
    excerpt = _excerpt(raw_text, limit=2000)
    review_payload_schema = props.get("review_payload", {}) if isinstance(props.get("review_payload"), Mapping) else {}
    return {
        "schema_version": _const_for(props.get("schema_version"), "arc.llm.review_envelope.v1"),
        "controller": _controller_fallback(props.get("controller"), excerpt),
        "proposer_messages": {
            proposer_id: {"message": excerpt or "Reviewer output was malformed; recalculate or inspect manually."}
            for proposer_id in proposer_ids
        },
        "review_payload": _review_payload_fallback(review_payload_schema, proposer_ids=proposer_ids, excerpt=excerpt),
    }


def _controller_fallback(schema: Any, excerpt: str) -> dict[str, Any]:
    props = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    result = {
        "message": "Reviewer output required structured recovery. Treat this review as low confidence.",
        "stop_requested": False,
    }
    if "stop_reason" in props:
        result["stop_reason"] = None
    return result


def _review_payload_fallback(schema: Any, *, proposer_ids: list[str], excerpt: str) -> dict[str, Any]:
    props = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    result: dict[str, Any] = {}
    if "marks" in props:
        mark_schema = props.get("marks") if isinstance(props.get("marks"), Mapping) else {}
        fields = [str(item) for item in mark_schema.get("required", [])]
        if not fields and isinstance(mark_schema.get("properties"), Mapping):
            fields = [str(key) for key in mark_schema["properties"]]
        result["marks"] = {field: 0 for field in fields}
    if "reviewer_benchmark" in props:
        result["reviewer_benchmark"] = {
            "same_direction_alternative": "Recovered reviewer output did not provide a benchmark.",
            "preserves_proposer_direction": False,
            "comparison": excerpt or "Reviewer output was malformed.",
        }
    if "improvement_comments" in props:
        result["improvement_comments"] = [excerpt or "Reviewer output required structured recovery."]
    if "evidence_checked" in props:
        result["evidence_checked"] = []
    if "tool_queries_used" in props:
        result["tool_queries_used"] = []
    if "consensus" in props:
        result["consensus"] = _consensus_fallback(props["consensus"], proposer_ids=proposer_ids, excerpt=excerpt)
    return result


def _consensus_fallback(schema: Mapping[str, Any], *, proposer_ids: list[str], excerpt: str) -> dict[str, Any]:
    props = schema.get("properties", {}) if isinstance(schema.get("properties"), Mapping) else {}
    status_schema = props.get("status", {}) if isinstance(props.get("status"), Mapping) else {}
    workflow_action_schema = props.get("workflow_action", {}) if isinstance(props.get("workflow_action"), Mapping) else {}
    return {
        "status": _safe_enum(list(status_schema.get("enum") or ["unresolved"])),
        "accepted_result": None,
        "agreed_proposer_ids": [],
        "likely_wrong_proposer_ids": proposer_ids,
        "recalculate_proposer_ids": proposer_ids,
        "validity_scope": "Uncertain: reviewer output required structured recovery.",
        "analysis": excerpt or "Reviewer output required structured recovery; forcing unresolved status.",
        "agreement_assessment": _agreement_assessment_fallback(props.get("agreement_assessment")),
        "best_written_proposer_id": None,
        "best_written_selection_reason": "",
        "source_discrepancies": [],
        "workflow_action": _workflow_action_fallback(workflow_action_schema),
    }


def _agreement_assessment_fallback(schema: Any) -> dict[str, Any]:
    props = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    result: dict[str, Any] = {}
    for key in props:
        child = props[key]
        if isinstance(child, Mapping) and child.get("type") == "boolean":
            result[str(key)] = False
        elif isinstance(child, Mapping) and child.get("type") == "array":
            result[str(key)] = []
        else:
            result[str(key)] = "Reviewer output required structured recovery."
    return result


def _workflow_action_fallback(schema: Mapping[str, Any]) -> dict[str, Any]:
    props = schema.get("properties", {}) if isinstance(schema.get("properties"), Mapping) else {}
    action_schema = props.get("action", {}) if isinstance(props.get("action"), Mapping) else {}
    issue_schema = props.get("issue_type", {}) if isinstance(props.get("issue_type"), Mapping) else {}
    actions = list(action_schema.get("enum") or [])
    issues = list(issue_schema.get("enum") or [])
    action = "pause_for_human" if "pause_for_human" in actions else ("retry" if "retry" in actions else _safe_enum(actions))
    return {
        "action": action,
        "requires_human": True,
        "issue_type": "worker_failure" if "worker_failure" in issues else _safe_enum(issues),
        "proposed_revision": None,
        "reason": "Reviewer output required major structured-output recovery.",
        "expert_question": "The reviewer returned malformed structured output. Should this step be retried or inspected manually?",
    }


def _schema_default(schema: Mapping[str, Any], *, warnings: list[str]) -> Any:
    return _normalize_value(None, schema, warnings=warnings, fill_required=True)


def _validates(value: Any, schema: Mapping[str, Any]) -> bool:
    try:
        validate_json_schema(instance=value, schema=dict(schema))
    except (JsonSchemaValidationError, JsonSchemaError):
        return False
    return True


def _const_for(schema: Any, fallback: str) -> str:
    if isinstance(schema, Mapping) and isinstance(schema.get("const"), str):
        return schema["const"]
    return fallback


def _schema_allows_null(schema: Mapping[str, Any]) -> bool:
    schema_type = schema.get("type")
    return schema_type == "null" or (isinstance(schema_type, list) and "null" in schema_type)


def _safe_enum(values: list[Any]) -> Any:
    for preferred in ("unresolved", "retry", "none", None):
        if preferred in values:
            return preferred
    return values[0] if values else None


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().strip("#*- ")
        if stripped:
            return stripped
    return ""


def _excerpt(text: str, *, limit: int) -> str:
    cleaned = text.strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3].rstrip() + "..."


def _max_severity(left: str, right: str) -> str:
    order = {"none": 0, "minor": 1, "major": 2, "fatal": 3}
    return left if order.get(left, 0) >= order.get(right, 0) else right
