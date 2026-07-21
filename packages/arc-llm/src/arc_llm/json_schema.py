from __future__ import annotations

from copy import deepcopy
from typing import Any

from .call_record import ARC_LLM_CALL_RECORD_FIELD


class CodexSchemaError(ValueError):
    """Raised before provider invocation for an unsupported strict schema."""


_SCHEMA_MAP_KEYS = ("$defs", "definitions", "dependentSchemas", "patternProperties", "properties")
_SCHEMA_KEYS = (
    "additionalItems",
    "additionalProperties",
    "contains",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
)
_SCHEMA_LIST_KEYS = ("allOf", "anyOf", "items", "oneOf", "prefixItems")


def to_provider_json_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a provider-facing schema.

    `arc_llm_call_record` is ARC audit metadata attached after provider output,
    so the model should not be asked to generate it. Codex structured output
    also requires every object schema to set additionalProperties=false.
    """
    if schema is None:
        return None
    normalized = deepcopy(schema)
    _normalize_schema_node(normalized)
    return normalized


def validate_codex_strict_schema(schema: dict[str, Any] | None) -> None:
    """Reject schemas Codex structured output cannot accept."""

    if schema is None:
        return
    errors: list[str] = []
    _validate_codex_node(schema, path="$", errors=errors)
    if errors:
        raise CodexSchemaError("Codex strict JSON schema is invalid: " + "; ".join(errors))


def _validate_codex_node(node: Any, *, path: str, errors: list[str]) -> None:
    if isinstance(node, list):
        for index, child in enumerate(node):
            _validate_codex_node(child, path=f"{path}[{index}]", errors=errors)
        return
    if not isinstance(node, dict):
        return
    if "oneOf" in node:
        errors.append(f"{path}.oneOf is not supported; use anyOf or a type union")
    if _is_object_schema(node):
        properties = node.get("properties", {})
        if not isinstance(properties, dict):
            errors.append(f"{path}.properties must be an object")
        else:
            required = node.get("required")
            if properties and not isinstance(required, list):
                errors.append(f"{path}.required must list every property")
            elif isinstance(required, list):
                names = {item for item in required if isinstance(item, str)}
                missing = sorted(set(properties) - names)
                if missing:
                    errors.append(f"{path}.required is missing {missing}")
        if node.get("additionalProperties") is not False:
            errors.append(f"{path}.additionalProperties must be false")
    for key in _SCHEMA_MAP_KEYS:
        value = node.get(key)
        if isinstance(value, dict):
            for child_key, child in value.items():
                _validate_codex_node(child, path=f"{path}.{key}.{child_key}", errors=errors)
    for key in _SCHEMA_KEYS:
        value = node.get(key)
        if isinstance(value, dict):
            _validate_codex_node(value, path=f"{path}.{key}", errors=errors)
    for key in _SCHEMA_LIST_KEYS:
        value = node.get(key)
        if isinstance(value, list):
            for index, child in enumerate(value):
                _validate_codex_node(child, path=f"{path}.{key}[{index}]", errors=errors)


def _normalize_schema_node(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _normalize_schema_node(item)
        return
    if not isinstance(node, dict):
        return

    if _is_object_schema(node):
        node["additionalProperties"] = False
        required = node.get("required")
        if isinstance(required, list):
            node["required"] = [item for item in required if item != ARC_LLM_CALL_RECORD_FIELD]
        properties = node.get("properties")
        if isinstance(properties, dict):
            properties.pop(ARC_LLM_CALL_RECORD_FIELD, None)

    _normalize_child_schemas(node)


def _normalize_child_schemas(node: dict[str, Any]) -> None:
    for key in _SCHEMA_MAP_KEYS:
        value = node.get(key)
        if isinstance(value, dict):
            for child in value.values():
                _normalize_schema_node(child)

    for key in _SCHEMA_KEYS:
        value = node.get(key)
        if isinstance(value, dict):
            _normalize_schema_node(value)

    for key in _SCHEMA_LIST_KEYS:
        value = node.get(key)
        if isinstance(value, list):
            for child in value:
                _normalize_schema_node(child)


def _is_object_schema(node: dict[str, Any]) -> bool:
    schema_type = node.get("type")
    if schema_type == "object":
        return True
    if isinstance(schema_type, list) and "object" in schema_type:
        return True
    return "properties" in node
