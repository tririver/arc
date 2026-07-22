"""Portable JSON Schema transport policy for ARC LLM providers.

Closed object schemas may use a provider's native strict structured-output
transport.  If any object is open, native normalization would change caller
semantics, so ARC sends the canonical schema in the prompt, parses the JSON
locally, validates against the original schema, and records
``structured_output.open_object_prompt_fallback`` in the call record.  A
provider with no native schema support uses prompt transport normally and does
not emit the fallback warning.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from .call_record import ARC_LLM_CALL_RECORD_FIELD
from .schema_cache import canonical_json


STRICT_SCHEMA_PROMPT_FALLBACK_WARNING = "structured_output.open_object_prompt_fallback"


class CodexSchemaError(ValueError):
    """Raised before provider invocation for an unsupported strict schema."""


@dataclass(frozen=True)
class ProviderJSONSchemaPlan:
    """Provider-facing schema transport selected without changing caller semantics."""

    provider_schema: dict[str, Any] | None
    checkpoint_schema: dict[str, Any] | None
    prompt_fallback: bool = False
    warnings: tuple[str, ...] = ()


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
    normalized = to_prompt_json_schema(schema)
    assert normalized is not None
    _normalize_schema_node(normalized)
    return normalized


def to_prompt_json_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the canonical prompt/local-validation schema without closing objects.

    ARC attaches ``arc_llm_call_record`` after provider output.  Removing that
    metadata is safe for every transport, while changing ``additionalProperties``
    is only safe for native strict-schema transports when the caller already
    supplied a closed object schema.
    """

    if schema is None:
        return None
    prompt_schema = deepcopy(schema)
    _strip_arc_metadata(prompt_schema)
    return prompt_schema


def plan_provider_json_schema(
    schema: dict[str, Any] | None,
    *,
    provider: str,
    uses_native_schema: bool,
) -> ProviderJSONSchemaPlan:
    """Choose native strict schema or canonical prompt transport locally.

    Native strict structured-output dialects cannot preserve JSON Schema's
    open-object semantics.  For providers which may use such a dialect, ARC
    therefore sends the original schema in the prompt and performs parsing,
    validation, and configured recovery locally.  Providers without native
    schema support always receive the semantic prompt schema and need no
    fallback warning because prompt transport is their normal contract.
    """

    prompt_schema = to_prompt_json_schema(schema)
    if prompt_schema is None:
        return ProviderJSONSchemaPlan(provider_schema=None, checkpoint_schema=None)
    if not uses_native_schema:
        return ProviderJSONSchemaPlan(
            provider_schema=prompt_schema,
            checkpoint_schema=prompt_schema,
        )
    if schema_has_open_object(prompt_schema):
        return ProviderJSONSchemaPlan(
            provider_schema=None,
            checkpoint_schema=prompt_schema,
            prompt_fallback=True,
            warnings=(STRICT_SCHEMA_PROMPT_FALLBACK_WARNING,),
        )
    provider_schema = to_provider_json_schema(prompt_schema)
    if provider == "codex-cli":
        validate_codex_strict_schema(provider_schema)
    return ProviderJSONSchemaPlan(
        provider_schema=provider_schema,
        checkpoint_schema=prompt_schema,
    )


def provider_uses_native_schema(
    provider: str,
    *,
    supports_native_schema: bool,
    env: Mapping[str, str] | None,
    output_recovery: str,
) -> bool:
    """Resolve whether this call would select a native strict-schema transport."""

    if not supports_native_schema:
        return False
    if provider != "claude-cli":
        return True
    values = env or {}
    mode = str(values.get("ARC_CLAUDE_JSON_SCHEMA_MODE", "auto")).strip().lower()
    if mode not in {"auto", "provider", "prompt"}:
        raise ValueError("ARC_CLAUDE_JSON_SCHEMA_MODE must be auto, provider, or prompt")
    if mode != "auto":
        return mode == "provider"
    if output_recovery != "warn":
        return False
    warn_mode = str(values.get("ARC_CLAUDE_WARN_JSON_SCHEMA_MODE", "prompt")).strip().lower()
    if warn_mode not in {"provider", "prompt"}:
        raise ValueError("ARC_CLAUDE_WARN_JSON_SCHEMA_MODE must be provider or prompt")
    return warn_mode == "provider"


def schema_has_open_object(schema: dict[str, Any] | None) -> bool:
    """Return whether a schema contains object keys not fixed by ``properties``."""

    if schema is None:
        return False
    return _node_has_open_object(schema)


def validate_local_json_schema(schema: dict[str, Any] | None) -> None:
    """Check the caller schema before creating artifacts or invoking a provider."""

    if schema is None:
        return
    from jsonschema.exceptions import SchemaError
    from jsonschema.validators import validator_for

    try:
        validator_for(schema).check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"JSON schema is invalid: {exc.message}") from exc


def with_canonical_json_schema_contract(prompt: str, schema: dict[str, Any]) -> str:
    """Append ARC's portable JSON-object contract for prompt fallback."""

    return (
        prompt.rstrip()
        + "\n\n## JSON output contract for this turn\n"
        + "Return exactly one JSON object and no surrounding prose. Do not wrap it in Markdown.\n"
        + "The object must satisfy this canonical JSON Schema. ARC will validate it locally:\n"
        + canonical_json(schema)
        + "\n"
    )


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


def _node_has_open_object(node: Any) -> bool:
    if isinstance(node, list):
        return any(_node_has_open_object(child) for child in node)
    if not isinstance(node, dict):
        return False
    if _is_object_schema(node):
        if node.get("additionalProperties", True) is not False:
            return True
        if node.get("patternProperties"):
            return True
        if "unevaluatedProperties" in node and node.get("unevaluatedProperties") is not False:
            return True
    return any(_node_has_open_object(child) for child in _schema_children(node))


def _strip_arc_metadata(node: Any) -> None:
    if isinstance(node, list):
        for child in node:
            _strip_arc_metadata(child)
        return
    if not isinstance(node, dict):
        return
    if _is_object_schema(node):
        required = node.get("required")
        if isinstance(required, list):
            node["required"] = [item for item in required if item != ARC_LLM_CALL_RECORD_FIELD]
        properties = node.get("properties")
        if isinstance(properties, dict):
            properties.pop(ARC_LLM_CALL_RECORD_FIELD, None)
    for child in _schema_children(node):
        _strip_arc_metadata(child)


def _schema_children(node: dict[str, Any]) -> list[Any]:
    children: list[Any] = []
    for key in _SCHEMA_MAP_KEYS:
        value = node.get(key)
        if isinstance(value, dict):
            children.extend(value.values())
    for key in _SCHEMA_KEYS:
        value = node.get(key)
        if isinstance(value, dict):
            children.append(value)
    for key in _SCHEMA_LIST_KEYS:
        value = node.get(key)
        if isinstance(value, list):
            children.extend(value)
    return children


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
    return any(
        key in node
        for key in (
            "properties",
            "patternProperties",
            "additionalProperties",
            "unevaluatedProperties",
        )
    )
