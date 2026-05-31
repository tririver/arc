from __future__ import annotations

from copy import deepcopy
from typing import Any


ARC_LLM_CALL_RECORD_FIELD = "arc_llm_call_record"
ARC_LLM_CALL_RECORD_SCHEMA_VERSION = "arc.llm.call_record.v2"

ARC_LLM_CALL_RECORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "provider_requested",
        "model_requested",
        "model_tier_requested",
        "provider_used",
        "model_used",
        "fallback_index",
        "attempt",
        "host",
        "signals",
        "attempts",
        "session_policy",
        "usage",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": ARC_LLM_CALL_RECORD_SCHEMA_VERSION},
        "provider_requested": {"type": "string"},
        "model_requested": {"type": ["string", "null"]},
        "model_tier_requested": {"type": ["string", "null"]},
        "provider_used": {"type": "string"},
        "model_used": {"type": ["string", "null"]},
        "fallback_index": {"type": "integer", "minimum": 0},
        "attempt": {"type": "integer", "minimum": 1},
        "host": {"type": "string"},
        "signals": {"type": "array", "items": {"type": "string"}},
        "attempts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "provider",
                    "model",
                    "fallback_index",
                    "attempt",
                    "status",
                    "error_type",
                    "message",
                ],
                "properties": {
                    "provider": {"type": "string"},
                    "model": {"type": ["string", "null"]},
                    "fallback_index": {"type": "integer", "minimum": 0},
                    "attempt": {"type": "integer", "minimum": 1},
                    "status": {"type": "string"},
                    "error_type": {"type": ["string", "null"]},
                    "message": {"type": ["string", "null"]},
                },
            },
        },
        "session_policy": {"type": "string"},
        "session_key": {"type": ["string", "null"]},
        "native_session_id": {"type": ["string", "null"]},
        "call_label": {"type": ["string", "null"]},
        "prompt_sha256": {"type": ["string", "null"]},
        "static_prefix_sha256": {"type": ["string", "null"]},
        "schema_sha256": {"type": ["string", "null"]},
        "runtime_fingerprint": {"type": ["string", "null"]},
        "usage": {"type": "object"},
    },
}


def allow_arc_llm_call_record(schema: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(schema)


def attach_arc_llm_call_record(payload: dict[str, Any], call_record: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result[ARC_LLM_CALL_RECORD_FIELD] = call_record
    return result


def strip_arc_llm_call_records(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_arc_llm_call_records(item)
            for key, item in value.items()
            if key != ARC_LLM_CALL_RECORD_FIELD
        }
    if isinstance(value, list):
        return [strip_arc_llm_call_records(item) for item in value]
    return value
