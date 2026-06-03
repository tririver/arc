from __future__ import annotations

import pytest

from arc_llm.call_record import ARC_LLM_CALL_RECORD_FIELD
from arc_llm.schema_formatter import SchemaFormatError, format_to_schema, format_to_schema_or_retry


def _review_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "review_payload"],
        "properties": {
            "schema_version": {"type": "string", "const": "arc.llm.review_envelope.v1"},
            "review_payload": {
                "type": "object",
                "additionalProperties": False,
                "required": ["marks"],
                "properties": {
                    "marks": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["total_score", "novelty"],
                        "properties": {
                            "total_score": {"type": "number"},
                            "novelty": {"type": "number"},
                        },
                    }
                },
            },
        },
    }


def test_schema_formatter_preserves_explicit_numbers() -> None:
    calls = []

    def fake_runner(prompt: str, **kwargs):
        calls.append({"prompt": prompt, "schema": kwargs["schema"]})
        return {
            "schema_version": "arc.llm.review_envelope.v1",
            "review_payload": {"marks": {"total_score": 92, "novelty": 13}},
        }

    result = format_to_schema(
        raw_text="Final review. Total 92/100. Novelty 13/15.",
        schema=_review_schema(),
        role_hint="reviewer",
        json_runner=fake_runner,
    )

    assert result.value["review_payload"]["marks"] == {"total_score": 92, "novelty": 13}
    assert result.structured_output["recovery_strategy"] == "schema_formatter"
    assert calls


def test_schema_formatter_rejects_numbers_not_present_in_source() -> None:
    def fake_runner(prompt: str, **kwargs):
        return {
            "schema_version": "arc.llm.review_envelope.v1",
            "review_payload": {"marks": {"total_score": 88, "novelty": 12}},
        }

    with pytest.raises(SchemaFormatError, match="missing_required_numeric_fields"):
        format_to_schema(
            raw_text="Final review says ready for execution, but gives no numeric scores.",
            schema=_review_schema(),
            role_hint="reviewer",
            json_runner=fake_runner,
        )


def test_schema_formatter_strips_call_record_before_validation() -> None:
    def fake_runner(prompt: str, **kwargs):
        return {
            "schema_version": "arc.llm.review_envelope.v1",
            "review_payload": {"marks": {"total_score": 92, "novelty": 13}},
            ARC_LLM_CALL_RECORD_FIELD: {"provider_used": "manual", "attempt": 1},
        }

    result = format_to_schema(
        raw_text="Final review. Total 92/100. Novelty 13/15.",
        schema=_review_schema(),
        role_hint="reviewer",
        json_runner=fake_runner,
    )

    assert ARC_LLM_CALL_RECORD_FIELD not in result.value
    assert result.value["review_payload"]["marks"]["total_score"] == 92


def test_schema_formatter_decision_formats_valid_output() -> None:
    calls = []

    def fake_runner(prompt: str, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return {
            "action": "format",
            "reason": "numbers present",
            "formatted_output": {
                "schema_version": "arc.llm.review_envelope.v1",
                "review_payload": {"marks": {"total_score": 92, "novelty": 13}},
            },
        }

    result = format_to_schema_or_retry(
        raw_text="Final review. Total 92/100. Novelty 13/15.",
        schema=_review_schema(),
        role_hint="reviewer",
        json_runner=fake_runner,
        process_chain=["pytest"],
    )

    assert result.action == "format"
    assert result.value["review_payload"]["marks"]["total_score"] == 92
    assert result.structured_output["recovery_strategy"] == "schema_formatter"
    assert calls[0]["output_recovery"] == "strict"
    assert calls[0]["role_hint"] == "schema_formatter"
    assert calls[0]["process_chain"] == ["pytest"]


def test_schema_formatter_decision_can_request_retry() -> None:
    def fake_runner(prompt: str, **kwargs):
        return {"action": "retry", "reason": "metadata only", "formatted_output": None}

    result = format_to_schema_or_retry(
        raw_text='{"type":"result","result":""}',
        schema=_review_schema(),
        role_hint="reviewer",
        json_runner=fake_runner,
    )

    assert result.action == "retry"
    assert result.value is None
    assert result.structured_output["recovery_strategy"] == "schema_formatter_retry"


@pytest.mark.parametrize(
    "schema",
    [
        {
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["score"],
                    "properties": {"score": {"type": "number"}},
                }
            ]
        },
        {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["score"],
                    "properties": {"score": {"type": "number"}},
                }
            ]
        },
        {
            "allOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["score"],
                    "properties": {"score": {"type": "number"}},
                }
            ]
        },
        {
            "$defs": {"score_value": {"type": "number"}},
            "type": "object",
            "additionalProperties": False,
            "required": ["score"],
            "properties": {"score": {"$ref": "#/$defs/score_value"}},
        },
        {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
    ],
)
def test_schema_formatter_rejects_fabricated_numbers_for_schema_applicators(schema: dict) -> None:
    def fake_runner(prompt: str, **kwargs):
        return {"score": 77}

    with pytest.raises(SchemaFormatError, match="missing_required_numeric_fields"):
        format_to_schema(
            raw_text="Review says acceptable, but gives no numeric score.",
            schema=schema,
            role_hint="reviewer",
            json_runner=fake_runner,
        )
