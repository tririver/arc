import json

from arc_llm.structured_recovery import recover_json_output


def test_recover_json_output_valid_schema_object_is_unchanged():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
    }
    value = {"ok": True}

    recovered = recover_json_output(
        value=value,
        schema=schema,
        raw_text=json.dumps(value),
        strict_first=True,
    )

    assert recovered.value == value
    assert recovered.structured_output is None


def test_recovery_preserves_additional_properties_when_allowed():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": True,
    }

    recovered = recover_json_output(
        value={"a": "x", "b": "y"},
        schema=schema,
        raw_text="",
        strict_first=False,
    )

    assert recovered.value == {"a": "x", "b": "y"}


def test_recovery_normalizes_additional_properties_schema():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": {"type": "string"},
    }

    recovered = recover_json_output(
        value={"a": "x", "b": 3},
        schema=schema,
        raw_text="",
        strict_first=False,
    )

    assert recovered.value == {"a": "x", "b": "3"}


def test_recovery_still_drops_additional_properties_when_forbidden():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": False,
    }

    recovered = recover_json_output(
        value={"a": "x", "b": "y"},
        schema=schema,
        raw_text="",
        strict_first=False,
    )

    assert recovered.value == {"a": "x"}
    assert any("Dropped extra properties" in warning for warning in recovered.structured_output["warnings"])
