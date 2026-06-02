import json
import builtins

from arc_llm.structured_recovery import parse_json_object_relaxed, recover_json_output


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


def test_relaxed_parser_repairs_truncated_fenced_json_object():
    text = """```json
{"title": "x", "items": ["a", "b"]
```"""

    parsed, warnings = parse_json_object_relaxed(text)

    assert parsed == {"title": "x", "items": ["a", "b"]}
    assert any("repair" in warning.lower() for warning in warnings)


def test_relaxed_parser_repairs_truncated_fenced_json_without_json_repair(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "json_repair":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    parsed, warnings = parse_json_object_relaxed('```json\n{"title": "x", "items": ["a", "b"]\n```')

    assert parsed == {"title": "x", "items": ["a", "b"]}
    assert any("repair" in warning.lower() for warning in warnings)


def test_relaxed_parser_keeps_plain_text_as_unstructured_fallback():
    parsed, warnings = parse_json_object_relaxed("plain calculation answer")

    assert parsed is None
    assert warnings == ["No JSON object could be extracted."]


def test_relaxed_parser_does_not_repair_rootless_json_fragment():
    text = """Continuing from where output was cut off:

```json
    "Massless limit $m=0$ ($\\nu=3/2$): $K_{3/2}(t)$": "check",
    "risks": []
}
```"""

    parsed, warnings = parse_json_object_relaxed(text)

    assert parsed is None
    assert warnings == ["No JSON object could be extracted."]


def test_workflow_action_fallback_prefers_manual_action_over_retry():
    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["retry", "manual_inspection", "continue_with_warning"],
            },
            "issue_type": {"type": "string", "enum": ["worker_failure"]},
        },
    }
    review_schema = {
        "type": "object",
        "required": ["schema_version", "controller", "proposer_messages", "review_payload"],
        "properties": {
            "schema_version": {"type": "string"},
            "controller": {"type": "object", "properties": {}},
            "proposer_messages": {"type": "object", "properties": {}, "required": []},
            "review_payload": {
                "type": "object",
                "properties": {
                    "consensus": {
                        "type": "object",
                        "properties": {
                            "workflow_action": schema,
                        },
                    }
                },
            },
        },
    }

    recovered = recover_json_output(value={}, schema=review_schema, raw_text="bad reviewer", role_hint="reviewer", strict_first=False)

    assert recovered.value["review_payload"]["consensus"]["workflow_action"]["action"] == "manual_inspection"


def test_workflow_action_fallback_uses_retry_only_when_only_choice():
    review_schema = {
        "type": "object",
        "required": ["schema_version", "controller", "proposer_messages", "review_payload"],
        "properties": {
            "schema_version": {"type": "string"},
            "controller": {"type": "object", "properties": {}},
            "proposer_messages": {"type": "object", "properties": {}, "required": []},
            "review_payload": {
                "type": "object",
                "properties": {
                    "consensus": {
                        "type": "object",
                        "properties": {
                            "workflow_action": {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string", "enum": ["retry"]},
                                    "issue_type": {"type": "string", "enum": ["worker_failure"]},
                                },
                            },
                        },
                    }
                },
            },
        },
    }

    recovered = recover_json_output(value={}, schema=review_schema, raw_text="bad reviewer", role_hint="reviewer", strict_first=False)

    assert recovered.value["review_payload"]["consensus"]["workflow_action"]["action"] == "retry"


def test_calculation_recovery_requires_revision_before_continuing():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "result_summary",
            "derivation",
            "assumptions",
            "validity_scope",
            "final_result",
            "work_note_assessment",
        ],
        "properties": {
            "result_summary": {"type": "string"},
            "derivation": {"type": "string"},
            "assumptions": {"type": "string"},
            "validity_scope": {"type": "string"},
            "final_result": {"type": "string"},
            "work_note_assessment": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "needs_revision",
                    "issue_type",
                    "proposed_revision",
                    "rationale",
                    "can_continue_without_revision",
                ],
                "properties": {
                    "needs_revision": {"type": "boolean"},
                    "issue_type": {"type": "string"},
                    "proposed_revision": {"type": ["string", "null"]},
                    "rationale": {"type": "string"},
                    "can_continue_without_revision": {"type": "boolean"},
                },
            },
        },
    }

    recovered = recover_json_output(value={}, schema=schema, raw_text="plain calculation text", strict_first=False)

    assert recovered.value["work_note_assessment"]["needs_revision"] is True
    assert recovered.value["work_note_assessment"]["can_continue_without_revision"] is False
