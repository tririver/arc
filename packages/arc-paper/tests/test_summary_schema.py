import pytest
from importlib.resources import files
from jsonschema import ValidationError

from arc_paper.summary.schema import load_summary_prompt, load_summary_schema, validate_summary


def valid_summary():
    return {
        "schema_version": "arc.paper_llm_summary.v1",
        "paper_id": "arXiv:0911.3380",
        "title": "A Test Paper",
        "authors_short": "Alice and Bob",
        "high_value_summary": ["The paper computes a useful result."],
        "toc": [
            {
                "section_id": "S1",
                "title": "1 Introduction",
                "level": 2,
            }
        ],
        "section_summaries": [
            {
                "section_id": "S1",
                "title": "1 Introduction",
                "summary": "Introduces the problem.",
                "warnings": [],
            }
        ],
        "reading_guide": [
            {
                "purpose": "Understand the main result",
                "sections": ["S1"],
                "reason": "This section defines the setup.",
            }
        ],
        "warnings": [],
        "provenance": {
            "created_at": "2026-05-22T00:00:00Z",
            "method": "manual",
            "model": "test-model",
            "prompt_version": "paper-summary-v1",
            "source_hash": "a" * 64,
        },
    }


def test_summary_schema_loads():
    schema = load_summary_schema()
    assert schema["$id"] == "arc.paper-summary-v1"


def test_summary_resources_are_package_local():
    summary_package = files("arc_paper.summary")
    assert (summary_package / "schemas" / "paper-summary-v1.schema.json").is_file()
    assert (summary_package / "prompts" / "paper-summary-v1.md").is_file()


def test_summary_prompt_loads_from_package_resource():
    prompt = load_summary_prompt()
    assert "You are summarizing a physics paper" in prompt


def test_valid_summary_passes_schema():
    validate_summary(valid_summary())


def test_missing_title_fails_schema():
    payload = valid_summary()
    payload.pop("title")
    with pytest.raises(ValidationError):
        validate_summary(payload)
