import pytest
from jsonschema import ValidationError

from arc_paper_query.summary.schema import load_summary_schema, validate_summary


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
                "one_sentence_summary": "Introduces the problem.",
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


def test_valid_summary_passes_schema():
    validate_summary(valid_summary())


def test_missing_title_fails_schema():
    payload = valid_summary()
    payload.pop("title")
    with pytest.raises(ValidationError):
        validate_summary(payload)
