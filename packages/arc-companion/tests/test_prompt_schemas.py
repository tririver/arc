from __future__ import annotations

from typing import Any

import jsonschema

from arc_llm.json_schema import to_provider_json_schema

from arc_companion.prompts import (
    ANNOTATION_SCHEMA,
    CUT_SCHEMA,
    GLOSSARY_SCHEMA,
    REVIEW_SCHEMA,
    SECTION_REVIEW_SCHEMA,
    TRANSLATION_COVERAGE_REPAIR_SCHEMA,
    TRANSLATION_SCHEMA,
    TRANSLATION_SLOT_REPAIR_SCHEMA,
    annotation_prompt,
    review_prompt,
    section_review_prompt,
)


def _assert_codex_strict_objects(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _assert_codex_strict_objects(item)
        return
    if not isinstance(node, dict):
        return
    schema_type = node.get("type")
    if schema_type == "object" or isinstance(schema_type, list) and "object" in schema_type:
        assert node.get("additionalProperties") is False
        properties = node.get("properties") or {}
        assert set(node.get("required") or []) == set(properties)
    for value in node.values():
        _assert_codex_strict_objects(value)


def test_all_companion_schemas_satisfy_codex_strict_object_contract() -> None:
    schemas = (
        CUT_SCHEMA,
        GLOSSARY_SCHEMA,
        TRANSLATION_SCHEMA,
        TRANSLATION_COVERAGE_REPAIR_SCHEMA,
        TRANSLATION_SLOT_REPAIR_SCHEMA,
        ANNOTATION_SCHEMA,
        REVIEW_SCHEMA,
        SECTION_REVIEW_SCHEMA,
    )

    for schema in schemas:
        provider_schema = to_provider_json_schema(schema)
        assert provider_schema is not None
        _assert_codex_strict_objects(provider_schema)


def test_review_patch_uses_null_for_unchanged_optional_replacements() -> None:
    patch = REVIEW_SCHEMA["properties"]["patches"]["items"]

    assert set(patch["required"]) == set(patch["properties"])
    for field in (
        "translation_blocks",
        "commentary",
        "explanation",
        "prior_work",
        "later_work",
        "evidence_ids",
    ):
        assert "null" in patch["properties"][field]["type"]


def test_annotation_and_review_schemas_allow_intentionally_empty_explanation() -> None:
    annotation = {
        "explanation": "",
        "prior_work": [],
        "later_work": [],
        "context_claims": [],
        "commentary": "",
        "evidence_ids": [],
        "key_points": [],
        "source_notes": [],
        "evidence_requests": [],
    }
    jsonschema.validate(annotation, ANNOTATION_SCHEMA)

    patch = {
        "segment_id": "seg-1",
        "translation_blocks": None,
        "commentary": "",
        "explanation": "",
        "prior_work": None,
        "later_work": None,
        "evidence_ids": None,
        "reason": "remove commentary that only repeats an evident passage",
    }
    jsonschema.validate({"patches": [patch], "issues": []}, REVIEW_SCHEMA)


def test_generation_and_review_prompts_treat_explanation_as_reader_driven() -> None:
    generation = annotation_prompt(
        {"segment_id": "seg-1", "block_ids": ["b1"]},
        [{"block_id": "b1", "text": "A direct statement."}],
        language="zh-CN",
        metadata={},
        evidence={"papers": []},
        glossary={"entries": []},
        protected_names=[],
        paper_context={},
    )
    reviews = " ".join(
        (
            review_prompt({"segments": []}, language="zh-CN"),
            section_review_prompt({"segments": []}, language="zh-CN"),
        )
    )

    assert "Explanation is optional" in generation
    assert "opening of a section or chapter" in generation
    assert "alternative presentation" in generation
    assert "copied verbatim" in generation
    assert "separate from source_locators" in generation
    assert "equivalent formulation as an inconsistency" in generation
    assert "intermediate mathematics" in generation
    assert "materially useful current understanding or development" in generation
    assert "registered, verifiable evidence" in generation
    assert "Do not chase novelty" in generation
    assert "empty explanation/commentary is valid" in reviews
    assert "notation, convention, normalization" in reviews
    assert "materially useful current understanding or developments" in reviews
    assert "registered, verifiable evidence" in reviews
