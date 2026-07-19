from __future__ import annotations

from typing import Any

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
