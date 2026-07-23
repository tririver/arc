from __future__ import annotations

from typing import Any

import jsonschema
import pytest

from arc_llm.json_schema import to_provider_json_schema

from arc_companion.chapter_guide import CHAPTER_GUIDE_SCHEMA
from arc_companion.prompts import (
    ANNOTATION_SCHEMA,
    COMMENTARY_PROMPT_VERSION,
    COMMENTARY_REVIEW_SCHEMA,
    CUT_SCHEMA,
    GLOSSARY_SCHEMA,
    REVIEW_SCHEMA,
    SECTION_REVIEW_SCHEMA,
    TRANSLATION_COVERAGE_REPAIR_SCHEMA,
    TRANSLATION_PROMPT_VERSION,
    TRANSLATION_SCHEMA,
    TRANSLATION_SLOT_REPAIR_SCHEMA,
    annotation_prompt,
    commentary_review_prompt,
    glossary_consolidation_prompt,
    glossary_prompt,
    review_prompt,
    section_review_prompt,
    translation_coverage_repair_prompt,
    translation_prompt,
    translation_reference_prompt,
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
        COMMENTARY_REVIEW_SCHEMA,
        SECTION_REVIEW_SCHEMA,
        CHAPTER_GUIDE_SCHEMA,
    )

    for schema in schemas:
        provider_schema = to_provider_json_schema(schema)
        assert provider_schema is not None
        _assert_codex_strict_objects(provider_schema)


def test_reference_translation_prompt_keeps_source_authoritative() -> None:
    prompt = translation_reference_prompt(
        {"segment_id": "seg-1"},
        [{"block_id": "b1", "text": "Source"}],
        language="zh-CN",
        glossary={},
        protected_names=["Alice"],
        paper_context={},
        reference_translation={"sections": [{"text": "Working draft"}]},
        source_language="en",
    )
    assert "working draft" in prompt
    assert "original source is authoritative" in prompt
    assert "remove its additions" in prompt
    assert '"block_id": "b1"' in prompt


def test_review_patch_uses_null_for_unchanged_optional_replacements() -> None:
    patch = REVIEW_SCHEMA["properties"]["patches"]["items"]

    assert set(patch["required"]) == set(patch["properties"])
    for field in (
        "translation_blocks",
        "commentary",
        "explanation",
        "commentary_sources",
        "prior_work",
        "later_work",
    ):
        assert "null" in patch["properties"][field]["type"]


@pytest.mark.parametrize(
    "schema",
    [REVIEW_SCHEMA, COMMENTARY_REVIEW_SCHEMA, SECTION_REVIEW_SCHEMA],
)
def test_review_schemas_use_strict_nullable_arrays_without_one_of(schema) -> None:
    provider_schema = to_provider_json_schema(schema)
    assert provider_schema is not None
    assert "oneOf" not in str(provider_schema)

    patch = provider_schema["properties"]["patches"]["items"]
    source_patch = patch["properties"]["commentary_sources"]
    assert source_patch["type"] == ["array", "null"]
    assert source_patch["maxItems"] == 3
    assert source_patch["items"] == ANNOTATION_SCHEMA["properties"]["commentary_sources"]["items"]
    assert patch["properties"]["prior_work"]["type"] == ["array", "null"]
    assert patch["properties"]["later_work"]["type"] == ["array", "null"]


def test_commentary_review_schema_rejects_translation_patch_fields() -> None:
    patch = {
        "segment_id": "seg-1",
        "commentary": None,
        "explanation": "Revised explanation.",
        "commentary_sources": None,
        "prior_work": None,
        "later_work": None,
        "reason": "clarify the passage",
    }
    jsonschema.validate({"patches": [patch], "issues": []}, COMMENTARY_REVIEW_SCHEMA)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {
                "patches": [{**patch, "translation_blocks": None}],
                "issues": [],
            },
            COMMENTARY_REVIEW_SCHEMA,
        )


def test_annotation_and_review_schemas_allow_intentionally_empty_explanation() -> None:
    annotation = {
        "explanation": "",
        "commentary": "",
        "commentary_sources": [],
        "prior_work": [],
        "later_work": [],
    }
    jsonschema.validate(annotation, ANNOTATION_SCHEMA)

    patch = {
        "segment_id": "seg-1",
        "translation_blocks": None,
        "commentary": "",
        "explanation": "",
        "commentary_sources": None,
        "prior_work": None,
        "later_work": None,
        "reason": "remove commentary that only repeats an evident passage",
    }
    jsonschema.validate({"patches": [patch], "issues": []}, REVIEW_SCHEMA)


def test_annotation_sources_require_direct_http_url_and_reader_locator() -> None:
    annotation = {
        "explanation": "A sourced fact.",
        "commentary": "",
        "commentary_sources": [{
            "title": "Primary source",
            "url": "https://example.test/paper",
            "locator": "Section 3",
        }],
        "prior_work": [{
            "text": "An earlier treatment used this convention.",
            "sources": [{
                "title": "Earlier paper",
                "url": "http://example.test/earlier",
                "locator": "p. 12",
            }],
        }],
        "later_work": [],
    }
    jsonschema.validate(annotation, ANNOTATION_SCHEMA)

    source = annotation["commentary_sources"][0]
    for missing in ("title", "url", "locator"):
        invalid = {**annotation, "commentary_sources": [{
            key: value for key, value in source.items() if key != missing
        }]}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, ANNOTATION_SCHEMA)
    for url in ("ftp://example.test/paper", "example.test/paper", ""):
        invalid = {**annotation, "commentary_sources": [{**source, "url": url}]}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, ANNOTATION_SCHEMA)


def test_annotation_limits_each_claim_to_three_sources_and_has_no_legacy_fields() -> None:
    source = {"title": "Paper", "url": "https://example.test/p", "locator": "Abstract"}
    annotation = {
        "explanation": "",
        "commentary": "",
        "commentary_sources": [],
        "prior_work": [{"text": "Claim", "sources": [source] * 4}],
        "later_work": [],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(annotation, ANNOTATION_SCHEMA)

    properties = set(ANNOTATION_SCHEMA["properties"])
    assert properties == {
        "explanation", "commentary", "commentary_sources", "prior_work", "later_work",
    }
    assert not properties.intersection({
        "key_points", "source_notes", "covered_points", "context_claims",
        "evidence_ids", "evidence_requests", "source_locators", "request_key",
    })


def test_section_review_schema_is_sparse_and_requires_exact_coverage_ids() -> None:
    value = {
        "reviewed_segment_ids": ["seg-1", "seg-2"],
        "findings": [{"segment_id": "seg-2", "issue": "incorrect sign"}],
        "patches": [],
    }
    jsonschema.validate(value, SECTION_REVIEW_SCHEMA)
    assert "reviewed_segments" not in SECTION_REVIEW_SCHEMA["properties"]


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
            commentary_review_prompt({"segments": []}, language="zh-CN"),
            section_review_prompt({"segments": []}, language="zh-CN"),
        )
    )

    assert "Explanation is optional" in generation
    assert "opening of a section or chapter" in generation
    assert "alternative presentation" in generation
    assert "same meaning" in generation
    assert "logical starting point" in generation
    assert "historical story" in generation
    assert "concept, course, or discipline" in generation
    assert "what specifically changed" in generation
    assert "reader-understandable locator" in generation
    assert "equivalent formulation as an inconsistency" in generation
    assert "intermediate mathematics" in generation
    assert "materially useful current understanding or development" in generation
    assert "host internet search" in generation
    assert "{{ARC_NESTED_SHELL_CAPABILITY}}" in generation
    assert "arc-paper-worker" not in generation
    assert "search-results page" in generation
    assert "HTTP(S) URL" in generation
    assert "native session" in generation
    assert "avoid unnecessary repetition" in generation
    assert "If internet access is disabled" in generation
    assert "positive, direct statements" in generation
    assert "Never invent a mistaken belief" in generation
    assert "full-paper equation navigation" in generation
    assert "historical importance, influence, or later status" in generation
    assert "exact title, direct HTTP(S) URL, and locator" in generation
    assert "Do not chase novelty" in generation
    assert "empty explanation/commentary is valid" in reviews
    assert "notation, convention, normalization" in reviews
    assert "materially useful current understanding or developments" in reviews
    assert "same-meaning paraphrase" in reviews
    assert "must not invent or add a source" in reviews
    assert "reviewed_segment_ids" in reviews
    assert "Never echo complete unchanged translations or annotations" in reviews
    assert "unsupported corrective contrast" in reviews
    assert "Never invent a reader's prior misconception" in reviews
    assert "equation is landmark" in reviews


def test_commentary_and_translation_prompt_recipes_are_independent() -> None:
    translation = translation_prompt(
        {"segment_id": "seg-1"}, [{"block_id": "b1", "text": "Source"}],
        language="zh-CN", glossary={}, protected_names=[], paper_context={},
    )

    assert TRANSLATION_PROMPT_VERSION == "arc.companion.prompts.v15"
    assert COMMENTARY_PROMPT_VERSION != TRANSLATION_PROMPT_VERSION
    assert "corrective contrast" not in translation
    assert "landmark equation" not in translation


def test_non_english_prompt_contract_is_source_language_neutral() -> None:
    glossary = glossary_prompt(
        [{"block_id": "b1", "text": "Le théorème de Noether"}],
        language="zh-CN",
        source_language="fr",
        protected_names=["Noether"],
        entry_limit=50,
    )
    consolidation = glossary_consolidation_prompt(
        [{"entries": []}],
        language="zh-CN",
        source_language="fr",
        protected_names=["Noether"],
        entry_limit=50,
    )
    translation = translation_prompt(
        {"segment_id": "seg-1"},
        [{"block_id": "b1", "text": "Le théorème de Noether"}],
        language="zh-CN",
        source_language="fr",
        glossary={},
        protected_names=["Noether"],
        paper_context={},
    )
    coverage_repair = translation_coverage_repair_prompt(
        {"segment_id": "seg-1"},
        [],
        language="zh-CN",
        source_language="fr",
        glossary={},
        protected_names=["Noether"],
        paper_context={},
        repair_model_tier="strong",
    )
    annotation = annotation_prompt(
        {"segment_id": "seg-1"},
        [{"block_id": "b1", "text": "Le théorème de Noether"}],
        language="zh-CN",
        source_language="fr",
        metadata={},
        evidence={},
        glossary={},
        protected_names=["Noether"],
        paper_context={},
    )

    prompts = "\n".join((glossary, consolidation, translation, coverage_repair, annotation))
    assert "source-language term exactly as it appears" in glossary
    assert "exactly in its source spelling" in prompts
    assert "English source term" not in prompts
    assert "Latin spelling" not in prompts
