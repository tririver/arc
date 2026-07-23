from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from arc_companion.source_credit import (
    SOURCE_CREDIT_VERSION,
    SourceCreditError,
    normalize_source_credit,
    ordered_source_credit_items,
    project_source_credit,
    source_credit_hash,
    validate_source_credit,
)


def _field_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _document() -> dict:
    return {
        "front_matter": {
            "authors": [
                {"source_id": "author-a", "source_name": "Alice 原"},
                {"source_id": "author-b", "source_name": "Bob"},
            ],
            "affiliations": [
                {"source_id": "aff-a", "text": "Institute A", "block_id": "f2"},
                {"source_id": "aff-b", "text": "Institute B"},
            ],
            "author_profiles": [
                {
                    "source_id": "bio-a",
                    "text": "Alice studies long-form theory.\nSecond line.",
                    "author_id": "author-a",
                    "block_id": "f3",
                },
            ],
            "author_affiliations": [
                {"author_id": "author-a", "affiliation_id": "aff-a"},
            ],
            "author_name_variants": [
                {
                    "author_id": "author-a",
                    "localized_name": "爱丽丝",
                    "source_identity": "source:fixture#alice-localized",
                },
            ],
            "block_ids": {"authors": ["f1"]},
        },
        "blocks": [
            {"block_id": "title", "order": 0, "text": "Title"},
            {"block_id": "f1", "order": 1, "text": "Alice 原; Bob"},
            {"block_id": "f2", "order": 2, "text": "Institute A"},
            {"block_id": "f3", "order": 3, "text": "Profile"},
        ],
    }


def test_original_names_remain_primary_and_source_variant_is_adjacent() -> None:
    credit = normalize_source_credit(_document())

    assert credit["schema_version"] == SOURCE_CREDIT_VERSION
    assert [item["source_name"] for item in credit["authors"]] == ["Alice 原", "Bob"]
    assert credit["authors"][0]["localized_name"] == "爱丽丝"
    assert credit["authors"][0]["localized_evidence"] == {
        "evidence_class": "source_variant",
        "reference_identity": "source:fixture#alice-localized",
        "field_sha256": credit["authors"][0]["localized_evidence"]["field_sha256"],
    }
    assert credit["authors"][1]["localized_name"] is None


def test_singleton_cached_reference_maps_only_one_to_one() -> None:
    source = {"front_matter": {"authors": ["Original"]}, "blocks": []}
    reference = {
        "identity": "arxiv:1234.5678",
        "document": {"front_matter": {"author_records": [{
            "source_id": "ref-author",
            "source_name": "本地名",
            "field_sha256": _field_sha256("本地名"),
        }]}},
    }

    credit = normalize_source_credit(source, cached_reference=reference)

    assert credit["authors"][0]["source_name"] == "Original"
    assert credit["authors"][0]["localized_name"] == "本地名"
    assert credit["authors"][0]["localized_evidence"]["evidence_class"] == (
        "cached_reference"
    )
    assert credit["authors"][0]["localized_evidence"]["reference_identity"] == (
        "arxiv:1234.5678"
    )


def test_multi_author_cached_reference_is_ignored_without_explicit_mapping() -> None:
    source = {"front_matter": {"authors": ["A", "B"]}, "blocks": []}
    reference = {
        "identity": "arxiv:ref",
        "authors": ["甲", "乙"],
    }

    credit = normalize_source_credit(source, cached_reference=reference)

    assert [item["localized_name"] for item in credit["authors"]] == [None, None]


def test_valid_explicit_multi_author_mapping_requires_complete_unique_evidence() -> None:
    source = {
        "front_matter": {
            "authors": [
                {"source_id": "a", "name": "A"},
                {"source_id": "b", "name": "B"},
            ],
        },
        "blocks": [],
    }
    reference = {
        "identity": "arxiv:ref",
        "document": {
            "front_matter": {
                "author_records": [
                    {
                        "source_id": "ref-a",
                        "source_name": "甲",
                        "field_sha256": _field_sha256("甲"),
                    },
                    {
                        "source_id": "ref-b",
                        "source_name": "乙",
                        "field_sha256": _field_sha256("乙"),
                    },
                ],
            },
        },
    }
    mapping = [
        {
            "source_author_id": "a",
            "reference_author_id": "ref-b",
            "reference_identity": "arxiv:ref",
        },
        {
            "source_author_id": "b",
            "reference_author_id": "ref-a",
            "reference_identity": "arxiv:ref",
        },
    ]

    credit = normalize_source_credit(
        source,
        cached_reference=reference,
        explicit_author_mapping=mapping,
    )

    assert [item["localized_name"] for item in credit["authors"]] == ["乙", "甲"]


@pytest.mark.parametrize(
    ("mapping", "diagnostic_code"),
    [
        (
            [
                {
                    "source_author_id": "a",
                    "reference_author_id": "ref-a",
                    "reference_identity": "arxiv:ref",
                },
                {
                    "source_author_id": "a",
                    "reference_author_id": "ref-b",
                    "reference_identity": "arxiv:ref",
                },
            ],
            "source_credit_author_mapping_duplicate_identity",
        ),
        (
            [
                {
                    "source_author_id": "a",
                    "reference_author_id": "ref-a",
                    "reference_identity": "arxiv:ref",
                },
                {
                    "source_author_id": "b",
                    "reference_author_id": "ref-a",
                    "reference_identity": "arxiv:ref",
                },
            ],
            "source_credit_author_mapping_duplicate_identity",
        ),
        (
            [{
                "source_author_id": "a",
                "reference_author_id": "ref-a",
                "reference_identity": "arxiv:ref",
            }],
            "source_credit_author_mapping_cardinality_mismatch",
        ),
        (
            [
                {
                    "source_author_id": "a",
                    "reference_author_id": "unknown",
                    "reference_identity": "arxiv:ref",
                },
                {
                    "source_author_id": "b",
                    "reference_author_id": "ref-b",
                    "reference_identity": "arxiv:ref",
                },
            ],
            "source_credit_author_mapping_target_evidence_invalid",
        ),
        (
            [
                {
                    "source_author_id": "a",
                    "reference_author_id": "ref-a",
                    "reference_identity": "",
                },
                {
                    "source_author_id": "b",
                    "reference_author_id": "ref-b",
                    "reference_identity": "arxiv:ref",
                },
            ],
            "source_credit_author_mapping_reference_identity_invalid",
        ),
        (
            [
                {
                    "author_index": 0,
                    "reference_name": "甲",
                    "reference_identity": "arxiv:ref",
                },
                {
                    "author_index": 1,
                    "reference_name": "乙",
                    "reference_identity": "arxiv:ref",
                },
            ],
            "source_credit_author_mapping_identity_keys_required",
        ),
    ],
)
def test_duplicate_ambiguous_incomplete_or_unverified_mapping_is_ignored(
    mapping: list[dict],
    diagnostic_code: str,
) -> None:
    source = {
        "front_matter": {
            "authors": [
                {"source_id": "a", "name": "A"},
                {"source_id": "b", "name": "B"},
            ],
        },
        "blocks": [],
    }
    reference = {
        "identity": "arxiv:ref",
        "author_records": [
            {
                "source_id": "ref-a",
                "source_name": "甲",
                "field_sha256": _field_sha256("甲"),
            },
            {
                "source_id": "ref-b",
                "source_name": "乙",
                "field_sha256": _field_sha256("乙"),
            },
        ],
    }
    diagnostics: list[dict[str, str]] = []

    credit = normalize_source_credit(
        source,
        cached_reference=reference,
        explicit_author_mapping=mapping,
        diagnostics=diagnostics,
    )

    assert [item["localized_name"] for item in credit["authors"]] == [None, None]
    assert [item["code"] for item in diagnostics] == [diagnostic_code]


def test_reference_author_records_preserve_same_name_distinct_identities() -> None:
    source = {
        "front_matter": {
            "author_records": [
                {"source_id": "source-a", "source_name": "A"},
                {"source_id": "source-b", "source_name": "B"},
            ],
        },
        "blocks": [],
    }
    reference = {
        "identity": "arxiv:ref",
        "document": {
            "front_matter": {"authors": ["collapsed display"]},
        },
        "metadata": {
            "author_records": [
                {
                    "source_id": "target-a",
                    "source_name": "同名",
                    "field_sha256": _field_sha256("同名"),
                },
                {
                    "source_id": "target-b",
                    "source_name": "同名",
                    "field_sha256": _field_sha256("同名"),
                },
            ],
        },
    }
    mapping = [
        {
            "source_author_id": "source-a",
            "reference_author_id": "target-a",
            "reference_identity": "arxiv:ref",
        },
        {
            "source_author_id": "source-b",
            "reference_author_id": "target-b",
            "reference_identity": "arxiv:ref",
        },
    ]

    credit = normalize_source_credit(
        source,
        cached_reference=reference,
        explicit_author_mapping=mapping,
    )

    assert [item["localized_name"] for item in credit["authors"]] == [
        "同名", "同名",
    ]


def test_multi_author_name_mapping_is_rejected_with_stable_diagnostic() -> None:
    source = {
        "front_matter": {
            "author_records": [
                {"source_id": "source-a", "source_name": "Same"},
                {"source_id": "source-b", "source_name": "Same"},
            ],
        },
        "blocks": [],
    }
    reference = {
        "identity": "arxiv:ref",
        "author_records": [
            {
                "source_id": "target-a",
                "source_name": "同名",
                "field_sha256": _field_sha256("同名"),
            },
            {
                "source_id": "target-b",
                "source_name": "同名",
                "field_sha256": _field_sha256("同名"),
            },
        ],
    }
    diagnostics: list[dict[str, str]] = []

    credit = normalize_source_credit(
        source,
        cached_reference=reference,
        explicit_author_mapping=[
            {
                "source_name": "Same",
                "reference_name": "同名",
                "reference_identity": "arxiv:ref",
            },
            {
                "source_name": "Same",
                "reference_name": "同名",
                "reference_identity": "arxiv:ref",
            },
        ],
        diagnostics=diagnostics,
    )

    assert [item["localized_name"] for item in credit["authors"]] == [None, None]
    assert diagnostics == [{
        "severity": "warning",
        "code": "source_credit_author_mapping_identity_keys_required",
        "source": "source-credit",
        "message": (
            "Multi-author mapping was ignored because only stable source and "
            "reference author identities are accepted."
        ),
    }]


def test_multi_author_mapping_rejects_unvalidated_target_field_evidence() -> None:
    source = {
        "front_matter": {
            "author_records": [
                {"source_id": "source-a", "source_name": "A"},
                {"source_id": "source-b", "source_name": "B"},
            ],
        },
        "blocks": [],
    }
    reference = {
        "identity": "arxiv:ref",
        "author_records": [
            {
                "source_id": "target-a",
                "source_name": "甲",
                "field_sha256": _field_sha256("甲"),
            },
            {
                "source_id": "target-b",
                "source_name": "乙",
                "field_sha256": "0" * 64,
            },
        ],
    }
    diagnostics: list[dict[str, str]] = []

    credit = normalize_source_credit(
        source,
        cached_reference=reference,
        explicit_author_mapping=[
            {
                "source_author_id": "source-a",
                "reference_author_id": "target-a",
                "reference_identity": "arxiv:ref",
            },
            {
                "source_author_id": "source-b",
                "reference_author_id": "target-b",
                "reference_identity": "arxiv:ref",
            },
        ],
        diagnostics=diagnostics,
    )

    assert [item["localized_name"] for item in credit["authors"]] == [None, None]
    assert [item["code"] for item in diagnostics] == [
        "source_credit_author_mapping_target_evidence_invalid",
    ]


@pytest.mark.parametrize(
    ("evidence", "expected_identity"),
    [
        ({"record_id": "variant-1"}, "record:variant-1"),
        ({"block_id": "front-author-block"}, "block:front-author-block"),
        ({"field_path": "front_matter.author_variants.0"}, (
            "field:front_matter.author_variants.0"
        )),
        (
            {"field_sha256": _field_sha256("甲")},
            f"field-sha256:{_field_sha256('甲')}",
        ),
    ],
)
def test_source_variant_identity_comes_from_explicit_source_evidence(
    evidence: dict[str, str],
    expected_identity: str,
) -> None:
    document = {
        "front_matter": {
            "authors": [{"source_id": "a", "source_name": "A"}],
            "author_name_variants": [{
                "author_id": "a",
                "localized_name": "甲",
                **evidence,
            }],
        },
        "blocks": [],
    }

    credit = normalize_source_credit(document)

    assert credit["authors"][0]["localized_evidence"]["reference_identity"] == (
        expected_identity
    )


def test_source_variant_without_stable_source_identity_is_ignored() -> None:
    document = {
        "front_matter": {
            "authors": [{"source_id": "a", "source_name": "A"}],
            "author_name_variants": [{
                "author_id": "a",
                "localized_name": "甲",
            }],
        },
        "blocks": [],
    }
    diagnostics: list[dict[str, str]] = []

    credit = normalize_source_credit(document, diagnostics=diagnostics)

    assert credit["authors"][0]["localized_name"] is None
    assert [item["code"] for item in diagnostics] == [
        "source_credit_source_variant_identity_missing",
    ]
    assert credit["canonical_sha256"] == normalize_source_credit(document)[
        "canonical_sha256"
    ]


def test_reliable_metadata_profiles_are_used_when_front_profiles_are_absent() -> None:
    document = {
        "front_matter": {
            "authors": [{"source_id": "a", "source_name": "A"}],
        },
        "blocks": [],
    }
    metadata = {
        "profiles": [{
            "source_id": "profile-a",
            "text": "Metadata-authored profile.",
            "author_id": "a",
        }],
    }

    credit = normalize_source_credit(document, metadata)

    assert [item["text"] for item in credit["profiles"]] == [
        "Metadata-authored profile.",
    ]
    assert credit["profiles"][0]["author_id"] == credit["authors"][0]["id"]


def test_ambiguous_category_anchors_are_never_assigned_by_position() -> None:
    document = {
        "front_matter": {
            "author_records": [
                {"source_id": "a", "source_name": "A"},
                {"source_id": "b", "source_name": "B", "block_id": "author-b"},
            ],
            "block_ids": {"authors": ["author-a", "author-b"]},
        },
        "blocks": [
            {"block_id": "author-a", "text": "A"},
            {"block_id": "author-b", "text": "B"},
        ],
    }

    credit = normalize_source_credit(document)
    anchors = {
        author["source_name"]: next(
            anchor
            for anchor in credit["anchors"]
            if anchor["id"] == author["anchor_id"]
        )
        for author in credit["authors"]
    }

    assert anchors["A"]["placement"] == "after_title"
    assert anchors["A"]["block_id"] is None
    assert anchors["B"]["placement"] == "source"
    assert anchors["B"]["block_id"] == "author-b"


def test_reference_never_replaces_profiles_or_affiliations() -> None:
    source = {
        "front_matter": {
            "authors": ["Original"],
            "affiliations": ["Original institute"],
            "profiles": ["Original profile"],
        },
        "blocks": [],
    }
    reference = {
        "identity": "arxiv:ref",
        "authors": ["本地名"],
        "document": {
            "front_matter": {
                "authors": ["本地名"],
                "affiliations": ["Reference institute"],
                "profiles": ["Rewritten profile"],
            },
        },
    }

    credit = normalize_source_credit(source, cached_reference=reference)

    assert [item["text"] for item in credit["affiliations"]] == [
        "Original institute"
    ]
    assert [item["text"] for item in credit["profiles"]] == ["Original profile"]


def test_source_anchors_and_fallback_order_are_output_neutral() -> None:
    credit = normalize_source_credit(_document())
    items = ordered_source_credit_items(credit)

    assert [(kind, anchor["placement"]) for kind, _, anchor in items] == [
        ("affiliation", "after_title"),
        ("author", "source"),
        ("author", "source"),
        ("affiliation", "source"),
        ("profile", "source"),
    ]
    assert project_source_credit(credit) == credit
    assert source_credit_hash(project_source_credit(credit)) == credit[
        "canonical_sha256"
    ]


def test_fallback_order_is_authors_then_affiliations_then_profiles() -> None:
    document = {
        "front_matter": {
            "authors": ["A", "B"],
            "affiliations": ["X", "Y"],
            "profiles": ["P", "Q"],
        },
        "blocks": [],
    }

    items = ordered_source_credit_items(normalize_source_credit(document))

    assert [kind for kind, _, _ in items] == [
        "author", "author", "affiliation", "affiliation", "profile", "profile"
    ]
    assert all(anchor["placement"] == "after_title" for _, _, anchor in items)


def test_identity_and_content_dedupes_same_projection_but_not_equal_text() -> None:
    document = {
        "front_matter": {
            "authors": [
                {"source_id": "same", "name": "Same"},
                {"source_id": "same", "name": "Same"},
                {"source_id": "distinct", "name": "Same"},
            ],
        },
        "blocks": [],
    }

    credit = normalize_source_credit(document)

    assert len(credit["authors"]) == 2
    assert credit["authors"][0]["source_name"] == credit["authors"][1]["source_name"]
    assert credit["authors"][0]["id"] != credit["authors"][1]["id"]


def test_closed_shape_and_bound_hashes_reject_tampering() -> None:
    credit = normalize_source_credit(_document())
    extra = deepcopy(credit)
    extra["private_renderer_field"] = True
    tampered = deepcopy(credit)
    tampered["authors"][0]["source_name"] = "Changed"

    with pytest.raises(SourceCreditError, match="shape"):
        validate_source_credit(extra)
    with pytest.raises(SourceCreditError, match="hash"):
        validate_source_credit(tampered)
