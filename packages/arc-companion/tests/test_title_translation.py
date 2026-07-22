from __future__ import annotations

import json

import pytest

from arc_companion.title_translation import (
    TITLE_TRANSLATION_SCHEMA,
    TitleTranslationError,
    chunk_title_records,
    collect_title_records,
    merge_title_translation_chunks,
    normalize_title_translation_response,
    title_translation_prompt,
    validate_title_translations,
)


def test_collects_document_and_all_structural_titles_but_not_captions_or_citations() -> None:
    digest = "a" * 64
    document = {
        "front_matter": {
            "title": "Plain fallback must not win",
            "block_ids": {"title": ["paper-title"]},
        },
        "blocks": [
            {
                "block_id": "paper-title",
                "kind": "heading",
                "source_role": "front_matter_title",
                "text": "ignored because rich runs win",
                "inline_runs": [
                    {"kind": "text", "content": "Geometry "},
                    {
                        "kind": "math",
                        "token_id": "math-1",
                        "content_hash": digest,
                    },
                ],
            },
            {"block_id": "authors", "kind": "prose", "text": "A. Author"},
            {"block_id": "part", "kind": "part", "text": "I Foundations"},
            {"block_id": "chapter", "kind": "chapter", "title": "1 Spacetime"},
            {"block_id": "section", "kind": "section", "text": "1.1 Events"},
            {"block_id": "subsection", "kind": "subsection", "text": "Observers"},
            {
                "block_id": "subsubsection",
                "kind": "subsubsection",
                "text": "Clocks",
            },
            {"block_id": "references", "kind": "heading", "text": "References"},
            {"block_id": "index", "kind": "heading", "text": "Index"},
            {
                "block_id": "figure",
                "kind": "figure",
                "caption": "A diagram title",
            },
            {
                "block_id": "table",
                "kind": "table",
                "title": "Measured values",
            },
            {
                "block_id": "citation",
                "kind": "bibliography_item",
                "title": "A cited paper",
            },
            {"block_id": "body-2", "kind": "prose", "text": "Second chapter."},
        ],
    }
    chapters = {"chapters": [
        {
            "chapter_id": "ch-0001",
            "title": "1 Spacetime",
            "title_block_ids": ["chapter"],
            "block_ids": [
                "part", "chapter", "section", "subsection", "subsubsection",
                "references", "index",
            ],
        },
        {
            "chapter_id": "ch-0002",
            "title": "2 Conclusions",
            "title_block_ids": [],
            "block_ids": ["body-2"],
        },
    ]}

    records = collect_title_records(document, chapters)

    assert [item["title_id"] for item in records] == [
        "document:title",
        "block:part",
        "block:chapter",
        "block:section",
        "block:subsection",
        "block:subsubsection",
        "block:references",
        "block:index",
        "chapter:ch-0002",
    ]
    assert records[0]["source_block_ids"] == ["paper-title"]
    assert records[0]["source_text"] == (
        f"Geometry [[ARC_INLINE:math-1:{digest}]]"
    )
    assert records[1]["number_prefix"] == "I "
    assert records[2]["number_prefix"] == "1 "
    assert records[3]["number_prefix"] == "1.1 "
    assert records[-1]["block_id"] is None
    assert records[-1]["chapter_id"] == "ch-0002"


def test_document_title_falls_back_to_metadata_and_matching_chapter_heading_dedupes() -> None:
    document = {
        "metadata": {"title": "Metadata Title"},
        "blocks": [
            {"block_id": "h1", "kind": "heading", "text": "Opening"},
            {"block_id": "p1", "kind": "prose", "text": "Text"},
        ],
    }
    chapters = [{
        "chapter_id": "ch-0001",
        "title": "Opening",
        "block_ids": ["h1", "p1"],
        "title_block_ids": [],
    }]

    records = collect_title_records(document, chapters)

    assert [item["title_id"] for item in records] == [
        "document:title", "block:h1",
    ]
    assert records[0]["source_text"] == "Metadata Title"
    assert records[1]["chapter_id"] == "ch-0001"


def test_document_title_projection_ignores_nonstructural_blocks_with_stale_title_roles() -> None:
    document = {
        "front_matter": {
            "title": "Canonical Title",
            "block_ids": {"title": ["title", "author", "intro"]},
        },
        "blocks": [
            {
                "block_id": "title", "kind": "heading",
                "text": "Canonical Title", "source_role": "front_matter_title",
            },
            {
                "block_id": "author", "kind": "prose",
                "text": "By An Author", "source_role": "front_matter_title",
            },
            {
                "block_id": "intro", "kind": "prose",
                "text": "Maxwell appears only in the introduction.",
                "source_role": "front_matter_title",
            },
        ],
    }

    records = collect_title_records(document)

    assert records == [{
        "title_id": "document:title",
        "source_text": "Canonical Title",
        "role": "document",
        "block_id": None,
        "chapter_id": None,
        "source_block_ids": ["title"],
        "number_prefix": "",
        "opaque_tokens": [],
    }]


def test_document_title_projection_matches_canonical_prose_title_without_body_contamination() -> None:
    document = {
        "front_matter": {
            "title": "A Prose Title",
            "block_ids": {"title": ["title", "intro"]},
        },
        "blocks": [
            {
                "block_id": "title", "kind": "prose",
                "text": "A Prose Title", "source_role": "front_matter_title",
            },
            {
                "block_id": "intro", "kind": "prose",
                "text": "Body text.", "source_role": "front_matter_title",
            },
        ],
    }

    record = collect_title_records(document)[0]

    assert record["source_text"] == "A Prose Title"
    assert record["source_block_ids"] == ["title"]


def test_title_projection_rejects_ambiguous_source_identity() -> None:
    document = {
        "blocks": [
            {"block_id": "same", "kind": "heading", "text": "One"},
            {"block_id": "same", "kind": "heading", "text": "Two"},
        ]
    }
    with pytest.raises(TitleTranslationError, match="unique non-empty block ids"):
        collect_title_records(document)


def test_chunking_is_ordered_bounded_and_rejects_one_oversized_title() -> None:
    records = [
        {
            "title_id": f"block:{index}",
            "source_text": character * 80,
            "role": "heading",
            "number_prefix": "",
            "opaque_tokens": [],
        }
        for index, character in enumerate(("α", "Ж", "文"), 1)
    ]

    chunks = chunk_title_records(records, max_bytes=360)

    assert [item["title_id"] for chunk in chunks for item in chunk] == [
        "block:1", "block:2", "block:3",
    ]
    assert len(chunks) >= 2
    assert all(
        len(json.dumps({"titles": chunk}, ensure_ascii=False, sort_keys=True).encode())
        <= 360
        for chunk in chunks
    )
    with pytest.raises(TitleTranslationError, match="exceeds the bounded prompt size"):
        chunk_title_records(records[:1], max_bytes=20)


def test_prompt_is_source_language_neutral_and_forbids_commentary() -> None:
    records = [{
        "title_id": "block:h",
        "source_text": "La relativité d’Einstein",
        "role": "section",
        "number_prefix": "",
        "opaque_tokens": [],
    }]
    prompt = title_translation_prompt(
        records,
        source_language="fr",
        target_language="zh-CN",
        glossary={"entries": [{"source_term": "relativité", "target_term": "相对论"}]},
        protected_names=["Einstein"],
    )

    assert '"source_language": "fr"' in prompt
    assert '"target_language": "zh-CN"' in prompt
    assert "La relativité d’Einstein" in prompt
    assert "Einstein" in prompt and "相对论" in prompt
    assert "English source term" not in prompt
    assert "do not add explanations, annotations, commentary" in prompt
    assert TITLE_TRANSLATION_SCHEMA["additionalProperties"] is False


def test_normalize_accepts_host_aliases_but_emits_the_canonical_shape() -> None:
    result = normalize_title_translation_response({
        "translations": [{"id": "block:h", "translated_title": "标题"}],
    })
    assert result == {"titles": [{"title_id": "block:h", "text": "标题"}]}


def test_validation_preserves_order_prefix_tokens_and_unicode_names() -> None:
    token = f"[[ARC_INLINE:eq:{'b' * 64}]]"
    records = [
        {
            "title_id": "block:one",
            "source_text": f"1.2 Einstein and {token}",
            "number_prefix": "1.2 ",
            "opaque_tokens": [token],
        },
        {
            "title_id": "block:two",
            "source_text": "Жуковский transformation",
            "number_prefix": "",
            "opaque_tokens": [],
        },
        {
            "title_id": "block:three",
            "source_text": "爱因斯坦的理论",
            "number_prefix": "",
            "opaque_tokens": [],
        },
    ]
    response = {"titles": [
        {"title_id": "block:one", "text": f"1.2 Einstein 与 {token}"},
        {"title_id": "block:two", "text": "Жуковский 变换"},
        {"title_id": "block:three", "text": "爱因斯坦理论"},
    ]}

    assert validate_title_translations(
        records,
        response,
        protected_names=["Einstein", "Жуковский", "爱因斯坦"],
    ) == response


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"titles": []}, "exactly once"),
        (
            {"titles": [
                {"title_id": "block:b", "text": "B"},
                {"title_id": "block:a", "text": "A"},
            ]},
            "source order",
        ),
        (
            {"titles": [
                {"title_id": "block:a", "text": ""},
                {"title_id": "block:b", "text": "B"},
            ]},
            "non-empty",
        ),
    ],
)
def test_validation_fails_closed_on_coverage_order_and_empty_text(
    response: dict[str, object], message: str,
) -> None:
    records = [
        {"title_id": "block:a", "source_text": "A"},
        {"title_id": "block:b", "source_text": "B"},
    ]
    with pytest.raises(TitleTranslationError, match=message):
        validate_title_translations(records, response)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("2 新标题", "number prefix"),
        ("1 新标题", "opaque inline tokens"),
        ("1 新标题 [[ARC_INLINE:other:deadbeef]]", "opaque inline tokens"),
    ],
)
def test_validation_rejects_changed_prefix_or_opaque_tokens(text: str, message: str) -> None:
    token = f"[[ARC_INLINE:math:{'c' * 64}]]"
    records = [{
        "title_id": "block:h",
        "source_text": f"1 Einstein {token}",
        "number_prefix": "1 ",
        "opaque_tokens": [token],
    }]
    with pytest.raises(TitleTranslationError, match=message):
        validate_title_translations(
            records,
            {"titles": [{"title_id": "block:h", "text": text}]},
        )


def test_validation_restores_an_entirely_omitted_number_prefix() -> None:
    records = [{
        "title_id": "block:part",
        "source_text": "II. ELECTRODYNAMICAL PART",
        "number_prefix": "II. ",
        "opaque_tokens": [],
    }]

    assert validate_title_translations(
        records,
        {"titles": [{"title_id": "block:part", "text": "电动力学部分"}]},
    ) == {
        "titles": [{"title_id": "block:part", "text": "II. 电动力学部分"}],
    }


def test_validation_rejects_translated_or_dropped_protected_name() -> None:
    records = [{
        "title_id": "block:h",
        "source_text": "Einstein relativity",
        "number_prefix": "",
        "opaque_tokens": [],
    }]
    with pytest.raises(TitleTranslationError, match="protected names"):
        validate_title_translations(
            records,
            {"titles": [{"title_id": "block:h", "text": "爱因斯坦相对论"}]},
            protected_names=["Einstein"],
        )


def test_merge_title_translation_chunks_validates_full_document_order() -> None:
    records = [
        {"title_id": "block:a", "source_text": "A"},
        {"title_id": "block:b", "source_text": "B"},
    ]
    result = merge_title_translation_chunks(
        records,
        [
            {"titles": [{"title_id": "block:a", "text": "甲"}]},
            [{"title_id": "block:b", "text": "乙"}],
        ],
    )
    assert result == {"titles": [
        {"title_id": "block:a", "text": "甲"},
        {"title_id": "block:b", "text": "乙"},
    ]}
