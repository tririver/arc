from __future__ import annotations

import pytest

from arc_paper.parse.pdf_structure import (
    parse_index_entries,
    parse_printed_toc,
    read_embedded_outline,
    reconcile_blocks_to_pages,
    reconcile_headings_to_pages,
    title_fingerprint,
)


class _Destination:
    def __init__(self, title: str, page: int):
        self.title = title
        self.page = page


class _Reader:
    page_labels = ["i", "ii", "1", "2", "3"]

    def __init__(self, _path):
        self.introduction = _Destination("1 Introduction", 2)
        self.detail = _Destination("1.1 Detail", 3)
        self.outline = [self.introduction, [self.detail]]

    def get_destination_page_number(self, destination):
        return destination.page


def test_embedded_outline_reader_is_injectable_and_preserves_hierarchy(tmp_path) -> None:
    entries, labels = read_embedded_outline(tmp_path / "book.pdf", reader_factory=_Reader)

    assert labels == ["i", "ii", "1", "2", "3"]
    assert entries == [
        {"title": "1 Introduction", "level": 1, "physical_page": 3, "printed_page": "1"},
        {"title": "1.1 Detail", "level": 2, "physical_page": 4, "printed_page": "2"},
    ]


def test_printed_toc_fallback_extracts_entries_and_resolves_page_labels() -> None:
    pages = [
        "Preface",
        "1 Introduction ........ 1\n2 Methods ........ 2\n3 Results ........ 3",
        "1 Introduction\nBody",
        "2 Methods\nBody",
        "3 Results\nBody",
    ]
    entries, source_pages = parse_printed_toc(pages, page_labels=["i", "ii", "1", "2", "3"])

    assert source_pages == [2]
    assert [item["physical_page"] for item in entries] == [3, 4, 5]
    assert [item["title"] for item in entries] == ["1 Introduction", "2 Methods", "3 Results"]


def test_title_fingerprint_and_monotonic_reconciliation_are_deterministic() -> None:
    headings = [
        {"section_id": "s1", "title": "1. Introduction"},
        {"section_id": "s2", "title": "2. Methods"},
    ]
    pages = ["INTRODUCTION\nBody", "Methods\nBody", "Appendix"]

    anchors = reconcile_headings_to_pages(headings, pages)

    assert title_fingerprint("1. Introduction") == "introduction"
    assert [(item["pdf_page_start"], item["pdf_page_end"]) for item in anchors] == [(1, 1), (2, 3)]


def test_reconciliation_fails_when_alignment_is_ambiguous() -> None:
    headings = [{"section_id": "s1", "title": "Introduction"}]
    with pytest.raises(ValueError, match="ambiguous"):
        reconcile_headings_to_pages(headings, ["Introduction", "Introduction"])


def test_reconciliation_prefers_unique_text_anchor_over_equation_page_hint() -> None:
    headings = [
        {
            "section_id": "s1",
            "title": "First Principle",
            "text": "First Principle\nA distinctive opening argument begins the section.",
            "pdf_page_start": 2,
        },
        {
            "section_id": "s2",
            "title": "Second Principle",
            "text": "Second Principle\nA separate conclusion begins the next section.",
            "pdf_page_start": 2,
        },
    ]
    pages = [
        "First Principle\nA distinctive opening argument begins the section.",
        "Second Principle\nA separate conclusion begins the next section.\nEquations",
    ]

    anchors = reconcile_headings_to_pages(headings, pages)

    assert [item["pdf_page_start"] for item in anchors] == [1, 2]


def test_reconciliation_uses_equation_page_hint_to_disambiguate_running_heading() -> None:
    headings = [{"section_id": "s1", "title": "Introduction", "pdf_page_start": 2}]

    anchors = reconcile_headings_to_pages(headings, ["Introduction", "Introduction"])

    assert anchors[0]["pdf_page_start"] == 2


def test_block_reconciliation_uses_text_fingerprints_and_monotonic_order() -> None:
    blocks = [
        {"block_id": "h1", "kind": "heading", "section_id": "s1", "text": "Introduction"},
        {"block_id": "p1", "kind": "prose", "section_id": "s1", "text": "A unique opening argument appears here."},
        {"block_id": "h2", "kind": "heading", "section_id": "s2", "text": "Methods"},
        {"block_id": "p2", "kind": "prose", "section_id": "s2", "text": "The bounded calculation closes the discussion."},
    ]
    sections = [
        {"section_id": "s1", "pdf_page_start": 1, "pdf_page_end": 1},
        {"section_id": "s2", "pdf_page_start": 2, "pdf_page_end": 2},
    ]
    pages = [
        "Introduction\nA unique opening argument appears here.",
        "Methods\nThe bounded calculation closes the discussion.",
    ]

    anchors = reconcile_blocks_to_pages(blocks, pages, section_anchors=sections)

    assert [item["pdf_page_start"] for item in anchors] == [1, 1, 2, 2]


def test_block_reconciliation_ignores_equation_anchor_outside_section() -> None:
    blocks = [
        {
            "block_id": "eq1",
            "source_id": "eq1",
            "kind": "equation",
            "section_id": "s1",
            "text": "alpha beta gamma delta",
        }
    ]
    sections = [{"section_id": "s1", "pdf_page_start": 1, "pdf_page_end": 1}]
    equations = [{"id": "eq1", "pdf_page": 2}]

    anchors = reconcile_blocks_to_pages(
        blocks,
        ["alpha beta gamma delta", "unrelated material"],
        section_anchors=sections,
        equations=equations,
    )

    assert anchors[0]["pdf_page_start"] == 1
    assert anchors[0]["alignment_method"] == "text_fingerprint"


def test_block_reconciliation_fails_for_repeated_text_across_section_pages() -> None:
    blocks = [{"block_id": "p1", "kind": "prose", "section_id": "s1", "text": "Repeated block text"}]
    sections = [{"section_id": "s1", "pdf_page_start": 1, "pdf_page_end": 2}]
    with pytest.raises(ValueError, match="ambiguous"):
        reconcile_blocks_to_pages(
            blocks,
            ["Repeated block text", "Repeated block text"],
            section_anchors=sections,
        )


def test_nested_index_preserves_ranges_roman_pages_and_cross_references() -> None:
    pages = [
        "Front",
        "Body",
        "Gauge fields, i, 2-3\n  abelian case, ii\nGhosts, see Faddeev--Popov\nLoops, see also Diagrams; Renormalization",
    ]
    result = parse_index_entries(pages, page_labels=["i", "ii", "1"], start_page=3)

    assert result["source_pages"] == [3]
    assert [entry["term"] for entry in result["entries"]] == ["Gauge fields", "Ghosts", "Loops"]
    gauge = result["entries"][0]
    assert gauge["page_ranges"] == [
        {"printed_start": "i", "printed_end": "i", "physical_start": 1, "physical_end": 1},
        {"printed_start": "2", "printed_end": "3", "physical_start": None, "physical_end": None},
    ]
    assert gauge["children"][0]["term"] == "abelian case"
    assert gauge["children"][0]["page_ranges"][0]["physical_start"] == 2
    assert result["entries"][1]["see"] == ["Faddeev--Popov"]
    assert result["entries"][2]["see_also"] == ["Diagrams", "Renormalization"]


def test_index_keeps_unlocated_parent_and_sparse_trailing_page() -> None:
    pages = [
        "Body",
        "Fields\n  scalar fields, 2\n  vector fields, 3",
        "Waves, 4",
    ]

    result = parse_index_entries(pages, start_page=2)

    assert result["source_pages"] == [2, 3]
    assert result["entries"][0]["term"] == "Fields"
    assert [item["term"] for item in result["entries"][0]["children"]] == [
        "scalar fields", "vector fields"
    ]
    assert result["entries"][1]["term"] == "Waves"
    assert [line["text"] for line in result["raw_lines"]] == [
        "Fields", "  scalar fields, 2", "  vector fields, 3", "Waves, 4"
    ]


def test_locator_dense_appendix_is_not_an_index_without_structural_evidence() -> None:
    result = parse_index_entries([
        "Body", "Photon scattering, 151\nReferences, 152\nExercises, 153",
    ])
    assert result["entries"] == []
    assert result["source_pages"] == []
