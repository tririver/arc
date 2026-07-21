from __future__ import annotations

import pytest

from arc_paper.parse.structure import (
    INDEX_ENTRIES_SCHEMA_VERSION,
    STRUCTURE_SCHEMA_VERSION,
    build_structure,
    empty_index_entries,
    has_reconciliation_proof,
    resolve_document_kind,
    select_chapter_units,
    validate_exact_coverage,
)


def _section(section_id: str, title: str, level: int) -> dict:
    return {"section_id": section_id, "title": title, "level": level}


def test_article_structure_uses_top_level_sections() -> None:
    sections = [_section("s1", "Introduction", 1), _section("s2", "Method", 1)]
    assert resolve_document_kind(sections) == ("article", [])
    selected, diagnostics = select_chapter_units(sections, document_kind="article")
    assert [item["section_id"] for item in selected] == ["s1", "s2"]
    assert diagnostics == []


def test_article_skips_a_lone_document_title_when_sections_follow() -> None:
    sections = [
        _section("title", "Lecture Notes", 1),
        _section("s1", "Introduction", 2),
        _section("s2", "Method", 2),
    ]
    selected, _ = select_chapter_units(sections, document_kind="article")
    assert [item["section_id"] for item in selected] == ["s1", "s2"]


def test_book_structure_drills_below_three_parts_to_seven_chapters() -> None:
    sections: list[dict] = []
    chapter = 0
    for part in range(1, 4):
        sections.append(_section(f"part-{part}", f"Part {part}", 1))
        for _ in range(2 if part < 3 else 3):
            chapter += 1
            sections.append(_section(f"chapter-{chapter}", f"Chapter {chapter}", 2))
    assert resolve_document_kind(sections)[0] == "book"
    selected, diagnostics = select_chapter_units(sections, document_kind="book")
    assert [item["section_id"] for item in selected] == [f"chapter-{index}" for index in range(1, 8)]
    assert diagnostics == []


def test_book_without_five_units_uses_deepest_real_level_with_diagnostic() -> None:
    sections = [_section("p1", "Part A", 1), _section("c1", "One", 2), _section("c2", "Two", 2)]
    selected, diagnostics = select_chapter_units(sections, document_kind="book")
    assert [item["section_id"] for item in selected] == ["c1", "c2"]
    assert diagnostics


def test_exact_coverage_reports_duplicates_missing_and_order() -> None:
    receipt = validate_exact_coverage(
        ["s1", "s2", "s3"],
        [{"section_ids": ["s1", "s2"]}, {"section_ids": ["s2"]}],
    )
    assert receipt["status"] == "invalid"
    assert receipt["duplicates"] == ["s2"]
    assert receipt["missing"] == ["s3"]
    assert receipt["monotonic_order"] is False


def test_structure_and_index_contracts_are_versioned_and_proof_is_pdf_bound(tmp_path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF fixture")
    parsed = {"source_hash": "combined-source-hash", "sections": [_section("s1", "Introduction", 1)]}
    structure = build_structure(parsed, pdf_path=pdf, pdf_pages=["Introduction\nBody"])
    assert structure["schema_version"] == STRUCTURE_SCHEMA_VERSION
    assert structure["coverage"]["status"] == "complete"
    assert has_reconciliation_proof(structure)
    assert not has_reconciliation_proof(structure, pdf_hash="different")
    assert empty_index_entries() == {"schema_version": INDEX_ENTRIES_SCHEMA_VERSION, "entries": []}


def test_four_top_level_units_require_explicit_document_kind() -> None:
    sections = [_section(f"s{index}", f"Unit {index}", 1) for index in range(1, 5)]
    with pytest.raises(ValueError, match="--document-kind"):
        resolve_document_kind(sections)


def test_semantic_source_roles_exclude_contents_from_chapter_selection() -> None:
    parsed = {
        "source_hash": "source",
        "sections": [
            _section("toc", "Contents", 1),
            *[_section(f"chapter-{number}", f"Unit {number}", 1) for number in range(1, 6)],
        ],
        "document": {"blocks": [
            {"block_id": "toc.heading", "section_id": "toc", "source_role": "table_of_contents"},
        ]},
    }
    structure = build_structure(parsed, requested_document_kind="book")
    assert structure["excluded_block_ids"] == ["toc.heading"]
    assert [item["title"] for item in structure["chapters"]] == [f"Unit {number}" for number in range(1, 6)]


def test_pdf_contents_evidence_excludes_converter_toc_heading_with_page_suffix(tmp_path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF fixture")
    parsed = {
        "source_hash": "source",
        "sections": [
            _section("toc-intro", "Introduction 1", 1),
            _section("intro", "Introduction", 1),
        ],
    }
    structure = build_structure(
        parsed, requested_document_kind="article", pdf_path=pdf,
        pdf_pages=["Introduction\nBody"],
        embedded_outline=[{"title": "0. Introduction", "physical_page": 1, "printed_page": "1"}],
        pdf_page_labels=["1"],
    )
    assert structure["coverage"]["expected_count"] == 1
    assert structure["chapters"][0]["section_ids"] == ["intro"]
