from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .pdf_structure import (
    parse_printed_toc,
    read_embedded_outline,
    reconcile_blocks_to_pages,
    reconcile_headings_to_pages,
    title_fingerprint,
)


STRUCTURE_SCHEMA_VERSION = "arc.paper.structure.v1"
INDEX_ENTRIES_SCHEMA_VERSION = "arc.paper.index_entries.v1"
RECONCILIATION_PROOF_VERSION = "arc.paper.reconciliation.v1"
DOCUMENT_KINDS = {"auto", "article", "book"}


def normalize_document_kind(value: str) -> str:
    kind = str(value or "auto").strip().casefold()
    if kind not in DOCUMENT_KINDS:
        raise ValueError(
            f"Unsupported document kind {value!r}; choose auto, article, or book."
        )
    return kind


def resolve_document_kind(
    sections: Iterable[dict[str, Any]], *, requested: str = "auto"
) -> tuple[str, list[str]]:
    """Resolve a conservative article/book classification from real headings.

    A repeated structural level with at least five units is sufficient evidence
    for a book.  Four top-level units without such a deeper level are genuinely
    ambiguous: they can be either a short book or a sectioned article, so the
    caller must choose explicitly.  Smaller structures default to article.
    """

    requested = normalize_document_kind(requested)
    if requested != "auto":
        return requested, []
    levels = [
        int(item.get("level") or 0)
        for item in sections
        if int(item.get("level") or 0) > 0 and str(item.get("title") or "").strip()
    ]
    if not levels:
        return "article", ["No source headings were found; treating the document as one article unit."]
    counts = {level: levels.count(level) for level in sorted(set(levels))}
    if any(count >= 5 for count in counts.values()):
        return "book", []
    shallowest = min(levels)
    if counts.get(shallowest) == 4:
        raise ValueError(
            "Document kind is ambiguous from structure; rerun with --document-kind article or book."
        )
    return "article", []


def select_chapter_units(
    sections: Iterable[dict[str, Any]], *, document_kind: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select real chapter/section units without inventing source titles."""

    kind = normalize_document_kind(document_kind)
    if kind == "auto":
        kind, _ = resolve_document_kind(sections)
    items = [
        dict(item)
        for item in sections
        if str(item.get("section_id") or "") and str(item.get("title") or "").strip()
    ]
    diagnostics: list[str] = []
    if not items:
        return [], ["No real heading level was available for chapter selection."]
    levels = sorted({int(item.get("level") or 1) for item in items})
    if kind == "article":
        selected_level = levels[0]
        if (
            len(levels) > 1
            and sum(int(item.get("level") or 1) == selected_level for item in items) == 1
        ):
            # A lone shallow heading is commonly the document title.  Preserve
            # it as leading material, but use the next real heading level for
            # article sections.
            selected_level = levels[1]
    else:
        selected_level = next(
            (level for level in levels if sum(int(item.get("level") or 1) == level for item in items) >= 5),
            levels[-1],
        )
        if sum(int(item.get("level") or 1) == selected_level for item in items) < 5:
            diagnostics.append(
                "No heading level contained five units; selected the deepest real structure level."
            )
    selected = [item for item in items if int(item.get("level") or 1) == selected_level]
    return selected, diagnostics


def validate_exact_coverage(
    source_ids: Iterable[str], units: Iterable[dict[str, Any]], *, member_key: str = "section_ids"
) -> dict[str, Any]:
    expected = [str(value) for value in source_ids]
    covered = [
        str(value)
        for unit in units
        for value in (unit.get(member_key) or [])
    ]
    duplicates = sorted({value for value in covered if covered.count(value) > 1})
    missing = [value for value in expected if value not in covered]
    unexpected = [value for value in covered if value not in expected]
    ordered = [value for value in covered if value in expected] == expected
    return {
        "status": "complete" if not duplicates and not missing and not unexpected and ordered else "invalid",
        "expected_count": len(expected),
        "covered_count": len(covered),
        "duplicates": duplicates,
        "missing": missing,
        "unexpected": unexpected,
        "monotonic_order": ordered,
    }


def build_structure(
    parsed: dict[str, Any],
    *,
    requested_document_kind: str = "auto",
    pdf_path: str | Path | None = None,
    pdf_pages: Iterable[str] | None = None,
    index_source_pages: Iterable[int] = (),
    outline_reader_factory: Any = None,
    embedded_outline: Iterable[dict[str, Any]] | None = None,
    pdf_page_labels: Iterable[str] | None = None,
) -> dict[str, Any]:
    requested = normalize_document_kind(requested_document_kind)
    all_sections = list(parsed.get("sections") or [])
    excluded_section_ids, excluded_block_ids, index_block_ids = _excluded_source_structure(parsed)
    sections = [
        item for item in all_sections
        if str(item.get("section_id") or "") not in excluded_section_ids
    ]
    pdf_page_list = list(pdf_pages or [])
    outline_entries = list(embedded_outline or [])
    page_labels = list(pdf_page_labels or [])
    printed_toc: list[dict[str, Any]] = []
    toc_pages: list[int] = []
    reconciliation_anchors: list[dict[str, Any]] = []
    structure_source = "rich_source_headings"
    if pdf_path and pdf_page_list:
        if embedded_outline is None and pdf_page_labels is None:
            outline_entries, page_labels = read_embedded_outline(
                pdf_path, reader_factory=outline_reader_factory
            )
        if outline_entries:
            structure_source = "embedded_outline"
        else:
            printed_toc, toc_pages = parse_printed_toc(
                pdf_page_list,
                page_labels=page_labels,
                excluded_pages=index_source_pages,
            )
            if printed_toc:
                structure_source = "printed_toc"
        authority_entries = outline_entries or printed_toc
        toc_duplicate_ids = _source_toc_duplicate_section_ids(sections, authority_entries)
        if toc_duplicate_ids:
            excluded_section_ids.update(toc_duplicate_ids)
            sections = [
                item for item in sections
                if str(item.get("section_id") or "") not in toc_duplicate_ids
            ]
            for block in (parsed.get("document") or {}).get("blocks") or []:
                if str(block.get("section_id") or "") in toc_duplicate_ids:
                    identifier = str(block.get("block_id") or block.get("source_id") or "")
                    if identifier and identifier not in excluded_block_ids:
                        excluded_block_ids.append(identifier)
        reconciliation_anchors = reconcile_headings_to_pages(
            sections,
            pdf_page_list,
            authority_entries=authority_entries,
            excluded_pages={*toc_pages, *index_source_pages},
        )
        anchors_by_id = {item["section_id"]: item for item in reconciliation_anchors}
        for section in sections:
            anchor = anchors_by_id.get(str(section.get("section_id") or ""))
            if anchor:
                section["pdf_page_start"] = anchor["pdf_page_start"]
                section["pdf_page_end"] = anchor["pdf_page_end"]
        if authority_entries:
            _apply_authority_levels(sections, authority_entries)
    resolved, diagnostics = resolve_document_kind(sections, requested=requested)
    chapter_sections, chapter_diagnostics = select_chapter_units(sections, document_kind=resolved)
    diagnostics.extend(chapter_diagnostics)
    all_section_ids = [str(item.get("section_id") or "") for item in sections if item.get("section_id")]
    chapters = _chapter_records(sections, chapter_sections)
    coverage = validate_exact_coverage(all_section_ids, chapters)
    structure: dict[str, Any] = {
        "schema_version": STRUCTURE_SCHEMA_VERSION,
        "requested_document_kind": requested,
        "document_kind": resolved,
        "structure_source": structure_source,
        "chapters": chapters,
        "coverage": coverage,
        "diagnostics": diagnostics,
        "excluded_block_ids": excluded_block_ids,
        "index_block_ids": index_block_ids,
    }
    if pdf_path and pdf_page_list:
        pdf_hash = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
        blocks = [
            item for item in (parsed.get("document") or {}).get("blocks") or []
            if str(item.get("section_id") or "") not in excluded_section_ids
            and str(item.get("block_id") or item.get("source_id") or "") not in set(excluded_block_ids)
        ]
        if blocks:
            block_anchors = reconcile_blocks_to_pages(
                blocks,
                pdf_page_list,
                section_anchors=reconciliation_anchors,
                equations=parsed.get("equations") or [],
            )
        else:
            # Legacy TeX has section records but no rich block contract.  Each
            # section record is therefore the complete deterministic source
            # unit available to reconcile.
            block_anchors = [
                {
                    "block_id": str(item.get("section_id") or ""),
                    "section_id": str(item.get("section_id") or ""),
                    "source_fingerprint": str(item.get("title_fingerprint") or ""),
                    "pdf_page_start": item.get("pdf_page_start"),
                    "pdf_page_end": item.get("pdf_page_end"),
                }
                for item in reconciliation_anchors
            ]
        section_coverage = _identity_coverage(
            [str(item.get("section_id") or "") for item in sections],
            [str(item.get("section_id") or "") for item in reconciliation_anchors],
        )
        expected_block_ids = (
            [str(item.get("block_id") or "") for item in blocks]
            if blocks
            else [str(item.get("section_id") or "") for item in sections]
        )
        block_coverage = _identity_coverage(
            expected_block_ids,
            [str(item.get("block_id") or "") for item in block_anchors],
        )
        if section_coverage["status"] != "complete" or block_coverage["status"] != "complete":
            raise ValueError("PDF reconciliation did not cover source sections and blocks exactly once.")
        proof_material = {
            "pdf_sha256": pdf_hash,
            "source_hash": str(parsed.get("source_hash") or ""),
            "structure_source": structure_source,
            "page_count": len(pdf_page_list),
            "section_anchors": reconciliation_anchors,
            "block_anchors": block_anchors,
            "section_coverage": section_coverage,
            "block_coverage": block_coverage,
        }
        structure["pdf_sha256"] = pdf_hash
        structure["reconciliation"] = {
            "schema_version": RECONCILIATION_PROOF_VERSION,
            "status": "complete",
            "scope": "source-heading-and-block-page-alignment",
            "proof_sha256": hashlib.sha256(
                json.dumps(proof_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            **proof_material,
        }
    return structure


def _source_toc_duplicate_section_ids(
    sections: Iterable[dict[str, Any]], authority_entries: Iterable[dict[str, Any]]
) -> set[str]:
    """Recognize source-converter TOC headings by their authoritative page suffix.

    This is tied to the PDF's own outline/printed contents evidence, not to
    language-specific title keywords.
    """

    authority: set[tuple[str, str]] = set()
    for item in authority_entries:
        fingerprint = title_fingerprint(str(item.get("title") or ""))
        printed = str(item.get("printed_page") or "").strip().casefold()
        if fingerprint and printed:
            authority.add((fingerprint, printed))
    duplicates: set[str] = set()
    for section in sections:
        title = str(section.get("title") or "").strip()
        match = re.match(r"^(?P<title>.+?)\s+(?P<page>[ivxlcdm]+|\d+)\s*$", title, re.IGNORECASE)
        if not match:
            continue
        candidate = (title_fingerprint(match.group("title")), match.group("page").casefold())
        if candidate in authority:
            identifier = str(section.get("section_id") or "")
            if identifier:
                duplicates.add(identifier)
    return duplicates


def _apply_authority_levels(
    sections: list[dict[str, Any]], authority_entries: Iterable[dict[str, Any]]
) -> None:
    levels_by_fingerprint: dict[str, set[int]] = {}
    maximum = 1
    for item in authority_entries:
        fingerprint = title_fingerprint(str(item.get("title") or ""))
        level = int(item.get("level") or 1)
        maximum = max(maximum, level)
        if fingerprint:
            levels_by_fingerprint.setdefault(fingerprint, set()).add(level)
    for section in sections:
        fingerprint = title_fingerprint(str(section.get("title") or ""))
        levels = levels_by_fingerprint.get(fingerprint) or set()
        source_level = _source_numbering_level(
            str(section.get("title") or "") or str(section.get("text") or "")
        )
        if source_level is None:
            source_level = _source_numbering_level(str(section.get("text") or ""))
        # A unique PDF outline/printed-TOC level is authoritative. Source-only
        # reader headings remain nested below that real hierarchy.
        section["level"] = (
            source_level
            if source_level is not None and source_level in levels
            else (next(iter(levels)) if len(levels) == 1 else maximum + 1)
        )


def _source_numbering_level(value: str) -> int | None:
    match = re.match(r"^\s*#*\s*(?P<number>\d+(?:\.\d+)*)[.)\-:]?\s+\S", value)
    return match.group("number").count(".") + 1 if match else None


def _excluded_source_structure(
    parsed: dict[str, Any],
) -> tuple[set[str], list[str], list[str]]:
    """Use parser-owned semantic roles instead of title keyword guesses."""

    excluded_roles = {
        "cover", "title", "table_of_contents", "contents", "acknowledgements",
        "acknowledgments", "bibliography", "references", "index", "front_matter",
        "front_matter_title", "front_matter_authors", "front_matter_affiliations",
    }
    excluded_sections: set[str] = set()
    excluded_blocks: list[str] = []
    index_blocks: list[str] = []
    blocks = list((parsed.get("document") or {}).get("blocks") or [])
    for block in blocks:
        role = str(block.get("source_role") or block.get("role") or "").strip().casefold()
        if role not in excluded_roles:
            continue
        identifier = str(block.get("block_id") or block.get("source_id") or "")
        if identifier:
            excluded_blocks.append(identifier)
            if role == "index":
                index_blocks.append(identifier)
        section_id = str(block.get("section_id") or "")
        if section_id:
            excluded_sections.add(section_id)
    return excluded_sections, excluded_blocks, index_blocks


def empty_index_entries() -> dict[str, Any]:
    return {"schema_version": INDEX_ENTRIES_SCHEMA_VERSION, "entries": []}


def has_reconciliation_proof(structure: Any, *, pdf_hash: str | None = None) -> bool:
    if not isinstance(structure, dict):
        return False
    proof = structure.get("reconciliation")
    if not isinstance(proof, dict):
        return False
    valid = bool(
        proof.get("schema_version") == RECONCILIATION_PROOF_VERSION
        and proof.get("status") == "complete"
        and proof.get("proof_sha256")
        and proof.get("source_hash")
        and proof.get("pdf_sha256")
        and (proof.get("section_coverage") or {}).get("status") == "complete"
        and (proof.get("block_coverage") or {}).get("status") == "complete"
    )
    return valid and (not pdf_hash or proof.get("pdf_sha256") == pdf_hash)


def _identity_coverage(expected: list[str], actual: list[str]) -> dict[str, Any]:
    return {
        "status": "complete" if expected == actual and len(actual) == len(set(actual)) else "invalid",
        "expected_count": len(expected),
        "covered_count": len(actual),
        "missing": [value for value in expected if value not in actual],
        "unexpected": [value for value in actual if value not in expected],
        "duplicates": sorted({value for value in actual if actual.count(value) > 1}),
        "monotonic_order": actual == expected,
    }


def _chapter_records(
    sections: list[dict[str, Any]], selected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not sections:
        return []
    if not selected:
        return [
            {
                "chapter_id": "ch-0001",
                "title": str(sections[0].get("title") or ""),
                "level": int(sections[0].get("level") or 1),
                "section_ids": [str(item["section_id"]) for item in sections],
            }
        ]
    selected_ids = {str(item["section_id"]) for item in selected}
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for section in sections:
        section_id = str(section["section_id"])
        if section_id in selected_ids:
            current = {
                "chapter_id": f"ch-{len(records) + 1:04d}",
                "title": str(section.get("title") or ""),
                "level": int(section.get("level") or 1),
                "section_ids": [],
                "pdf_page_start": section.get("pdf_page_start"),
                "pdf_page_end": section.get("pdf_page_end"),
            }
            records.append(current)
        if current is None:
            # Leading material belongs to the first real unit for exact source
            # coverage; later structure work may mark it non-substantive.
            if records:
                current = records[0]
            else:
                continue
        current["section_ids"].append(section_id)
        if section.get("pdf_page_end") is not None:
            current["pdf_page_end"] = section.get("pdf_page_end")
    if records:
        leading = [
            str(item["section_id"])
            for item in sections
            if str(item["section_id"]) not in {value for record in records for value in record["section_ids"]}
        ]
        records[0]["section_ids"] = leading + records[0]["section_ids"]
    return records
