from __future__ import annotations

import hashlib
import re
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

from ..results import err, ok
from .equations import extract_equation_contexts


HEADING_NAMES = ("h1", "h2", "h3", "h4", "h5", "h6")
PARSER_VERSION = 6
INLINE_LABEL_TAG_NAMES = ("b", "strong", "em", "i", "span")
REFERENCE_LABELS = {"references", "bibliography"}
INLINE_SECTION_LABELS = {
    "acknowledgements",
    "acknowledgments",
    "conclusion",
    "conclusions",
    "discussion",
    "future directions",
    "introduction and summary",
    "outlook",
    "summary",
}


def parse_html(html: str, *, paper_id: str = "") -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    sections = []
    toc = []

    for index, section in enumerate(soup.find_all("section"), start=1):
        if not isinstance(section, Tag):
            continue
        heading = section.find(HEADING_NAMES)
        title = _clean_text(heading.get_text(" ", strip=True)) if heading else f"Section {index}"
        section_id = str(section.get("id") or f"section-{index}")
        level = _heading_level(heading)
        text = _clean_text(section.get_text("\n", strip=True))
        toc.append({"id": section_id, "title": title, "level": level})
        sections.append({"section_id": section_id, "title": title, "level": level, "text": text})

    for item in _inline_labeled_sections(soup):
        toc.append({"id": item["section_id"], "title": item["title"], "level": item["level"]})
        sections.append(item)

    return {
        "paper_id": paper_id,
        "parser_version": PARSER_VERSION,
        "source_hash": hashlib.sha256(html.encode("utf-8")).hexdigest(),
        "toc": toc,
        "sections": sections,
        "equations": extract_equation_contexts(soup),
    }


def get_section(parsed: dict[str, Any], selector: str) -> dict[str, Any]:
    needle = (selector or "").strip().lower()
    sections = parsed.get("sections") or []
    for section in sections:
        if str(section.get("section_id", "")).lower() == needle:
            return ok(section)
        if str(section.get("title", "")).lower() == needle:
            return ok(section)
    for section in sections:
        if needle and needle in str(section.get("title", "")).lower():
            return ok(section)
    return err(
        "section_not_found",
        f"Section not found: {selector}",
        toc=parsed.get("toc") or [],
    )


def _heading_level(heading: Tag | None) -> int:
    if heading is None:
        return 1
    try:
        return int(heading.name[1:])
    except (TypeError, ValueError):
        return 1


def _clean_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _inline_labeled_sections(soup: BeautifulSoup) -> list[dict[str, Any]]:
    blocks = [
        tag
        for tag in soup.find_all((*HEADING_NAMES, "p"))
        if isinstance(tag, Tag) and not tag.find_parent("section")
    ]
    candidates = []
    for index, paragraph in enumerate(blocks):
        if paragraph.name != "p":
            continue
        title = _inline_label_title(paragraph)
        if not title:
            continue
        candidates.append((index, title))

    sections = []
    for candidate_index, (block_index, title) in enumerate(candidates):
        next_block_index = candidates[candidate_index + 1][0] if candidate_index + 1 < len(candidates) else None
        text = _inline_section_text(blocks, block_index, next_block_index)
        sections.append(
            {
                "section_id": _inline_section_id(title),
                "title": title,
                "level": 2,
                "text": text,
            }
        )
    return sections


def _inline_label_title(paragraph: Tag) -> str | None:
    first = _first_meaningful_child(paragraph)
    if not isinstance(first, Tag) or not _is_inline_label_tag(first):
        return None
    raw = _clean_inline_label(first.get_text(" ", strip=True))
    normalized = _normalize_inline_label(raw)
    if normalized not in INLINE_SECTION_LABELS:
        return None
    if not _has_inline_label_separator(paragraph, raw):
        return None
    return _title_case_label(normalized)


def _first_meaningful_child(tag: Tag) -> Tag | NavigableString | None:
    for child in tag.children:
        if isinstance(child, NavigableString) and not str(child).strip():
            continue
        return child
    return None


def _is_inline_label_tag(tag: Tag) -> bool:
    if tag.name not in INLINE_LABEL_TAG_NAMES:
        return False
    classes = set(tag.get("class") or [])
    return bool(classes.intersection({"ltx_font_bold", "ltx_font_italic"})) or tag.name in {
        "b",
        "strong",
        "em",
        "i",
    }


def _has_inline_label_separator(paragraph: Tag, raw_label: str) -> bool:
    paragraph_text = _clean_inline_label(paragraph.get_text(" ", strip=True))
    if not paragraph_text.lower().startswith(raw_label.lower()):
        return False
    if raw_label[-1:] in ".:;—–-":
        return True
    remainder = paragraph_text[len(raw_label) :].lstrip()
    return not remainder or remainder[0] in ".:;—–-"


def _inline_section_text(blocks: list[Tag], start_index: int, next_block_index: int | None) -> str:
    parts = []
    stop_index = next_block_index if next_block_index is not None else len(blocks)
    for index in range(start_index, stop_index):
        current = blocks[index]
        if index > start_index and current.name in HEADING_NAMES:
            break
        if current.name == "p":
            if index > start_index and _is_reference_boundary_paragraph(current):
                break
            text, stopped = _paragraph_text_until_embedded_stop(current)
            if text:
                parts.append(text)
            if stopped:
                break
    return _clean_text("\n".join(parts))


def _inline_section_id(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f"inline-{slug or 'section'}"


def _clean_inline_label(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _normalize_inline_label(text: str) -> str:
    return _clean_inline_label(text).strip(".:;—–-").lower()


def _title_case_label(normalized: str) -> str:
    if normalized == "acknowledgements":
        return "Acknowledgements"
    if normalized == "acknowledgments":
        return "Acknowledgments"
    return " ".join(word.capitalize() for word in normalized.split())


def _paragraph_text_until_embedded_stop(paragraph: Tag) -> tuple[str, bool]:
    parts = []
    for child in paragraph.children:
        if isinstance(child, Tag) and _is_embedded_stop_label(child):
            return _clean_text("\n".join(parts)), True
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag) and child.name == "br":
            parts.append("\n")
        elif isinstance(child, Tag):
            parts.append(child.get_text("\n", strip=True))
    return _clean_text("\n".join(parts)), False


def _is_embedded_stop_label(tag: Tag) -> bool:
    if not _is_inline_label_tag(tag) or not _has_break_before(tag):
        return False
    raw = _clean_inline_label(tag.get_text(" ", strip=True))
    return _is_stop_label(raw)


def _has_break_before(tag: Tag) -> bool:
    sibling = tag.previous_sibling
    while sibling is not None:
        if isinstance(sibling, NavigableString) and not str(sibling).strip():
            sibling = sibling.previous_sibling
            continue
        return isinstance(sibling, Tag) and sibling.name == "br"
    return False


def _is_reference_boundary_paragraph(paragraph: Tag) -> bool:
    title = _inline_stop_label(paragraph)
    if title and _normalize_inline_label(title) in REFERENCE_LABELS:
        return True
    return _looks_like_reference_item(paragraph.get_text(" ", strip=True))


def _inline_stop_label(paragraph: Tag) -> str | None:
    first = _first_meaningful_child(paragraph)
    if not isinstance(first, Tag) or not _is_inline_label_tag(first):
        return None
    raw = _clean_inline_label(first.get_text(" ", strip=True))
    if not _is_stop_label(raw):
        return None
    if not _has_inline_label_separator(paragraph, raw):
        return None
    return raw


def _is_stop_label(raw_label: str) -> bool:
    normalized = _normalize_inline_label(raw_label)
    return normalized in INLINE_SECTION_LABELS or normalized in REFERENCE_LABELS


def _looks_like_reference_item(text: str) -> bool:
    compact = _clean_inline_label(text)
    return bool(re.match(r"^\[\s*\d+\s*\]", compact))
