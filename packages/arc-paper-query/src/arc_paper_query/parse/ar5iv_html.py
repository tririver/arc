from __future__ import annotations

import hashlib
from typing import Any

from bs4 import BeautifulSoup, Tag

from ..results import err, ok


HEADING_NAMES = ("h1", "h2", "h3", "h4", "h5", "h6")


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

    return {
        "paper_id": paper_id,
        "source_hash": hashlib.sha256(html.encode("utf-8")).hexdigest(),
        "toc": toc,
        "sections": sections,
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
