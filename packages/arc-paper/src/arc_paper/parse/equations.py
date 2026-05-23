from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from bs4 import BeautifulSoup, Tag


EQUATION_SELECTORS = "table.ltx_equation, div.ltx_equation, span.ltx_equation, math"


def extract_equation_contexts(html_or_soup: str | BeautifulSoup, *, window_paragraphs: int = 1) -> list[dict[str, Any]]:
    soup = html_or_soup if isinstance(html_or_soup, BeautifulSoup) else BeautifulSoup(html_or_soup, "lxml")
    contexts = []
    for element in soup.select(EQUATION_SELECTORS):
        text = element.get_text(" ", strip=True)
        section_id, section_title = _section_info(element)
        contexts.append(
            {
                "id": str(element.get("id") or ""),
                "equation": text,
                "before": "\n\n".join(_previous_paragraphs(element, window_paragraphs)),
                "after": "\n\n".join(_next_paragraphs(element, window_paragraphs)),
                "section_id": section_id,
                "section_title": section_title,
            }
        )
    return contexts


def find_equation_context(equations: Iterable[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    needle = _normalize(query)
    return [dict(item) for item in equations if needle and needle in _normalize(str(item.get("equation") or ""))]


def _section_info(element: Tag) -> tuple[str, str]:
    section = element.find_parent("section")
    if not isinstance(section, Tag):
        return "", ""
    heading = section.find(("h1", "h2", "h3", "h4", "h5", "h6"))
    title = heading.get_text(" ", strip=True) if isinstance(heading, Tag) else ""
    return str(section.get("id") or ""), " ".join(title.split())


def _previous_paragraphs(element: Tag, limit: int) -> list[str]:
    paragraphs: list[str] = []
    current = element
    while len(paragraphs) < limit:
        previous = current.find_previous("p")
        if previous is None:
            break
        text = previous.get_text(" ", strip=True)
        if text:
            paragraphs.append(text)
        current = previous
    paragraphs.reverse()
    return paragraphs


def _next_paragraphs(element: Tag, limit: int) -> list[str]:
    paragraphs: list[str] = []
    current = element
    while len(paragraphs) < limit:
        following = current.find_next("p")
        if following is None:
            break
        text = following.get_text(" ", strip=True)
        if text:
            paragraphs.append(text)
        current = following
    return paragraphs


def _normalize(text: str) -> str:
    return " ".join((text or "").split()).lower()
