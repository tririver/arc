from __future__ import annotations

from bs4 import BeautifulSoup, Tag


EQUATION_SELECTORS = "table.ltx_equation, div.ltx_equation, span.ltx_equation, math"


def find_equation_context(html: str, query: str, *, window_paragraphs: int = 1) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    needle = _normalize(query)
    contexts = []
    for element in soup.select(EQUATION_SELECTORS):
        text = element.get_text(" ", strip=True)
        if needle not in _normalize(text):
            continue
        contexts.append(
            {
                "id": str(element.get("id") or ""),
                "equation": text,
                "before": "\n\n".join(_previous_paragraphs(element, window_paragraphs)),
                "after": "\n\n".join(_next_paragraphs(element, window_paragraphs)),
            }
        )
    return contexts


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
