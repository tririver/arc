from __future__ import annotations

import math
import re
import hashlib
from collections import Counter
from typing import Iterable

from arc_paper.ids import arxiv_path_id, doi_value, inspire_recid, normalize_paper_id


STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "that", "this", "their",
    "paper", "papers", "study", "using", "based", "towards", "toward",
    "model", "models", "result", "results", "new", "general", "about",
    "between", "through", "within", "without", "field", "fields",
}


def tokens(text: str) -> set[str]:
    return {
        _stem(token)
        for token in re.findall(r"[a-z][a-z0-9+-]{2,}", (text or "").lower())
        if token not in STOPWORDS
    }


def token_overlap_score(text: str, intent: str) -> float:
    intent_tokens = tokens(intent)
    if not intent_tokens:
        return 0.0
    text_tokens = tokens(text)
    return len(intent_tokens & text_tokens) / max(1, len(intent_tokens))


def citation_per_year(paper: dict, current_year: int) -> float:
    year = int(paper.get("year") or current_year)
    citations = int(paper.get("citation_count") or paper.get("cited_by_count") or 0)
    age = max(1, current_year - year + 1)
    return citations / age


def deterministic_sample(items: list[dict], *, count: int, seed: str) -> list[dict]:
    if count <= 0:
        return []
    decorated = [
        (_hash_key(f"{seed}\n{paper_key(item)}\n{index}"), item)
        for index, item in enumerate(items)
    ]
    decorated.sort(key=lambda item: item[0])
    return [item for _, item in decorated[:count]]


def paper_key(paper: dict) -> str:
    identifiers = paper.get("identifiers") or {}
    for value in (
        paper.get("paper_id"),
        paper.get("id"),
        _prefixed_identifier("arXiv", paper.get("arxiv_id") or paper.get("arxiv")),
        _prefixed_identifier("inspire", paper.get("inspire_recid")),
        paper.get("doi"),
        identifiers.get("paper_id"),
        _prefixed_identifier("arXiv", identifiers.get("arxiv_id") or identifiers.get("arxiv")),
        _prefixed_identifier("inspire", identifiers.get("inspire")),
        _prefixed_identifier("inspire", identifiers.get("inspire_recid")),
        identifiers.get("doi"),
    ):
        stable_id = stable_paper_id(value)
        if stable_id:
            return stable_id
    return ""


def stable_paper_id(identifier: object) -> str:
    if identifier is None:
        return ""
    normalized = normalize_paper_id(str(identifier))
    if arxiv_path_id(normalized) or doi_value(normalized) or inspire_recid(normalized):
        return normalized
    return ""


def _prefixed_identifier(prefix: str, value: object) -> str:
    text = str(value or "").strip()
    if not text or ":" in text:
        return text
    return f"{prefix}:{text}"


def normalize_authors(authors: Iterable[str] | None, *, limit: int = 5) -> str:
    values = [str(author) for author in authors or [] if str(author).strip()]
    if not values:
        return ""
    if len(values) <= limit:
        return ", ".join(values)
    return f"{values[0]} et al."


def top_counts(counter: Counter[str], *, limit: int) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (item[1], item[0]), reverse=True)[:limit]


def log_score(value: int | float) -> float:
    return math.log1p(max(0.0, float(value or 0)))


def _stem(token: str) -> str:
    token = token.lower()
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        return token[:-1]
    return token


def _hash_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
