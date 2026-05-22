from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from ..cache import CachePaths, ONE_MONTH_SECONDS, read_json, write_json
from ..ids import arxiv_path_id, inspire_recid, normalize_paper_id
from .base import ProviderError


BASE_URL = "https://inspirehep.net/api"
MAX_PAGE_SIZE = 1000
MATHML_RE = re.compile(r"<math\b[^>]*>.*?</math>", re.IGNORECASE | re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
SUMMARY_FIELDS = ",".join(
    [
        "titles",
        "authors",
        "arxiv_eprints",
        "dois",
        "citation_count",
        "earliest_date",
        "preprint_date",
        "publication_info",
        "abstracts",
    ]
)


class InspireProvider:
    def __init__(self, *, client: httpx.Client | None = None, timeout: float = 60.0):
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self.timeout = timeout

    def get_metadata(self, paper_id: str, *, refresh: bool = False) -> dict[str, Any]:
        raw = self.get_raw_record(paper_id, refresh=refresh)
        return _normalize_record(raw)

    def get_references(self, paper_id: str, *, refresh: bool = False, enrich: bool = False) -> list[dict[str, Any]]:
        paths = CachePaths.for_paper(paper_id)
        references: list[dict[str, Any]]
        if not refresh and (cached := read_json(paths.inspire_references)) and _references_cache_is_current(cached):
            cached, changed = _clean_cached_paper_items(cached)
            if changed:
                write_json(paths.inspire_references, cached)
            if not enrich or _references_cache_is_enriched(cached):
                return cached
            references = cached
        else:
            raw = self.get_raw_record(paper_id, refresh=refresh)
            references = [
                normalized
                for item in raw.get("metadata", {}).get("references", [])
                if (normalized := _normalize_reference(item))
            ]

        if enrich:
            references = self.enrich_reference_metadata(references, refresh=refresh)
        write_json(paths.inspire_references, references)
        return references

    def enrich_reference_metadata(
        self,
        references: list[dict[str, Any]],
        *,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for reference in references:
            lookup_id = _reference_lookup_id(reference)
            if not lookup_id:
                enriched.append(reference)
                continue
            try:
                metadata = self.get_metadata(lookup_id, refresh=refresh)
            except ProviderError as exc:
                failed = dict(reference)
                failed["metadata_enriched"] = False
                failed["metadata_enrichment_error"] = {"code": exc.code, "message": exc.message}
                enriched.append(failed)
                continue
            enriched.append(_merge_reference_metadata(reference, metadata))
        return enriched

    def get_citers(
        self,
        paper_id: str,
        *,
        refresh: bool = False,
        limit: int = MAX_PAGE_SIZE,
        sort: str = "mostrecent",
    ) -> list[dict[str, Any]]:
        limit = _clamp_limit(limit)
        sort = _normalize_sort(sort)
        cache_path = _citers_cache_path(paper_id, sort, limit)
        if not refresh and (cached := read_json(cache_path, ttl_seconds=ONE_MONTH_SECONDS)):
            if isinstance(cached, list):
                cached, changed = _clean_cached_paper_items(cached)
                if changed:
                    write_json(cache_path, cached)
                return cached[:limit]

        metadata = self.get_metadata(paper_id, refresh=refresh)
        recid = metadata.get("inspire_recid")
        if not recid:
            return []

        response = self.client.get(
            f"{BASE_URL}/literature",
            params={
                "q": f"refersto:recid:{recid}",
                "size": str(limit),
                "sort": sort,
                "fields": SUMMARY_FIELDS,
                "format": "json",
            },
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError("inspire_citers_fetch_failed", str(exc)) from exc

        data = response.json()
        citers = [_normalize_record(hit) for hit in data.get("hits", {}).get("hits", [])]
        write_json(cache_path, citers)
        return citers

    def get_citer_count(self, paper_id: str, *, refresh: bool = False) -> int:
        return int(self.get_metadata(paper_id, refresh=refresh).get("citation_count") or 0)

    def get_raw_record(self, paper_id: str, *, refresh: bool = False) -> dict[str, Any]:
        normalized_id = normalize_paper_id(paper_id)
        paths = CachePaths.for_paper(normalized_id)
        if not refresh and (cached := read_json(paths.inspire_metadata)):
            return cached

        recid = inspire_recid(normalized_id)
        aid = arxiv_path_id(normalized_id)
        if recid:
            url = f"{BASE_URL}/literature/{recid}"
        elif aid:
            url = f"{BASE_URL}/arxiv/{aid}"
        else:
            raise ProviderError("unsupported_paper_id", f"INSPIRE requires an arXiv ID or INSPIRE recid: {paper_id}")

        response = self.client.get(url, timeout=self.timeout)
        if response.status_code == 404:
            raise ProviderError("inspire_not_found", f"INSPIRE record not found for {paper_id}")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError("inspire_fetch_failed", str(exc)) from exc

        raw = response.json()
        _cache_raw_record(raw, requested_id=normalized_id)
        return raw


def _normalize_record(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata", payload) or {}
    arxiv_id = _first_arxiv_id(metadata)
    recid = str(metadata.get("control_number") or payload.get("id") or "")
    paper_id = f"arXiv:{arxiv_id}" if arxiv_id else (f"inspire:{recid}" if recid else "")
    doi = _first_doi(metadata)
    return {
        "paper_id": paper_id,
        "title": _first_title(metadata),
        "abstract": _first_abstract(metadata),
        "authors": _authors(metadata),
        "arxiv_id": arxiv_id,
        "inspire_recid": recid,
        "doi": doi,
        "identifiers": _identifiers(paper_id=paper_id, arxiv_id=arxiv_id, inspire_recid=recid, doi=doi),
        "year": _year(metadata),
        "published": str(metadata.get("earliest_date") or metadata.get("preprint_date") or ""),
        "citation_count": int(metadata.get("citation_count") or 0),
    }


def _clamp_limit(limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = MAX_PAGE_SIZE
    return max(1, min(value, MAX_PAGE_SIZE))


def _normalize_sort(sort: str) -> str:
    normalized = (sort or "mostrecent").strip().lower()
    if normalized not in {"mostrecent", "mostcited"}:
        raise ProviderError("unsupported_citer_sort", f"Unsupported INSPIRE citer sort: {sort}")
    return normalized


def _citers_cache_path(paper_id: str, sort: str, limit: int):
    paths = CachePaths.for_paper(paper_id)
    if sort == "mostrecent" and limit == MAX_PAGE_SIZE:
        return paths.inspire_citers
    suffix = sort if limit == MAX_PAGE_SIZE else f"{sort}_{limit}"
    return paths.inspire_citers.with_name(f"citers_{suffix}.json")


def _normalize_reference(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("reference") or item
    recid = _reference_recid(item)
    arxiv_id = _reference_arxiv_id(item)
    title = raw.get("title") or raw.get("titles") or ""
    paper_id = normalize_paper_id(arxiv_id) if arxiv_id else (f"inspire:{recid}" if recid else "")
    doi = _first_doi(raw)
    if not paper_id and not title and not raw:
        return {}
    out = {
        "paper_id": paper_id,
        "title": _clean_inspire_text(_string_or_first(title)),
        "raw_inspire_reference": item,
    }
    record_ref = (item.get("record") or {}).get("$ref")
    if record_ref:
        out["record_ref"] = str(record_ref)
    if publication_info := raw.get("publication_info"):
        out["publication_info"] = publication_info
    if abstract := _first_abstract(raw):
        out["abstract"] = abstract
    if authors := _authors(raw):
        out["authors"] = authors
    if arxiv_id:
        out["arxiv_id"] = arxiv_id
    if recid:
        out["inspire_recid"] = recid
    if doi:
        out["doi"] = doi
    if identifiers := _identifiers(paper_id=paper_id, arxiv_id=arxiv_id, inspire_recid=recid, doi=doi):
        out["identifiers"] = identifiers
    if year := _year(raw):
        out["year"] = year
    published = str(raw.get("earliest_date") or raw.get("preprint_date") or "")
    if published:
        out["published"] = published
    if raw.get("citation_count") is not None:
        out["citation_count"] = int(raw.get("citation_count") or 0)
    return out


def _cache_raw_record(raw: dict[str, Any], *, requested_id: str) -> None:
    keys = {requested_id}
    metadata = _normalize_record(raw)
    if paper_id := metadata.get("paper_id"):
        keys.add(str(paper_id))
    if recid := metadata.get("inspire_recid"):
        keys.add(f"inspire:{recid}")
    if arxiv_id := metadata.get("arxiv_id"):
        keys.add(f"arXiv:{arxiv_id}")
    for key in {normalize_paper_id(item) for item in keys if item}:
        write_json(CachePaths.for_paper(key).inspire_metadata, raw)


def _reference_lookup_id(reference: dict[str, Any]) -> str:
    if recid := reference.get("inspire_recid"):
        return f"inspire:{recid}"
    if paper_id := reference.get("paper_id"):
        return normalize_paper_id(str(paper_id))
    return ""


def _merge_reference_metadata(reference: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    merged = dict(reference)
    merged["paper_id"] = metadata.get("paper_id") or merged.get("paper_id", "")
    merged["title"] = metadata.get("title") or merged.get("title", "")
    merged["abstract"] = metadata.get("abstract") or merged.get("abstract", "")
    merged["authors"] = metadata.get("authors") or merged.get("authors", [])
    for key in ("arxiv_id", "inspire_recid", "doi", "year", "published", "citation_count"):
        value = metadata.get(key)
        if value not in (None, "", []):
            merged[key] = value
        elif key not in merged:
            merged[key] = "" if key not in {"year", "citation_count"} else None
    merged["identifiers"] = metadata.get("identifiers") or _identifiers(
        paper_id=str(merged.get("paper_id") or ""),
        arxiv_id=str(merged.get("arxiv_id") or ""),
        inspire_recid=str(merged.get("inspire_recid") or ""),
        doi=str(merged.get("doi") or ""),
    )
    merged["metadata_enriched"] = True
    merged.pop("metadata_enrichment_error", None)
    return merged


def _identifiers(*, paper_id: str, arxiv_id: str, inspire_recid: str, doi: str) -> dict[str, str]:
    identifiers: dict[str, str] = {}
    if paper_id:
        identifiers["paper_id"] = paper_id
    if arxiv_id:
        identifiers["arxiv"] = f"arXiv:{arxiv_id}"
        identifiers["arxiv_id"] = arxiv_id
    if inspire_recid:
        identifiers["inspire"] = f"inspire:{inspire_recid}"
        identifiers["inspire_recid"] = inspire_recid
    if doi:
        identifiers["doi"] = doi
    return identifiers


def _reference_recid(item: dict[str, Any]) -> str:
    record = item.get("record") or {}
    return str(record.get("$ref", "").rstrip("/").split("/")[-1] or item.get("recid") or "")


def _reference_arxiv_id(item: dict[str, Any]) -> str:
    raw = item.get("reference") or item
    return str(raw.get("arxiv_eprint") or raw.get("arxiv_id") or raw.get("eprint") or "")


def _references_cache_is_current(cached: Any) -> bool:
    if not isinstance(cached, list):
        return False
    if not cached:
        return True
    return all(isinstance(item, dict) and "paper_id" in item and "title" in item for item in cached)


def _references_cache_is_enriched(cached: Any) -> bool:
    if not _references_cache_is_current(cached):
        return False
    for item in cached:
        if not _reference_lookup_id(item):
            continue
        if item.get("metadata_enriched") is not True:
            return False
        for key in ("title", "abstract", "authors", "arxiv_id", "inspire_recid"):
            if key not in item:
                return False
    return True


def _first_title(metadata: dict[str, Any]) -> str:
    titles = metadata.get("titles") or []
    if titles and isinstance(titles[0], dict):
        return _clean_inspire_text(titles[0].get("title"))
    return _clean_inspire_text(metadata.get("title"))


def _first_abstract(metadata: dict[str, Any]) -> str:
    abstracts = metadata.get("abstracts") or []
    if abstracts:
        first = abstracts[0]
        if isinstance(first, dict):
            return _clean_inspire_text(first.get("value") or first.get("summary"))
        return _clean_inspire_text(first)
    return _clean_inspire_text(metadata.get("abstract"))


def _first_arxiv_id(metadata: dict[str, Any]) -> str:
    for item in metadata.get("arxiv_eprints") or []:
        value = item.get("value") or item.get("eprint")
        if value:
            return str(value)
    return ""


def _first_doi(metadata: dict[str, Any]) -> str:
    for item in metadata.get("dois") or []:
        if isinstance(item, dict) and item.get("value"):
            return str(item["value"])
        if isinstance(item, str):
            return item
    return ""


def _year(metadata: dict[str, Any]) -> int | None:
    for key in ("earliest_date", "preprint_date"):
        value = str(metadata.get(key) or "")
        if len(value) >= 4 and value[:4].isdigit():
            return int(value[:4])
    for item in metadata.get("publication_info") or []:
        if isinstance(item, dict) and item.get("year"):
            return int(item["year"])
    return None


def _authors(metadata: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for author in metadata.get("authors") or []:
        name = author.get("full_name") or author.get("name") or author.get("display_name")
        if name:
            names.append(str(name))
    return names


def _string_or_first(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        first = value[0]
        if isinstance(first, dict):
            return str(first.get("title") or first.get("value") or first.get("summary") or "").strip()
        return str(first).strip()
    return str(value or "").strip()


def _clean_cached_paper_items(items: list[Any]) -> tuple[list[Any], bool]:
    changed = False
    cleaned: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            cleaned.append(item)
            continue
        record = dict(item)
        for key in ("title", "abstract"):
            if key not in record:
                continue
            value = _clean_inspire_text(record.get(key))
            if value != record.get(key):
                record[key] = value
                changed = True
        cleaned.append(record)
    return cleaned, changed


def _clean_inspire_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = MATHML_RE.sub(lambda match: _mathml_to_text(match.group(0)), text)
    text = HTML_TAG_RE.sub("", text)
    text = html.unescape(text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = re.sub(r"\s+([,.;:)\]\}])", r"\1", text)
    text = re.sub(r"([(\[\{])\s+", r"\1", text)
    return text


def _mathml_to_text(markup: str) -> str:
    try:
        root = ET.fromstring(markup)
    except ET.ParseError:
        return _clean_inspire_text(HTML_TAG_RE.sub("", markup))
    return _normalize_math_text(_math_node_text(root))


def _math_node_text(node: ET.Element) -> str:
    tag = _local_name(node.tag)
    children = list(node)
    if tag in {"math", "mrow", "mstyle", "mpadded", "mphantom"}:
        return _math_children_text(node)
    if tag in {"mi", "mn", "mo", "mtext"}:
        return _math_token_text((node.text or "") + _math_child_elements_text(node))
    if tag == "msub" and len(children) >= 2:
        return f"{_math_node_text(children[0])}_{_math_node_text(children[1])}"
    if tag == "msup" and len(children) >= 2:
        return f"{_math_node_text(children[0])}^{_math_node_text(children[1])}"
    if tag == "msubsup" and len(children) >= 3:
        return f"{_math_node_text(children[0])}_{_math_node_text(children[1])}^{_math_node_text(children[2])}"
    if tag == "mfrac" and len(children) >= 2:
        return f"({_math_node_text(children[0])})/({_math_node_text(children[1])})"
    if tag == "msqrt" and children:
        return f"sqrt({_math_children_text(node)})"
    if tag == "mroot" and len(children) >= 2:
        return f"root({_math_node_text(children[0])},{_math_node_text(children[1])})"
    if tag in {"semantics", "menclose"} and children:
        return _math_node_text(children[0])
    if tag in {"annotation", "annotation-xml"}:
        return ""
    return _math_children_text(node) or _math_token_text(node.text or "")


def _math_children_text(node: ET.Element) -> str:
    parts = []
    if node.text and node.text.strip():
        parts.append(node.text)
    parts.append(_math_child_elements_text(node))
    return "".join(parts)


def _math_child_elements_text(node: ET.Element) -> str:
    parts = []
    for child in list(node):
        if _local_name(child.tag) not in {"annotation", "annotation-xml"}:
            parts.append(_math_node_text(child))
        if child.tail and child.tail.strip():
            parts.append(child.tail)
    return "".join(parts)


def _math_token_text(text: str) -> str:
    return html.unescape(text).replace("\u2062", "").replace("\xa0", " ").strip()


def _normalize_math_text(text: str) -> str:
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = re.sub(r"\s+([,.;:)\]\}])", r"\1", text)
    text = re.sub(r"([(\[\{])\s+", r"\1", text)
    return text


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
