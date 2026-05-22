from __future__ import annotations

from typing import Any

import httpx

from ..cache import CachePaths, ONE_MONTH_SECONDS, read_json, write_json
from ..ids import arxiv_path_id, normalize_paper_id
from .base import ProviderError


BASE_URL = "https://inspirehep.net/api"
MAX_PAGE_SIZE = 1000
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

    def get_references(self, paper_id: str, *, refresh: bool = False) -> list[dict[str, Any]]:
        paths = CachePaths.for_paper(paper_id)
        if not refresh and (cached := read_json(paths.inspire_references)) and _references_cache_is_current(cached):
            return cached

        raw = self.get_raw_record(paper_id, refresh=refresh)
        references = [
            normalized
            for item in raw.get("metadata", {}).get("references", [])
            if (normalized := _normalize_reference(item))
        ]
        write_json(paths.inspire_references, references)
        return references

    def get_citers(self, paper_id: str, *, refresh: bool = False) -> list[dict[str, Any]]:
        paths = CachePaths.for_paper(paper_id)
        if not refresh and (cached := read_json(paths.inspire_citers, ttl_seconds=ONE_MONTH_SECONDS)):
            return cached

        metadata = self.get_metadata(paper_id, refresh=refresh)
        recid = metadata.get("inspire_recid")
        if not recid:
            return []

        response = self.client.get(
            f"{BASE_URL}/literature",
            params={
                "q": f"refersto:recid:{recid}",
                "size": str(MAX_PAGE_SIZE),
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
        write_json(paths.inspire_citers, citers)
        return citers

    def get_citer_count(self, paper_id: str, *, refresh: bool = False) -> int:
        return int(self.get_metadata(paper_id, refresh=refresh).get("citation_count") or 0)

    def get_raw_record(self, paper_id: str, *, refresh: bool = False) -> dict[str, Any]:
        paths = CachePaths.for_paper(paper_id)
        if not refresh and (cached := read_json(paths.inspire_metadata)):
            return cached

        aid = arxiv_path_id(paper_id)
        if not aid:
            raise ProviderError("not_arxiv_id", f"INSPIRE arXiv endpoint requires an arXiv ID: {paper_id}")

        response = self.client.get(f"{BASE_URL}/arxiv/{aid}", timeout=self.timeout)
        if response.status_code == 404:
            raise ProviderError("inspire_not_found", f"INSPIRE record not found for {paper_id}")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError("inspire_fetch_failed", str(exc)) from exc

        raw = response.json()
        write_json(paths.inspire_metadata, raw)
        return raw


def _normalize_record(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata", payload) or {}
    arxiv_id = _first_arxiv_id(metadata)
    recid = str(metadata.get("control_number") or payload.get("id") or "")
    return {
        "paper_id": f"arXiv:{arxiv_id}" if arxiv_id else (f"inspire:{recid}" if recid else ""),
        "title": _first_title(metadata),
        "abstract": _first_abstract(metadata),
        "authors": _authors(metadata),
        "arxiv_id": arxiv_id,
        "inspire_recid": recid,
        "doi": _first_doi(metadata),
        "year": _year(metadata),
        "published": str(metadata.get("earliest_date") or metadata.get("preprint_date") or ""),
        "citation_count": int(metadata.get("citation_count") or 0),
    }


def _normalize_reference(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("reference") or item
    recid = _reference_recid(item)
    arxiv_id = _reference_arxiv_id(item)
    title = raw.get("title") or raw.get("titles") or ""
    paper_id = normalize_paper_id(arxiv_id) if arxiv_id else (f"inspire:{recid}" if recid else "")
    if not paper_id and not title:
        return {}
    out = {"paper_id": paper_id, "title": _string_or_first(title)}
    if abstract := _first_abstract(raw):
        out["abstract"] = abstract
    if authors := _authors(raw):
        out["authors"] = authors
    if arxiv_id:
        out["arxiv_id"] = arxiv_id
    if recid:
        out["inspire_recid"] = recid
    if doi := _first_doi(raw):
        out["doi"] = doi
    if year := _year(raw):
        out["year"] = year
    published = str(raw.get("earliest_date") or raw.get("preprint_date") or "")
    if published:
        out["published"] = published
    if raw.get("citation_count") is not None:
        out["citation_count"] = int(raw.get("citation_count") or 0)
    return out


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


def _first_title(metadata: dict[str, Any]) -> str:
    titles = metadata.get("titles") or []
    if titles and isinstance(titles[0], dict):
        return str(titles[0].get("title") or "").strip()
    return str(metadata.get("title") or "").strip()


def _first_abstract(metadata: dict[str, Any]) -> str:
    abstracts = metadata.get("abstracts") or []
    if abstracts:
        first = abstracts[0]
        if isinstance(first, dict):
            return str(first.get("value") or first.get("summary") or "").strip()
        return str(first).strip()
    return str(metadata.get("abstract") or "").strip()


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
        return str(value[0] if value else "").strip()
    return str(value or "").strip()
