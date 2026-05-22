from __future__ import annotations

from typing import Any

import httpx

from ..cache import CachePaths, ONE_MONTH_SECONDS, read_json, write_json
from ..ids import arxiv_path_id, normalize_paper_id
from .base import ProviderError


BASE_URL = "https://inspirehep.net/api"
MAX_PAGE_SIZE = 1000


class InspireProvider:
    def __init__(self, *, client: httpx.Client | None = None, timeout: float = 60.0):
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self.timeout = timeout

    def get_metadata(self, paper_id: str, *, refresh: bool = False) -> dict[str, Any]:
        raw = self.get_raw_record(paper_id, refresh=refresh)
        return _normalize_record(raw)

    def get_references(self, paper_id: str, *, refresh: bool = False) -> list[dict[str, Any]]:
        paths = CachePaths.for_paper(paper_id)
        if not refresh and (cached := read_json(paths.inspire_references)):
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
            params={"q": f"refersto:recid:{recid}", "size": str(MAX_PAGE_SIZE), "format": "json"},
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
    recid = str(payload.get("id") or metadata.get("control_number") or "")
    return {
        "paper_id": f"arXiv:{arxiv_id}" if arxiv_id else (f"inspire:{recid}" if recid else ""),
        "title": _first_title(metadata),
        "abstract": _first_abstract(metadata),
        "authors": _authors(metadata),
        "arxiv_id": arxiv_id,
        "inspire_recid": recid,
        "citation_count": int(metadata.get("citation_count") or 0),
    }


def _normalize_reference(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("reference") or item
    record = item.get("record") or {}
    recid = str(record.get("$ref", "").rstrip("/").split("/")[-1] or item.get("recid") or "")
    arxiv_id = str(raw.get("arxiv_eprint") or raw.get("arxiv_id") or raw.get("eprint") or "")
    title = raw.get("title") or raw.get("titles") or ""
    paper_id = normalize_paper_id(arxiv_id) if arxiv_id else (f"inspire:{recid}" if recid else "")
    if not paper_id and not title:
        return {}
    out = {"paper_id": paper_id, "title": _string_or_first(title)}
    if recid:
        out["inspire_recid"] = recid
    return out


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
