from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from arc_paper_query import service as paper_service
from arc_paper_query.ids import normalize_paper_id


class PaperQueryError(RuntimeError):
    pass


def metadata(paper_id: str, *, refresh: bool = False) -> dict[str, Any]:
    return _data(paper_service.get_metadata(paper_id, refresh=refresh))


def references(paper_id: str, *, refresh: bool = False, enrich: bool = False) -> list[dict[str, Any]]:
    return list(_data(paper_service.get_references(paper_id, refresh=refresh, enrich=enrich)) or [])


def citers(
    paper_id: str,
    *,
    refresh: bool = False,
    limit: int = 1000,
    sort: str = "mostrecent",
) -> list[dict[str, Any]]:
    return list(_data(paper_service.get_citers(paper_id, refresh=refresh, limit=limit, sort=sort)) or [])


def section(paper_id: str, selector: str, *, refresh: bool = False) -> dict[str, Any]:
    result = paper_service.get_section(paper_id, selector, refresh=refresh)
    if result.get("ok"):
        return result["data"]
    raise PaperQueryError((result.get("error") or {}).get("message") or f"Section not found: {selector}")


def toc(paper_id: str, *, refresh: bool = False) -> list[dict[str, Any]]:
    return list(_data(paper_service.get_toc(paper_id, refresh=refresh)) or [])


def fetch_many(
    ids: list[str],
    func: Callable[[str], Any],
    *,
    workers: int = 8,
) -> dict[str, Any]:
    unique = list(dict.fromkeys(normalize_paper_id(item) for item in ids if item))
    out: dict[str, Any] = {}
    if not unique:
        return out
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(func, paper_id): paper_id for paper_id in unique}
        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                out[paper_id] = future.result()
            except Exception as exc:
                out[paper_id] = {"error": str(exc)}
    return out


def _data(result: dict[str, Any]) -> Any:
    if result.get("ok"):
        return result.get("data")
    error = result.get("error") or {}
    raise PaperQueryError(error.get("message") or error.get("code") or "paper-query failed")
