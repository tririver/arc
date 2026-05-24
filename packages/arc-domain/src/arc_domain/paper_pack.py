from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from arc_paper.ids import normalize_paper_id

from . import paper
from .cache import DomainPaths, now_iso, read_json, update_status, write_json


def build_paper_json_pack(
    *,
    paths: DomainPaths,
    refresh: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    update_status(paths, stage="paper_json_pack_started")
    graph = read_json(paths.domain_graph, {})
    nodes = graph.get("nodes") or []
    papers = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_paper_json, node, refresh): node.get("paper_id") or node.get("id")
            for node in nodes
            if node.get("paper_id") or node.get("id")
        }
        for future in as_completed(futures):
            papers.append(future.result())
    papers.sort(key=lambda item: (_role_order(item.get("role", "")), item.get("paper_id", "")))
    pack = {
        "schema_version": "arc.domain_paper_json_pack.v1",
        "domain_id": paths.domain_id,
        "foundation_paper": graph.get("foundation_paper", ""),
        "paper_count": len(papers),
        "papers": papers,
        "warnings": _pack_warnings(papers),
        "created_at": now_iso(),
    }
    write_json(paths.paper_json_pack, pack)
    update_status(paths, stage="paper_json_pack_done", paper_json_pack_paper_count=len(papers))
    return {
        "domain_id": paths.domain_id,
        "paper_json_pack_path": str(paths.paper_json_pack),
        "paper_json_pack": pack,
    }


def _paper_json(node: dict[str, Any], refresh: bool) -> dict[str, Any]:
    paper_id = normalize_paper_id(node.get("paper_id") or node.get("id") or "")
    warnings = []
    metadata: dict[str, Any] = {}
    references: list[dict[str, Any]] = []
    toc: list[dict[str, Any]] = []
    try:
        metadata = paper.metadata(paper_id, refresh=refresh)
    except Exception as exc:
        warnings.append(f"metadata_unavailable:{exc}")
    try:
        references = paper.references(paper_id, refresh=refresh, enrich=False)
    except Exception as exc:
        warnings.append(f"references_unavailable:{exc}")
    try:
        toc = paper.toc(paper_id, refresh=refresh)
    except Exception as exc:
        warnings.append(f"toc_unavailable:{exc}")
    return {
        "paper_id": paper_id,
        "role": node.get("role", ""),
        "metadata": metadata,
        "references": references,
        "toc": toc,
        "warnings": warnings,
    }


def _role_order(role: str) -> int:
    order = {
        "selected_foundation": 0,
        "parent_foundation": 1,
        "domain_paper": 2,
        "common_reference": 3,
    }
    return order.get(role, 9)


def _pack_warnings(papers: list[dict[str, Any]]) -> list[str]:
    warnings = []
    missing_toc = sum(
        1
        for item in papers
        if any(str(warning).startswith("toc_unavailable:") for warning in item.get("warnings", []))
    )
    if missing_toc:
        warnings.append(f"{missing_toc} papers have no cached ar5iv table of contents")
    missing_references = sum(
        1
        for item in papers
        if any(str(warning).startswith("references_unavailable:") for warning in item.get("warnings", []))
    )
    if missing_references:
        warnings.append(f"{missing_references} papers have no cached reference list")
    return warnings
