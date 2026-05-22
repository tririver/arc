from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from . import paper
from .cache import DomainPaths, now_iso, read_json, update_status, write_json


CONCLUSION_SELECTORS = (
    "conclusion",
    "conclusions",
    "summary and discussion",
    "summary",
    "discussion",
    "outlook",
)


def build_evidence_pack(
    *,
    paths: DomainPaths,
    refresh: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    update_status(paths, stage="evidence_started")
    graph = read_json(paths.domain_graph, {})
    nodes = graph.get("nodes") or []
    papers = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_paper_evidence, node, refresh): node.get("paper_id") or node.get("id")
            for node in nodes
            if node.get("paper_id") or node.get("id")
        }
        for future in as_completed(futures):
            papers.append(future.result())
    papers.sort(key=lambda item: _role_order(item.get("role", "")))
    pack = {
        "schema_version": "arc.domain_evidence_pack.v1",
        "domain_id": paths.domain_id,
        "foundation_paper": graph.get("foundation_paper", ""),
        "paper_count": len(papers),
        "papers": papers,
        "warnings": _pack_warnings(papers),
        "created_at": now_iso(),
    }
    write_json(paths.evidence_pack, pack)
    update_status(paths, stage="evidence_done", evidence_paper_count=len(papers))
    return {"domain_id": paths.domain_id, "evidence_pack_path": str(paths.evidence_pack), "evidence_pack": pack}


def _paper_evidence(node: dict[str, Any], refresh: bool) -> dict[str, Any]:
    paper_id = node.get("paper_id") or node.get("id") or ""
    warnings = []
    try:
        meta = paper.metadata(paper_id, refresh=refresh)
    except Exception as exc:
        meta = dict(node)
        warnings.append(f"metadata_unavailable:{exc}")
    conclusion = None
    for selector in CONCLUSION_SELECTORS:
        try:
            section = paper.section(paper_id, selector, refresh=refresh)
            conclusion = {
                "section_id": section.get("section_id", ""),
                "title": section.get("title", ""),
                "text": _compact(section.get("text", ""), 5000),
                "selector": selector,
            }
            break
        except Exception:
            continue
    if conclusion is None:
        warnings.append("conclusion_section_unavailable")
    return {
        "paper_id": paper_id,
        "role": node.get("role", ""),
        "title": meta.get("title") or node.get("title", ""),
        "abstract": meta.get("abstract") or node.get("abstract", ""),
        "authors": meta.get("authors") or node.get("authors", []),
        "year": meta.get("year") or node.get("year"),
        "citation_count": int(meta.get("citation_count") or node.get("citation_count") or 0),
        "selection_reason": node.get("selection_reason", ""),
        "conclusion": conclusion,
        "warnings": warnings,
    }


def _compact(text: str, limit: int) -> str:
    text = "\n".join(line.strip() for line in str(text or "").splitlines() if line.strip())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[truncated]"


def _role_order(role: str) -> tuple[int, str]:
    order = {
        "selected_foundation": 0,
        "parent_foundation": 1,
        "domain_paper": 2,
        "common_reference": 3,
    }
    return (order.get(role, 9), role)


def _pack_warnings(papers: list[dict[str, Any]]) -> list[str]:
    missing = sum(1 for item in papers if item.get("conclusion") is None)
    if not missing:
        return []
    return [f"{missing} papers have no cached conclusion/outlook/discussion section"]
