from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

from arc_llm import run_json
from arc_llm.call_record import ARC_LLM_CALL_RECORD_FIELD, ARC_LLM_CALL_RECORD_SCHEMA
from arc_paper.ids import arxiv_path_id, normalize_paper_id

from . import paper
from .cache import DomainPaths, now_iso, read_json, update_status, write_json
from .text import citation_per_year, log_score, paper_key, token_overlap_score


INTENT_RANKING_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "arc.domain-intent-ranking-v1",
    "type": "object",
    "additionalProperties": False,
    "required": ["ranked_paper_ids", "reasoning"],
    "properties": {
        "ranked_paper_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "reasoning": {"type": "string"},
        ARC_LLM_CALL_RECORD_FIELD: ARC_LLM_CALL_RECORD_SCHEMA,
    },
}
CITATION_RATE_WEIGHT = 0.1
RECENCY_WEIGHT = 0.5
INTENT_OVERLAP_WEIGHT = 1.0
GRAPH_CITER_WEIGHT = 2.0
REFERENCE_EDGE_WEIGHT = 0.5
RECENT_ARXIV_WINDOW_DAYS = 365


def build_network(
    *,
    seed_paper: str,
    intent: str,
    paths: DomainPaths,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    refresh: bool = False,
    workers: int = 8,
    max_citers: int = 1000,
    selected_count: int = 50,
    max_nodes: int = 60,
) -> dict[str, Any]:
    update_status(paths, stage="network_started")
    selection = read_json(paths.foundation_selection, {})
    selected_foundation = selection.get("selected_foundation") or {}
    foundation_id = normalize_paper_id(selected_foundation.get("paper_id") or seed_paper)
    foundation_meta = paper.metadata(foundation_id, refresh=refresh)
    _copy_selection_metadata(foundation_meta, selected_foundation)

    citer_pool = _merged_citers(foundation_id, refresh=refresh, limit=max_citers)
    write_json(
        paths.citer_pool,
        {
            "schema_version": "arc.domain_citer_pool.v1",
            "foundation_paper": foundation_id,
            "citers": citer_pool,
            "created_at": now_iso(),
        },
    )

    intent_ranking = _rank_by_intent(citer_pool, intent=intent, provider=provider, model=model, model_tier=model_tier)
    write_json(paths.intent_rankings, intent_ranking)

    selected = _select_domain_papers(
        citer_pool,
        foundation_id=foundation_id,
        intent_ranking=intent_ranking,
        intent=intent,
        selected_count=selected_count,
    )

    selected_ids = [item["paper_id"] for item in selected]
    refs_by_selected = paper.fetch_many(
        selected_ids,
        lambda paper_id: paper.references(paper_id, refresh=refresh, enrich=False),
        workers=workers,
    )
    selected = _add_in_graph_citer_scores(selected, refs_by_selected=refs_by_selected)
    selected_ids = [item["paper_id"] for item in selected]
    common_refs = _common_references(
        foundation_id=foundation_id,
        selected_ids=selected_ids,
        refs_by_selected=refs_by_selected,
        max_extra=max(0, max_nodes - 1 - len(selected)),
        refresh=refresh,
        workers=workers,
    )
    write_json(
        paths.reference_overlap,
        {
            "schema_version": "arc.domain_reference_overlap.v1",
            "selected_papers": selected_ids,
            "common_references": common_refs,
            "created_at": now_iso(),
        },
    )

    parent_foundations = _enrich_parent_foundations(
        selection.get("parent_foundations") or [],
        refresh=refresh,
        workers=workers,
    )
    selected = _add_reference_edge_scores(
        selected,
        foundation_id=foundation_id,
        parent_foundations=parent_foundations,
        common_references=common_refs,
        refs_by_selected=refs_by_selected,
    )
    selected_ids = [item["paper_id"] for item in selected]
    write_json(
        paths.selected_papers,
        {
            "schema_version": "arc.domain_selected_papers.v1",
            "foundation_paper": foundation_id,
            "papers": selected,
            "created_at": now_iso(),
        },
    )
    graph = _build_graph(
        foundation=foundation_meta,
        parent_foundations=parent_foundations,
        selected_papers=selected,
        common_references=common_refs,
        refs_by_selected=refs_by_selected,
        intent=intent,
    )
    write_json(paths.domain_graph, graph)
    update_status(paths, stage="network_done", node_count=len(graph["nodes"]), edge_count=len(graph["edges"]))
    return {
        "domain_id": paths.domain_id,
        "foundation_paper": foundation_id,
        "citer_pool_size": len(citer_pool),
        "selected_paper_count": len(selected),
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "graph_path": str(paths.domain_graph),
        "selected_papers_path": str(paths.selected_papers),
        "reference_overlap_path": str(paths.reference_overlap),
        "graph": graph,
    }


def _merged_citers(foundation_id: str, *, refresh: bool, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    recent = paper.citers(foundation_id, refresh=refresh, limit=limit, sort="mostrecent")
    cited = paper.citers(foundation_id, refresh=refresh, limit=limit, sort="mostcited")
    merged: dict[str, dict[str, Any]] = {}
    source_ids: dict[str, list[str]] = {"mostrecent": [], "mostcited": []}
    for source, items in (("mostrecent", recent), ("mostcited", cited)):
        seen_in_source: set[str] = set()
        for index, item in enumerate(items):
            paper_id = normalize_paper_id(paper_key(item))
            if not paper_id or paper_id == foundation_id:
                continue
            if paper_id not in seen_in_source:
                source_ids[source].append(paper_id)
                seen_in_source.add(paper_id)
            record = dict(item)
            record["paper_id"] = paper_id
            record.setdefault("citer_sources", [])
            if source not in record["citer_sources"]:
                record["citer_sources"].append(source)
            record[f"{source}_rank"] = index + 1
            if paper_id in merged:
                existing = merged[paper_id]
                existing.update(
                    {
                        key: value
                        for key, value in record.items()
                        if key != "citer_sources" and value not in ("", None, [])
                    }
                )
                for label in record["citer_sources"]:
                    if label not in existing["citer_sources"]:
                        existing["citer_sources"].append(label)
            else:
                merged[paper_id] = record

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_source_len = max((len(ids) for ids in source_ids.values()), default=0)
    for index in range(max_source_len):
        for source in ("mostrecent", "mostcited"):
            ids = source_ids[source]
            if index >= len(ids):
                continue
            paper_id = ids[index]
            if paper_id in seen:
                continue
            ordered.append(merged[paper_id])
            seen.add(paper_id)
            if len(ordered) >= limit:
                return ordered
    return ordered


def _rank_by_intent(
    citer_pool: list[dict[str, Any]],
    *,
    intent: str,
    provider: str,
    model: str | None,
    model_tier: str | None = None,
) -> dict[str, Any]:
    if not intent.strip():
        return {"schema_version": "arc.domain_intent_ranking.v1", "ranked_paper_ids": [], "reasoning": "no intent supplied"}
    shortlist = sorted(
        citer_pool,
        key=lambda item: (
            token_overlap_score(f"{item.get('title', '')} {item.get('abstract', '')}", intent),
            int(item.get("citation_count") or 0),
        ),
        reverse=True,
    )[:120]
    compact = [
        {
            "paper_id": item.get("paper_id"),
            "title": item.get("title", ""),
            "abstract": str(item.get("abstract") or "")[:800],
            "year": item.get("year"),
            "citation_count": item.get("citation_count", 0),
        }
        for item in shortlist
    ]
    prompt = "\n\n".join(
        [
            "Rank up to 10 papers whose titles and abstracts best match the user's research intent.",
            "Return only IDs from the supplied list. Prefer scientifically specific matches over generic review papers.",
            f"User intent:\n{intent}",
            f"Candidate papers:\n{compact}",
            "Return JSON only.",
        ]
    )
    try:
        result = run_json(prompt, schema=INTENT_RANKING_SCHEMA, provider=provider, model=model, model_tier=model_tier)
        ids = [normalize_paper_id(item) for item in result.get("ranked_paper_ids", []) if item]
        valid = {item["paper_id"] for item in citer_pool}
        ranking = {
            "schema_version": "arc.domain_intent_ranking.v1",
            "ranked_paper_ids": [item for item in ids if item in valid][:10],
            "reasoning": str(result.get("reasoning") or ""),
            "method": "llm",
        }
        if isinstance(result.get(ARC_LLM_CALL_RECORD_FIELD), dict):
            ranking[ARC_LLM_CALL_RECORD_FIELD] = result[ARC_LLM_CALL_RECORD_FIELD]
        return ranking
    except Exception as exc:
        ranked = [
            item["paper_id"]
            for item in shortlist
            if token_overlap_score(f"{item.get('title', '')} {item.get('abstract', '')}", intent) > 0
        ][:10]
        return {
            "schema_version": "arc.domain_intent_ranking.v1",
            "ranked_paper_ids": ranked,
            "reasoning": f"deterministic lexical fallback after LLM failure: {exc}",
            "method": "deterministic_fallback",
        }


def _select_domain_papers(
    citer_pool: list[dict[str, Any]],
    *,
    foundation_id: str,
    intent_ranking: dict[str, Any],
    intent: str,
    selected_count: int,
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    current_year = now.year
    intent_rank = {
        paper_id: index
        for index, paper_id in enumerate(intent_ranking.get("ranked_paper_ids") or [], start=1)
    }
    scored = []
    for item in citer_pool:
        paper_id = normalize_paper_id(item.get("paper_id") or paper_key(item))
        if not paper_id or paper_id == foundation_id:
            continue
        record = dict(item)
        record["paper_id"] = paper_id
        cpy = citation_per_year(record, current_year)
        age = max(1, current_year - int(record.get("year") or current_year) + 1)
        recency = 1.0 / age
        intent_overlap = token_overlap_score(f"{record.get('title', '')} {record.get('abstract', '')}", intent)
        intent_boost = 0.0
        if paper_id in intent_rank:
            intent_boost = 2.0 - 0.12 * (intent_rank[paper_id] - 1)
        score = _domain_score(
            citation_per_year=cpy,
            recency=recency,
            intent_overlap=intent_overlap,
            intent_boost=intent_boost,
            in_graph_citer_score=0.0,
            reference_edge_count=0,
        )
        record["domain_score"] = round(score, 4)
        record["citation_per_year"] = round(cpy, 4)
        record["citation_rate_score"] = round(CITATION_RATE_WEIGHT * log_score(cpy), 4)
        record["recency"] = round(recency, 4)
        record["recency_score"] = round(RECENCY_WEIGHT * recency, 4)
        record["intent_overlap"] = round(intent_overlap, 4)
        record["intent_overlap_score"] = round(INTENT_OVERLAP_WEIGHT * intent_overlap, 4)
        record["intent_boost"] = round(intent_boost, 4)
        record["in_graph_citer_count"] = 0
        record["in_graph_citer_score"] = 0.0
        record["reference_edge_count"] = 0
        record["reference_edge_score"] = 0.0
        record["recent_arxiv"] = _is_recent_arxiv_paper(record, now=now)
        record["selection_reason"] = _selection_reason(record, paper_id in intent_rank)
        scored.append(record)
    scored.sort(
        key=lambda item: (item["domain_score"], item.get("citation_count") or 0, item.get("year") or 0),
        reverse=True,
    )
    selected = scored[:selected_count]
    selected_ids = {item["paper_id"] for item in selected}
    recent = [
        item
        for item in scored[selected_count:]
        if item.get("recent_arxiv") and item["paper_id"] not in selected_ids
    ]
    return [*selected, *recent]


def _add_in_graph_citer_scores(
    selected_papers: list[dict[str, Any]],
    *,
    refs_by_selected: dict[str, Any],
) -> list[dict[str, Any]]:
    selected_ids = [normalize_paper_id(item.get("paper_id") or paper_key(item)) for item in selected_papers]
    selected_set = set(selected_ids)
    counts = Counter()
    for source_id, refs in refs_by_selected.items():
        source_id = normalize_paper_id(source_id)
        if source_id not in selected_set or not isinstance(refs, list):
            continue
        seen = set()
        for ref in refs:
            target_id = normalize_paper_id(paper_key(ref))
            if not target_id or target_id == source_id or target_id not in selected_set or target_id in seen:
                continue
            seen.add(target_id)
            counts[target_id] += 1

    max_count = max(counts.values(), default=0)
    scored = []
    for item in selected_papers:
        record = dict(item)
        paper_id = normalize_paper_id(record.get("paper_id") or paper_key(record))
        count = int(counts.get(paper_id, 0))
        normalized = count / max_count if max_count else 0.0
        record["in_graph_citer_count"] = count
        record["in_graph_citer_score"] = round(normalized, 4)
        score = _domain_score(
            citation_per_year=float(record.get("citation_per_year") or 0),
            recency=float(record.get("recency") or 0),
            intent_overlap=float(record.get("intent_overlap") or 0),
            intent_boost=float(record.get("intent_boost") or 0),
            in_graph_citer_score=normalized,
            reference_edge_count=int(record.get("reference_edge_count") or 0),
        )
        record["domain_score"] = round(score, 4)
        record["selection_reason"] = _selection_reason(record, bool(record.get("intent_boost")))
        scored.append(record)
    scored.sort(
        key=lambda item: (
            item["domain_score"],
            item.get("in_graph_citer_count") or 0,
            item.get("citation_count") or 0,
            item.get("year") or 0,
        ),
        reverse=True,
    )
    return scored


def _add_reference_edge_scores(
    selected_papers: list[dict[str, Any]],
    *,
    foundation_id: str,
    parent_foundations: list[dict[str, Any]],
    common_references: list[dict[str, Any]],
    refs_by_selected: dict[str, Any],
) -> list[dict[str, Any]]:
    selected_ids = [normalize_paper_id(item.get("paper_id") or paper_key(item)) for item in selected_papers]
    node_ids = set(selected_ids)
    if foundation_id:
        node_ids.add(normalize_paper_id(foundation_id))
    for item in parent_foundations:
        parent_id = normalize_paper_id(item.get("paper_id") or paper_key(item))
        if parent_id:
            node_ids.add(parent_id)
    for item in common_references:
        common_id = normalize_paper_id(item.get("paper_id") or paper_key(item))
        if common_id:
            node_ids.add(common_id)

    counts: Counter[str] = Counter()
    for source_id, refs in refs_by_selected.items():
        source_id = normalize_paper_id(source_id)
        if source_id not in node_ids or not isinstance(refs, list):
            continue
        seen = set()
        for ref in refs:
            target_id = normalize_paper_id(paper_key(ref))
            if not target_id or target_id == source_id or target_id not in node_ids or target_id in seen:
                continue
            seen.add(target_id)
            counts[source_id] += 1

    scored = []
    for item in selected_papers:
        record = dict(item)
        paper_id = normalize_paper_id(record.get("paper_id") or paper_key(record))
        count = int(counts.get(paper_id, 0))
        record["reference_edge_count"] = count
        record["reference_edge_score"] = round(REFERENCE_EDGE_WEIGHT * count, 4)
        score = _domain_score(
            citation_per_year=float(record.get("citation_per_year") or 0),
            recency=float(record.get("recency") or 0),
            intent_overlap=float(record.get("intent_overlap") or 0),
            intent_boost=float(record.get("intent_boost") or 0),
            in_graph_citer_score=float(record.get("in_graph_citer_score") or 0),
            reference_edge_count=count,
        )
        record["domain_score"] = round(score, 4)
        record["selection_reason"] = _selection_reason(record, bool(record.get("intent_boost")))
        scored.append(record)
    scored.sort(
        key=lambda item: (
            item["domain_score"],
            item.get("reference_edge_count") or 0,
            item.get("in_graph_citer_count") or 0,
            item.get("citation_count") or 0,
            item.get("year") or 0,
        ),
        reverse=True,
    )
    return scored


def _domain_score(
    *,
    citation_per_year: float,
    recency: float,
    intent_overlap: float,
    intent_boost: float,
    in_graph_citer_score: float,
    reference_edge_count: int,
) -> float:
    return (
        CITATION_RATE_WEIGHT * log_score(citation_per_year)
        + RECENCY_WEIGHT * recency
        + INTENT_OVERLAP_WEIGHT * intent_overlap
        + intent_boost
        + GRAPH_CITER_WEIGHT * in_graph_citer_score
        + REFERENCE_EDGE_WEIGHT * reference_edge_count
    )


def _common_references(
    *,
    foundation_id: str,
    selected_ids: list[str],
    refs_by_selected: dict[str, Any],
    max_extra: int,
    refresh: bool,
    workers: int,
) -> list[dict[str, Any]]:
    selected_set = {normalize_paper_id(item) for item in selected_ids}
    selected_set.add(normalize_paper_id(foundation_id))
    counts = Counter()
    support: dict[str, list[str]] = defaultdict(list)
    embedded: dict[str, dict[str, Any]] = {}
    for source_id, refs in refs_by_selected.items():
        if not isinstance(refs, list):
            continue
        seen = set()
        for ref in refs:
            ref_id = normalize_paper_id(paper_key(ref))
            if not ref_id or ref_id in selected_set or ref_id in seen:
                continue
            seen.add(ref_id)
            counts[ref_id] += 1
            support[ref_id].append(source_id)
            embedded.setdefault(ref_id, ref)
    top_ids = [
        ref_id
        for ref_id, count in sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
        if count >= 2
    ][:max_extra]
    metadata_by_id = paper.fetch_many(
        top_ids,
        lambda paper_id: paper.metadata(paper_id, refresh=refresh),
        workers=workers,
    )
    common = []
    for ref_id in top_ids:
        meta = metadata_by_id.get(ref_id)
        if not isinstance(meta, dict) or meta.get("error"):
            meta = embedded.get(ref_id, {})
        common.append(
            {
                "paper_id": normalize_paper_id(meta.get("paper_id") or ref_id),
                "title": meta.get("title") or embedded.get(ref_id, {}).get("title", ""),
                "abstract": meta.get("abstract", ""),
                "authors": meta.get("authors", []),
                "year": meta.get("year"),
                "citation_count": int(meta.get("citation_count") or 0),
                "support_count": int(counts[ref_id]),
                "supported_by": support.get(ref_id, [])[:50],
                "identifiers": meta.get("identifiers") or {},
            }
        )
    return common


def _enrich_parent_foundations(
    parent_foundations: list[dict[str, Any]],
    *,
    refresh: bool,
    workers: int,
) -> list[dict[str, Any]]:
    ids = [normalize_paper_id(paper_key(item)) for item in parent_foundations if paper_key(item)]
    metadata_by_id = paper.fetch_many(
        ids,
        lambda paper_id: paper.metadata(paper_id, refresh=refresh),
        workers=workers,
    )
    enriched = []
    for item in parent_foundations:
        parent_id = normalize_paper_id(paper_key(item))
        meta = metadata_by_id.get(parent_id)
        if not isinstance(meta, dict) or meta.get("error"):
            meta = {}
        record = dict(meta)
        record["paper_id"] = normalize_paper_id(record.get("paper_id") or parent_id)
        record["title"] = record.get("title") or item.get("title", "")
        record["reason"] = item.get("reason", "")
        record["selection_reason"] = item.get("reason", "")
        for key in ("abstract", "authors", "year", "citation_count", "identifiers"):
            if key not in record and key in item:
                record[key] = item[key]
        enriched.append(record)
    return enriched


def _build_graph(
    *,
    foundation: dict[str, Any],
    parent_foundations: list[dict[str, Any]],
    selected_papers: list[dict[str, Any]],
    common_references: list[dict[str, Any]],
    refs_by_selected: dict[str, Any],
    intent: str,
) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    foundation_id = normalize_paper_id(foundation.get("paper_id") or "")
    nodes[foundation_id] = _node(foundation, role="selected_foundation")
    for item in parent_foundations:
        parent_id = normalize_paper_id(item.get("paper_id") or item.get("upi") or "")
        if parent_id:
            nodes[parent_id] = _node(item, role="parent_foundation")
    for item in selected_papers:
        nodes[item["paper_id"]] = _node(item, role="domain_paper")
    for item in common_references:
        if item["paper_id"] not in nodes:
            nodes[item["paper_id"]] = _node(item, role="common_reference")

    edges = []
    seen_edges = set()
    node_ids = set(nodes)
    for source_id, refs in refs_by_selected.items():
        source_id = normalize_paper_id(source_id)
        if source_id not in node_ids:
            continue
        if foundation_id and source_id != foundation_id:
            _add_edge(edges, seen_edges, source_id, foundation_id, relation="cites_foundation")
        if not isinstance(refs, list):
            continue
        for ref in refs:
            target_id = normalize_paper_id(paper_key(ref))
            if target_id in node_ids and target_id != source_id:
                _add_edge(edges, seen_edges, source_id, target_id, relation="cites")
    return {
        "schema_version": "arc.domain_graph.v1",
        "intent": intent,
        "foundation_paper": foundation_id,
        "nodes": list(nodes.values()),
        "edges": edges,
        "created_at": now_iso(),
    }


def _node(paper_record: dict[str, Any], *, role: str) -> dict[str, Any]:
    paper_id = normalize_paper_id(paper_record.get("paper_id") or paper_record.get("upi") or "")
    node = {
        "id": paper_id,
        "paper_id": paper_id,
        "role": role,
        "title": paper_record.get("title", ""),
        "abstract": paper_record.get("abstract", ""),
        "authors": paper_record.get("authors", []),
        "year": paper_record.get("year"),
        "citation_count": int(paper_record.get("citation_count") or paper_record.get("cited_by_count") or 0),
        "citation_per_year": paper_record.get("citation_per_year"),
        "domain_score": paper_record.get("domain_score"),
        "citation_rate_score": paper_record.get("citation_rate_score"),
        "recency": paper_record.get("recency"),
        "recency_score": paper_record.get("recency_score"),
        "intent_overlap": paper_record.get("intent_overlap"),
        "intent_overlap_score": paper_record.get("intent_overlap_score"),
        "intent_boost": paper_record.get("intent_boost"),
        "in_graph_citer_count": paper_record.get("in_graph_citer_count"),
        "in_graph_citer_score": paper_record.get("in_graph_citer_score"),
        "reference_edge_count": paper_record.get("reference_edge_count"),
        "reference_edge_score": paper_record.get("reference_edge_score"),
        "selection_reason": paper_record.get("selection_reason") or paper_record.get("reason", ""),
        "support_count": paper_record.get("support_count"),
        "identifiers": paper_record.get("identifiers") or {},
    }
    for field in (
        "source_role",
        "llm_added",
        "llm_recommended",
        "llm_addition_reason",
        "llm_reference_query",
        "llm_verified_evidence_urls",
        "llm_reference_inference",
    ):
        if field in paper_record:
            node[field] = paper_record[field]
    return node


def _copy_selection_metadata(target: dict[str, Any], source: dict[str, Any]) -> None:
    if reason := source.get("reason"):
        target["reason"] = reason
    for field in (
        "source_role",
        "llm_added",
        "llm_recommended",
        "llm_addition_reason",
        "llm_reference_query",
        "llm_verified_evidence_urls",
        "llm_reference_inference",
    ):
        if field in source:
            target[field] = source[field]


def _add_edge(edges: list[dict[str, Any]], seen: set[tuple[str, str, str]], source: str, target: str, *, relation: str) -> None:
    key = (source, target, relation)
    if key in seen:
        return
    seen.add(key)
    edges.append({"source": source, "target": target, "relation": relation})


def _selection_reason(record: dict[str, Any], intent_ranked: bool) -> str:
    parts = []
    if intent_ranked:
        parts.append("LLM intent-ranked")
    if record.get("recent_arxiv"):
        parts.append("recent arXiv")
    if record.get("citation_per_year", 0) > 0:
        parts.append("citation-per-year")
    if record.get("in_graph_citer_count", 0) > 0:
        parts.append("cited-within-graph")
    if record.get("reference_edge_count", 0) > 0:
        parts.append("reference-connected")
    if record.get("year"):
        parts.append("recency")
    return ", ".join(parts) or "representative foundation citer"


def _is_recent_arxiv_paper(
    record: dict[str, Any],
    *,
    now: datetime | None = None,
    window_days: int = RECENT_ARXIV_WINDOW_DAYS,
) -> bool:
    if not _has_arxiv_id(record):
        return False
    paper_date = _paper_date(record)
    if paper_date is None:
        paper_date = _arxiv_month_date(record)
    if paper_date is None:
        return False
    current = (now or datetime.now(timezone.utc)).date()
    return paper_date >= current - timedelta(days=window_days)


def _has_arxiv_id(record: dict[str, Any]) -> bool:
    identifiers = record.get("identifiers") or {}
    values = (
        record.get("paper_id"),
        record.get("arxiv_id"),
        record.get("arxiv"),
        identifiers.get("paper_id"),
        identifiers.get("arxiv_id"),
        identifiers.get("arxiv"),
    )
    return any(arxiv_path_id(str(value or "")) for value in values)


def _paper_date(record: dict[str, Any]) -> date | None:
    for key in ("published", "preprint_date", "earliest_date", "created", "updated"):
        value = str(record.get(key) or "").strip()
        parsed = _parse_date(value)
        if parsed is not None:
            return parsed
    year = record.get("year")
    try:
        return date(int(year), 1, 1) if year else None
    except (TypeError, ValueError):
        return None


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    match = re.match(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", value)
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2) or 1)
    day = int(match.group(3) or 1)
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _arxiv_month_date(record: dict[str, Any]) -> date | None:
    paper_id = normalize_paper_id(record.get("paper_id") or paper_key(record))
    arxiv_id = arxiv_path_id(paper_id)
    match = re.match(r"^(\d{2})(\d{2})\.", arxiv_id)
    if not match:
        return None
    year = 2000 + int(match.group(1))
    month = int(match.group(2))
    try:
        return date(year, month, 1)
    except ValueError:
        return None
