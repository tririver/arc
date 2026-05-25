from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from arc_llm import run_json
from arc_paper.ids import normalize_paper_id

from . import paper
from .cache import DomainPaths, now_iso, update_status, write_json
from .text import deterministic_sample, normalize_authors, paper_key, token_overlap_score


@dataclass(frozen=True)
class FoundationHeuristics:
    """Configurable foundation-selection heuristics.

    min_citation_count is a prioritization threshold, not an exclusion rule.
    Lower-citation candidates remain eligible when they are the best same-scope
    foundation in the supplied evidence.
    """

    min_citation_count: int = 100


DEFAULT_FOUNDATION_HEURISTICS = FoundationHeuristics()
MIN_FOUNDATION_CITATION_COUNT = DEFAULT_FOUNDATION_HEURISTICS.min_citation_count


FOUNDATION_SELECTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "arc.domain-foundation-selection-v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "selected_foundation",
        "best_reference_paper",
        "parent_foundations",
        "rejected_candidates",
        "reasoning",
        "warnings",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": "arc.domain_foundation_selection.v1"},
        "selected_foundation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["paper_id", "title", "reason"],
            "properties": {
                "paper_id": {"type": "string"},
                "title": {"type": "string"},
                "reason": {"type": "string"},
            },
        },
        "best_reference_paper": {
            "type": "object",
            "additionalProperties": False,
            "required": ["paper_id", "title", "reason"],
            "properties": {
                "paper_id": {"type": "string"},
                "title": {"type": "string"},
                "reason": {"type": "string"},
            },
        },
        "parent_foundations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["paper_id", "title", "reason"],
                "properties": {
                    "paper_id": {"type": "string"},
                    "title": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "rejected_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["paper_id", "title", "reason"],
                "properties": {
                    "paper_id": {"type": "string"},
                    "title": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "reasoning": {"type": "string"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}


def identify_foundation(
    *,
    seed_paper: str,
    intent: str,
    paths: DomainPaths,
    provider: str = "auto",
    model: str | None = None,
    refresh: bool = False,
    workers: int = 8,
    newest_citer_count: int = 50,
    witness_size: int = 60,
    min_citation_count: int = MIN_FOUNDATION_CITATION_COUNT,
) -> dict[str, Any]:
    seed_id = normalize_paper_id(seed_paper)
    update_status(paths, stage="foundation_started", seed_paper=seed_id, intent=intent)

    seed_metadata = paper.metadata(seed_id, refresh=refresh)
    newest_citers = paper.citers(seed_id, refresh=refresh, limit=newest_citer_count, sort="mostrecent")
    seed_references = paper.references(seed_id, refresh=refresh, enrich=True)
    sampled_references = deterministic_sample(
        [item for item in seed_references if paper_key(item)],
        count=max(0, witness_size - len(newest_citers)),
        seed=f"{seed_id}\n{intent}",
    )

    witness_papers = [
        {"source": "newest_citer", "paper": item}
        for item in newest_citers[:newest_citer_count]
        if paper_key(item)
    ]
    witness_papers.extend({"source": "seed_reference_sample", "paper": item} for item in sampled_references)

    citer_ids = [paper_key(item["paper"]) for item in witness_papers if item["source"] == "newest_citer"]
    refs_by_citer = paper.fetch_many(
        citer_ids,
        lambda paper_id: paper.references(paper_id, refresh=refresh, enrich=False),
        workers=workers,
    )
    candidates = _candidate_records(
        seed_metadata=seed_metadata,
        seed_id=seed_id,
        seed_references=seed_references,
        refs_by_citer=refs_by_citer,
        intent=intent,
        refresh=refresh,
        workers=workers,
        min_citation_count=min_citation_count,
    )
    pool = {
        "schema_version": "arc.domain_foundation_pool.v1",
        "seed_paper": seed_id,
        "intent": intent,
        "seed_metadata": seed_metadata,
        "newest_citers": newest_citers,
        "seed_references": seed_references,
        "sampled_references": sampled_references,
        "witness_papers": witness_papers,
        "reference_lists_fetched": len(refs_by_citer),
        "created_at": now_iso(),
    }
    write_json(paths.foundation_pool, pool)
    write_json(paths.foundation_candidates, candidates)

    selection = _llm_select_foundation(
        seed_metadata=seed_metadata,
        candidates=candidates,
        intent=intent,
        provider=provider,
        model=model,
        min_citation_count=min_citation_count,
    )
    selection["seed_paper"] = seed_id
    selection["intent"] = intent
    selection["candidate_count"] = len(candidates)
    selection["created_at"] = now_iso()
    write_json(paths.foundation_selection, selection)
    update_status(
        paths,
        stage="foundation_done",
        selected_foundation=(selection.get("selected_foundation") or {}).get("paper_id"),
    )
    return {
        "domain_id": paths.domain_id,
        "foundation_pool_path": str(paths.foundation_pool),
        "foundation_candidates_path": str(paths.foundation_candidates),
        "foundation_selection_path": str(paths.foundation_selection),
        "selection": selection,
        "candidates": candidates,
    }


def _candidate_records(
    *,
    seed_metadata: dict[str, Any],
    seed_id: str,
    seed_references: list[dict[str, Any]],
    refs_by_citer: dict[str, Any],
    intent: str,
    refresh: bool,
    workers: int,
    min_citation_count: int = MIN_FOUNDATION_CITATION_COUNT,
) -> list[dict[str, Any]]:
    overlap = Counter()
    support: dict[str, list[str]] = defaultdict(list)
    embedded: dict[str, dict[str, Any]] = {}
    for citer_id, refs in refs_by_citer.items():
        if not isinstance(refs, list):
            continue
        seen_in_paper = set()
        for ref in refs:
            ref_id = paper_key(ref)
            if not ref_id:
                continue
            ref_id = normalize_paper_id(ref_id)
            if ref_id in seen_in_paper:
                continue
            seen_in_paper.add(ref_id)
            overlap[ref_id] += 1
            support[ref_id].append(citer_id)
            embedded.setdefault(ref_id, ref)

    seed_id = normalize_paper_id(seed_id)
    overlap.setdefault(seed_id, 0)
    embedded.setdefault(seed_id, seed_metadata)
    for ref in seed_references:
        ref_id = normalize_paper_id(paper_key(ref))
        if ref_id:
            embedded.setdefault(ref_id, ref)

    top_ids = [
        item_id
        for item_id, _count in sorted(
            overlap.items(),
            key=lambda item: (item[1], _embedded_citation_count(embedded.get(item[0], {})), item[0]),
            reverse=True,
        )[:20]
    ]
    metadata_by_id = paper.fetch_many(
        top_ids,
        lambda paper_id: paper.metadata(paper_id, refresh=refresh),
        workers=workers,
    )

    records = []
    for rank, candidate_id in enumerate(top_ids, start=1):
        meta = metadata_by_id.get(candidate_id)
        if not isinstance(meta, dict) or meta.get("error"):
            meta = embedded.get(candidate_id, {})
        title = str(meta.get("title") or embedded.get(candidate_id, {}).get("title") or "")
        abstract = str(meta.get("abstract") or "")
        citation_count = int(meta.get("citation_count") or meta.get("cited_by_count") or 0)
        year = meta.get("year") or embedded.get(candidate_id, {}).get("year")
        record = {
            "paper_id": normalize_paper_id(meta.get("paper_id") or candidate_id),
            "rank": rank,
            "title": title,
            "abstract": abstract,
            "authors": list(meta.get("authors") or []),
            "authors_short": normalize_authors(meta.get("authors") or []),
            "year": year,
            "citation_count": citation_count,
            "witness_citation_overlap": int(overlap[candidate_id]),
            "supported_by": support.get(candidate_id, [])[:50],
            "intent_overlap": round(token_overlap_score(f"{title} {abstract}", intent), 4),
            "identifiers": meta.get("identifiers") or {},
            "warnings": [],
        }
        if candidate_id == seed_id:
            record["source_role"] = "seed"
        elif candidate_id in {normalize_paper_id(paper_key(item)) for item in seed_references}:
            record["source_role"] = "seed_reference"
        else:
            record["source_role"] = "common_reference"
        if citation_count < min_citation_count:
            record["warnings"].append("low_citation_foundation_priority")
        if citation_count >= 1000:
            record["warnings"].append("high_citation_parent_domain_risk")
        records.append(record)
    return records[:10]


def _llm_select_foundation(
    *,
    seed_metadata: dict[str, Any],
    candidates: list[dict[str, Any]],
    intent: str,
    provider: str,
    model: str | None,
    min_citation_count: int = MIN_FOUNDATION_CITATION_COUNT,
) -> dict[str, Any]:
    prompt = _foundation_prompt(
        seed_metadata=seed_metadata,
        candidates=candidates,
        intent=intent,
        min_citation_count=min_citation_count,
    )
    try:
        selection = run_json(prompt, schema=FOUNDATION_SELECTION_SCHEMA, provider=provider, model=model)
        return _repair_selection(selection, candidates, method="llm")
    except Exception as exc:
        selection = _deterministic_selection(
            candidates,
            intent=intent,
            min_citation_count=min_citation_count,
        )
        selection["warnings"].append(f"llm_selection_failed:{exc}")
        return selection


def _foundation_prompt(
    *,
    seed_metadata: dict[str, Any],
    candidates: list[dict[str, Any]],
    intent: str,
    min_citation_count: int = MIN_FOUNDATION_CITATION_COUNT,
) -> str:
    return "\n\n".join(
        [
            "You are selecting the foundation paper for a theoretical-physics research domain.",
            "Choose two papers from the supplied candidates. They may be the same paper.",
            "First, choose selected_foundation: the same-scope foundation paper that best defines the research field represented by the seed paper and its citers.",
            "Second, choose best_reference_paper, the best reference: the easiest useful reference for an agent to read before proposing or calculating in the user's intended methodology. Prefer a candidate with a modern method, clear exposition, or comprehensive review-style coverage when that better serves the user's intent.",
            "If an older high-citation candidate is broader than the user's intent, keep it as a parent foundation rather than the selected foundation.",
            "A parent foundation must be earlier than, or at the latest from the same year as, the selected foundation. A later paper can be a child, extension, or successful descendant, but never a parent foundation.",
            f"Citation support heuristic: candidates with fewer than {min_citation_count} citations should normally have low priority as the selected foundation, because there is usually not enough literature built on top of them to define a research field. Select such a candidate only if the supplied candidates contain no better-supported same-scope foundation.",
            "Use only the supplied candidates. Prefer a candidate that defines the domain represented by the seed paper and its newest citers.",
            f"User intent:\n{intent or '(none)'}",
            f"Seed paper:\n{seed_metadata}",
            f"Candidate papers:\n{candidates}",
            "Return JSON only.",
        ]
    )


def _deterministic_selection(
    candidates: list[dict[str, Any]],
    *,
    intent: str,
    min_citation_count: int = MIN_FOUNDATION_CITATION_COUNT,
) -> dict[str, Any]:
    if not candidates:
        selected = {"paper_id": "", "title": "", "reason": "no candidates were available"}
    else:
        ranked = sorted(
            candidates,
            key=lambda item: (
                int(item.get("citation_count") or 0) >= min_citation_count,
                item.get("witness_citation_overlap", 0),
                item.get("intent_overlap", 0),
                item.get("citation_count", 0),
            ),
            reverse=True,
        )
        best = ranked[0]
        selected = {
            "paper_id": best.get("paper_id", ""),
            "title": best.get("title", ""),
            "year": best.get("year"),
            "reason": "highest deterministic combination of witness citation overlap, intent overlap, and citation count",
        }
    best_reference = _deterministic_best_reference(candidates, selected)
    parent_foundations = [
        {
            "paper_id": item.get("paper_id", ""),
            "title": item.get("title", ""),
            "reason": "high-citation candidate kept as possible broader parent foundation",
        }
        for item in candidates
        if item.get("paper_id") != selected.get("paper_id") and "high_citation_parent_domain_risk" in item.get("warnings", [])
    ]
    return {
        "schema_version": "arc.domain_foundation_selection.v1",
        "selected_foundation": selected,
        "best_reference_paper": best_reference,
        "parent_foundations": parent_foundations[:5],
        "rejected_candidates": [],
        "reasoning": f"Deterministic fallback selection. User intent: {intent or '(none)'}.",
        "warnings": [],
        "selection_method": "deterministic_fallback",
    }


def _repair_selection(selection: dict[str, Any], candidates: list[dict[str, Any]], *, method: str) -> dict[str, Any]:
    candidate_by_id = {item.get("paper_id"): item for item in candidates}
    selected = dict(selection.get("selected_foundation") or {})
    selected_id = normalize_paper_id(str(selected.get("paper_id") or ""))
    if selected_id not in candidate_by_id and candidates:
        selected_id = candidates[0]["paper_id"]
        selected = {
            "paper_id": selected_id,
            "title": candidates[0].get("title", ""),
            "reason": "LLM selected an unknown id; repaired to the top candidate",
        }
    else:
        selected["paper_id"] = selected_id
        if selected_id in candidate_by_id:
            selected.setdefault("title", candidate_by_id[selected_id].get("title", ""))
    selection["selected_foundation"] = selected
    selection["best_reference_paper"] = _repair_best_reference(
        selection.get("best_reference_paper"),
        selected=selected,
        candidate_by_id=candidate_by_id,
    )
    selection["parent_foundations"], moved = _valid_parent_foundations(
        selection.get("parent_foundations") or [],
        selected_id=selected_id,
        candidate_by_id=candidate_by_id,
    )
    selection["rejected_candidates"] = [*(selection.get("rejected_candidates") or []), *moved]
    selection.setdefault("warnings", [])
    selection["selection_method"] = method
    selection["schema_version"] = "arc.domain_foundation_selection.v1"
    return selection


def _deterministic_best_reference(
    candidates: list[dict[str, Any]],
    selected: dict[str, Any],
) -> dict[str, Any]:
    if not candidates:
        return {
            "paper_id": selected.get("paper_id", ""),
            "title": selected.get("title", ""),
            "reason": "no separate candidates were available",
        }
    ranked = sorted(
        candidates,
        key=lambda item: (
            item.get("intent_overlap", 0),
            _candidate_year(item) or 0,
            item.get("citation_count", 0),
            item.get("witness_citation_overlap", 0),
        ),
        reverse=True,
    )
    best = ranked[0]
    return {
        "paper_id": best.get("paper_id", ""),
        "title": best.get("title", ""),
        "reason": (
            "highest deterministic combination of intent overlap, recency, "
            "citation count, and witness support for a readable methodology reference"
        ),
    }


def _repair_best_reference(
    best_reference: Any,
    *,
    selected: dict[str, Any],
    candidate_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate = dict(best_reference or selected)
    candidate_id = normalize_paper_id(str(candidate.get("paper_id") or ""))
    if candidate_id not in candidate_by_id:
        selected_id = normalize_paper_id(str(selected.get("paper_id") or ""))
        selected_candidate = candidate_by_id.get(selected_id, {})
        return {
            "paper_id": selected_id,
            "title": selected.get("title") or selected_candidate.get("title", ""),
            "reason": "Best-reference LLM selected an unknown id; repaired to the selected foundation",
        }
    source = candidate_by_id[candidate_id]
    return {
        "paper_id": candidate_id,
        "title": candidate.get("title") or source.get("title", ""),
        "reason": candidate.get("reason") or "selected as the best methodology reference",
    }


def _valid_parent_foundations(
    parent_foundations: list[dict[str, Any]],
    *,
    selected_id: str,
    candidate_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_year = _candidate_year(candidate_by_id.get(selected_id, {}))
    valid: list[dict[str, Any]] = []
    moved: list[dict[str, Any]] = []
    seen_valid: set[str] = set()
    seen_moved: set[str] = set()
    for item in parent_foundations:
        parent = dict(item)
        parent_id = normalize_paper_id(str(parent.get("paper_id") or ""))
        parent["paper_id"] = parent_id
        candidate = candidate_by_id.get(parent_id, {})
        parent_year = _candidate_year(candidate)
        if (
            selected_year is not None
            and parent_year is not None
            and parent_year > selected_year
        ):
            if parent_id not in seen_moved:
                moved.append(
                    {
                        "paper_id": parent_id,
                        "title": parent.get("title") or candidate.get("title", ""),
                        "reason": (
                            f"Cannot be a parent foundation because it is from {parent_year}, "
                            f"later than the selected foundation year {selected_year}."
                        ),
                    }
                )
                seen_moved.add(parent_id)
            continue
        if parent_id and parent_id not in seen_valid:
            valid.append(parent)
            seen_valid.add(parent_id)
    return valid, moved


def _candidate_year(candidate: dict[str, Any]) -> int | None:
    try:
        return int(candidate.get("year"))
    except (TypeError, ValueError):
        return None


def _embedded_citation_count(item: dict[str, Any]) -> int:
    try:
        return int(item.get("citation_count") or item.get("cited_by_count") or 0)
    except (TypeError, ValueError):
        return 0
