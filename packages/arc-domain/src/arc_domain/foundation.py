from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from jsonschema.exceptions import SchemaError as JsonSchemaError

from arc_llm import run_json
from arc_llm.call_record import ARC_LLM_CALL_RECORD_FIELD, ARC_LLM_CALL_RECORD_SCHEMA
from arc_paper.ids import extract_paper_ids, normalize_paper_id

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
MAX_AUDIT_SEARCH_QUERIES = 3
LLM_CANDIDATE_SOURCE_ROLE = "llm_added_foundation_candidate"
LLM_SELECTION_MARK_FIELDS = (
    "source_role",
    "llm_added",
    "llm_recommended",
    "llm_addition_reason",
    "llm_reference_query",
    "llm_verified_evidence_urls",
    "llm_reference_inference",
)


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
        ARC_LLM_CALL_RECORD_FIELD: ARC_LLM_CALL_RECORD_SCHEMA,
    },
}


FOUNDATION_CANDIDATE_AUDIT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "arc.domain-foundation-candidate-audit-v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "candidate_set_sufficient",
        "confidence",
        "search_queries",
        "citation_directions",
        "reasoning",
        "warnings",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": "arc.domain_foundation_candidate_audit.v1"},
        "candidate_set_sufficient": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["complete", "high", "medium", "low"]},
        "search_queries": {
            "type": "array",
            "maxItems": MAX_AUDIT_SEARCH_QUERIES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query", "reason", "confidence"],
                "properties": {
                    "query": {"type": "string"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["complete", "high", "medium", "low"]},
                },
            },
        },
        "citation_directions": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "reasoning": {"type": "string"},
        "warnings": {"type": "array", "items": {"type": "string"}},
        ARC_LLM_CALL_RECORD_FIELD: ARC_LLM_CALL_RECORD_SCHEMA,
    },
}


def _schema_error(payload: dict[str, Any], schema: dict[str, Any]) -> str | None:
    try:
        validate_json_schema(instance=payload, schema=schema)
        return None
    except (JsonSchemaValidationError, JsonSchemaError) as exc:
        return str(exc)


def _domain_llm_recovered(payload: dict[str, Any]) -> bool:
    record = payload.get(ARC_LLM_CALL_RECORD_FIELD) if isinstance(payload, dict) else None
    if not isinstance(record, dict):
        return False
    structured = record.get("structured_output")
    return isinstance(structured, dict) and structured.get("mode") == "recovered"


def identify_foundation(
    *,
    seed_paper: str,
    intent: str,
    paths: DomainPaths,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
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
    initial_candidate_count = len(candidates)
    candidate_audit = _llm_audit_candidates(
        seed_metadata=seed_metadata,
        candidates=candidates,
        intent=intent,
        provider=provider,
        model=model,
        model_tier=model_tier,
        min_citation_count=min_citation_count,
    )
    candidates, expansion_report = _expand_candidates_from_audit(
        candidates=candidates,
        audit=candidate_audit,
        intent=intent,
        provider=provider,
        model=model,
        model_tier=model_tier,
        refresh=refresh,
        workers=workers,
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
        "candidate_audit": candidate_audit,
        "candidate_expansion": expansion_report,
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
        model_tier=model_tier,
        min_citation_count=min_citation_count,
    )
    selection["seed_paper"] = seed_id
    selection["intent"] = intent
    selection["initial_candidate_count"] = initial_candidate_count
    selection["candidate_count"] = len(candidates)
    selection["candidate_audit"] = candidate_audit
    selection["candidate_expansion"] = expansion_report
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
        if candidate_id == seed_id:
            source_role = "seed"
        elif candidate_id in {normalize_paper_id(paper_key(item)) for item in seed_references}:
            source_role = "seed_reference"
        else:
            source_role = "common_reference"
        record = _metadata_candidate_record(
            candidate_id=candidate_id,
            meta=meta,
            fallback=embedded.get(candidate_id, {}),
            rank=rank,
            intent=intent,
            source_role=source_role,
            witness_citation_overlap=int(overlap[candidate_id]),
            supported_by=support.get(candidate_id, []),
            min_citation_count=min_citation_count,
        )
        records.append(record)
    return records[:10]


def _llm_audit_candidates(
    *,
    seed_metadata: dict[str, Any],
    candidates: list[dict[str, Any]],
    intent: str,
    provider: str,
    model: str | None,
    model_tier: str | None = None,
    min_citation_count: int = MIN_FOUNDATION_CITATION_COUNT,
) -> dict[str, Any]:
    prompt = _candidate_audit_prompt(
        seed_metadata=seed_metadata,
        candidates=candidates,
        intent=intent,
        min_citation_count=min_citation_count,
    )
    try:
        audit = run_json(
            prompt,
            schema=FOUNDATION_CANDIDATE_AUDIT_SCHEMA,
            provider=provider,
            model=model,
            model_tier=model_tier,
            validate_schema=False,
            output_recovery="warn",
        )
        method = "llm_relaxed" if _domain_llm_recovered(audit) or _schema_error(audit, FOUNDATION_CANDIDATE_AUDIT_SCHEMA) else "llm"
        return _repair_candidate_audit(audit, method=method)
    except Exception as exc:
        audit = _default_candidate_audit()
        audit["warnings"].append(f"llm_candidate_audit_failed:{exc}")
        audit["audit_method"] = "llm_failed"
        return audit


def _candidate_audit_prompt(
    *,
    seed_metadata: dict[str, Any],
    candidates: list[dict[str, Any]],
    intent: str,
    min_citation_count: int = MIN_FOUNDATION_CITATION_COUNT,
) -> str:
    return "\n\n".join(
        [
            "You audit a theoretical-physics foundation-paper candidate set before selection.",
            "Decide whether the supplied candidates are sufficient for choosing the same-scope foundation paper.",
            "You may propose search_queries only if you are completely sure a likely foundational or canonical same-scope paper is missing from the candidates.",
            "If you have any doubt, leave search_queries empty and let the later selector use the heavily data-suggested candidates.",
            "Do not invent papers. Do not return paper IDs from memory. Return search terms that a separate web-search verifier can check.",
            "Citation directions are optional hints such as references/citers to inspect; they are not selected papers.",
            f"Low-citation heuristic: fewer than {min_citation_count} citations normally means low priority as selected foundation unless no better-supported same-scope foundation is available.",
            f"User intent:\n{intent or '(none)'}",
            f"Seed paper:\n{seed_metadata}",
            f"Candidate papers:\n{candidates}",
            "Return JSON only.",
        ]
    )


def _default_candidate_audit() -> dict[str, Any]:
    return {
        "schema_version": "arc.domain_foundation_candidate_audit.v1",
        "candidate_set_sufficient": True,
        "confidence": "low",
        "search_queries": [],
        "citation_directions": [],
        "reasoning": "No reliable audit expansion; use deterministic candidates.",
        "warnings": [],
        "audit_method": "deterministic_no_expansion",
    }


def _repair_candidate_audit(audit: dict[str, Any], *, method: str) -> dict[str, Any]:
    schema_error = _schema_error(audit if isinstance(audit, dict) else {}, FOUNDATION_CANDIDATE_AUDIT_SCHEMA)
    repaired = dict(audit or {})
    repaired["schema_version"] = "arc.domain_foundation_candidate_audit.v1"
    repaired["candidate_set_sufficient"] = _relaxed_bool(repaired.get("candidate_set_sufficient"), default=True)
    repaired["confidence"] = _confidence(repaired.get("confidence"))
    search_queries, skipped = _complete_audit_search_queries(repaired.get("search_queries"))
    repaired["search_queries"] = search_queries
    repaired["citation_directions"] = [
        str(item).strip()
        for item in repaired.get("citation_directions", [])
        if str(item).strip()
    ][:5]
    repaired["reasoning"] = str(repaired.get("reasoning") or "")
    repaired["warnings"] = [str(item) for item in repaired.get("warnings", [])]
    repaired["warnings"].extend(skipped)
    if schema_error:
        repaired["warnings"].append("candidate_audit_schema_relaxed:" + schema_error[:500])
    repaired["audit_method"] = method
    return repaired


def _relaxed_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off", ""}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    return default


def _complete_audit_search_queries(raw_queries: Any) -> tuple[list[dict[str, str]], list[str]]:
    queries: list[dict[str, str]] = []
    warnings: list[str] = []
    for query in _audit_search_queries(raw_queries):
        if query["confidence"] == "complete":
            queries.append(query)
        else:
            warnings.append(
                "Dropped non-complete audit search query: "
                f"{query['query']} ({query['confidence']})."
            )
    return queries, warnings


def _audit_search_queries(raw_queries: Any) -> list[dict[str, str]]:
    if not isinstance(raw_queries, list):
        return []
    queries: list[dict[str, str]] = []
    for item in raw_queries[:MAX_AUDIT_SEARCH_QUERIES]:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        queries.append(
            {
                "query": query,
                "reason": str(item.get("reason") or "").strip(),
                "confidence": _confidence(item.get("confidence")),
            }
        )
    return queries


def _confidence(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"complete", "high", "medium", "low"}:
        return normalized
    return "low"


def _expand_candidates_from_audit(
    *,
    candidates: list[dict[str, Any]],
    audit: dict[str, Any],
    intent: str,
    provider: str,
    model: str | None,
    model_tier: str | None = None,
    refresh: bool,
    workers: int,
    min_citation_count: int = MIN_FOUNDATION_CITATION_COUNT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expanded = [dict(item) for item in candidates]
    candidate_by_id = {
        normalize_paper_id(str(item.get("paper_id") or "")): item
        for item in expanded
        if item.get("paper_id")
    }
    report: dict[str, Any] = {
        "schema_version": "arc.domain_foundation_candidate_expansion.v1",
        "initial_candidate_count": len(candidates),
        "expanded_candidate_count": len(expanded),
        "added_candidate_count": 0,
        "added_papers": [],
        "searches": [],
        "warnings": [],
    }
    queries = _audit_search_queries(audit.get("search_queries"))
    if not queries:
        return expanded, report
    if audit.get("candidate_set_sufficient") is not False or _confidence(audit.get("confidence")) != "complete":
        for query in queries:
            report["searches"].append(
                {
                    "query": query["query"],
                    "reason": query.get("reason", ""),
                    "status": "skipped_uncertain_audit",
                }
            )
        return expanded, report

    for query in queries:
        if extract_paper_ids(query["query"]):
            report["searches"].append(
                {
                    "query": query["query"],
                    "reason": query.get("reason", ""),
                    "status": "skipped_explicit_id_query",
                    "warnings": ["audit search query contained a paper identifier; web-search verifier requires search terms"],
                }
            )
            continue
        if query["confidence"] != "complete":
            report["searches"].append(
                {
                    "query": query["query"],
                    "reason": query.get("reason", ""),
                    "status": "skipped_uncertain_query",
                }
            )
            continue
        search = _run_reference_inference_query(
            query,
            intent=intent,
            provider=provider,
            model=model,
            model_tier=model_tier,
            refresh=refresh,
        )
        paper_ids = search.get("paper_ids", [])
        metadata_by_id = paper.fetch_many(
            paper_ids,
            lambda paper_id: paper.metadata(paper_id, refresh=refresh),
            workers=workers,
        )
        verified_by_id = _verified_references_by_id(search.get("result", {}))
        added_ids: list[str] = []
        already_present: list[str] = []
        metadata_failed: list[str] = []
        for paper_id in paper_ids:
            if paper_id in candidate_by_id:
                _mark_existing_llm_recommended(candidate_by_id[paper_id], query=query, verified=verified_by_id.get(paper_id))
                already_present.append(paper_id)
                continue
            meta = metadata_by_id.get(paper_id)
            if not isinstance(meta, dict) or meta.get("error"):
                metadata_failed.append(paper_id)
                continue
            record = _llm_added_candidate_record(
                paper_id=paper_id,
                metadata=meta,
                rank=len(expanded) + 1,
                intent=intent,
                query=query,
                verified=verified_by_id.get(paper_id),
                result=search.get("result", {}),
                min_citation_count=min_citation_count,
            )
            expanded.append(record)
            candidate_by_id[record["paper_id"]] = record
            added_ids.append(record["paper_id"])
        status = "added" if added_ids else ("already_candidate" if already_present else search["status"])
        search_report = {
            "query": query["query"],
            "reason": query.get("reason", ""),
            "status": status,
            "paper_ids": paper_ids,
            "added_papers": added_ids,
            "already_present": already_present,
            "metadata_failed": metadata_failed,
            "warnings": search.get("warnings", []),
        }
        if search.get("error"):
            search_report["error"] = search["error"]
        report["searches"].append(search_report)

    report["expanded_candidate_count"] = len(expanded)
    report["added_papers"] = [item["paper_id"] for item in expanded if item.get("llm_added")]
    report["added_candidate_count"] = len(report["added_papers"])
    return expanded, report


def _run_reference_inference_query(
    query: dict[str, str],
    *,
    intent: str,
    provider: str,
    model: str | None,
    model_tier: str | None = None,
    refresh: bool,
) -> dict[str, Any]:
    inference_kwargs: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "refresh": refresh,
    }
    if model_tier is not None:
        inference_kwargs["model_tier"] = model_tier
    try:
        result = paper.infer_main_references(
            _reference_inference_request(query, intent=intent),
            **inference_kwargs,
        )
    except Exception as exc:
        return {"status": "reference_inference_failed", "paper_ids": [], "warnings": [], "error": str(exc), "result": {}}
    if not result.get("ok"):
        error = result.get("error") or {}
        return {
            "status": "reference_inference_failed",
            "paper_ids": [],
            "warnings": [],
            "error": error.get("message") or error.get("code") or "reference inference failed",
            "result": result,
        }
    meta = result.get("meta") or {}
    verified_by_id = _verified_references_by_id(result)
    if meta.get("llm_used") is not True or not verified_by_id:
        return {
            "status": "reference_inference_unverified",
            "paper_ids": [],
            "warnings": list(meta.get("warnings", [])),
            "result": result,
        }
    paper_ids = [
        paper_id
        for item in result.get("data", [])
        if (paper_id := normalize_paper_id(str(item))) in verified_by_id
    ]
    if not paper_ids:
        return {
            "status": "reference_inference_unverified",
            "paper_ids": [],
            "warnings": list(meta.get("warnings", [])),
            "result": result,
        }
    return {
        "status": "verified" if paper_ids else "no_verified_papers",
        "paper_ids": list(dict.fromkeys(paper_ids)),
        "warnings": list(meta.get("warnings", [])),
        "result": result,
        "intent": intent,
    }


def _reference_inference_request(query: dict[str, str], *, intent: str) -> str:
    parts = [
        "Find and verify the single strongest missing foundational or canonical same-scope paper.",
        f"Search hint: {query['query']}",
    ]
    if (reason := query.get("reason")) and not extract_paper_ids(reason):
        parts.append(f"Reason this may be missing: {reason}")
    if intent and not extract_paper_ids(intent):
        parts.append(f"User intent: {intent}")
    return "\n".join(parts)


def _verified_references_by_id(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    verified = (result.get("meta") or {}).get("verified_references", [])
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(verified, list):
        return out
    for item in verified:
        if not isinstance(item, dict):
            continue
        paper_id = normalize_paper_id(str(item.get("paper_id") or item.get("input_paper_id") or ""))
        if paper_id:
            out[paper_id] = item
    return out


def _mark_existing_llm_recommended(
    record: dict[str, Any],
    *,
    query: dict[str, str],
    verified: dict[str, Any] | None,
) -> None:
    record["llm_recommended"] = True
    record.setdefault("llm_reference_query", query["query"])
    record.setdefault("llm_addition_reason", query.get("reason", ""))
    if verified:
        urls = [str(item) for item in verified.get("evidence_urls", []) if str(item).strip()]
        if urls:
            record.setdefault("llm_verified_evidence_urls", urls)


def _llm_added_candidate_record(
    *,
    paper_id: str,
    metadata: dict[str, Any],
    rank: int,
    intent: str,
    query: dict[str, str],
    verified: dict[str, Any] | None,
    result: dict[str, Any],
    min_citation_count: int,
) -> dict[str, Any]:
    record = _metadata_candidate_record(
        candidate_id=paper_id,
        meta=metadata,
        fallback={},
        rank=rank,
        intent=intent,
        source_role=LLM_CANDIDATE_SOURCE_ROLE,
        witness_citation_overlap=0,
        supported_by=[],
        min_citation_count=min_citation_count,
    )
    record["llm_added"] = True
    record["llm_addition_reason"] = query.get("reason", "")
    record["llm_reference_query"] = query["query"]
    if verified:
        urls = [str(item) for item in verified.get("evidence_urls", []) if str(item).strip()]
        if urls:
            record["llm_verified_evidence_urls"] = urls
        if reasoning := str(verified.get("reasoning") or "").strip():
            record["llm_reference_reasoning"] = reasoning
    record["llm_reference_inference"] = _reference_inference_summary(result)
    return record


def _metadata_candidate_record(
    *,
    candidate_id: str,
    meta: dict[str, Any],
    fallback: dict[str, Any],
    rank: int,
    intent: str,
    source_role: str,
    witness_citation_overlap: int,
    supported_by: list[str],
    min_citation_count: int,
) -> dict[str, Any]:
    title = str(meta.get("title") or fallback.get("title") or "")
    abstract = str(meta.get("abstract") or "")
    citation_count = int(meta.get("citation_count") or meta.get("cited_by_count") or 0)
    record = {
        "paper_id": normalize_paper_id(meta.get("paper_id") or candidate_id),
        "rank": rank,
        "title": title,
        "abstract": abstract,
        "authors": list(meta.get("authors") or []),
        "authors_short": normalize_authors(meta.get("authors") or []),
        "year": meta.get("year") or fallback.get("year"),
        "citation_count": citation_count,
        "witness_citation_overlap": witness_citation_overlap,
        "supported_by": supported_by[:50],
        "intent_overlap": round(token_overlap_score(f"{title} {abstract}", intent), 4),
        "identifiers": meta.get("identifiers") or {},
        "warnings": [],
        "source_role": source_role,
    }
    if citation_count < min_citation_count:
        record["warnings"].append("low_citation_foundation_priority")
    if citation_count >= 1000:
        record["warnings"].append("high_citation_parent_domain_risk")
    return record


def _reference_inference_summary(result: dict[str, Any]) -> dict[str, Any]:
    meta = result.get("meta") or {}
    return {
        "provider": meta.get("provider"),
        "model": meta.get("model"),
        "focus_scope": meta.get("focus_scope"),
        "warnings": meta.get("warnings", []),
        "verified_references": meta.get("verified_references", []),
        "rejected_candidates": meta.get("rejected_candidates", []),
    }


def _llm_select_foundation(
    *,
    seed_metadata: dict[str, Any],
    candidates: list[dict[str, Any]],
    intent: str,
    provider: str,
    model: str | None,
    model_tier: str | None = None,
    min_citation_count: int = MIN_FOUNDATION_CITATION_COUNT,
) -> dict[str, Any]:
    prompt = _foundation_prompt(
        seed_metadata=seed_metadata,
        candidates=candidates,
        intent=intent,
        min_citation_count=min_citation_count,
    )
    try:
        selection = run_json(
            prompt,
            schema=FOUNDATION_SELECTION_SCHEMA,
            provider=provider,
            model=model,
            model_tier=model_tier,
            validate_schema=False,
            output_recovery="warn",
        )
        method = "llm_relaxed" if _domain_llm_recovered(selection) or _schema_error(selection, FOUNDATION_SELECTION_SCHEMA) else "llm"
        return _repair_selection(
            selection,
            candidates,
            method=method,
            intent=intent,
            min_citation_count=min_citation_count,
        )
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
            "Candidates marked llm_added were added only after a separate web-search verifier returned INSPIRE-verified metadata; they may be selected when the evidence is stronger than the original candidate set.",
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
        _copy_candidate_marks(selected, best)
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


def _repair_selection(
    selection: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    method: str,
    intent: str = "",
    min_citation_count: int = MIN_FOUNDATION_CITATION_COUNT,
) -> dict[str, Any]:
    if not isinstance(selection, dict):
        selection = {}
    selection.setdefault("warnings", [])
    if "selected_foundation" not in selection and isinstance(selection.get("foundation_paper"), dict):
        selection["selected_foundation"] = selection["foundation_paper"]
    candidate_by_id = {
        normalize_paper_id(str(item.get("paper_id") or "")): item
        for item in candidates
        if item.get("paper_id")
    }
    selected = dict(selection.get("selected_foundation") or {})
    selected_id = normalize_paper_id(str(selected.get("paper_id") or ""))
    if selected_id not in candidate_by_id and candidates:
        unknown_id = selected_id or str(selected.get("paper_id") or "")
        fallback = _deterministic_selection(
            candidates,
            intent=intent,
            min_citation_count=min_citation_count,
        )["selected_foundation"]
        selected_id = normalize_paper_id(str(fallback.get("paper_id") or ""))
        selected = dict(fallback)
        selected["reason"] = "LLM selected an unknown id; repaired via deterministic fallback ranking"
        selection.setdefault("warnings", [])
        selection["warnings"].append(f"llm_selected_unknown_id:{unknown_id}")
    else:
        selected["paper_id"] = selected_id
        if selected_id in candidate_by_id:
            selected.setdefault("title", candidate_by_id[selected_id].get("title", ""))
    if selected_id in candidate_by_id:
        _copy_candidate_marks(selected, candidate_by_id[selected_id])
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
    selected = {
        "paper_id": best.get("paper_id", ""),
        "title": best.get("title", ""),
        "reason": (
            "highest deterministic combination of intent overlap, recency, "
            "citation count, and witness support for a readable methodology reference"
        ),
    }
    _copy_candidate_marks(selected, best)
    return selected


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
    repaired = {
        "paper_id": candidate_id,
        "title": candidate.get("title") or source.get("title", ""),
        "reason": candidate.get("reason") or "selected as the best methodology reference",
    }
    _copy_candidate_marks(repaired, source)
    return repaired


def _copy_candidate_marks(target: dict[str, Any], source: dict[str, Any]) -> None:
    for field in LLM_SELECTION_MARK_FIELDS:
        if field in source:
            target[field] = source[field]


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
