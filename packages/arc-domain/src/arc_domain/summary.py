from __future__ import annotations

from typing import Any

from arc_llm import run_json

from .cache import DomainPaths, now_iso, read_json, update_status, write_json


DOMAIN_SUMMARY_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "arc.domain-summary-v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "domain_title",
        "brief_introduction",
        "foundation_paper",
        "methodology",
        "mainstream_directions",
        "open_questions",
        "reading_guide",
        "warnings",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": "arc.domain_summary.v1"},
        "domain_title": {"type": "string"},
        "brief_introduction": {"type": "string"},
        "foundation_paper": {"type": "object"},
        "methodology": {"type": "array", "items": {"type": "object"}},
        "mainstream_directions": {"type": "array", "items": {"type": "object"}},
        "open_questions": {"type": "array", "items": {"type": "object"}},
        "reading_guide": {"type": "array", "items": {"type": "object"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}


def summarize_domain(
    *,
    paths: DomainPaths,
    provider: str = "auto",
    model: str | None = None,
) -> dict[str, Any]:
    update_status(paths, stage="summary_started")
    graph = read_json(paths.domain_graph, {})
    evidence = read_json(paths.evidence_pack, {})
    selection = read_json(paths.foundation_selection, {})
    prompt = _summary_prompt(graph=graph, evidence=evidence, selection=selection)
    try:
        summary = run_json(prompt, schema=DOMAIN_SUMMARY_SCHEMA, provider=provider, model=model)
        summary["summary_method"] = "llm"
    except Exception as exc:
        summary = _fallback_summary(graph=graph, evidence=evidence, selection=selection, error=str(exc))
    summary["schema_version"] = "arc.domain_summary.v1"
    summary["domain_id"] = paths.domain_id
    summary["created_at"] = now_iso()
    write_json(paths.domain_summary, summary)
    update_status(paths, stage="summary_done", domain_summary_path=str(paths.domain_summary))
    return {"domain_id": paths.domain_id, "domain_summary_path": str(paths.domain_summary), "summary": summary}


def _summary_prompt(*, graph: dict[str, Any], evidence: dict[str, Any], selection: dict[str, Any]) -> str:
    compact_evidence = {
        "foundation_selection": selection,
        "graph": {
            "foundation_paper": graph.get("foundation_paper"),
            "nodes": [
                {
                    "paper_id": node.get("paper_id"),
                    "role": node.get("role"),
                    "title": node.get("title"),
                    "year": node.get("year"),
                    "citation_count": node.get("citation_count"),
                    "selection_reason": node.get("selection_reason"),
                }
                for node in graph.get("nodes", [])
            ],
            "edges": graph.get("edges", [])[:200],
        },
        "papers": [
            {
                "paper_id": item.get("paper_id"),
                "role": item.get("role"),
                "title": item.get("title"),
                "abstract": item.get("abstract"),
                "conclusion": (item.get("conclusion") or {}).get("text", ""),
                "warnings": item.get("warnings", []),
            }
            for item in evidence.get("papers", [])
        ],
        "warnings": evidence.get("warnings", []),
    }
    return "\n\n".join(
        [
            "Write a compact field briefing for an LLM physicist and a human researcher.",
            "Use the supplied titles, abstracts, graph roles, and conclusion/outlook/discussion text. Do not invent papers.",
            "The summary should explain the domain, the selected foundation, core methodology, mainstream directions with reference paper IDs, and open questions mentioned in conclusions.",
            "Keep the result concise enough to fit comfortably in a research-agent context.",
            f"Evidence pack:\n{compact_evidence}",
            "Return JSON only.",
        ]
    )


def _fallback_summary(
    *,
    graph: dict[str, Any],
    evidence: dict[str, Any],
    selection: dict[str, Any],
    error: str,
) -> dict[str, Any]:
    foundation = selection.get("selected_foundation") or {}
    papers = evidence.get("papers", [])
    domain_papers = [item for item in papers if item.get("role") == "domain_paper"]
    return {
        "schema_version": "arc.domain_summary.v1",
        "domain_title": foundation.get("title") or "Research domain",
        "brief_introduction": (
            "LLM summary generation was unavailable; this deterministic briefing lists the selected "
            "foundation and the cached domain paper set for follow-up reading."
        ),
        "foundation_paper": foundation,
        "methodology": [
            {
                "claim": "Read the selected foundation and high-scoring domain papers to identify methodology.",
                "papers": [foundation.get("paper_id")] if foundation.get("paper_id") else [],
            }
        ],
        "mainstream_directions": [
            {
                "direction": "High-scoring foundation citers",
                "papers": [item.get("paper_id") for item in domain_papers[:15]],
            }
        ],
        "open_questions": [],
        "reading_guide": [
            {
                "purpose": "Start with the selected foundation and then the highest-scoring citing papers.",
                "papers": [item.get("paper_id") for item in papers[:10]],
            }
        ],
        "warnings": [f"llm_summary_failed:{error}", *list(evidence.get("warnings") or [])],
        "summary_method": "deterministic_fallback",
        "graph_node_count": len(graph.get("nodes", [])),
    }
