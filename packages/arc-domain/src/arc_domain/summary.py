from __future__ import annotations

import json
from collections import Counter
from typing import Any

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from jsonschema.exceptions import SchemaError as JsonSchemaError

from arc_llm import run_json
from arc_llm.call_record import ARC_LLM_CALL_RECORD_FIELD, ARC_LLM_CALL_RECORD_SCHEMA, strip_arc_llm_call_records

from .cache import DomainPaths, now_iso, read_json, update_status, write_json, write_text


SUMMARY_ABSTRACT_CHAR_LIMIT = 1600
SUMMARY_CONCLUSION_CHAR_LIMIT = 1600
SUMMARY_WARNING_CHAR_LIMIT = 160
SUMMARY_REASON_CHAR_LIMIT = 1200
SUMMARY_LIST_ITEM_LIMIT = 12
SUMMARY_DETAILED_PAPER_LIMIT = 150
SUMMARY_FALLBACK_DETAILED_PAPER_LIMIT = 80
SUMMARY_GRAPH_NODE_LIMIT = 150
SUMMARY_FALLBACK_GRAPH_NODE_LIMIT = 80
SUMMARY_GRAPH_EDGE_LIMIT = 200
SUMMARY_PROMPT_CHAR_LIMIT = 900_000
SUMMARY_FALLBACK_ABSTRACT_CHAR_LIMIT = 800
SUMMARY_FALLBACK_CONCLUSION_CHAR_LIMIT = 800


DOMAIN_SUMMARY_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "arc.domain-summary-v4",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "domain_title",
        "brief_introduction",
        "task_focus",
        "foundation_paper",
        "best_reference_paper",
        "methodology",
        "known_solved_cases",
        "open_axes_for_new_work",
        "warnings",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": "arc.domain_summary.v4"},
        "domain_title": {"type": "string"},
        "brief_introduction": {"type": "string"},
        "task_focus": {
            "type": "object",
            "additionalProperties": False,
            "required": ["user_intent", "research_scope", "priority_rules"],
            "properties": {
                "user_intent": {"type": "string"},
                "research_scope": {"type": "string"},
                "priority_rules": {"type": "array", "items": {"type": "string"}},
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
        "foundation_paper": {
            "type": "object",
            "additionalProperties": False,
            "required": ["paper_id", "title", "reason"],
            "properties": {
                "paper_id": {"type": "string"},
                "title": {"type": "string"},
                "reason": {"type": "string"},
            },
        },
        "methodology": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "papers"],
                "properties": {
                    "claim": {"type": "string"},
                    "papers": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "known_solved_cases": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "solved_case",
                    "why_it_is_solved",
                    "transferable_form",
                    "forbidden_reuse",
                    "valid_new_axes",
                    "papers",
                ],
                "properties": {
                    "solved_case": {"type": "string"},
                    "why_it_is_solved": {"type": "string"},
                    "transferable_form": {"type": "string"},
                    "forbidden_reuse": {"type": "string"},
                    "valid_new_axes": {"type": "array", "items": {"type": "string"}},
                    "papers": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "open_axes_for_new_work": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["axis", "guidance", "example_variations", "papers"],
                "properties": {
                    "axis": {"type": "string"},
                    "guidance": {"type": "string"},
                    "example_variations": {"type": "array", "items": {"type": "string"}},
                    "papers": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
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


def _call_record_warning(payload: dict[str, Any]) -> str | None:
    record = payload.get(ARC_LLM_CALL_RECORD_FIELD)
    if not isinstance(record, dict):
        return None
    structured = record.get("structured_output")
    if not isinstance(structured, dict) or structured.get("mode") != "recovered":
        return None
    bits = []
    if structured.get("severity"):
        bits.append(f"severity={structured['severity']}")
    if structured.get("recovery_strategy"):
        bits.append(f"strategy={structured['recovery_strategy']}")
    warnings = structured.get("warnings")
    if isinstance(warnings, list) and warnings:
        bits.append("; ".join(str(item) for item in warnings[:3]))
    return "domain_summary_structured_recovery:" + " | ".join(bits) if bits else None


def _raw_text_from_call_record(payload: dict[str, Any]) -> str:
    record = payload.get(ARC_LLM_CALL_RECORD_FIELD)
    if not isinstance(record, dict):
        return ""
    structured = record.get("structured_output")
    if not isinstance(structured, dict):
        return ""
    return str(structured.get("raw_text_excerpt") or "").strip()


def summarize_domain(
    *,
    paths: DomainPaths,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
) -> dict[str, Any]:
    update_status(paths, stage="summary_started")
    graph = read_json(paths.domain_graph, {})
    evidence = read_json(paths.evidence_pack, {})
    selection = read_json(paths.foundation_selection, {})
    prompt = _summary_prompt(graph=graph, evidence=evidence, selection=selection)
    try:
        raw_summary = run_json(
            prompt,
            schema=DOMAIN_SUMMARY_SCHEMA,
            provider=provider,
            model=model,
            model_tier=model_tier,
            output_recovery="warn",
        )
    except Exception as exc:
        warning = {
            "code": "domain_summary_llm_failed",
            "message": f"LLM domain summary failed; proceeding without domain summary: {type(exc).__name__}: {exc}",
            "created_at": now_iso(),
        }
        _append_status_warnings(paths, [warning])
        _remove_stale_domain_summary_artifacts(paths)
        update_status(
            paths,
            stage="summary_warning_no_summary",
            domain_summary_path=None,
            domain_summary_markdown_path=None,
            summary_available=False,
            domain_summary_available=False,
        )
        return {
            "domain_id": paths.domain_id,
            "summary_available": False,
            "domain_summary_path": None,
            "domain_summary_markdown_path": None,
            "summary": None,
            "warnings": [warning],
        }
    summary, method, relaxed_warnings = _normalize_domain_summary_output(
        raw_summary,
        paths=paths,
        graph=graph,
        evidence=evidence,
        selection=selection,
    )
    summary["summary_method"] = method
    summary["schema_version"] = "arc.domain_summary.v4"
    summary["domain_id"] = paths.domain_id
    summary["created_at"] = now_iso()
    write_json(paths.domain_summary, summary)
    write_text(paths.domain_summary_markdown, render_summary_markdown(summary))
    update_status(
        paths,
        stage="summary_done",
        domain_summary_path=str(paths.domain_summary),
        domain_summary_markdown_path=str(paths.domain_summary_markdown),
        summary_available=True,
        domain_summary_available=True,
    )
    if relaxed_warnings:
        status = read_json(paths.status, {}) or {}
        prior = status.get("warnings") if isinstance(status.get("warnings"), list) else []
        update_status(
            paths,
            warnings=[
                *prior,
                *[
                    {"code": "domain_summary_relaxed", "message": warning, "created_at": now_iso()}
                    for warning in relaxed_warnings
                ],
            ],
        )
    return {
        "domain_id": paths.domain_id,
        "summary_available": True,
        "domain_summary_path": str(paths.domain_summary),
        "domain_summary_markdown_path": str(paths.domain_summary_markdown),
        "summary": summary,
    }


def _append_status_warnings(paths: DomainPaths, warnings: list[dict[str, Any]]) -> None:
    status = read_json(paths.status, {}) or {}
    prior = status.get("warnings") if isinstance(status.get("warnings"), list) else []
    update_status(paths, warnings=[*prior, *warnings])


def _remove_stale_domain_summary_artifacts(paths: DomainPaths) -> None:
    for path in (paths.domain_summary, paths.domain_summary_markdown):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _normalize_domain_summary_output(
    raw: Any,
    *,
    paths: DomainPaths,
    graph: dict[str, Any],
    evidence: dict[str, Any],
    selection: dict[str, Any],
) -> tuple[dict[str, Any], str, list[str]]:
    del paths, graph, evidence
    warnings: list[str] = []
    if isinstance(raw, dict):
        raw_dict = dict(raw)
        raw_non_dict_text = ""
    else:
        raw_dict = {}
        raw_non_dict_text = _compact_text(str(raw or ""), SUMMARY_REASON_CHAR_LIMIT)
        warnings.append("domain_summary_non_object_relaxed: LLM returned non-object output; preserved as text.")
    raw = raw_dict
    schema_error = _schema_error(raw, DOMAIN_SUMMARY_SCHEMA)
    recovery_warning = _call_record_warning(raw)
    if schema_error is None and recovery_warning is None:
        return raw, "llm", []

    if schema_error is not None:
        warnings.append("domain_summary_schema_relaxed:" + _compact_text(schema_error, SUMMARY_REASON_CHAR_LIMIT))
    if recovery_warning:
        warnings.append(recovery_warning)

    raw_without_record = strip_arc_llm_call_records(raw)
    raw_text = _raw_text_from_call_record(raw) or raw_non_dict_text or _best_relaxed_summary_text(raw_without_record)
    foundation = _paper_summary_from_any(
        raw_without_record.get("foundation_paper") or selection.get("selected_foundation") or {},
        fallback_reason="Foundation paper from ARC selection.",
    )
    best_reference = _paper_summary_from_any(
        raw_without_record.get("best_reference_paper")
        or raw_without_record.get("best_reference")
        or selection.get("best_reference_paper")
        or selection.get("selected_foundation")
        or {},
        fallback_reason="Best reference paper from ARC selection.",
    )
    raw_warnings = raw_without_record.get("warnings")
    normalized_warnings = [str(item) for item in raw_warnings if item] if isinstance(raw_warnings, list) else []
    normalized = {
        "schema_version": "arc.domain_summary.v4",
        "domain_title": str(
            raw_without_record.get("domain_title")
            or raw_without_record.get("domain")
            or raw_without_record.get("title")
            or "Research Domain"
        ),
        "brief_introduction": raw_text or "LLM returned a malformed domain summary; inspect relaxed_payload for details.",
        "task_focus": _task_focus_from_relaxed(raw_without_record, selection=selection),
        "foundation_paper": foundation,
        "best_reference_paper": best_reference,
        "methodology": _methodology_from_relaxed(raw_without_record),
        "known_solved_cases": _solved_cases_from_relaxed(raw_without_record),
        "open_axes_for_new_work": _open_axes_from_relaxed(raw_without_record),
        "warnings": [*normalized_warnings, *warnings],
        "relaxed_payload": raw_without_record,
    }
    if isinstance(raw.get(ARC_LLM_CALL_RECORD_FIELD), dict):
        normalized[ARC_LLM_CALL_RECORD_FIELD] = raw[ARC_LLM_CALL_RECORD_FIELD]
    method = "llm_relaxed_text" if (_raw_text_from_call_record(raw) or raw_non_dict_text) and not raw_without_record else "llm_relaxed"
    return normalized, method, warnings


def _best_relaxed_summary_text(raw: dict[str, Any]) -> str:
    for key in (
        "brief_introduction",
        "summary",
        "overview",
        "domain",
        "core_methodology",
        "methodology",
        "priority_rules",
        "open_axes",
        "open_axes_for_new_work",
    ):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if raw:
        return json.dumps(raw, ensure_ascii=False, indent=2, default=str)[:8000]
    return ""


def _paper_summary_from_any(value: Any, *, fallback_reason: str) -> dict[str, str]:
    if isinstance(value, dict):
        return {
            "paper_id": str(value.get("paper_id") or value.get("id") or ""),
            "title": str(value.get("title") or ""),
            "reason": str(value.get("reason") or fallback_reason),
        }
    return {"paper_id": "", "title": "", "reason": fallback_reason}


def _task_focus_from_relaxed(raw: dict[str, Any], *, selection: dict[str, Any]) -> dict[str, Any]:
    task_focus = raw.get("task_focus") if isinstance(raw.get("task_focus"), dict) else {}
    priority_rules = task_focus.get("priority_rules") or raw.get("priority_rules") or []
    if isinstance(priority_rules, str):
        priority_rules = [priority_rules]
    if not isinstance(priority_rules, list):
        priority_rules = []
    return {
        "user_intent": str(task_focus.get("user_intent") or selection.get("intent") or ""),
        "research_scope": str(task_focus.get("research_scope") or raw.get("research_scope") or raw.get("domain") or ""),
        "priority_rules": [str(item) for item in priority_rules if item],
    }


def _methodology_from_relaxed(raw: dict[str, Any]) -> list[dict[str, Any]]:
    source = raw.get("methodology") or raw.get("core_methodology") or []
    return _items_as_claims(source, claim_key="claim")


def _solved_cases_from_relaxed(raw: dict[str, Any]) -> list[dict[str, Any]]:
    source = raw.get("known_solved_cases") or raw.get("solved_cases") or []
    items = []
    for item in _listify(source):
        if isinstance(item, dict):
            items.append(
                {
                    "solved_case": str(item.get("solved_case") or item.get("title") or item.get("case") or ""),
                    "why_it_is_solved": str(item.get("why_it_is_solved") or item.get("description") or ""),
                    "transferable_form": str(item.get("transferable_form") or ""),
                    "forbidden_reuse": str(item.get("forbidden_reuse") or ""),
                    "valid_new_axes": [str(value) for value in _listify(item.get("valid_new_axes"))],
                    "papers": [str(value) for value in _listify(item.get("papers"))],
                }
            )
        else:
            text = str(item)
            items.append(
                {
                    "solved_case": text,
                    "why_it_is_solved": "Recovered from relaxed domain summary text.",
                    "transferable_form": "",
                    "forbidden_reuse": "",
                    "valid_new_axes": [],
                    "papers": [],
                }
            )
    return items[:SUMMARY_LIST_ITEM_LIMIT]


def _open_axes_from_relaxed(raw: dict[str, Any]) -> list[dict[str, Any]]:
    source = raw.get("open_axes_for_new_work") or raw.get("open_axes") or []
    items = []
    for item in _listify(source):
        if isinstance(item, dict):
            items.append(
                {
                    "axis": str(item.get("axis") or item.get("title") or item.get("direction") or ""),
                    "guidance": str(item.get("guidance") or item.get("description") or ""),
                    "example_variations": [str(value) for value in _listify(item.get("example_variations"))],
                    "papers": [str(value) for value in _listify(item.get("papers"))],
                }
            )
        else:
            items.append(
                {
                    "axis": str(item),
                    "guidance": "Recovered from relaxed domain summary text.",
                    "example_variations": [],
                    "papers": [],
                }
            )
    return items[:SUMMARY_LIST_ITEM_LIMIT]


def _items_as_claims(source: Any, *, claim_key: str) -> list[dict[str, Any]]:
    items = []
    for item in _listify(source):
        if isinstance(item, dict):
            items.append(
                {
                    "claim": str(item.get(claim_key) or item.get("description") or item.get("method") or item),
                    "papers": [str(value) for value in _listify(item.get("papers"))],
                }
            )
        else:
            items.append({"claim": str(item), "papers": []})
    return items[:SUMMARY_LIST_ITEM_LIMIT]


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _summary_prompt(*, graph: dict[str, Any], evidence: dict[str, Any], selection: dict[str, Any]) -> str:
    compact_evidence = _compact_summary_evidence(
        graph=graph,
        evidence=evidence,
        selection=selection,
        paper_limit=SUMMARY_DETAILED_PAPER_LIMIT,
        graph_node_limit=SUMMARY_GRAPH_NODE_LIMIT,
        abstract_limit=SUMMARY_ABSTRACT_CHAR_LIMIT,
        conclusion_limit=SUMMARY_CONCLUSION_CHAR_LIMIT,
    )
    prompt = _render_summary_prompt(compact_evidence)
    if len(prompt) <= SUMMARY_PROMPT_CHAR_LIMIT:
        return prompt

    compact_evidence = _compact_summary_evidence(
        graph=graph,
        evidence=evidence,
        selection=selection,
        paper_limit=SUMMARY_FALLBACK_DETAILED_PAPER_LIMIT,
        graph_node_limit=SUMMARY_FALLBACK_GRAPH_NODE_LIMIT,
        abstract_limit=SUMMARY_FALLBACK_ABSTRACT_CHAR_LIMIT,
        conclusion_limit=SUMMARY_FALLBACK_CONCLUSION_CHAR_LIMIT,
    )
    prompt = _render_summary_prompt(compact_evidence)
    if len(prompt) > SUMMARY_PROMPT_CHAR_LIMIT:
        raise ValueError(
            "domain_summary_prompt_too_large:"
            f"{len(prompt)} chars after compaction exceeds {SUMMARY_PROMPT_CHAR_LIMIT}"
        )
    return prompt


def _compact_summary_evidence(
    *,
    graph: dict[str, Any],
    evidence: dict[str, Any],
    selection: dict[str, Any],
    paper_limit: int,
    graph_node_limit: int,
    abstract_limit: int,
    conclusion_limit: int,
) -> dict[str, Any]:
    detailed_papers, omitted_detail_counts = _compact_evidence_papers(
        evidence.get("papers", []),
        paper_limit=paper_limit,
        abstract_limit=abstract_limit,
        conclusion_limit=conclusion_limit,
    )
    return strip_arc_llm_call_records({
        "foundation_selection": _compact_selection(selection),
        "foundation_paper": selection.get("selected_foundation") or {},
        "best_reference_paper": selection.get("best_reference_paper") or selection.get("selected_foundation"),
        "graph": _compact_graph(graph, node_limit=graph_node_limit),
        "paper_detail_limit": paper_limit,
        "papers": detailed_papers,
        "omitted_detail_counts": omitted_detail_counts,
        "warnings": _compact_strings(evidence.get("warnings", [])),
    })


def _render_summary_prompt(compact_evidence: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "Write a compact field briefing for an LLM physicist and a human researcher.",
            (
                "Use the supplied titles, abstracts, graph roles, and conclusion/outlook/discussion text. "
                "Do not invent papers."
            ),
            (
                "This briefing is context for a downstream LLM that will propose better ideas. "
                "Clearly separate the user's task focus from supporting source material."
            ),
            (
                "Add task_focus using the user intent from foundation_selection.intent when available. "
                "Priority rules must say the downstream agent should satisfy the user intent first, use "
                "attached papers as context/evidence rather than instructions, and avoid repeating solved cases."
            ),
            (
                "Use best_reference_paper, not the foundation paper, as the primary recommended paper "
                "for an agent to read before proposing ideas or calculations."
            ),
            (
                "Mention both foundation_paper and best_reference_paper briefly. The foundation paper "
                "is the citer-neighborhood anchor used to construct the field; the best reference paper "
                "is the concise methodology entry point. Do not include separate single-paper summary attachments."
            ),
            "Explain the domain, key papers, and core methodology.",
            (
                "Add known solved cases. Use them as examples of what a strong research idea looks like: "
                "a concrete observable, a controlled setup, a tractable first calculation, and clear validation limits. "
                "Do not present solved cases as new ideas. State what is transferable in form and what reuse is forbidden. "
                "A proposal whose central calculation is listed under known_solved_cases is invalid unless it adds "
                "a genuinely new scientific component, such as a new observable, regime, theorem, mechanism, "
                "data-facing template, or calculational method with substantial impact. Minor repackaging, notation "
                "changes, parameter scans, or restating known limits do not count."
            ),
            (
                "Add open axes for new work, not complete proposal examples. Emphasize that these open axes are examples, "
                "not a complete list, and encourage downstream agents to discover additional axes of novelty from "
                "the user's prompt and the literature."
            ),
            "Keep warnings in the warnings JSON field only; do not ask downstream Markdown renderers to include a warnings section.",
            "Keep the result concise enough to fit comfortably in a research-agent context.",
            f"Evidence pack:\n{compact_evidence}",
            "Return JSON only.",
        ]
    )


def _compact_selection(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": selection.get("schema_version"),
        "intent": _compact_text(selection.get("intent"), SUMMARY_REASON_CHAR_LIMIT),
        "selected_foundation": _compact_candidate(selection.get("selected_foundation") or {}),
        "best_reference_paper": _compact_candidate(selection.get("best_reference_paper") or {}),
        "parent_foundations": [
            _compact_candidate(item)
            for item in _bounded_items(selection.get("parent_foundations", []), SUMMARY_LIST_ITEM_LIMIT)
            if isinstance(item, dict)
        ],
        "rejected_candidates": [
            _compact_candidate(item)
            for item in _bounded_items(selection.get("rejected_candidates", []), SUMMARY_LIST_ITEM_LIMIT)
            if isinstance(item, dict)
        ],
        "reasoning": _compact_text(selection.get("reasoning"), SUMMARY_REASON_CHAR_LIMIT),
        "warnings": _compact_strings(selection.get("warnings", [])),
    }


def _compact_candidate(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": item.get("paper_id"),
        "title": item.get("title"),
        "year": item.get("year"),
        "reason": _compact_text(item.get("reason"), SUMMARY_REASON_CHAR_LIMIT),
        "source_role": item.get("source_role"),
    }


def _compact_graph(graph: dict[str, Any], *, node_limit: int) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        nodes = []
    edges = graph.get("edges", [])
    if not isinstance(edges, list):
        edges = []
    return {
        "foundation_paper": graph.get("foundation_paper"),
        "node_limit": node_limit,
        "omitted_node_count": max(0, len(nodes) - node_limit),
        "nodes": [
            {
                "paper_id": node.get("paper_id"),
                "role": node.get("role"),
                "title": node.get("title"),
                "year": node.get("year"),
                "citation_count": node.get("citation_count"),
                "selection_reason": node.get("selection_reason"),
            }
            for node in _bounded_items(nodes, node_limit)
            if isinstance(node, dict)
        ],
        "edge_limit": SUMMARY_GRAPH_EDGE_LIMIT,
        "omitted_edge_count": max(0, len(edges) - SUMMARY_GRAPH_EDGE_LIMIT),
        "edges": edges[:SUMMARY_GRAPH_EDGE_LIMIT],
    }


def _compact_evidence_papers(
    values: Any,
    *,
    paper_limit: int,
    abstract_limit: int,
    conclusion_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    papers = values if isinstance(values, list) else []
    detailed = [
        _compact_evidence_paper(item, abstract_limit=abstract_limit, conclusion_limit=conclusion_limit)
        for item in _bounded_items(papers, paper_limit)
        if isinstance(item, dict)
    ]
    omitted = [item for item in papers[paper_limit:] if isinstance(item, dict)]
    return detailed, _omitted_detail_counts(omitted, total_paper_count=len(papers), detail_limit=paper_limit)


def _omitted_detail_counts(items: list[dict[str, Any]], *, total_paper_count: int, detail_limit: int) -> dict[str, Any]:
    return {
        "total_paper_count": total_paper_count,
        "paper_detail_limit": detail_limit,
        "omitted_paper_count": len(items),
        "by_role": _counts_by_field(items, "role"),
        "by_year": _counts_by_field(items, "year"),
    }


def _counts_by_field(items: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(item.get(field) or "unknown") for item in items)
    return dict(sorted(counts.items(), key=lambda entry: entry[0]))


def _compact_evidence_paper(item: dict[str, Any], *, abstract_limit: int, conclusion_limit: int) -> dict[str, Any]:
    conclusion = item.get("conclusion") or {}
    conclusion_text = conclusion.get("text", "") if isinstance(conclusion, dict) else conclusion
    return {
        "paper_id": item.get("paper_id"),
        "role": item.get("role"),
        "title": item.get("title"),
        "abstract": _compact_text(item.get("abstract"), abstract_limit),
        "conclusion": _compact_text(conclusion_text, conclusion_limit),
        "warnings": _compact_strings(item.get("warnings", []), max_items=4),
    }


def _compact_strings(values: Any, *, max_items: int = SUMMARY_LIST_ITEM_LIMIT) -> list[str]:
    if not isinstance(values, list):
        values = [values] if values else []
    compacted = [
        _compact_text(item, SUMMARY_WARNING_CHAR_LIMIT)
        for item in _bounded_items(values, max_items)
        if item
    ]
    if len(values) > max_items:
        compacted.append(f"[truncated list: {len(values) - max_items} more item(s)]")
    return compacted


def _bounded_items(values: Any, max_items: int) -> list[Any]:
    if not isinstance(values, list):
        return []
    return values[:max_items]


def _compact_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[truncated]"


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = [f"# {summary.get('domain_title') or 'Research Domain'}", ""]
    if intro := summary.get("brief_introduction"):
        lines.extend([str(intro), ""])
    task_focus = summary.get("task_focus") or {}
    if task_focus:
        lines.extend(["## Task Focus for Idea Generation", ""])
        if intent := task_focus.get("user_intent"):
            lines.append(f"- User intent: {intent}")
        if scope := task_focus.get("research_scope"):
            lines.append(f"- Research scope: {scope}")
        rules = task_focus.get("priority_rules") or []
        if rules:
            lines.append("- Priority rules:")
            for rule in rules:
                lines.append(f"  - {rule}")
        lines.append("")
    foundation_paper = summary.get("foundation_paper") or {}
    best_reference = summary.get("best_reference_paper") or {}
    if foundation_paper or best_reference:
        lines.extend(["## Key Papers", ""])
        _append_key_paper(lines, "Foundation paper", foundation_paper)
        _append_key_paper(lines, "Best reference paper", best_reference)
        lines.append("")
    methodology = summary.get("methodology") or []
    if methodology:
        lines.extend(["## Methodology", ""])
        for item in methodology:
            lines.append(f"- {item.get('claim', '')}")
            _append_papers(lines, item.get("papers"))
        lines.append("")
    solved_cases = summary.get("known_solved_cases") or []
    if solved_cases:
        lines.extend(
            [
                "## Known Solved Cases",
                "",
                (
                    "Use these solved cases as examples of strong research form, not as new ideas. "
                    "Do not propose a solved case itself as the core deliverable unless the proposal "
                    "adds a genuinely new scientific component with substantial impact."
                ),
                "",
            ]
        )
        for item in solved_cases:
            lines.append(f"- {item.get('solved_case', '')}")
            if why := item.get("why_it_is_solved"):
                lines.append(f"  Why solved: {why}")
            if form := item.get("transferable_form"):
                lines.append(f"  Transferable form: {form}")
            if forbidden := item.get("forbidden_reuse"):
                lines.append(f"  Forbidden reuse: {forbidden}")
            if axes := item.get("valid_new_axes"):
                lines.append(f"  Valid new axes: {', '.join(str(axis) for axis in axes if axis)}")
            _append_papers(lines, item.get("papers"))
        lines.append("")
    open_axes = summary.get("open_axes_for_new_work") or []
    if open_axes:
        lines.extend(
            [
                "## Open Axes for New Work",
                "",
                (
                    "These axes are examples, not a complete list. Use them to look for substantial "
                    "differences from solved work, and actively discover additional axes from the "
                    "user prompt, source papers, and novelty checks."
                ),
                "",
            ]
        )
        for item in open_axes:
            lines.append(f"- {item.get('axis', '')}")
            if guidance := item.get("guidance"):
                lines.append(f"  Guidance: {guidance}")
            if variations := item.get("example_variations"):
                joined_variations = ", ".join(str(variation) for variation in variations if variation)
                lines.append(f"  Example variations: {joined_variations}")
            _append_papers(lines, item.get("papers"))
        lines.append("")
    relaxed_payload = summary.get("relaxed_payload")
    if relaxed_payload:
        lines.extend(
            [
                "## Relaxed LLM Output Warning",
                "",
                (
                    "The domain summary did not fully match the strict schema. ARC preserved the recovered content "
                    "below for downstream reading."
                ),
                "",
                "```json",
                json.dumps(relaxed_payload, ensure_ascii=False, indent=2, default=str)[:12000],
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _append_papers(lines: list[str], papers: Any) -> None:
    if papers:
        lines.append(f"  Papers: {', '.join(str(item) for item in papers if item)}")


def _append_key_paper(lines: list[str], label: str, paper: dict[str, Any]) -> None:
    if not paper:
        return
    title = paper.get("title") or ""
    paper_id = paper.get("paper_id") or ""
    identifier = ": ".join(part for part in [paper_id, title] if part)
    lines.append(f"- {label}: {identifier}".rstrip())
    if reason := paper.get("reason"):
        lines.append(f"  Reason: {reason}")
