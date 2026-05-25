from __future__ import annotations

from typing import Any

from arc_llm import run_json

from .cache import DomainPaths, now_iso, read_json, update_status, write_json, write_text


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
    summary = run_json(prompt, schema=DOMAIN_SUMMARY_SCHEMA, provider=provider, model=model)
    summary["summary_method"] = "llm"
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
    )
    return {
        "domain_id": paths.domain_id,
        "domain_summary_path": str(paths.domain_summary),
        "domain_summary_markdown_path": str(paths.domain_summary_markdown),
        "summary": summary,
    }


def _summary_prompt(*, graph: dict[str, Any], evidence: dict[str, Any], selection: dict[str, Any]) -> str:
    compact_evidence = {
        "foundation_selection": selection,
        "foundation_paper": selection.get("selected_foundation") or {},
        "best_reference_paper": selection.get("best_reference_paper") or selection.get("selected_foundation"),
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
            (
                "Use the supplied titles, abstracts, graph roles, and conclusion/outlook/discussion text. "
                "Do not invent papers."
            ),
            (
                "This briefing is context for a downstream LLM that will propose better research ideas. "
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
