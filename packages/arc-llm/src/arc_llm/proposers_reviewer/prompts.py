from __future__ import annotations

import json
from typing import Any

from arc_llm.call_record import strip_arc_llm_call_records
from arc_llm.progress_prompt import ensure_runtime_progress_contract

from .config import LoopConfig, WorkerConfig


def proposer_context(
    *,
    loop: LoopConfig,
    worker: WorkerConfig,
    round_number: int,
    correspondence: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "role": "proposer",
        "loop_id": loop.loop_id,
        "loop_metadata": _loop_metadata(loop),
        "worker_id": worker.id,
        "round_number": round_number,
        "caller_context": strip_arc_llm_call_records(loop.caller_context),
        "correspondence": strip_arc_llm_call_records(correspondence),
    }


def reviewer_context(
    *,
    loop: LoopConfig,
    worker: WorkerConfig,
    round_number: int,
    correspondence: list[dict[str, Any]],
    current_proposer_outputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": "reviewer",
        "loop_id": loop.loop_id,
        "loop_metadata": _loop_metadata(loop),
        "worker_id": worker.id,
        "round_number": round_number,
        "caller_context": strip_arc_llm_call_records(loop.caller_context),
        "correspondence": strip_arc_llm_call_records(correspondence),
        "current_proposer_outputs": strip_arc_llm_call_records(current_proposer_outputs),
    }


def render_prompt(worker: WorkerConfig, context: dict[str, Any]) -> str:
    rendered_template = _replace_known_placeholders(worker.prompt.template, context)
    sections = [
        "## ARC Worker Instructions",
    ]
    if worker.prompt.system:
        sections.extend(["### System", worker.prompt.system])
    sections.extend(["### Task", rendered_template])
    paper_cli_access = str(worker.runtime.get("arc_paper_cli_access", "full"))
    if paper_cli_access == "full":
        sections.extend(
            [
                "## ARC Paper CLI",
                "When the host exposes a sandboxed shell and the controller-created worker guard validates, you may invoke arc-paper-worker directly and repeatedly for non-LLM paper lookup, search, citation relations, full text, equations, parsing, cache access, and missing-paper retrieval. Commands stage results in a writable run overlay; the trusted controller validates and promotes them only after the model call ends. Use its pagination command when a result returns an artifact handle. Never invoke raw arc-paper, Python arc_paper modules, arc-llm, arc-jobs, arc-domain, summary generation, inference, batch-model commands, aliases, or another route to a nested LLM. If the host has no sandboxed shell (including a host without the required sandbox helper), do not attempt a shell bypass; request the equivalent operation through arc_evidence_requests.",
            ]
        )
    if worker.evidence_enabled:
        sections.extend(
            [
                "## Controller Evidence Protocol",
                "When controller evidence is needed, add arc_evidence_requests using the provided schema. Give each request a worker-prefixed request_id unique in this loop round, an operation from caller_context.controller_evidence_operations when that list is present, JSON arguments, and a precise reason. Return [] or omit the field when no check is needed. The controller may resolve at most three evidence rounds and will return responses with provenance in a later turn. "
                + (
                    "Except for arc-paper-worker, do not invoke shell commands, ARC CLIs, or MCP tools yourself."
                    if paper_cli_access == "full"
                    else "Do not invoke shell commands, ARC CLIs, or MCP tools yourself."
                ),
            ]
        )
    if worker.runtime.get("append_context", True):
        sections.extend(["## ARC Worker Context", json.dumps(context, indent=2, ensure_ascii=False, sort_keys=True)])
    prompt = "\n".join(sections).rstrip() + "\n"
    if paper_cli_access == "full":
        prompt = _remove_legacy_paper_cli_blankets(prompt)
    return ensure_runtime_progress_contract(prompt)


def _remove_legacy_paper_cli_blankets(prompt: str) -> str:
    """Translate older controller-only wording for direct paper CLI workers."""
    replacement = (
        "Use arc-paper-worker for ARC paper evidence; do not invoke other ARC CLIs, "
        "nested LLM commands, or MCP tools"
    )
    for legacy in (
        "Do not invoke ARC CLIs, shell commands, or MCP tools",
        "do not invoke ARC CLIs, shell commands, or MCP tools",
        "Do not invoke ARC CLI, shell, or MCP tools",
        "do not invoke ARC CLI, shell, or MCP tools",
    ):
        prompt = prompt.replace(legacy, replacement)
    return prompt


def _replace_known_placeholders(template: str, context: dict[str, Any]) -> str:
    replacements = {
        "{loop_id}": str(context.get("loop_id", "")),
        "{worker_id}": str(context.get("worker_id", "")),
        "{round_number}": str(context.get("round_number", "")),
        "{caller_context_json}": json.dumps(context.get("caller_context", {}), indent=2, ensure_ascii=False, sort_keys=True),
        "{correspondence_json}": json.dumps(context.get("correspondence", []), indent=2, ensure_ascii=False, sort_keys=True),
        "{current_proposer_outputs_json}": json.dumps(
            context.get("current_proposer_outputs", {}),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def _loop_metadata(loop: LoopConfig) -> dict[str, Any]:
    return {
        "loop_id": loop.loop_id,
        "max_rounds": loop.max_rounds,
        "early_stop_enabled": loop.early_stop_enabled,
        "evidence_enabled": loop.evidence_enabled,
        "proposer_ids": [worker.id for worker in loop.proposers],
        "reviewer_ids": [worker.id for worker in loop.reviewers],
    }
