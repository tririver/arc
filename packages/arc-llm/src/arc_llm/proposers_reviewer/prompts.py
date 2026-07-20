from __future__ import annotations

import json
from typing import Any

from arc_llm.call_record import strip_arc_llm_call_records

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
    if worker.evidence_enabled:
        sections.extend(
            [
                "## Controller Evidence Protocol",
                "When controller evidence is needed, add arc_evidence_requests using the provided schema. Give each request a worker-prefixed request_id unique in this loop round, an operation from caller_context.controller_evidence_operations when that list is present, JSON arguments, and a precise reason. Return [] or omit the field when no check is needed. The controller may resolve at most three evidence rounds and will return responses with provenance in a later turn. Do not invoke shell commands, ARC CLIs, or MCP tools yourself.",
            ]
        )
    if worker.runtime.get("append_context", True):
        sections.extend(["## ARC Worker Context", json.dumps(context, indent=2, ensure_ascii=False, sort_keys=True)])
    return "\n".join(sections).rstrip() + "\n"


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
