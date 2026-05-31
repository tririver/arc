from __future__ import annotations

import json
from typing import Any

from arc_llm.call_record import strip_arc_llm_call_records

from .config import LoopConfig, WorkerConfig
from .context_pack import split_caller_context
from .prompts import render_prompt


CONTEXT_PLACEHOLDERS = (
    "{caller_context_json}",
    "{correspondence_json}",
    "{current_proposer_outputs_json}",
)


def render_legacy_full_prompt(worker: WorkerConfig, context: dict[str, Any]) -> str:
    return render_prompt(worker, context)


def render_initial_worker_prompt(
    *,
    loop: LoopConfig,
    worker: WorkerConfig,
    role: str,
    round_number: int,
) -> tuple[str, dict[str, Any], str]:
    pack = split_caller_context(strip_arc_llm_call_records(loop.caller_context), _cache_context_dict(loop))
    shared = {
        "caller_context": pack.static,
        "loop_metadata": _static_loop_metadata(loop),
    }
    variable = {
        "role": role,
        "loop_id": loop.loop_id,
        "worker_id": worker.id,
        "round_number": round_number,
        "caller_context": pack.volatile,
    }
    rendered_template = _replace_safe_placeholders(worker.prompt.template, variable)
    prompt = "\n".join(
        [
            "## ARC-LLM Worker Session ABI v2",
            "This initializes a persistent ARC worker session. Remember the static task context and your worker instructions for later turns in this session.",
            "",
            "## Shared Static Task Context",
            _canonical_json(shared),
            "",
            "## Variable Initial Context",
            _canonical_json(variable),
            "",
            "## Worker Instructions",
            "### System",
            worker.prompt.system,
            "",
            "### Task",
            rendered_template,
            "",
            "## Output Contract",
            "Return exactly one JSON object matching the provided schema. Do not wrap it in Markdown.",
        ]
    ).rstrip() + "\n"
    context = {
        "turn_kind": "initial",
        "role": role,
        "loop_id": loop.loop_id,
        "worker_id": worker.id,
        "round_number": round_number,
        "static_context": shared,
        "variable_context": variable,
    }
    return prompt, context, _canonical_json(shared)


def render_proposer_delta_prompt(
    *,
    loop: LoopConfig,
    worker: WorkerConfig,
    round_number: int,
    correspondence: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    last_round = round_number - 1
    delta = {
        "role": "proposer",
        "loop_id": loop.loop_id,
        "worker_id": worker.id,
        "round_number": round_number,
        "controller_message": _last_event(correspondence, event_type="controller_message", round_number=last_round),
        "proposer_message": _last_event(
            correspondence,
            event_type="proposer_message",
            round_number=last_round,
            worker_id=worker.id,
        ),
        "caller_context_delta": _volatile_caller_context(loop),
    }
    prompt = "\n".join(
        [
            "## ARC-LLM Worker Delta Turn v2",
            "You are continuing the same proposer session. Use the static task context and your previous work already present in this session.",
            "",
            "## Delta Context",
            _canonical_json(strip_arc_llm_call_records(delta)),
            "",
            "Return the revised proposer JSON object for this round.",
        ]
    ) + "\n"
    context = {"turn_kind": "delta", **strip_arc_llm_call_records(delta)}
    return prompt, context


def render_reviewer_delta_prompt(
    *,
    loop: LoopConfig,
    worker: WorkerConfig,
    round_number: int,
    current_proposer_outputs: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    delta = {
        "role": "reviewer",
        "loop_id": loop.loop_id,
        "worker_id": worker.id,
        "round_number": round_number,
        "active_proposer_ids": [proposer.id for proposer in loop.proposers],
        "caller_context_delta": _volatile_caller_context(loop),
    }
    outputs = strip_arc_llm_call_records(current_proposer_outputs)
    prompt = "\n".join(
        [
            "## ARC-LLM Reviewer Delta Turn v2",
            "You are continuing the same reviewer session. Use the static task context and previous review history already present in this session.",
            "",
            "## Current Proposer Outputs To Review",
            _canonical_json(outputs),
            "",
            "## Delta Context",
            _canonical_json(delta),
            "",
            "Return exactly one arc.llm.review_envelope.v1 JSON object for this round.",
        ]
    ) + "\n"
    context = {"turn_kind": "delta", **delta, "current_proposer_outputs": outputs}
    return prompt, context


def render_reviewer_validation_retry_delta(exc: Exception) -> str:
    return (
        "## ARC-LLM Reviewer Validation Retry v2\n"
        f"Your previous reviewer JSON failed validation:\n{exc}\n\n"
        "Use the same current proposer outputs already present in this session. Return exactly one corrected "
        "arc.llm.review_envelope.v1 JSON object.\n"
    )


def _static_loop_metadata(loop: LoopConfig) -> dict[str, Any]:
    return {
        "max_rounds": loop.max_rounds,
        "early_stop_enabled": loop.early_stop_enabled,
        "proposer_ids": [worker.id for worker in loop.proposers],
        "reviewer_ids": [worker.id for worker in loop.reviewers],
    }


def _volatile_caller_context(loop: LoopConfig) -> dict[str, Any]:
    return split_caller_context(strip_arc_llm_call_records(loop.caller_context), _cache_context_dict(loop)).volatile


def _cache_context_dict(loop: LoopConfig) -> dict[str, Any] | None:
    cache_context = getattr(loop, "cache_context", None)
    if cache_context is None:
        return None
    return {
        "static_caller_context_keys": list(cache_context.static_caller_context_keys),
        "volatile_caller_context_keys": list(cache_context.volatile_caller_context_keys),
    }


def _replace_safe_placeholders(template: str, context: dict[str, Any]) -> str:
    rendered = template
    replacements = {
        "{loop_id}": str(context.get("loop_id", "")),
        "{worker_id}": str(context.get("worker_id", "")),
        "{round_number}": str(context.get("round_number", "")),
    }
    for placeholder in CONTEXT_PLACEHOLDERS:
        rendered = rendered.replace(placeholder, "")
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def _last_event(
    correspondence: list[dict[str, Any]],
    *,
    event_type: str,
    round_number: int,
    worker_id: str | None = None,
) -> dict[str, Any] | None:
    for event in reversed(correspondence):
        if event.get("type") != event_type:
            continue
        if event.get("round_number") != round_number:
            continue
        if worker_id is not None and event.get("worker_id") != worker_id:
            continue
        return strip_arc_llm_call_records(event)
    return None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
