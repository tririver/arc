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
    rendered_template = _replace_safe_placeholders(worker.prompt.template, variable, stable_role=role)
    static_parts = [
        "## ARC-LLM Worker Session ABI v2",
        "This initializes a persistent ARC worker session. Remember the static task context and your worker instructions for later turns in this session.",
        "When Worker Instructions mention caller_context.X, read X from the union of Shared Static Task Context.caller_context and Variable Initial Context.caller_context. Static caller_context fields are remembered for this session; volatile fields are provided in Variable Initial Context or later Delta Context.",
        "",
        "## Shared Static Task Context",
        _canonical_json(shared),
        "",
        "## Worker Instructions",
        "### System",
        worker.prompt.system,
        "",
        "### Task",
        rendered_template,
        "",
        "## Output Contract",
        "Return exactly one JSON object matching the provided schema. Do not wrap it in Markdown. If unsure about a field, still include the field with your best useful value. For arrays, return [] when no items are available. For booleans, return true or false. For nullable fields, use null when appropriate. Do not return explanatory prose outside the JSON object.",
    ]
    static_prefix = "\n".join(static_parts).rstrip() + "\n\n"
    variable_suffix = "\n".join(
        [
            "## Variable Initial Context",
            _canonical_json(variable),
        ]
    ).rstrip() + "\n"
    prompt = static_prefix + variable_suffix
    context = {
        "turn_kind": "initial",
        "role": role,
        "loop_id": loop.loop_id,
        "worker_id": worker.id,
        "round_number": round_number,
        "static_context": shared,
        "variable_context": variable,
    }
    return prompt, context, static_prefix


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
            "You are continuing the same proposer session. Use the remembered static task context, worker instructions, and current delta context.",
            "When Worker Instructions mention caller_context.X, read X from the remembered static caller_context plus the current Delta Context caller_context_delta. Current delta fields override older volatile fields.",
            "Treat prior unaccepted proposer outputs as tentative scratch work, not as facts. If reviewer or controller feedback identifies an error, asks for recalculation, changes the target, changes active proposer ids, or points to a convention/source-mapping issue, recompute the relevant reasoning from the original task context rather than merely patching your previous answer.",
            "Locked outputs and accepted_prior_step_outputs are accepted context unless current delta feedback explicitly asks you to re-check them.",
            "The JSON Schema/output contract for this turn may differ from earlier turns. Obey the schema provided for this turn and current turn context, not any older schema or older active proposer list in the session history.",
            "",
            "## Delta Context",
            _canonical_json(strip_arc_llm_call_records(delta)),
            "",
            "Return the revised proposer JSON object for this round. If you cannot complete the derivation or proposal, still return every required JSON field and put the partial work or difficulty inside the appropriate field rather than prose outside JSON.",
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
            "You are continuing the same reviewer session. Use the remembered static task context and worker instructions.",
            "When Worker Instructions mention caller_context.X, read X from the remembered static caller_context plus the current Delta Context caller_context_delta. Current delta fields override older volatile fields.",
            "Review the Current Proposer Outputs section independently for this turn. Previous review history is background only; do not let an older accepted/rejected judgment override the current proposer outputs, current active_proposer_ids, caller_context_delta, or current JSON schema.",
            "The JSON Schema/output contract for this turn may differ from earlier turns. Obey the schema provided for this turn and the current active_proposer_ids, not any older schema or older active proposer list in the session history.",
            "",
            "## Current Proposer Outputs To Review",
            _canonical_json(outputs),
            "",
            "## Delta Context",
            _canonical_json(delta),
            "",
            "Return exactly one arc.llm.review_envelope.v1 JSON object for this round. If you cannot complete a full review, still return a valid review envelope with a non-accepting status, explanatory analysis/controller.message, and proposer_messages for every active proposer id.",
        ]
    ) + "\n"
    context = {"turn_kind": "delta", **delta, "current_proposer_outputs": outputs}
    return prompt, context


def render_reviewer_validation_retry_delta(exc: Exception) -> str:
    return (
        "## ARC-LLM Reviewer Validation Retry v2\n"
        f"Your previous reviewer JSON failed validation:\n{exc}\n\n"
        "The JSON Schema/output contract for this turn may differ from earlier turns. Obey the schema provided for this turn and the current active_proposer_ids, not any older schema or older active proposer list in the session history.\n"
        "Use the same current proposer outputs already present in this session, but validate them against the current schema and current active proposer ids. Do not reuse an older review envelope. Return exactly one corrected "
        "arc.llm.review_envelope.v1 JSON object. If you cannot complete the review, use a non-accepting status and explain why inside the envelope.\n"
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


def _replace_safe_placeholders(template: str, context: dict[str, Any], *, stable_role: str) -> str:
    rendered = template
    replacements = {
        "{loop_id}": "current loop",
        "{worker_id}": f"current {stable_role}",
        "{round_number}": "current round",
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
