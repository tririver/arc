from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from _arc_script_bootstrap import bootstrap_arc_pythonpath

bootstrap_arc_pythonpath()

from arc_llm.proposers_reviewer.artifacts import RunPaths, atomic_write_json
from arc_llm.proposers_reviewer.config import ConfigError, SAFE_ID_RE
from arc_llm.proposers_reviewer.runner import JsonRunner, run_proposers_reviewer_batch


CALCULATE_CONFIG_SCHEMA = "arc.workflow.calculate.config.v1"
CALCULATE_RESULT_SCHEMA = "arc.workflow.calculate.result.v1"
REVIEW_ENVELOPE_SCHEMA = "arc.llm.review_envelope.v1"
DEFAULT_HUMAN_GATE_PAUSE_STATUSES = (
    "reference_disagrees",
    "two_agree",
    "all_disagree",
    "unresolved",
    "failed",
)
RETRYABLE_CONSENSUS_STATUSES = {"reference_disagrees", "two_agree", "all_disagree", "unresolved"}
REVISION_ACTIONS = {"revise_plan", "split_step"}
SOURCE_DISCREPANCY_STATUSES = {
    "confirmed_source_error",
    "likely_source_error",
    "ambiguous_convention",
}
LEGACY_ALLOWED_CONTEXT_KEYS = {"foundation_file", "allowed_foundation", "target_equation_id"}
CALLER_ALLOWED_CONTEXT_OMIT_KEYS = {
    "source_commands",
    "shell_commands",
    "mcp_call_instructions",
    "cli_invocations",
}

BatchRunner = Callable[..., dict[str, Any]]


def _relaxed_output_recovery_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": "warn",
        "allow_natural_language": True,
        "schema_violation_policy": "peer_visible",
        "reviewer_validation_retries": 0,
    }


@dataclass(frozen=True)
class CalculateStep:
    step_id: str
    prompt: str
    kind: str
    allowed_context: dict[str, Any]
    proposer_runtime: dict[str, Any]
    reviewer_reference_claim: dict[str, Any] | None


@dataclass(frozen=True)
class CalculateConfig:
    schema_version: str
    run_id: str
    run_dir: Path
    workflow_json_dir: Path
    proposer_count: int
    max_recalculations: int
    human_gate: dict[str, Any]
    defaults: dict[str, Any]
    artifact_options: dict[str, Any]
    steps: list[CalculateStep]


def load_calculation_config(payload: Mapping[str, Any]) -> CalculateConfig:
    data = copy.deepcopy(dict(payload))
    schema_version = _required_text(data, "schema_version")
    if schema_version != CALCULATE_CONFIG_SCHEMA:
        raise ConfigError(f"schema_version must be {CALCULATE_CONFIG_SCHEMA}")

    run_id = _safe_id(_required_text(data, "run_id"), "run_id")
    run_dir = Path(_required_text(data, "run_dir")).expanduser()
    workflow_json_dir = Path(str(data.get("workflow_json_dir") or _default_workflow_json_dir())).expanduser()
    proposer_count = _positive_int(data.get("proposer_count", 2), "proposer_count")
    max_recalculations = _nonnegative_int(data.get("max_recalculations", 1), "max_recalculations")
    human_gate = _parse_human_gate(data.get("human_gate", {}))
    defaults = _dict(data.get("defaults", {}), "defaults")
    if defaults.get("model") is not None and str(defaults.get("provider", "auto") or "auto") == "auto":
        raise ConfigError("defaults.model requires explicit provider")
    artifact_options = _dict(data.get("artifact_options", {"save_prompts": True}), "artifact_options")
    artifact_options["save_prompts"] = _bool(
        artifact_options.get("save_prompts", True),
        "artifact_options.save_prompts",
    )

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ConfigError("steps must be a non-empty list")

    steps: list[CalculateStep] = []
    seen_step_ids: set[str] = set()
    for raw_step in raw_steps:
        step_data = _dict(raw_step, "steps[]")
        step_id = _safe_id(_required_text(step_data, "step_id"), "step_id")
        if step_id in seen_step_ids:
            raise ConfigError(f"duplicate step_id: {step_id}")
        seen_step_ids.add(step_id)
        kind = str(step_data.get("kind", "new_calculation") or "new_calculation")
        if kind != "new_calculation":
            raise ConfigError("step.kind must be new_calculation")
        allowed_context = _dict(step_data.get("allowed_context", {}), f"{step_id}.allowed_context")
        for legacy_key in sorted(LEGACY_ALLOWED_CONTEXT_KEYS):
            if legacy_key in allowed_context:
                raise ConfigError(f"allowed_context.{legacy_key} is no longer supported")
        steps.append(
            CalculateStep(
                step_id=step_id,
                prompt=_required_text(step_data, "prompt"),
                kind=kind,
                allowed_context=allowed_context,
                proposer_runtime=_dict(
                    step_data.get("proposer_runtime", {}),
                    f"{step_id}.proposer_runtime",
                ),
                reviewer_reference_claim=_optional_dict(
                    step_data.get("reviewer_reference_claim"),
                    f"{step_id}.reviewer_reference_claim",
                ),
            )
        )

    return CalculateConfig(
        schema_version=schema_version,
        run_id=run_id,
        run_dir=run_dir,
        workflow_json_dir=workflow_json_dir,
        proposer_count=proposer_count,
        max_recalculations=max_recalculations,
        human_gate=human_gate,
        defaults=defaults,
        artifact_options=artifact_options,
        steps=steps,
    )


def run_calculation(
    config: CalculateConfig | Mapping[str, Any],
    *,
    batch_runner: BatchRunner | None = None,
    json_runner: JsonRunner | None = None,
    base_env: Mapping[str, str] | None = None,
    process_chain: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    calculation = config if isinstance(config, CalculateConfig) else load_calculation_config(config)
    paths = RunPaths(run_dir=calculation.run_dir, run_id=calculation.run_id)
    if dry_run:
        return _dry_run_result(calculation, paths)

    paths.run_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths.config, _jsonable(calculation))

    runner = batch_runner or run_proposers_reviewer_batch
    step_results: list[dict[str, Any]] = []
    accepted_step_outputs: dict[str, Any] = {}
    overall_status = "completed"
    for step in calculation.steps:
        step_result = _run_calculation_step(
            calculation,
            step,
            runner=runner,
            json_runner=json_runner,
            base_env=base_env,
            process_chain=process_chain,
            run_root=paths.run_root,
            accepted_step_outputs=accepted_step_outputs,
        )
        step_results.append(step_result)
        if step_result["status"] in {"blocked_for_user", "blocked_for_revision"}:
            overall_status = step_result["status"]
            break
        if step_result["status"] == "failed":
            overall_status = "failed"
            break
        if step_result["status"] == "accepted":
            accepted_step_outputs[step.step_id] = copy.deepcopy(step_result["accepted_output"])

    result = {
        "schema_version": CALCULATE_RESULT_SCHEMA,
        "status": overall_status,
        "run_id": calculation.run_id,
        "run_root": str(paths.run_root),
        "proposer_count": calculation.proposer_count,
        "max_recalculations": calculation.max_recalculations,
        "human_gate": copy.deepcopy(calculation.human_gate),
        "steps": step_results,
        "warnings_summary": _aggregate_warnings_summary(step_results),
    }
    atomic_write_json(paths.state, result)
    return result


def _run_calculation_step(
    config: CalculateConfig,
    step: CalculateStep,
    *,
    runner: BatchRunner,
    json_runner: JsonRunner | None,
    base_env: Mapping[str, str] | None,
    process_chain: list[str] | None,
    run_root: Path,
    accepted_step_outputs: Mapping[str, Any],
) -> dict[str, Any]:
    all_proposer_ids = _proposer_ids(config.proposer_count)
    active_proposer_ids = list(all_proposer_ids)
    locked_outputs: dict[str, Any] = {}
    retry_feedback: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    max_attempts = config.max_recalculations + 1

    for attempt_number in range(1, max_attempts + 1):
        try:
            batch_config = _attempt_batch_config(
                config,
                step,
                attempt_number=attempt_number,
                active_proposer_ids=active_proposer_ids,
                locked_outputs=locked_outputs,
                retry_feedback=retry_feedback,
                run_root=run_root,
                accepted_step_outputs=accepted_step_outputs,
            )
        except Exception as exc:
            return _failed_step_result(config, step, attempts=attempts, error=str(exc))
        batch_result = runner(
            batch_config,
            json_runner=json_runner,
            base_env=base_env,
            process_chain=process_chain,
            dry_run=False,
            max_concurrent_loops=1,
        )
        attempt_paths = _attempt_paths(batch_config)
        attempt_record = {
            "attempt_number": attempt_number,
            "active_proposer_ids": list(active_proposer_ids),
            "batch_run_id": batch_config["run_id"],
            "batch_loop_id": batch_config["loops"][0]["loop_id"],
            "batch_root": str(attempt_paths["run_root"]),
            "review_path": str(attempt_paths["review_path"]),
            "warnings_summary": _batch_warnings_summary(batch_result),
        }
        if batch_result.get("status") != "completed":
            attempt_record["batch_result"] = batch_result
            attempts.append(attempt_record)
            return _failed_step_result(
                config,
                step,
                attempts=attempts,
                error="attempt batch did not complete",
            )

        try:
            review = _read_json(attempt_paths["review_path"])
            proposer_outputs = _read_proposer_outputs(attempt_paths["round_root"], active_proposer_ids)
            review_consensus = _review_consensus(
                review,
                active_proposer_ids=active_proposer_ids,
                selectable_proposer_ids=list(
                    dict.fromkeys([*active_proposer_ids, *[proposer_id for proposer_id in locked_outputs]])
                ),
                reviewer_reference_claim=step.reviewer_reference_claim,
            )
        except Exception as exc:
            attempt_record["error"] = str(exc)
            attempts.append(attempt_record)
            return _failed_step_result(config, step, attempts=attempts, error=str(exc))
        attempt_record["consensus"] = review_consensus
        attempt_record["proposer_output_paths"] = {
            proposer_id: str(attempt_paths["round_root"] / "proposer_outputs" / f"{proposer_id}.json")
            for proposer_id in active_proposer_ids
        }
        attempts.append(attempt_record)

        if _reviewer_output_requires_manual_block(review):
            return _structured_recovery_blocked_step_result(
                step,
                attempts=attempts,
                consensus=review_consensus,
            )

        status = str(review_consensus.get("status", "unresolved"))
        retryable_status = status in RETRYABLE_CONSENSUS_STATUSES
        retry_budget_available = attempt_number < max_attempts

        if status == "all_agree":
            source_discrepancy_block = _source_discrepancy_blocked_step_result(
                step,
                attempts=attempts,
                consensus=review_consensus,
            )
            if source_discrepancy_block is not None:
                return source_discrepancy_block
            accepted_result = review_consensus.get("accepted_result")
            return {
                "step_id": step.step_id,
                "kind": step.kind,
                "status": "accepted",
                "attempts": attempts,
                "accepted_output": accepted_result
                if accepted_result is not None
                else {"proposer_outputs": proposer_outputs},
                "blocked_output": None,
                "reviewer_consensus": review_consensus,
            }

        if retryable_status and retry_budget_available:
            retry_feedback.append(
                _retry_feedback_record(
                    review,
                    review_consensus,
                    attempt_number=attempt_number,
                )
            )
            if status == "two_agree":
                next_active = _next_active_for_two_agree(review_consensus, all_proposer_ids)
                if next_active is not None:
                    agreed_ids = _valid_ids(review_consensus.get("agreed_proposer_ids", []), all_proposer_ids)
                    for proposer_id in agreed_ids:
                        if proposer_id in proposer_outputs:
                            locked_outputs[proposer_id] = proposer_outputs[proposer_id]
                    active_proposer_ids = next_active
                    continue

            active_proposer_ids = list(all_proposer_ids)
            locked_outputs = {}
            continue

        gated_block = _human_gate_blocked_step_result(
            config,
            step,
            attempts=attempts,
            consensus=review_consensus,
            trigger_status=status,
        )
        if gated_block is not None:
            return gated_block

        if status == "reference_disagrees":
            return _reference_disagrees_step_result(
                step,
                attempts=attempts,
                consensus=review_consensus,
            )

        if retryable_status and attempt_number >= max_attempts:
            return {
                "step_id": step.step_id,
                "kind": step.kind,
                "status": "blocked_for_user",
                "attempts": attempts,
                "accepted_output": None,
                "blocked_output": {
                    "analysis": str(review_consensus.get("analysis", "")),
                    "last_consensus": review_consensus,
                },
                "reviewer_consensus": review_consensus,
            }

        if status == "two_agree":
            next_active = _next_active_for_two_agree(review_consensus, all_proposer_ids)
            if next_active is not None:
                agreed_ids = _valid_ids(review_consensus.get("agreed_proposer_ids", []), all_proposer_ids)
                for proposer_id in agreed_ids:
                    if proposer_id in proposer_outputs:
                        locked_outputs[proposer_id] = proposer_outputs[proposer_id]
                active_proposer_ids = next_active
                continue

        active_proposer_ids = list(all_proposer_ids)
        locked_outputs = {}

    raise AssertionError("unreachable calculation loop exit")


def _batch_warnings_summary(batch_result: Mapping[str, Any]) -> dict[str, Any]:
    summary = batch_result.get("warnings_summary")
    return dict(summary) if isinstance(summary, Mapping) else {
        "structured_output_warning_count": 0,
        "structured_output_warnings_path": "",
        "cache_warning_count": 0,
        "cache_warnings_path": "",
    }


def _aggregate_warnings_summary(step_results: list[dict[str, Any]]) -> dict[str, Any]:
    structured_count = 0
    cache_count = 0
    structured_paths: list[str] = []
    cache_paths: list[str] = []
    for step in step_results:
        for attempt in step.get("attempts", []):
            if not isinstance(attempt, Mapping):
                continue
            summary = attempt.get("warnings_summary")
            if not isinstance(summary, Mapping):
                continue
            structured_count += int(summary.get("structured_output_warning_count") or 0)
            cache_count += int(summary.get("cache_warning_count") or 0)
            if path := str(summary.get("structured_output_warnings_path") or ""):
                structured_paths.append(path)
            if path := str(summary.get("cache_warnings_path") or ""):
                cache_paths.append(path)
    return {
        "structured_output_warning_count": structured_count,
        "structured_output_warnings_paths": sorted(set(structured_paths)),
        "cache_warning_count": cache_count,
        "cache_warnings_paths": sorted(set(cache_paths)),
    }


def _attempt_batch_config(
    config: CalculateConfig,
    step: CalculateStep,
    *,
    attempt_number: int,
    active_proposer_ids: list[str],
    locked_outputs: dict[str, Any],
    retry_feedback: list[dict[str, Any]],
    run_root: Path,
    accepted_step_outputs: Mapping[str, Any],
) -> dict[str, Any]:
    attempt_id = f"{step.step_id}_attempt_{attempt_number:03d}"
    selectable_proposer_ids = list(
        dict.fromkeys([*active_proposer_ids, *[proposer_id for proposer_id in locked_outputs]])
    )
    caller_context = _caller_context(
        config,
        step,
        attempt_number=attempt_number,
        active_proposer_ids=active_proposer_ids,
        locked_outputs=locked_outputs,
        retry_feedback=retry_feedback,
        accepted_step_outputs=accepted_step_outputs,
    )
    return {
        "schema_version": "arc.llm.proposers_reviewer_batch.config.v1",
        "run_id": attempt_id,
        "run_dir": str(run_root / "attempt_batches"),
        "max_concurrent_loops": 1,
        "session": {
            "policy": "stateful",
            "history_mode": "delta",
            "scope_id": f"calculate/{config.run_id}/{step.step_id}",
            "reuse_across_batch_calls": True,
            "max_concurrent_same_prefix": 4,
            "root": str(run_root / "llm_sessions"),
        },
        "artifact_options": {"save_prompts": config.artifact_options.get("save_prompts", True)},
        "output_recovery": _relaxed_output_recovery_config(),
        "defaults": copy.deepcopy(config.defaults),
        "loops": [
            {
                "loop_id": attempt_id,
                "max_rounds": 1,
                "early_stop": {"enabled": False},
                "caller_context": caller_context,
                "cache_context": {
                    "static_caller_context_keys": [
                        "step_id",
                        "step_kind",
                        "step_prompt",
                        "allowed_context",
                        "accepted_prior_step_outputs",
                        "max_recalculations",
                        "integrity_reference",
                        "consensus_instruction",
                    ],
                    "volatile_caller_context_keys": [
                        "attempt_number",
                        "active_proposer_ids",
                        "locked_outputs",
                        "retry_feedback",
                    ],
                },
                "proposers": [
                    _proposer_config(
                        config,
                        proposer_id,
                        runtime=_proposer_runtime(config, step),
                    )
                    for proposer_id in active_proposer_ids
                ],
                "reviewers": [
                    _reviewer_config(
                        config,
                        active_proposer_ids,
                        selectable_proposer_ids,
                        reviewer_reference_claim=step.reviewer_reference_claim,
                        human_gate=config.human_gate,
                    )
                ],
            }
        ],
    }


def _caller_context(
    config: CalculateConfig,
    step: CalculateStep,
    *,
    attempt_number: int,
    active_proposer_ids: list[str],
    locked_outputs: dict[str, Any],
    retry_feedback: list[dict[str, Any]],
    accepted_step_outputs: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "step_id": step.step_id,
        "step_kind": step.kind,
        "step_prompt": step.prompt,
        "allowed_context": _sanitize_caller_allowed_context(step.allowed_context),
        "attempt_number": attempt_number,
        "active_proposer_ids": active_proposer_ids,
        "locked_outputs": copy.deepcopy(locked_outputs),
        "retry_feedback": copy.deepcopy(retry_feedback),
        "accepted_prior_step_outputs": copy.deepcopy(dict(accepted_step_outputs)),
        "max_recalculations": config.max_recalculations,
        "integrity_reference": _integrity_reference(config.defaults.get("integrity_reference_path")),
        "consensus_instruction": "Work only on this calculation step. Respect accepted_prior_step_outputs and locked_outputs as already accepted unless explicitly asked to check them.",
    }


def _proposer_config(config: CalculateConfig, proposer_id: str, *, runtime: Mapping[str, Any]) -> dict[str, Any]:
    payload = _read_template(config.workflow_json_dir / "calculate-proposer.template.json")
    payload["id"] = proposer_id
    prompt = _dict(payload.get("prompt"), "calculate-proposer.template.prompt")
    prompt["template"] = str(prompt.get("template", "")).replace("{source_policy}", _proposer_source_policy(runtime))
    payload["prompt"] = prompt
    payload["runtime"] = dict(runtime)
    return payload


def _reviewer_config(
    config: CalculateConfig,
    active_proposer_ids: list[str],
    selectable_proposer_ids: list[str],
    *,
    reviewer_reference_claim: Mapping[str, Any] | None = None,
    human_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _read_template(config.workflow_json_dir / "calculate-reviewer.template.json")
    prompt = _dict(payload.get("prompt"), "calculate-reviewer.template.prompt")
    replacements = {
        "{active_proposer_ids}": ", ".join(active_proposer_ids),
        "{reviewer_status_instruction}": _reviewer_status_instruction(
            allow_reference_disagrees=bool(reviewer_reference_claim)
        ),
        "{reference_instruction}": _reviewer_reference_instruction(
            reviewer_reference_claim,
            active_proposer_ids=active_proposer_ids,
        ),
        "{workflow_instruction}": _reviewer_workflow_instruction(human_gate or {}),
    }
    template = str(prompt.get("template", ""))
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    prompt["template"] = template
    payload["prompt"] = prompt
    payload["output_schema"] = _reviewer_output_schema(
        config,
        active_proposer_ids,
        selectable_proposer_ids,
        allow_reference_disagrees=bool(reviewer_reference_claim),
    )
    return payload


def _proposer_runtime(config: CalculateConfig, step: CalculateStep) -> dict[str, Any]:
    if step.reviewer_reference_claim:
        runtime = {
            "allow_internet": False,
            "allow_mcp": False,
            "codex_sandbox": "read-only",
        }
    elif step.kind == "new_calculation":
        runtime = {
            "allow_internet": True,
            "allow_mcp": False,
            "codex_sandbox": "read-only",
        }
    else:
        runtime = {
            "allow_internet": False,
            "allow_mcp": False,
            "codex_sandbox": "read-only",
        }
    runtime.update(_dict(config.defaults.get("proposer_runtime", {}), "defaults.proposer_runtime"))
    runtime.update(step.proposer_runtime)
    return runtime


def _proposer_source_policy(runtime: Mapping[str, Any]) -> str:
    allow_mcp = _bool_default(runtime.get("allow_mcp", False), False)
    allow_internet = _bool_default(runtime.get("allow_internet", False), False)
    if not allow_mcp and not allow_internet:
        return (
            "Do not use internet search. Do not use ARC paper MCP tools. "
            "Do not read paper source sections, arXiv pages, INSPIRE pages, "
            "cached paper text, or any external source. Use only the supplied "
            "caller_context, accepted locked_outputs, and your own local algebra. "
            "Do not use validation-only final formulas as derivation inputs."
        )
    parts = []
    if allow_mcp:
        parts.append(
            "You may use ARC paper MCP tools only to read the main reference and cited "
            "sections explicitly named in caller_context."
        )
    else:
        parts.append("Do not use ARC paper MCP tools or cached paper text.")
    if allow_internet:
        parts.append("Internet search is allowed only for source discovery or uncached paper access.")
    else:
        parts.append("Do not use internet search.")
    parts.append("Cite any paper tool or internet source you use.")
    parts.append("Do not use validation-only final formulas as derivation inputs.")
    return " ".join(parts)


def _reviewer_reference_instruction(
    reviewer_reference_claim: Mapping[str, Any] | None,
    *,
    active_proposer_ids: list[str],
) -> str:
    if not reviewer_reference_claim:
        return ""
    claim_json = json.dumps(reviewer_reference_claim, indent=2, ensure_ascii=False, sort_keys=True)
    first = active_proposer_ids[0] if len(active_proposer_ids) >= 1 else "proposer_001"
    second = active_proposer_ids[1] if len(active_proposer_ids) >= 2 else "proposer_002"
    return (
        "Reviewer-only blind reference check is active. Do not reveal the reference claim "
        "to proposers through proposer_messages. Compare the final result from "
        f"{first}, the final result from {second}, and reviewer_reference_claim. "
        "When blind proposers and the reference agree, set status=all_agree. When blind "
        "proposers agree with each other but disagree with the reference claim, "
        "set status=reference_disagrees, set agreed_proposer_ids to the agreeing proposer ids, "
        "put the blind proposer result in accepted_result with reference_claim_status='disagrees', "
        "set agreement_assessment.accepted_by_reviewer_judgment=false, and set "
        "one or more agreement_assessment match fields false according to the mismatch. "
        "Then set workflow_action according to the workflow instruction below. "
        "If blind proposers disagree, do not accept the reference claim merely because one proposer matches it; "
        "set status=unresolved or all_disagree and request recalculation.\n\n"
        f"reviewer_reference_claim:\n{claim_json}"
    )


def _reviewer_status_instruction(*, allow_reference_disagrees: bool) -> str:
    statuses = ["all_agree", "two_agree", "all_disagree", "unresolved"]
    if allow_reference_disagrees:
        statuses.append("reference_disagrees")
    status_text = ", ".join(statuses[:-1]) + f", or {statuses[-1]}"
    return f"set status to {status_text}."


def _reviewer_workflow_instruction(human_gate: Mapping[str, Any]) -> str:
    if not _bool_default(human_gate.get("enabled", False), False):
        return (
            "workflow_action is still required. In normal mode, choose continue for "
            "all_agree and reference_disagrees when legacy acceptance applies; for other "
            "statuses, choose retry or pause_for_human with a concise expert_question."
        )
    pause_statuses = ", ".join(_human_gate_pause_statuses_from_mapping(human_gate))
    return (
        "Human gate is active. Statuses that trigger a stop: "
        f"{pause_statuses}. When a stop is triggered, workflow_action decides whether "
        "the main agent should ask the human expert or revise project artifacts. Use "
        "pause_for_human with requires_human=true unless all proposers' assessments and "
        "your review agree on the same work-note or plan revision. Only then use "
        "revise_plan or split_step with requires_human=false."
    )


def _reviewer_output_schema(
    config: CalculateConfig,
    active_proposer_ids: list[str],
    selectable_proposer_ids: list[str] | None = None,
    *,
    allow_reference_disagrees: bool = False,
) -> dict[str, Any]:
    if selectable_proposer_ids is None:
        selectable_proposer_ids = active_proposer_ids
    schema = _read_template(config.workflow_json_dir / "calculate-reviewer-output.schema.json")
    status_values = ["all_agree", "two_agree", "all_disagree", "unresolved"]
    if allow_reference_disagrees:
        status_values.append("reference_disagrees")
    proposer_message_properties = {
        proposer_id: {
            "type": "object",
            "required": ["message"],
            "properties": {"message": {"type": "string"}},
            "additionalProperties": False,
        }
        for proposer_id in active_proposer_ids
    }
    schema["properties"]["proposer_messages"]["required"] = active_proposer_ids
    schema["properties"]["proposer_messages"]["properties"] = proposer_message_properties
    consensus_properties = schema["properties"]["review_payload"]["properties"]["consensus"]["properties"]
    consensus_properties["status"]["enum"] = status_values
    for field in ["agreed_proposer_ids", "likely_wrong_proposer_ids", "recalculate_proposer_ids"]:
        consensus_properties[field]["items"]["type"] = "string"
        consensus_properties[field]["items"]["enum"] = active_proposer_ids
    consensus_properties["best_written_proposer_id"]["anyOf"] = [
        {"type": "string", "enum": selectable_proposer_ids},
        {"type": "null"},
    ]
    return schema


def _failed_step_result(
    config: CalculateConfig,
    step: CalculateStep,
    *,
    attempts: list[dict[str, Any]],
    error: str,
) -> dict[str, Any]:
    gated_block = _human_gate_blocked_step_result(
        config,
        step,
        attempts=attempts,
        consensus={
            "status": "failed",
            "analysis": error,
            "workflow_action": _default_workflow_action("failed", error),
        },
        trigger_status="failed",
        error=error,
    )
    if gated_block is not None:
        return gated_block
    return {
        "step_id": step.step_id,
        "kind": step.kind,
        "status": "failed",
        "attempts": attempts,
        "accepted_output": None,
        "blocked_output": None,
        "reviewer_consensus": None,
        "error": error,
    }


def _human_gate_blocked_step_result(
    config: CalculateConfig,
    step: CalculateStep,
    *,
    attempts: list[dict[str, Any]],
    consensus: Mapping[str, Any],
    trigger_status: str,
    error: str | None = None,
) -> dict[str, Any] | None:
    if not _human_gate_enabled(config):
        return None
    if trigger_status not in _human_gate_pause_statuses(config):
        return None

    workflow_action = _normalized_workflow_action(consensus.get("workflow_action"), trigger_status)
    requires_human = _workflow_action_requires_human(workflow_action, allow_nonhuman_control=True)
    if requires_human:
        workflow_action = copy.deepcopy(workflow_action)
        workflow_action["action"] = "pause_for_human"
        workflow_action["requires_human"] = True
    step_status = "blocked_for_user" if requires_human else "blocked_for_revision"
    expert_question = str(workflow_action.get("expert_question", "")).strip()
    if not expert_question:
        expert_question = _default_expert_question(trigger_status, workflow_action)
    blocked_output = {
        "reason": "human_gate",
        "trigger_status": trigger_status,
        "requires_human": requires_human,
        "workflow_action": workflow_action,
        "expert_question": expert_question,
        "analysis": str(consensus.get("analysis", "")),
        "last_consensus": copy.deepcopy(dict(consensus)),
    }
    if error:
        blocked_output["error"] = error
    return {
        "step_id": step.step_id,
        "kind": step.kind,
        "status": step_status,
        "attempts": attempts,
        "accepted_output": None,
        "blocked_output": blocked_output,
        "reviewer_consensus": dict(consensus),
        "error": error,
    }


def _structured_recovery_blocked_step_result(
    step: CalculateStep,
    *,
    attempts: list[dict[str, Any]],
    consensus: Mapping[str, Any],
) -> dict[str, Any]:
    workflow_action = _normalized_workflow_action(
        {
            "action": "pause_for_human",
            "requires_human": True,
            "issue_type": "worker_failure",
            "proposed_revision": None,
            "reason": "Reviewer output required major structured-output recovery; automatic retry is disabled.",
            "expert_question": (
                "The reviewer returned malformed or unstructured output. "
                "Inspect the reviewer raw text and decide whether to accept, revise, or rerun."
            ),
        },
        "unresolved",
    )
    return {
        "step_id": step.step_id,
        "kind": step.kind,
        "status": "blocked_for_user",
        "attempts": attempts,
        "accepted_output": None,
        "blocked_output": {
            "reason": "reviewer_structured_output_recovery",
            "trigger_status": "unresolved",
            "requires_human": True,
            "workflow_action": workflow_action,
            "expert_question": workflow_action["expert_question"],
            "analysis": str(consensus.get("analysis", "")),
            "last_consensus": copy.deepcopy(dict(consensus)),
        },
        "reviewer_consensus": dict(consensus),
    }


def _attempt_paths(batch_config: Mapping[str, Any]) -> dict[str, Path]:
    run_root = Path(str(batch_config["run_dir"])) / str(batch_config["run_id"])
    loop_id = str(batch_config["loops"][0]["loop_id"])
    round_root = run_root / "loops" / loop_id / "rounds" / "round_001"
    return {
        "run_root": run_root,
        "round_root": round_root,
        "review_path": round_root / "reviews" / "reviewer_001.json",
    }


def _review_consensus(
    review: Mapping[str, Any],
    *,
    active_proposer_ids: list[str],
    selectable_proposer_ids: list[str] | None = None,
    reviewer_reference_claim: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if selectable_proposer_ids is None:
        selectable_proposer_ids = active_proposer_ids
    if review.get("schema_version") != REVIEW_ENVELOPE_SCHEMA:
        raise ValueError(f"review schema_version must be {REVIEW_ENVELOPE_SCHEMA}")
    payload = review.get("review_payload")
    if not isinstance(payload, dict):
        raise ValueError("review.review_payload must be an object")
    consensus = payload.get("consensus")
    if not isinstance(consensus, dict):
        raise ValueError("review.review_payload.consensus must be an object")
    status = consensus.get("status")
    allowed_statuses = {"all_agree", "two_agree", "all_disagree", "unresolved"}
    if reviewer_reference_claim:
        allowed_statuses.add("reference_disagrees")
    if status not in allowed_statuses:
        message = "consensus.status must be all_agree, two_agree, all_disagree, or unresolved"
        if reviewer_reference_claim:
            message += ", or reference_disagrees"
        raise ValueError(message)
    consensus = dict(consensus)
    _validate_source_discrepancies(consensus)
    consensus["workflow_action"] = _normalized_workflow_action(consensus.get("workflow_action"), str(status))
    if _structured_recovery_severity(review) in {"major", "fatal"}:
        consensus = _force_unresolved_after_recovered_review(consensus, active_proposer_ids=active_proposer_ids)
        status = "unresolved"
    if status == "all_agree":
        _validate_best_written_selection(
            consensus,
            active_proposer_ids=active_proposer_ids,
            selectable_proposer_ids=selectable_proposer_ids,
        )
        _validate_all_agree_agreement_assessment(consensus)
    if status == "reference_disagrees":
        _validate_best_written_selection(
            consensus,
            active_proposer_ids=active_proposer_ids,
            selectable_proposer_ids=selectable_proposer_ids,
        )
        _validate_reference_disagrees_agreement_assessment(
            consensus,
            active_proposer_ids=active_proposer_ids,
        )
    return consensus


def _structured_recovery_severity(payload: Mapping[str, Any]) -> str:
    record = payload.get("arc_llm_call_record")
    if not isinstance(record, Mapping):
        return "none"
    structured = record.get("structured_output")
    if not isinstance(structured, Mapping):
        return "none"
    return str(structured.get("severity") or "none")


def _reviewer_output_requires_manual_block(review: Mapping[str, Any]) -> bool:
    return _structured_recovery_severity(review) in {"major", "fatal"}


def _force_unresolved_after_recovered_review(
    consensus: Mapping[str, Any],
    *,
    active_proposer_ids: list[str],
) -> dict[str, Any]:
    result = dict(consensus)
    analysis = str(result.get("analysis", "")).strip()
    warning = (
        "ARC warning: reviewer output required major structured-output recovery; "
        "forcing unresolved/manual inspection instead of accepting."
    )
    result.update(
        {
            "status": "unresolved",
            "accepted_result": None,
            "agreed_proposer_ids": [],
            "likely_wrong_proposer_ids": list(active_proposer_ids),
            "recalculate_proposer_ids": list(active_proposer_ids),
            "analysis": f"{analysis}\n\n{warning}".strip(),
            "best_written_proposer_id": None,
            "best_written_selection_reason": "",
            "workflow_action": _normalized_workflow_action(
                {
                    "action": "pause_for_human",
                    "requires_human": True,
                    "issue_type": "worker_failure",
                    "proposed_revision": None,
                    "reason": "Reviewer output required major structured-output recovery; automatic retry is disabled.",
                    "expert_question": "Inspect the reviewer raw text and decide whether to accept, revise, or rerun.",
                },
                "unresolved",
            ),
        }
    )
    return result


def _validate_best_written_selection(
    consensus: Mapping[str, Any],
    *,
    active_proposer_ids: list[str],
    selectable_proposer_ids: list[str],
) -> None:
    best_written = consensus.get("best_written_proposer_id")
    if not isinstance(best_written, str) or not best_written.strip():
        raise ValueError("best_written_proposer_id is required for all_agree consensus")
    if best_written not in selectable_proposer_ids:
        raise ValueError("best_written_proposer_id must identify an active or locked proposer output")
    agreed_ids = _valid_ids(consensus.get("agreed_proposer_ids", []), active_proposer_ids)
    if best_written in active_proposer_ids and best_written not in agreed_ids:
        raise ValueError("best_written_proposer_id must be one of agreed_proposer_ids for all_agree consensus")
    reason = consensus.get("best_written_selection_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("best_written_selection_reason is required for all_agree consensus")


def _reference_disagrees_step_result(
    step: CalculateStep,
    *,
    attempts: list[dict[str, Any]],
    consensus: Mapping[str, Any],
) -> dict[str, Any]:
    workflow_action = _normalized_workflow_action(consensus.get("workflow_action"), "reference_disagrees")
    requires_human = _workflow_action_requires_human(workflow_action)
    step_status = "blocked_for_user" if requires_human else "blocked_for_revision"
    expert_question = str(workflow_action.get("expert_question", "")).strip()
    if not expert_question:
        expert_question = _default_expert_question("reference_disagrees", workflow_action)
    return {
        "step_id": step.step_id,
        "kind": step.kind,
        "status": step_status,
        "attempts": attempts,
        "accepted_output": None,
        "blocked_output": {
            "reason": "reference_disagrees",
            "trigger_status": "reference_disagrees",
            "requires_human": requires_human,
            "workflow_action": workflow_action,
            "expert_question": expert_question,
            "analysis": str(consensus.get("analysis", "")),
            "last_consensus": copy.deepcopy(dict(consensus)),
        },
        "reviewer_consensus": dict(consensus),
    }


def _agreement_assessment(
    consensus: Mapping[str, Any],
    *,
    status: str,
    reject_special_limit_only: bool = True,
) -> Mapping[str, Any]:
    assessment = consensus.get("agreement_assessment")
    if not isinstance(assessment, dict):
        raise ValueError(f"{status} requires review_payload.consensus.agreement_assessment")
    summary = assessment.get("comparison_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError(f"{status} requires agreement_assessment.comparison_summary")
    lowered_summary = summary.lower()
    if _has_weak_reliance_marker(lowered_summary):
        raise ValueError(
            f"{status} cannot rely on formatting, spacing, visual similarity, looks identical, or string equality"
        )
    if reject_special_limit_only and assessment.get("special_limit_only") is True:
        raise ValueError(f"{status} cannot accept agreement_assessment.special_limit_only=true")
    return assessment


def _has_weak_reliance_marker(lowered_summary: str) -> bool:
    weak_reliance_patterns = [
        r"\bby\s+visual\s+inspection\b",
        r"\bbased\s+on\s+visual\s+inspection\b",
        r"\brel(?:y|ies|ied|ying)\s+on\s+visual\s+inspection\b",
        r"\blooks?\s+identical\b",
        r"\bvisually\s+identical\b",
        r"\bstring[-\s]+equality\b",
        r"\bonly\s+spacing\b",
        r"\bonly\s+formatting\b",
        r"\bformatting\s+differences\b",
    ]
    for pattern in weak_reliance_patterns:
        for match in re.finditer(pattern, lowered_summary):
            if not _has_nearby_negation(lowered_summary, match.start()):
                return True
    return False


def _has_nearby_negation(lowered_summary: str, match_start: int) -> bool:
    prefix = lowered_summary[max(0, match_start - 32) : match_start]
    return re.search(r"\b(?:not|do\s+not|does\s+not|without|never)\b(?:\W+\w+){0,3}\W*$", prefix) is not None


def _validate_all_agree_agreement_assessment(consensus: Mapping[str, Any]) -> None:
    assessment = _agreement_assessment(consensus, status="all_agree")
    for field in [
        "target_quantity_match",
        "convention_match",
        "declared_scope_match",
        "agreement_covers_full_target",
        "accepted_by_reviewer_judgment",
    ]:
        if assessment.get(field) is not True:
            raise ValueError(f"all_agree requires agreement_assessment.{field}=true")


def _validate_reference_disagrees_agreement_assessment(
    consensus: Mapping[str, Any],
    *,
    active_proposer_ids: list[str],
) -> None:
    assessment = _agreement_assessment(
        consensus,
        status="reference_disagrees",
        reject_special_limit_only=False,
    )
    agreed_ids = _valid_ids(consensus.get("agreed_proposer_ids", []), active_proposer_ids)
    if len(set(agreed_ids)) < 2:
        raise ValueError("reference_disagrees requires two agreeing blind proposer ids")
    if assessment.get("accepted_by_reviewer_judgment") is not False:
        raise ValueError("reference_disagrees requires agreement_assessment.accepted_by_reviewer_judgment=false")
    mismatch_fields = [
        "target_quantity_match",
        "convention_match",
        "declared_scope_match",
        "agreement_covers_full_target",
    ]
    if not any(assessment.get(field) is False for field in mismatch_fields):
        raise ValueError(
            "reference_disagrees requires at least one agreement_assessment match field=false"
        )


def _sanitize_caller_allowed_context(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_caller_allowed_context(item)
            for key, item in value.items()
            if str(key) not in CALLER_ALLOWED_CONTEXT_OMIT_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_caller_allowed_context(item) for item in value]
    return copy.deepcopy(value)


def _integrity_reference(path_value: Any = None) -> dict[str, str]:
    path = _resolve_integrity_path(path_value)
    if path is None:
        raise FileNotFoundError("integrity.md was not found")
    return {"path": str(path), "content": path.read_text(encoding="utf-8")}


def _resolve_integrity_path(path_value: Any = None) -> Path | None:
    if path_value:
        requested = Path(str(path_value)).expanduser()
        candidates = [requested] if requested.is_absolute() else []
        if not requested.is_absolute():
            candidates.extend(root / requested for root in [Path.cwd(), *Path.cwd().parents])
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None
    path = Path(__file__).resolve().parents[2] / "rules/integrity.md"
    return path if path.exists() else None


def _read_proposer_outputs(round_root: Path, proposer_ids: list[str]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for proposer_id in proposer_ids:
        path = round_root / "proposer_outputs" / f"{proposer_id}.json"
        if path.exists():
            outputs[proposer_id] = _read_json(path)
    return outputs


def _retry_feedback_record(
    review: Mapping[str, Any],
    consensus: Mapping[str, Any],
    *,
    attempt_number: int,
) -> dict[str, Any]:
    proposer_messages = review.get("proposer_messages", {})
    if not isinstance(proposer_messages, dict):
        proposer_messages = {}
    return {
        "attempt_number": attempt_number,
        "status": str(consensus.get("status", "")),
        "analysis": str(consensus.get("analysis", "")),
        "likely_wrong_proposer_ids": copy.deepcopy(list(consensus.get("likely_wrong_proposer_ids", []))),
        "recalculate_proposer_ids": copy.deepcopy(list(consensus.get("recalculate_proposer_ids", []))),
        "proposer_messages": copy.deepcopy(proposer_messages),
    }


def _next_active_for_two_agree(consensus: Mapping[str, Any], all_proposer_ids: list[str]) -> list[str] | None:
    recalculate = _valid_ids(consensus.get("recalculate_proposer_ids", []), all_proposer_ids)
    likely_wrong = _valid_ids(consensus.get("likely_wrong_proposer_ids", []), all_proposer_ids)
    next_active = recalculate or likely_wrong
    if len(next_active) == 1:
        return next_active
    return None


def _human_gate_enabled(config: CalculateConfig) -> bool:
    return _bool_default(config.human_gate.get("enabled", False), False)


def _human_gate_pause_statuses(config: CalculateConfig) -> tuple[str, ...]:
    return _human_gate_pause_statuses_from_mapping(config.human_gate)


def _human_gate_pause_statuses_from_mapping(human_gate: Mapping[str, Any]) -> tuple[str, ...]:
    statuses = human_gate.get("pause_on_statuses", DEFAULT_HUMAN_GATE_PAUSE_STATUSES)
    if not isinstance(statuses, (list, tuple)):
        return DEFAULT_HUMAN_GATE_PAUSE_STATUSES
    return tuple(str(status) for status in statuses)


def _normalized_workflow_action(raw: Any, trigger_status: str) -> dict[str, Any]:
    default = _default_workflow_action(trigger_status)
    if not isinstance(raw, dict):
        return default

    allowed_actions = {"continue", "pause_for_human", "revise_plan", "split_step", "retry"}
    allowed_issue_types = {
        "none",
        "work_note_inadequate",
        "work_note_conflict",
        "plan_wrong",
        "step_too_coarse",
        "target_ambiguous",
        "source_mapping_error",
        "calculation_disagreement",
        "reference_disagreement",
        "source_discrepancy",
        "worker_failure",
        "other",
    }
    action = str(raw.get("action", default["action"])).strip()
    if action not in allowed_actions:
        action = default["action"]
    issue_type = str(raw.get("issue_type", default["issue_type"])).strip()
    if issue_type not in allowed_issue_types:
        issue_type = default["issue_type"]
    requires_human = _bool_default(
        raw.get("requires_human", default["requires_human"]),
        bool(default["requires_human"]),
    )

    normalized = copy.deepcopy(raw)
    normalized["action"] = action
    normalized["requires_human"] = requires_human
    normalized["issue_type"] = issue_type
    normalized["reason"] = str(raw.get("reason", default["reason"]) or default["reason"])
    normalized.setdefault("proposed_revision", None)
    normalized["expert_question"] = str(raw.get("expert_question", default["expert_question"]) or "")
    return normalized


def _normalized_source_discrepancy_item(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("source_discrepancies[] items must be objects")
    status = str(raw.get("status", "") or "").strip()
    if status not in SOURCE_DISCREPANCY_STATUSES:
        raise ValueError("source_discrepancies[].status is invalid")
    return {
        "item_id": str(raw.get("item_id", "") or f"source_discrepancy_{index + 1}"),
        "status": status,
        "source_claim": str(raw.get("source_claim", "") or ""),
        "derived_result": str(raw.get("derived_result", "") or ""),
        "confidence_reason": str(raw.get("confidence_reason", "") or ""),
        "reviewer_says_no_human_convention_choice_needed": _bool_default(
            raw.get("reviewer_says_no_human_convention_choice_needed", False),
            False,
        ),
        "decision_question": str(raw.get("decision_question", "") or ""),
    }


def _normalized_source_discrepancies(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("consensus.source_discrepancies must be an array")
    return [_normalized_source_discrepancy_item(item, index) for index, item in enumerate(raw)]


def _validate_source_discrepancies(consensus: dict[str, Any]) -> None:
    if "source_discrepancy" in consensus:
        raise ValueError("consensus.source_discrepancy is not supported; use source_discrepancies")
    source_discrepancies = _normalized_source_discrepancies(consensus.get("source_discrepancies"))
    for item in source_discrepancies:
        if not item["confidence_reason"].strip():
            raise ValueError("source_discrepancies[].confidence_reason is required")
        if item["status"] in {"likely_source_error", "ambiguous_convention"} and not item[
            "decision_question"
        ].strip():
            raise ValueError("source_discrepancies[].decision_question is required for unresolved items")
        if item["status"] == "confirmed_source_error" and not item[
            "reviewer_says_no_human_convention_choice_needed"
        ]:
            item["status"] = "likely_source_error"
            item["confidence_reason"] = (
                item["confidence_reason"].rstrip()
                + " Reviewer did not explicitly state that no human convention choice is needed."
            )
    consensus["source_discrepancies"] = source_discrepancies


def _source_discrepancy_blocked_step_result(
    step: CalculateStep,
    *,
    attempts: list[dict[str, Any]],
    consensus: Mapping[str, Any],
) -> dict[str, Any] | None:
    source_discrepancies = _normalized_source_discrepancies(consensus.get("source_discrepancies"))
    unresolved = [
        item for item in source_discrepancies if item["status"] in {"likely_source_error", "ambiguous_convention"}
    ]
    if not unresolved:
        return None
    questions = []
    for item in unresolved:
        question = item["decision_question"].strip()
        if not question:
            source_claim = item["source_claim"].strip() or "the source claim"
            derived_result = item["derived_result"].strip() or "the derived result"
            question = f"Should ARC treat {source_claim} or {derived_result} as the premise?"
        questions.append(f"- {item['item_id']}: {question}")
    expert_question = (
        "Accepted derivation has source discrepancies that need human resolution before "
        f"step `{step.step_id}` can become an accepted premise:\n" + "\n".join(questions)
    )
    workflow_action = {
        "action": "pause_for_human",
        "requires_human": True,
        "issue_type": "source_discrepancy",
        "proposed_revision": None,
        "reason": "accepted result has non-confirmed source discrepancy",
        "expert_question": expert_question,
    }
    return {
        "step_id": step.step_id,
        "kind": step.kind,
        "status": "blocked_for_user",
        "attempts": attempts,
        "accepted_output": None,
        "blocked_output": {
            "reason": "source_discrepancy_requires_human",
            "trigger_status": "all_agree",
            "requires_human": True,
            "workflow_action": workflow_action,
            "expert_question": expert_question,
            "source_discrepancies": copy.deepcopy(unresolved),
            "analysis": str(consensus.get("analysis", "")),
            "last_consensus": copy.deepcopy(dict(consensus)),
        },
        "reviewer_consensus": dict(consensus),
    }


def _default_workflow_action(trigger_status: str, reason: str | None = None) -> dict[str, Any]:
    issue_type_by_status = {
        "all_agree": "none",
        "reference_disagrees": "reference_disagreement",
        "two_agree": "calculation_disagreement",
        "all_disagree": "calculation_disagreement",
        "unresolved": "calculation_disagreement",
        "failed": "worker_failure",
    }
    if trigger_status == "all_agree":
        return {
            "action": "continue",
            "requires_human": False,
            "issue_type": "none",
            "proposed_revision": None,
            "reason": reason or "reviewer accepted all_agree consensus",
            "expert_question": "",
        }
    issue_type = issue_type_by_status.get(trigger_status, "other")
    return {
        "action": "pause_for_human",
        "requires_human": True,
        "issue_type": issue_type,
        "proposed_revision": None,
        "reason": reason or f"consensus status {trigger_status} requires expert decision",
        "expert_question": _default_expert_question(
            trigger_status,
            {"issue_type": issue_type},
        ),
    }


def _workflow_action_requires_human(
    workflow_action: Mapping[str, Any],
    *,
    allow_nonhuman_control: bool = False,
) -> bool:
    action = str(workflow_action.get("action", "")).strip()
    nonhuman_actions = set(REVISION_ACTIONS)
    if allow_nonhuman_control:
        nonhuman_actions.update({"continue", "retry"})
    if action in nonhuman_actions and workflow_action.get("requires_human") is False:
        return False
    return True


def _default_expert_question(trigger_status: str, workflow_action: Mapping[str, Any]) -> str:
    issue_type = str(workflow_action.get("issue_type", "other")).strip() or "other"
    if trigger_status == "reference_disagrees":
        return "Blind derivation disagrees with the note/reference claim. Which formula or premise should ARC use next?"
    if trigger_status == "failed":
        return "A worker or validation failure stopped this step. Should ARC retry, revise the work note or plan, or use a corrected premise?"
    return (
        "Proposers did not reach accepted consensus "
        f"({trigger_status}, {issue_type}). What correction, premise, work-note revision, or plan revision should ARC use?"
    )


def _valid_ids(raw_ids: Any, all_proposer_ids: list[str]) -> list[str]:
    if not isinstance(raw_ids, list):
        return []
    allowed = set(all_proposer_ids)
    return [str(item) for item in raw_ids if str(item) in allowed]


def _proposer_ids(count: int) -> list[str]:
    return [f"proposer_{index:03d}" for index in range(1, count + 1)]


def _dry_run_result(config: CalculateConfig, paths: RunPaths) -> dict[str, Any]:
    return {
        "schema_version": CALCULATE_RESULT_SCHEMA,
        "status": "dry_run",
        "run_id": config.run_id,
        "run_root": str(paths.run_root),
        "proposer_count": config.proposer_count,
        "max_recalculations": config.max_recalculations,
        "human_gate": copy.deepcopy(config.human_gate),
        "steps": [{"step_id": step.step_id, "kind": step.kind} for step in config.steps],
    }


def _read_template(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    return copy.deepcopy(payload)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _default_workflow_json_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "json"


def _required_text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None:
        raise ConfigError(f"{key} is required")
    text = str(value).strip()
    if not text:
        raise ConfigError(f"{key} is required")
    return text


def _safe_id(value: str, field_name: str) -> str:
    if not SAFE_ID_RE.match(value):
        raise ConfigError(f"{field_name} must match {SAFE_ID_RE.pattern}")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ConfigError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ConfigError(f"{field_name} must be a positive integer")
    return parsed


def _nonnegative_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ConfigError(f"{field_name} must be a nonnegative integer") from exc
    if parsed < 0:
        raise ConfigError(f"{field_name} must be a nonnegative integer")
    return parsed


def _dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{field_name} must be an object")
    return copy.deepcopy(value)


def _bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    raise ConfigError(f"{field_name} must be a boolean")


def _bool_default(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return default


def _optional_dict(value: Any, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _dict(value, field_name)


def _parse_human_gate(value: Any) -> dict[str, Any]:
    data = _dict(value, "human_gate")
    enabled = _bool(data.get("enabled", False), "human_gate.enabled")
    pause_statuses = data.get("pause_on_statuses", DEFAULT_HUMAN_GATE_PAUSE_STATUSES)
    if not isinstance(pause_statuses, (list, tuple)):
        raise ConfigError("human_gate.pause_on_statuses must be a list")
    return {
        "enabled": enabled,
        "pause_on_statuses": [str(item) for item in pause_statuses],
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARC calculate workflow runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    result = run_calculation(
        _read_json(Path(args.config)),
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
