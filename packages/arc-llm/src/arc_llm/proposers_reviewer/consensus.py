from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifacts import RunPaths, atomic_write_json
from .config import ConfigError, SAFE_ID_RE
from .runner import JsonRunner, run_proposers_reviewer_batch


CONSENSUS_CONFIG_SCHEMA = "arc.llm.proposers_reviewer_consensus.config.v1"
CONSENSUS_RESULT_SCHEMA = "arc.llm.proposers_reviewer_consensus.result.v1"
REVIEW_ENVELOPE_SCHEMA = "arc.llm.review_envelope.v1"
DEFAULT_HUMAN_GATE_PAUSE_STATUSES = (
    "reference_disagrees",
    "two_agree",
    "all_disagree",
    "unresolved",
    "failed",
)
REVISION_ACTIONS = {"revise_plan", "split_step"}
LEGACY_ALLOWED_CONTEXT_KEYS = {"foundation_file", "allowed_foundation", "target_equation_id"}

BatchRunner = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ConsensusStep:
    step_id: str
    prompt: str
    kind: str
    allowed_context: dict[str, Any]
    proposer_runtime: dict[str, Any]
    reviewer_reference_claim: dict[str, Any] | None


@dataclass(frozen=True)
class ConsensusConfig:
    schema_version: str
    run_id: str
    run_dir: Path
    proposer_count: int
    max_recalculations: int
    human_gate: dict[str, Any]
    defaults: dict[str, Any]
    artifact_options: dict[str, Any]
    steps: list[ConsensusStep]


def load_consensus_config(payload: Mapping[str, Any]) -> ConsensusConfig:
    data = copy.deepcopy(dict(payload))
    schema_version = _required_text(data, "schema_version")
    if schema_version != CONSENSUS_CONFIG_SCHEMA:
        raise ConfigError(f"schema_version must be {CONSENSUS_CONFIG_SCHEMA}")

    run_id = _safe_id(_required_text(data, "run_id"), "run_id")
    run_dir = Path(_required_text(data, "run_dir")).expanduser()
    proposer_count = _positive_int(data.get("proposer_count", 2), "proposer_count")
    max_recalculations = _nonnegative_int(data.get("max_recalculations", 2), "max_recalculations")
    human_gate = _parse_human_gate(data.get("human_gate", {}))
    defaults = _dict(data.get("defaults", {}), "defaults")
    if defaults.get("model") is not None and str(defaults.get("provider", "auto") or "auto") == "auto":
        raise ConfigError("defaults.model requires explicit provider")
    artifact_options = _dict(data.get("artifact_options", {"save_prompts": True}), "artifact_options")
    if "save_prompts" not in artifact_options:
        artifact_options["save_prompts"] = True
    if not isinstance(artifact_options.get("save_prompts"), bool):
        raise ConfigError("artifact_options.save_prompts must be a boolean")

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ConfigError("steps must be a non-empty list")

    steps: list[ConsensusStep] = []
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
            ConsensusStep(
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

    return ConsensusConfig(
        schema_version=schema_version,
        run_id=run_id,
        run_dir=run_dir,
        proposer_count=proposer_count,
        max_recalculations=max_recalculations,
        human_gate=human_gate,
        defaults=defaults,
        artifact_options=artifact_options,
        steps=steps,
    )


def run_proposers_reviewer_consensus(
    config: ConsensusConfig | Mapping[str, Any],
    *,
    batch_runner: BatchRunner | None = None,
    json_runner: JsonRunner | None = None,
    base_env: Mapping[str, str] | None = None,
    process_chain: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    consensus = config if isinstance(config, ConsensusConfig) else load_consensus_config(config)
    paths = RunPaths(run_dir=consensus.run_dir, run_id=consensus.run_id)
    if dry_run:
        return _dry_run_result(consensus, paths)

    paths.run_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths.config, _jsonable(consensus))

    runner = batch_runner or run_proposers_reviewer_batch
    step_results: list[dict[str, Any]] = []
    accepted_step_outputs: dict[str, Any] = {}
    overall_status = "completed"
    for step in consensus.steps:
        step_result = _run_consensus_step(
            consensus,
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
        "schema_version": CONSENSUS_RESULT_SCHEMA,
        "status": overall_status,
        "run_id": consensus.run_id,
        "run_root": str(paths.run_root),
        "proposer_count": consensus.proposer_count,
        "max_recalculations": consensus.max_recalculations,
        "human_gate": copy.deepcopy(consensus.human_gate),
        "steps": step_results,
    }
    atomic_write_json(paths.state, result)
    return result


def _run_consensus_step(
    config: ConsensusConfig,
    step: ConsensusStep,
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
            consensus = _review_consensus(
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
        attempt_record["consensus"] = consensus
        attempt_record["proposer_output_paths"] = {
            proposer_id: str(attempt_paths["round_root"] / "proposer_outputs" / f"{proposer_id}.json")
            for proposer_id in active_proposer_ids
        }
        attempts.append(attempt_record)

        status = str(consensus.get("status", "unresolved"))
        gated_block = _human_gate_blocked_step_result(
            config,
            step,
            attempts=attempts,
            consensus=consensus,
            trigger_status=status,
        )
        if gated_block is not None:
            return gated_block

        if status in {"all_agree", "reference_disagrees"}:
            accepted_result = consensus.get("accepted_result")
            return {
                "step_id": step.step_id,
                "kind": step.kind,
                "status": "accepted",
                "attempts": attempts,
                "accepted_output": accepted_result
                if accepted_result is not None
                else {"proposer_outputs": proposer_outputs},
                "blocked_output": None,
                "reviewer_consensus": consensus,
            }

        if attempt_number >= max_attempts:
            return {
                "step_id": step.step_id,
                "kind": step.kind,
                "status": "blocked_for_user",
                "attempts": attempts,
                "accepted_output": None,
                "blocked_output": {
                    "analysis": str(consensus.get("analysis", "")),
                    "last_consensus": consensus,
                },
                "reviewer_consensus": consensus,
            }

        if status == "two_agree":
            next_active = _next_active_for_two_agree(consensus, all_proposer_ids)
            if next_active is not None:
                agreed_ids = _valid_ids(consensus.get("agreed_proposer_ids", []), all_proposer_ids)
                for proposer_id in agreed_ids:
                    if proposer_id in proposer_outputs:
                        locked_outputs[proposer_id] = proposer_outputs[proposer_id]
                active_proposer_ids = next_active
                continue

        active_proposer_ids = list(all_proposer_ids)
        locked_outputs = {}

    raise AssertionError("unreachable consensus loop exit")


def _failed_step_result(
    config: ConsensusConfig,
    step: ConsensusStep,
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
    config: ConsensusConfig,
    step: ConsensusStep,
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
    requires_human = _workflow_action_requires_human(workflow_action)
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


def _attempt_batch_config(
    config: ConsensusConfig,
    step: ConsensusStep,
    *,
    attempt_number: int,
    active_proposer_ids: list[str],
    locked_outputs: dict[str, Any],
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
        accepted_step_outputs=accepted_step_outputs,
    )
    return {
        "schema_version": "arc.llm.proposers_reviewer_batch.config.v1",
        "run_id": attempt_id,
        "run_dir": str(run_root / "attempt_batches"),
        "max_concurrent_loops": 1,
        "artifact_options": {"save_prompts": bool(config.artifact_options.get("save_prompts", True))},
        "defaults": copy.deepcopy(config.defaults),
        "loops": [
            {
                "loop_id": attempt_id,
                "max_rounds": 1,
                "early_stop": {"enabled": False},
                "caller_context": caller_context,
                "proposers": [
                    _proposer_config(
                        proposer_id,
                        runtime=_proposer_runtime(config, step),
                    )
                    for proposer_id in active_proposer_ids
                ],
                "reviewers": [
                    _reviewer_config(
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
    config: ConsensusConfig,
    step: ConsensusStep,
    *,
    attempt_number: int,
    active_proposer_ids: list[str],
    locked_outputs: dict[str, Any],
    accepted_step_outputs: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_context = _sanitize_caller_allowed_context(step.allowed_context)
    context = {
        "step_id": step.step_id,
        "step_kind": step.kind,
        "step_prompt": step.prompt,
        "allowed_context": allowed_context,
        "attempt_number": attempt_number,
        "active_proposer_ids": active_proposer_ids,
        "locked_outputs": copy.deepcopy(locked_outputs),
        "accepted_prior_step_outputs": copy.deepcopy(dict(accepted_step_outputs)),
        "max_recalculations": config.max_recalculations,
        "integrity_reference": _integrity_reference(config.defaults.get("integrity_reference_path")),
        "consensus_instruction": "Work only on this calculation step. Respect accepted_prior_step_outputs and locked_outputs as already accepted unless explicitly asked to check them.",
    }
    return context


def _proposer_runtime(config: ConsensusConfig, step: ConsensusStep) -> dict[str, Any]:
    if step.reviewer_reference_claim:
        runtime = {
            "allow_internet": False,
            "allow_mcp": False,
            "codex_sandbox": "read-only",
        }
    elif step.kind == "new_calculation":
        runtime = {
            "allow_internet": True,
            "allow_mcp": True,
            "mcp_mode": "arc-only",
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
    if runtime.get("allow_mcp") and "mcp_mode" not in runtime:
        runtime["mcp_mode"] = "arc-only"
    return runtime


def _proposer_config(proposer_id: str, *, runtime: Mapping[str, Any]) -> dict[str, Any]:
    source_policy = _proposer_source_policy(runtime)
    return {
        "id": proposer_id,
        "prompt": {
            "system": "You are an independent theoretical-physics calculation proposer.",
            "template": (
                "First read and follow caller_context.integrity_reference.content. "
                "Use only caller_context.step_prompt, caller_context.allowed_context, "
                "accepted locked_outputs, "
                "caller_context.accepted_prior_step_outputs, and your own SymPy/local algebra. "
                f"{source_policy} Wolfram may be used only for algebraic "
                "verification. You must strictly derive from the supplied work note, allowed context, "
                "accepted prior step outputs, and locked_outputs. External sources may inspire methods, but do not "
                "directly use any result from papers or the internet unless it appears in "
                "the supplied work note, allowed context, accepted prior step outputs, or accepted locked_outputs. If you need an external "
                "identity or intermediate result, derive it here. External sources may use "
                "different conventions; map notation back to work-note conventions before "
                "using it. For coordinate transformations or relabeling, explicitly track "
                "which symbols are old coordinates and which are newly introduced symbols "
                "before substituting. Do the calculation very clearly step by step; never skip a step. "
                "Write all mathematical expressions in derivation, assumptions, and final_result "
                "as display-ready LaTeX inside Markdown math delimiters or as LaTeX strings in "
                "JSON fields; avoid ASCII-only math such as rho_2, eta_prime, or T_ab when a "
                "LaTeX form such as \\rho_2, \\eta', or T_{ab} is intended for the report. "
                "Return one JSON object with result_summary, derivation, assumptions, "
                "validity_scope, final_result, and work_note_assessment. "
                "In work_note_assessment, state whether the supplied work note or plan must "
                "change before this step can be checked; include needs_revision "
                "(boolean), issue_type, proposed_revision, rationale, and "
                "can_continue_without_revision. Use issue_type=none when no revision is "
                "needed. Use validity_scope for assumptions, "
                "conventions, limits, and unresolved dependencies; do not use date-like "
                "reliability fields. Put the final mathematical result in final_result using "
                "explicit symbols so a reviewer can compare it with other proposers. "
                "Do not coordinate with other proposers.\n\n"
                "{caller_context_json}"
            ),
        },
        "output_schema": _proposer_output_schema(),
        "runtime": dict(runtime),
    }


def _proposer_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "result_summary",
            "derivation",
            "assumptions",
            "validity_scope",
            "final_result",
            "work_note_assessment",
        ],
        "properties": {
            "result_summary": {"type": "string"},
            "derivation": {"type": "string"},
            "assumptions": {"type": ["string", "array", "object"]},
            "validity_scope": {"type": "string"},
            "final_result": {"type": ["object", "array", "string", "number", "boolean", "null"]},
            "work_note_assessment": {
                "type": "object",
                "required": [
                    "needs_revision",
                    "issue_type",
                    "proposed_revision",
                    "rationale",
                    "can_continue_without_revision",
                ],
                "properties": {
                    "needs_revision": {"type": "boolean"},
                    "issue_type": {
                        "enum": [
                            "none",
                            "work_note_inadequate",
                            "work_note_conflict",
                            "plan_wrong",
                            "step_too_coarse",
                            "target_ambiguous",
                            "source_mapping_error",
                            "human_needed",
                            "other",
                        ]
                    },
                    "proposed_revision": {
                        "type": ["object", "array", "string", "number", "boolean", "null"]
                    },
                    "rationale": {"type": "string"},
                    "can_continue_without_revision": {"type": "boolean"},
                },
                "additionalProperties": True,
            },
        },
        "additionalProperties": True,
    }


def _proposer_source_policy(runtime: Mapping[str, Any]) -> str:
    allow_mcp = bool(runtime.get("allow_mcp"))
    allow_internet = bool(runtime.get("allow_internet"))
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


def _reviewer_config(
    active_proposer_ids: list[str],
    selectable_proposer_ids: list[str],
    *,
    reviewer_reference_claim: Mapping[str, Any] | None = None,
    human_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reference_instruction = _reviewer_reference_instruction(
        reviewer_reference_claim,
        active_proposer_ids=active_proposer_ids,
    )
    workflow_instruction = _reviewer_workflow_instruction(human_gate or {})
    reviewer_status_instruction = _reviewer_status_instruction(
        allow_reference_disagrees=bool(reviewer_reference_claim)
    )
    return {
        "id": "reviewer_001",
        "prompt": {
            "system": "You are a skeptical theoretical-physics consensus reviewer.",
            "template": (
                "First read and follow caller_context.integrity_reference.content. "
                "Compare current_proposer_outputs_json for the current calculation step. "
                "Treat caller_context.accepted_prior_step_outputs as accepted context from "
                "earlier steps, not as current proposer outputs. "
                "Acceptance is physics and mathematics judgment. SymPy, Wolfram, explicit "
                "algebra, or numerical checks are optional tools when useful, not mandatory "
                "gates. Special limits are sanity checks, not proof, unless the target itself "
                "is limiting, asymptotic, or leading-order. "
                "Return exactly one arc.llm.review_envelope.v1 JSON object. The top-level object "
                "must contain schema_version, controller, proposer_messages, and review_payload. "
                f"proposer_messages must contain these keys: {', '.join(active_proposer_ids)}. "
                "In review_payload.consensus, "
                f"{reviewer_status_instruction} Include "
                "accepted_result, agreed_proposer_ids, likely_wrong_proposer_ids, "
                "recalculate_proposer_ids, validity_scope, analysis, and "
                "agreement_assessment. Also include best_written_proposer_id and "
                "best_written_selection_reason. When status is all_agree, choose "
                "best_written_proposer_id from agreed_proposer_ids or accepted "
                "caller_context.locked_outputs by clearest logic, most complete "
                "details, and best readability; this copy will be used "
                "for the full calculation appendix. When status is reference_disagrees, "
                "choose best_written_proposer_id from the agreeing blind proposer ids by "
                "the same clarity rule for report evidence if the step is later accepted. "
                "If the status is neither all_agree nor reference_disagrees, set "
                "best_written_proposer_id to null and explain why. "
                "Before marking all_agree, agreement_assessment must show target quantity, "
                "conventions, declared scope, and full target coverage all match, with a "
                "nonempty comparison_summary and accepted_by_reviewer_judgment=true. "
                "Never mark all_agree by string equality, spacing, formatting, visual "
                "similarity, or because outputs merely look identical. "
                "For coordinate transformations or relabeling, first apply the "
                "source-declared old/new variable definitions from caller_context "
                "and the step prompt, then compare metric components or scalar "
                "expressions. Do not infer a source typo from raw variable-name "
                "differences until the declared substitution has been checked. "
                "For reference claims written as proportionalities, approximations, "
                "or implicit relations, convert C into a testable relation such as "
                "a constant quotient, residual, or stated limiting condition when useful. "
                "If exactly one proposer is likely wrong, name only that proposer for "
                "recalculation; otherwise ask all proposers to recalculate. "
                "Also inspect each proposer's work_note_assessment. In "
                "workflow_action, choose continue for all_agree. For any failure to "
                "agree, target/source ambiguity, worker failure, or suspected bad "
                "premise, choose pause_for_human unless the proposers and your review "
                "agree on the same work-note or plan revision; then choose "
                "revise_plan or split_step with "
                "requires_human=false and include proposed_revision. For "
                "reference_disagrees, follow the "
                "human-gate instruction below. "
                f"{reference_instruction}\n\n"
                f"{workflow_instruction}\n\n"
                "{current_proposer_outputs_json}"
            ),
        },
        "output_schema": _reviewer_output_schema(
            active_proposer_ids,
            selectable_proposer_ids,
            allow_reference_disagrees=bool(reviewer_reference_claim),
        ),
        "runtime": {"allow_mcp": False, "codex_sandbox": "read-only"},
    }


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
        "and set workflow_action according to the workflow instruction below. "
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
    if not bool(human_gate.get("enabled", False)):
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
    active_proposer_ids: list[str],
    selectable_proposer_ids: list[str] | None = None,
    *,
    allow_reference_disagrees: bool = False,
) -> dict[str, Any]:
    if selectable_proposer_ids is None:
        selectable_proposer_ids = active_proposer_ids
    status_values = ["all_agree", "two_agree", "all_disagree", "unresolved"]
    if allow_reference_disagrees:
        status_values.append("reference_disagrees")
    proposer_message_properties = {
        proposer_id: {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "additionalProperties": True,
        }
        for proposer_id in active_proposer_ids
    }
    return {
        "type": "object",
        "required": ["schema_version", "controller", "proposer_messages", "review_payload"],
        "properties": {
            "schema_version": {"const": REVIEW_ENVELOPE_SCHEMA},
            "controller": {
                "type": "object",
                "required": ["message", "stop_requested"],
                "properties": {
                    "message": {"type": "string"},
                    "stop_requested": {"type": "boolean"},
                    "stop_reason": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "proposer_messages": {
                "type": "object",
                "required": active_proposer_ids,
                "properties": proposer_message_properties,
                "additionalProperties": True,
            },
            "review_payload": {
                "type": "object",
                "required": ["consensus"],
                "properties": {
                    "consensus": {
                        "type": "object",
                        "required": [
                            "status",
                            "accepted_result",
                            "agreed_proposer_ids",
                            "likely_wrong_proposer_ids",
                            "recalculate_proposer_ids",
                            "validity_scope",
                            "analysis",
                            "agreement_assessment",
                            "best_written_proposer_id",
                            "best_written_selection_reason",
                            "workflow_action",
                        ],
                        "properties": {
                            "status": {
                                "enum": status_values
                            },
                            "accepted_result": {"type": ["object", "array", "string", "number", "boolean", "null"]},
                            "agreed_proposer_ids": {
                                "type": "array",
                                "items": {"enum": active_proposer_ids},
                            },
                            "likely_wrong_proposer_ids": {
                                "type": "array",
                                "items": {"enum": active_proposer_ids},
                            },
                            "recalculate_proposer_ids": {
                                "type": "array",
                                "items": {"enum": active_proposer_ids},
                            },
                            "validity_scope": {"type": "string"},
                            "analysis": {"type": "string"},
                            "best_written_proposer_id": {
                                "anyOf": [
                                    {"enum": selectable_proposer_ids},
                                    {"type": "null"},
                                ]
                            },
                            "best_written_selection_reason": {"type": "string"},
                            "workflow_action": _workflow_action_schema(),
                            "agreement_assessment": {
                                "type": "object",
                                "required": [
                                    "target_quantity_match",
                                    "convention_match",
                                    "declared_scope_match",
                                    "agreement_covers_full_target",
                                    "comparison_summary",
                                    "accepted_by_reviewer_judgment",
                                ],
                                "properties": {
                                    "target_quantity_match": {"type": "boolean"},
                                    "convention_match": {"type": "boolean"},
                                    "declared_scope_match": {"type": "boolean"},
                                    "agreement_covers_full_target": {"type": "boolean"},
                                    "comparison_summary": {"type": "string", "minLength": 1},
                                    "accepted_by_reviewer_judgment": {"type": "boolean"},
                                    "tool_checks": {
                                        "type": "array",
                                        "items": {
                                            "type": ["object", "string", "number", "boolean", "null"]
                                        },
                                    },
                                    "sanity_checks": {
                                        "type": "array",
                                        "items": {
                                            "type": ["object", "string", "number", "boolean", "null"]
                                        },
                                    },
                                    "special_limit_only": {"type": "boolean"},
                                    "notes": {"type": "string"},
                                },
                                "additionalProperties": True,
                            },
                        },
                        "additionalProperties": True,
                    }
                },
                "additionalProperties": True,
            },
        },
        "additionalProperties": True,
    }


def _workflow_action_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["action", "requires_human", "issue_type", "reason"],
        "properties": {
            "action": {
                "enum": [
                    "continue",
                    "pause_for_human",
                    "revise_plan",
                    "split_step",
                    "retry",
                ]
            },
            "requires_human": {"type": "boolean"},
            "issue_type": {
                "enum": [
                    "none",
                    "work_note_inadequate",
                    "work_note_conflict",
                    "plan_wrong",
                    "step_too_coarse",
                    "target_ambiguous",
                    "source_mapping_error",
                    "calculation_disagreement",
                    "reference_disagreement",
                    "worker_failure",
                    "other",
                ]
            },
            "proposed_revision": {
                "type": ["object", "array", "string", "number", "boolean", "null"]
            },
            "reason": {"type": "string"},
            "expert_question": {"type": "string"},
        },
        "additionalProperties": True,
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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


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
    consensus["workflow_action"] = _normalized_workflow_action(consensus.get("workflow_action"), str(status))
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


def _agreement_assessment(consensus: Mapping[str, Any], *, status: str) -> Mapping[str, Any]:
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
    if assessment.get("special_limit_only") is True:
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
    true_fields = [
        "target_quantity_match",
        "convention_match",
        "declared_scope_match",
        "agreement_covers_full_target",
        "accepted_by_reviewer_judgment",
    ]
    for field in true_fields:
        if assessment.get(field) is not True:
            raise ValueError(f"all_agree requires agreement_assessment.{field}=true")


def _validate_reference_disagrees_agreement_assessment(
    consensus: Mapping[str, Any],
    *,
    active_proposer_ids: list[str],
) -> None:
    assessment = _agreement_assessment(consensus, status="reference_disagrees")
    agreed_ids = _valid_ids(consensus.get("agreed_proposer_ids", []), active_proposer_ids)
    if len(set(agreed_ids)) < 2:
        raise ValueError("reference_disagrees requires two agreeing blind proposer ids")
    for field in ["target_quantity_match", "convention_match"]:
        if assessment.get(field) is not True:
            raise ValueError(f"reference_disagrees requires agreement_assessment.{field}=true")


_CALLER_ALLOWED_CONTEXT_OMIT_KEYS = {"sources", "mcp", "cli", "cache_path", "source_path", "source_commands"}


def _sanitize_caller_allowed_context(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_caller_allowed_context(item)
            for key, item in value.items()
            if str(key) not in _CALLER_ALLOWED_CONTEXT_OMIT_KEYS
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
    return _default_integrity_path()


def _default_integrity_path() -> Path | None:
    for root in [Path.cwd(), *Path.cwd().parents]:
        candidate = root / "skills/arc/rules/integrity.md"
        if candidate.exists():
            return candidate
    return None


def _read_proposer_outputs(round_root: Path, proposer_ids: list[str]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for proposer_id in proposer_ids:
        path = round_root / "proposer_outputs" / f"{proposer_id}.json"
        if path.exists():
            outputs[proposer_id] = _read_json(path)
    return outputs


def _next_active_for_two_agree(consensus: Mapping[str, Any], all_proposer_ids: list[str]) -> list[str] | None:
    recalculate = _valid_ids(consensus.get("recalculate_proposer_ids", []), all_proposer_ids)
    likely_wrong = _valid_ids(consensus.get("likely_wrong_proposer_ids", []), all_proposer_ids)
    next_active = recalculate or likely_wrong
    if len(next_active) == 1:
        return next_active
    return None


def _human_gate_enabled(config: ConsensusConfig) -> bool:
    return bool(config.human_gate.get("enabled", False))


def _human_gate_pause_statuses(config: ConsensusConfig) -> tuple[str, ...]:
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
        "worker_failure",
        "other",
    }
    action = str(raw.get("action", default["action"])).strip()
    if action not in allowed_actions:
        action = default["action"]
    issue_type = str(raw.get("issue_type", default["issue_type"])).strip()
    if issue_type not in allowed_issue_types:
        issue_type = default["issue_type"]
    requires_human = raw.get("requires_human", default["requires_human"])
    if not isinstance(requires_human, bool):
        requires_human = bool(default["requires_human"])

    normalized = copy.deepcopy(raw)
    normalized["action"] = action
    normalized["requires_human"] = requires_human
    normalized["issue_type"] = issue_type
    normalized["reason"] = str(raw.get("reason", default["reason"]) or default["reason"])
    normalized.setdefault("proposed_revision", None)
    normalized["expert_question"] = str(raw.get("expert_question", default["expert_question"]) or "")
    return normalized


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


def _workflow_action_requires_human(workflow_action: Mapping[str, Any]) -> bool:
    action = str(workflow_action.get("action", "")).strip()
    if action in REVISION_ACTIONS and workflow_action.get("requires_human") is False:
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


def _dry_run_result(config: ConsensusConfig, paths: RunPaths) -> dict[str, Any]:
    return {
        "schema_version": CONSENSUS_RESULT_SCHEMA,
        "status": "dry_run",
        "run_id": config.run_id,
        "run_root": str(paths.run_root),
        "proposer_count": config.proposer_count,
        "max_recalculations": config.max_recalculations,
        "human_gate": copy.deepcopy(config.human_gate),
        "steps": [{"step_id": step.step_id, "kind": step.kind} for step in config.steps],
    }


def _required_text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None:
        raise ConfigError(f"{key} is required")
    text = str(value).strip()
    if not text:
        raise ConfigError(f"{key} is required")
    return text


def _parse_human_gate(value: Any) -> dict[str, Any]:
    raw = _dict(value, "human_gate")
    parsed: dict[str, Any] = {"enabled": _bool(raw.get("enabled", False), "human_gate.enabled")}

    raw_statuses = raw.get("pause_on_statuses", list(DEFAULT_HUMAN_GATE_PAUSE_STATUSES))
    if raw_statuses is None:
        raw_statuses = list(DEFAULT_HUMAN_GATE_PAUSE_STATUSES)
    if not isinstance(raw_statuses, list):
        raise ConfigError("human_gate.pause_on_statuses must be an array")
    allowed_statuses = set(DEFAULT_HUMAN_GATE_PAUSE_STATUSES)
    statuses: list[str] = []
    for item in raw_statuses:
        status = str(item).strip()
        if status not in allowed_statuses:
            raise ConfigError(
                "human_gate.pause_on_statuses values must be one of "
                + ", ".join(sorted(allowed_statuses))
            )
        if status not in statuses:
            statuses.append(status)
    parsed["pause_on_statuses"] = statuses or list(DEFAULT_HUMAN_GATE_PAUSE_STATUSES)
    return parsed


def _dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{field_name} must be an object")
    return copy.deepcopy(value)


def _optional_dict(value: Any, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    parsed = _dict(value, field_name)
    if not parsed:
        raise ConfigError(f"{field_name} must not be empty when provided")
    return parsed


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a boolean")
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
        raise ConfigError(f"{field_name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ConfigError(f"{field_name} must be a non-negative integer")
    return parsed


def _safe_id(value: str, field_name: str) -> str:
    if not SAFE_ID_RE.fullmatch(value):
        raise ConfigError(f"{field_name} must contain only letters, numbers, dot, underscore, or dash")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
