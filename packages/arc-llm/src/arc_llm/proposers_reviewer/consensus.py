from __future__ import annotations

import copy
import json
import math
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

BatchRunner = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ConsensusStep:
    step_id: str
    prompt: str
    kind: str
    allowed_context: dict[str, Any]


@dataclass(frozen=True)
class ConsensusConfig:
    schema_version: str
    run_id: str
    run_dir: Path
    proposer_count: int
    max_recalculations: int
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
    proposer_count = _positive_int(data.get("proposer_count", 3), "proposer_count")
    max_recalculations = _positive_int(data.get("max_recalculations", 3), "max_recalculations")
    defaults = _dict(data.get("defaults", {}), "defaults")
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
        if kind not in {"foundation_check", "new_calculation"}:
            raise ConfigError("step.kind must be foundation_check or new_calculation")
        steps.append(
            ConsensusStep(
                step_id=step_id,
                prompt=_required_text(step_data, "prompt"),
                kind=kind,
                allowed_context=_dict(step_data.get("allowed_context", {}), f"{step_id}.allowed_context"),
            )
        )

    return ConsensusConfig(
        schema_version=schema_version,
        run_id=run_id,
        run_dir=run_dir,
        proposer_count=proposer_count,
        max_recalculations=max_recalculations,
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
        )
        step_results.append(step_result)
        if step_result["status"] == "blocked_for_user":
            overall_status = "blocked_for_user"
            break
        if step_result["status"] == "failed":
            overall_status = "failed"
            break

    result = {
        "schema_version": CONSENSUS_RESULT_SCHEMA,
        "status": overall_status,
        "run_id": consensus.run_id,
        "run_root": str(paths.run_root),
        "proposer_count": consensus.proposer_count,
        "max_recalculations": consensus.max_recalculations,
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
            )
        except Exception as exc:
            return {
                "step_id": step.step_id,
                "kind": step.kind,
                "status": "failed",
                "attempts": attempts,
                "accepted_output": None,
                "blocked_output": None,
                "reviewer_consensus": None,
                "error": str(exc),
            }
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
            return {
                "step_id": step.step_id,
                "kind": step.kind,
                "status": "failed",
                "attempts": attempts,
                "accepted_output": None,
                "blocked_output": None,
                "reviewer_consensus": None,
                "error": "attempt batch did not complete",
            }

        try:
            review = _read_json(attempt_paths["review_path"])
            proposer_outputs = _read_proposer_outputs(attempt_paths["round_root"], active_proposer_ids)
            consensus = _review_consensus(
                review,
                active_proposer_ids=active_proposer_ids,
                proposer_outputs=proposer_outputs,
            )
        except Exception as exc:
            attempt_record["error"] = str(exc)
            attempts.append(attempt_record)
            return {
                "step_id": step.step_id,
                "kind": step.kind,
                "status": "failed",
                "attempts": attempts,
                "accepted_output": None,
                "blocked_output": None,
                "reviewer_consensus": None,
                "error": str(exc),
            }
        attempt_record["consensus"] = consensus
        attempt_record["proposer_output_paths"] = {
            proposer_id: str(attempt_paths["round_root"] / "proposer_outputs" / f"{proposer_id}.json")
            for proposer_id in active_proposer_ids
        }
        attempts.append(attempt_record)

        status = str(consensus.get("status", "unresolved"))
        if status == "all_agree":
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


def _attempt_batch_config(
    config: ConsensusConfig,
    step: ConsensusStep,
    *,
    attempt_number: int,
    active_proposer_ids: list[str],
    locked_outputs: dict[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    attempt_id = f"{step.step_id}_attempt_{attempt_number:03d}"
    caller_context = _caller_context(
        config,
        step,
        attempt_number=attempt_number,
        active_proposer_ids=active_proposer_ids,
        locked_outputs=locked_outputs,
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
                "proposers": [_proposer_config(proposer_id) for proposer_id in active_proposer_ids],
                "reviewers": [_reviewer_config(active_proposer_ids)],
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
) -> dict[str, Any]:
    allowed_context = _sanitize_caller_allowed_context(step.allowed_context)
    foundation_context = _foundation_context_for_step(step)
    if foundation_context is not None:
        allowed_context.pop("foundation_file", None)
    context = {
        "step_id": step.step_id,
        "step_kind": step.kind,
        "step_prompt": step.prompt,
        "allowed_context": allowed_context,
        "attempt_number": attempt_number,
        "active_proposer_ids": active_proposer_ids,
        "locked_outputs": copy.deepcopy(locked_outputs),
        "max_recalculations": config.max_recalculations,
        "integrity_reference": _integrity_reference(config.defaults.get("integrity_reference_path")),
        "consensus_instruction": "Work only on this calculation step. Respect locked_outputs as already accepted unless explicitly asked to check them.",
    }
    if foundation_context is not None:
        context["foundation_context"] = foundation_context
    return context


def _proposer_config(proposer_id: str) -> dict[str, Any]:
    return {
        "id": proposer_id,
        "prompt": {
            "system": "You are an independent theoretical-physics calculation proposer.",
            "template": (
                "First read and follow caller_context.integrity_reference.content. "
                "Use only caller_context.step_prompt, caller_context.allowed_context, "
                "caller_context.foundation_context when present, accepted locked_outputs, "
                "your own SymPy/local algebra, and cited source context you inspect. "
                "You may use ARC paper MCP tools to read the main reference and cited "
                "sections named in the plan or foundation context. Internet search is "
                "allowed only for source discovery or uncached paper access. Cite any "
                "paper tool or internet source you use. Do not use validation-only final "
                "formulas as derivation inputs. Wolfram may be used only for algebraic "
                "verification. You must strictly derive from the foundation context and "
                "accepted locked_outputs. External sources may inspire methods, but do not "
                "directly use any result from papers or the internet unless it appears in "
                "the foundation file or accepted locked_outputs. If you need an external "
                "identity or intermediate result, derive it here. External sources may use "
                "different conventions; map notation back to foundation conventions before "
                "using it. Do the calculation very clearly step by step; never skip a step. "
                "Return one JSON object with result_summary, derivation, assumptions, "
                "reliable_until, and final_result. Put the final mathematical result in "
                "final_result using explicit symbols so a reviewer can compare it with "
                "other proposers by evaluating A-B, B-C, and A-C. Do not coordinate with "
                "other proposers.\n\n"
                "{caller_context_json}"
            ),
        },
        "output_schema": {"type": "object"},
        "runtime": {
            "allow_internet": True,
            "allow_mcp": True,
            "mcp_mode": "arc-only",
            "codex_sandbox": "read-only",
        },
    }


def _reviewer_config(active_proposer_ids: list[str]) -> dict[str, Any]:
    return {
        "id": "reviewer_001",
        "prompt": {
            "system": "You are a skeptical theoretical-physics consensus reviewer.",
            "template": (
                "First read and follow caller_context.integrity_reference.content. "
                "Compare current_proposer_outputs_json for the current calculation step. "
                "Do not modify original equations. For analytic checks, first use expand, "
                "then simplify, then substitutions from checked equations in "
                "caller_context.foundation_context. Document the substitution/check history "
                "in pairwise_symbolic_checks.check_history. "
                "Return exactly one arc.llm.review_envelope.v1 JSON object. The top-level object "
                "must contain schema_version, controller, proposer_messages, and review_payload. "
                f"proposer_messages must contain these keys: {', '.join(active_proposer_ids)}. "
                "In review_payload.consensus, "
                "set status to all_agree, two_agree, all_disagree, or unresolved. Include "
                "accepted_result, agreed_proposer_ids, likely_wrong_proposer_ids, "
                "recalculate_proposer_ids, reliable_until, analysis, and "
                "pairwise_symbolic_checks. Let A, B, and C be the final mathematical results "
                "from proposer_001, proposer_002, and proposer_003 when those proposer ids "
                "are active. Use SymPy whenever available to simplify A-B, B-C, and A-C. "
                "If used_sympy=true, pairwise_symbolic_checks.sympy_code must include "
                "the actual code for A-B, B-C, and A-C using expand first and then "
                "simplify, for example simplify(expand(A-B)). "
                "Before marking all_agree, at least two of A-B=0, B-C=0, and A-C=0 must "
                "be true. Never mark all_agree by visual inspection or because the results "
                "seem to agree. If SymPy is unavailable, perform explicit algebraic checks "
                "and set used_sympy=false with the reason in pairwise_symbolic_checks.notes. "
                "Manual all_agree must show explicit A-B, B-C, and A-C algebraic "
                "differences reducing to zero; never accept by string equality, spacing, "
                "formatting, or visual comparison. If you cannot write those explicit "
                "differences, use the numerical fallback instead. "
                "If an analytic check cannot be done, perform numerical checks on at least "
                "10 randomly selected data points and report check_method=numerical, "
                "sample_count, numerical_relative_error as a relative error, and the sampled check history. "
                "If exactly one proposer is likely wrong, name only that proposer for "
                "recalculation; otherwise ask all proposers to recalculate.\n\n"
                "{current_proposer_outputs_json}"
            ),
        },
        "output_schema": _reviewer_output_schema(active_proposer_ids),
        "runtime": {"allow_mcp": False, "codex_sandbox": "read-only"},
    }


def _reviewer_output_schema(active_proposer_ids: list[str]) -> dict[str, Any]:
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
                            "reliable_until",
                            "analysis",
                            "pairwise_symbolic_checks",
                        ],
                        "properties": {
                            "status": {
                                "enum": ["all_agree", "two_agree", "all_disagree", "unresolved"]
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
                            "reliable_until": {"type": "string"},
                            "analysis": {"type": "string"},
                            "pairwise_symbolic_checks": {
                                "type": "object",
                                "required": [
                                    "used_sympy",
                                    "A_minus_B_zero",
                                    "B_minus_C_zero",
                                    "A_minus_C_zero",
                                    "true_count",
                                    "notes",
                                    "check_method",
                                    "check_history",
                                ],
                                "properties": {
                                    "used_sympy": {"type": "boolean"},
                                    "A_minus_B_zero": {"type": "boolean"},
                                    "B_minus_C_zero": {"type": "boolean"},
                                    "A_minus_C_zero": {"type": "boolean"},
                                    "true_count": {"type": "integer", "minimum": 0, "maximum": 3},
                                    "sympy_code": {"type": "string"},
                                    "notes": {"type": "string"},
                                    "check_method": {
                                        "enum": ["analytic", "numerical", "mixed"]
                                    },
                                    "sample_count": {"type": "integer", "minimum": 0},
                                    "numerical_relative_error": {
                                        "type": ["number", "null"],
                                        "minimum": 0
                                    },
                                    "check_history": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
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
    proposer_outputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if review.get("schema_version") != REVIEW_ENVELOPE_SCHEMA:
        raise ValueError(f"review schema_version must be {REVIEW_ENVELOPE_SCHEMA}")
    payload = review.get("review_payload")
    if not isinstance(payload, dict):
        raise ValueError("review.review_payload must be an object")
    consensus = payload.get("consensus")
    if not isinstance(consensus, dict):
        raise ValueError("review.review_payload.consensus must be an object")
    status = consensus.get("status")
    if status not in {"all_agree", "two_agree", "all_disagree", "unresolved"}:
        raise ValueError("consensus.status must be all_agree, two_agree, all_disagree, or unresolved")
    if status == "all_agree" and len(active_proposer_ids) >= 3:
        try:
            _validate_all_agree_pairwise_checks(consensus)
        except ValueError as exc:
            fallback_checks = _main_agent_sympy_agreement_check(
                proposer_outputs or {},
                active_proposer_ids=active_proposer_ids,
                validation_error=str(exc),
            )
            if fallback_checks is None:
                raise
            consensus = dict(consensus)
            consensus["pairwise_symbolic_checks"] = fallback_checks
            consensus["main_agent_agreement_check"] = {
                "status": "accepted_by_sympy_fallback",
                "reason": str(exc),
            }
            _validate_all_agree_pairwise_checks(consensus)
    return dict(consensus)


def _main_agent_sympy_agreement_check(
    proposer_outputs: Mapping[str, Any],
    *,
    active_proposer_ids: list[str],
    validation_error: str,
) -> dict[str, Any] | None:
    try:
        from sympy import Symbol, expand, simplify, sympify
    except Exception:
        return None

    parsed_by_id: dict[str, dict[str, Any]] = {}
    for proposer_id in active_proposer_ids[:3]:
        payload = proposer_outputs.get(proposer_id)
        if not isinstance(payload, dict):
            return None
        parsed = _extract_named_final_expressions(payload.get("final_result"))
        if not parsed:
            return None
        parsed_by_id[proposer_id] = parsed

    common_names = set.intersection(*(set(parsed) for parsed in parsed_by_id.values()))
    if not common_names:
        return None

    pair_ids = [
        ("A_minus_B_zero", active_proposer_ids[0], active_proposer_ids[1], "A-B"),
        ("B_minus_C_zero", active_proposer_ids[1], active_proposer_ids[2], "B-C"),
        ("A_minus_C_zero", active_proposer_ids[0], active_proposer_ids[2], "A-C"),
    ]
    pair_results: dict[str, bool] = {}
    history: list[str] = [
        f"Main-agent SymPy fallback used after reviewer validation failed: {validation_error}",
        f"Parsed common named final_result expressions: {', '.join(sorted(common_names))}.",
    ]
    code_lines = ["from sympy import Symbol, expand, simplify, sympify"]

    for result_key, left_id, right_id, label in pair_ids:
        pair_zero = True
        for expression_name in sorted(common_names):
            locals_map: dict[str, Any] = {}
            left_expr = _sympify_named_expression(
                str(parsed_by_id[left_id][expression_name]),
                Symbol=Symbol,
                sympify=sympify,
                locals_map=locals_map,
            )
            right_expr = _sympify_named_expression(
                str(parsed_by_id[right_id][expression_name]),
                Symbol=Symbol,
                sympify=sympify,
                locals_map=locals_map,
            )
            if left_expr is None or right_expr is None:
                return None
            difference = simplify(expand(left_expr - right_expr))
            is_zero = bool(difference == 0)
            pair_zero = pair_zero and is_zero
            history.append(f"{label} for {expression_name}: simplify(expand(left-right)) -> {difference}.")
            code_lines.append(
                f"# {label} for {expression_name}: simplify(expand(({parsed_by_id[left_id][expression_name]}) - ({parsed_by_id[right_id][expression_name]})))"
            )
        pair_results[result_key] = pair_zero

    true_count = sum(1 for value in pair_results.values() if value)
    if true_count < 2:
        return None
    return {
        "used_sympy": True,
        "A_minus_B_zero": pair_results["A_minus_B_zero"],
        "B_minus_C_zero": pair_results["B_minus_C_zero"],
        "A_minus_C_zero": pair_results["A_minus_C_zero"],
        "true_count": true_count,
        "sympy_code": "\n".join(code_lines),
        "notes": "Main agent fallback proved proposer agreement with SymPy after reviewer evidence was below standard.",
        "check_method": "analytic",
        "sample_count": 0,
        "numerical_relative_error": None,
        "check_history": history,
        "fallback_source": "main_agent_sympy",
    }


def _extract_named_final_expressions(final_result: Any) -> dict[str, str]:
    if isinstance(final_result, dict):
        return {
            str(key): str(value).strip()
            for key, value in final_result.items()
            if isinstance(value, str) and _looks_like_expression(value)
        }
    if not isinstance(final_result, str):
        return {}
    expressions: dict[str, str] = {}
    for raw_line in final_result.splitlines():
        line = raw_line.strip()
        if "=" not in line:
            continue
        left, right = line.split("=", 1)
        name_match = re.search(r"([A-Za-z][A-Za-z0-9_]*)\s*$", left.strip())
        if not name_match:
            continue
        expression = _strip_expression_comment(right)
        if expression and _looks_like_expression(expression):
            expressions[name_match.group(1)] = expression
    return expressions


def _strip_expression_comment(value: str) -> str:
    expression = value.strip()
    expression = expression.split("#", 1)[0].strip()
    expression = expression.split(",", 1)[0].strip()
    expression = expression.split(";", 1)[0].strip()
    expression = re.split(r"\s+\(", expression, maxsplit=1)[0].strip()
    return expression.rstrip(".")


def _looks_like_expression(value: str) -> bool:
    if not value.strip():
        return False
    without_names = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", "", value.replace("^", "**"))
    return re.fullmatch(r"[\d\s+\-*/().]*", without_names) is not None


def _sympify_named_expression(value: str, *, Symbol: Any, sympify: Any, locals_map: dict[str, Any]) -> Any | None:
    expression = value.replace("^", "**")
    names = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression))
    for name in names:
        locals_map.setdefault(name, Symbol(name))
    try:
        return sympify(expression, locals=locals_map)
    except Exception:
        return None


def _validate_all_agree_pairwise_checks(consensus: Mapping[str, Any]) -> None:
    checks = consensus.get("pairwise_symbolic_checks")
    if not isinstance(checks, dict):
        raise ValueError("all_agree requires review_payload.consensus.pairwise_symbolic_checks")
    pairwise_keys = ["A_minus_B_zero", "B_minus_C_zero", "A_minus_C_zero"]
    pairwise_true_count = sum(1 for key in pairwise_keys if checks.get(key) is True)
    reported_true_count = checks.get("true_count")
    if isinstance(reported_true_count, int):
        pairwise_true_count = min(pairwise_true_count, reported_true_count)
    if pairwise_true_count < 2:
        raise ValueError("all_agree requires at least two true pairwise symbolic checks")
    method = str(checks.get("check_method", "")).strip().lower()
    if method not in {"analytic", "numerical", "mixed"}:
        raise ValueError("all_agree requires check_method to be analytic, numerical, or mixed")
    used_sympy = checks.get("used_sympy")
    if not isinstance(used_sympy, bool):
        raise ValueError("all_agree requires used_sympy to be true or false")
    if used_sympy:
        sympy_code = str(checks.get("sympy_code", "")).lower()
        if "expand" not in sympy_code or "simplify" not in sympy_code:
            raise ValueError("SymPy all_agree requires sympy_code showing expand and simplify checks")
    elif method in {"analytic", "mixed"}:
        notes = str(checks.get("notes", "")).strip()
        if not notes:
            raise ValueError("analytic all_agree without SymPy requires explicit notes")
        history_text = "\n".join(str(item) for item in checks.get("check_history", []))
        lowered_evidence = f"{notes}\n{history_text}".lower()
        has_explicit_differences = all(marker in lowered_evidence for marker in ["a-b", "b-c", "a-c"])
        has_zero_results = lowered_evidence.count("=0") >= 2 or lowered_evidence.count("= 0") >= 2
        says_differences_reduce_to_zero = (
            "differences reduce to zero" in lowered_evidence
            or "difference reduce to zero" in lowered_evidence
            or "reduce to zero symbolically" in lowered_evidence
        )
        has_term_by_term_check = "term-by-term" in lowered_evidence and (
            "overall factor" in lowered_evidence
            or "every term matches" in lowered_evidence
            or "all terms match" in lowered_evidence
        )
        weak_markers = ["spacing", "string", "formatting", "visual", "inspection", "identical"]
        if any(marker in lowered_evidence for marker in weak_markers) and not (
            (has_explicit_differences and has_zero_results)
            or says_differences_reduce_to_zero
            or has_term_by_term_check
        ):
            raise ValueError("manual all_agree cannot rely on string, spacing, formatting, or visual comparison")
        if not (has_explicit_differences or says_differences_reduce_to_zero or has_term_by_term_check):
            raise ValueError("manual all_agree requires explicit A-B, B-C, and A-C algebraic differences")
    if method in {"numerical", "mixed"}:
        sample_count = checks.get("sample_count")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 10:
            raise ValueError("numerical all_agree requires at least 10 randomly selected data points")
        relative_error = checks.get("numerical_relative_error")
        if (
            not isinstance(relative_error, (int, float))
            or isinstance(relative_error, bool)
            or not math.isfinite(float(relative_error))
        ):
            raise ValueError("numerical all_agree requires numerical_relative_error")
    history = checks.get("check_history")
    if not isinstance(history, list) or not history:
        raise ValueError("all_agree requires documented pairwise check history")


def _foundation_context_for_step(step: ConsensusStep) -> dict[str, Any] | None:
    foundation_path = step.allowed_context.get("foundation_file")
    if not foundation_path or step.kind != "foundation_check":
        return None
    target_equation_id = str(step.allowed_context.get("target_equation_id", "")).strip()
    if not target_equation_id:
        raise ValueError("target_equation_id is required for foundation_check steps")
    path = Path(str(foundation_path))
    payload = _read_json(path)
    return filter_foundation_context(
        payload,
        target_equation_id=target_equation_id,
    )


def filter_foundation_context(
    payload: Mapping[str, Any],
    *,
    target_equation_id: str = "",
) -> dict[str, Any]:
    equations = payload.get("equations", [])
    if not isinstance(equations, list):
        equations = []
    allowed_equations = []
    omitted_equation_ids = []
    target_equation = None
    target_equation_id = target_equation_id.strip()
    if not target_equation_id:
        raise ValueError("target_equation_id is required")
    for item in equations:
        if not isinstance(item, dict):
            continue
        equation_id = str(item.get("id", ""))
        if target_equation_id and equation_id == target_equation_id:
            target_equation = _sanitize_foundation_context_item(item)
            continue
        if _equation_is_axiom_or_checked(item):
            allowed_equations.append(_sanitize_foundation_context_item(item))
        elif equation_id:
            omitted_equation_ids.append(equation_id)
    if target_equation is None:
        raise ValueError(f"target_equation_id {target_equation_id} was not found in foundation equations")

    conventions = payload.get("conventions", [])
    if not isinstance(conventions, list):
        conventions = []
    allowed_conventions = [
        _sanitize_foundation_context_item(item)
        for item in conventions
        if isinstance(item, dict) and _convention_is_checked(item)
    ]
    return {
        "schema_version": "arc.research_foundation_context.v1",
        "target_equation_id": target_equation_id,
        "target_equation": target_equation,
        "allowed_equations": allowed_equations,
        "allowed_conventions": allowed_conventions,
        "omitted_equation_ids": omitted_equation_ids,
        "filter_rule": "Only the target equation plus axiom or checked foundation items are provided.",
    }


def _equation_is_axiom_or_checked(item: Mapping[str, Any]) -> bool:
    if item.get("axiom_status") == "axiom":
        return True
    check_status = str(item.get("check_status", "")).strip().lower()
    return check_status == "checked" or check_status.startswith("checked_")


def _convention_is_checked(item: Mapping[str, Any]) -> bool:
    check_status = str(item.get("check_status", "")).strip().lower()
    return check_status == "checked" or check_status.startswith("checked_")


_FOUNDATION_CONTEXT_OMIT_KEYS = {"sources", "mcp", "cli", "cache_path", "source_path"}
_CALLER_ALLOWED_CONTEXT_OMIT_KEYS = _FOUNDATION_CONTEXT_OMIT_KEYS | {"source_commands"}


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


def _sanitize_foundation_context_item(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_foundation_context_item(item)
            for key, item in value.items()
            if str(key) not in _FOUNDATION_CONTEXT_OMIT_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_foundation_context_item(item) for item in value]
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
        candidate = root / "skills/arc/references/rules/integrity.md"
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


def _dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{field_name} must be an object")
    return copy.deepcopy(value)


def _positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ConfigError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ConfigError(f"{field_name} must be a positive integer")
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
