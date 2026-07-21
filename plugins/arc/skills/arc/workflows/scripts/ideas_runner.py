from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from _arc_script_bootstrap import bootstrap_arc_pythonpath

bootstrap_arc_pythonpath()

from arc_domain.summary import mathematical_opportunities_validation_error
from arc_llm.evidence import EvidenceControllerCallback, EvidenceRequest, EvidenceResponse
from arc_llm.proposers_reviewer.artifacts import atomic_write_json, atomic_write_text
from arc_llm.proposers_reviewer.runner import run_proposers_reviewer_batch
from arc_llm.proposers_reviewer.template_materializer import (
    deep_merge,
    materialize_batch,
    materialize_loop,
    materialize_worker,
    replace_placeholders,
)
from arc_paper import service as arc_paper_service

from ideas_config import ConfigError, IdeasConfig, VariantConfig, load_ideas_config
from ideas_marking import load_marking_scheme, marking_scheme_for_context, marks_schema, normalized_marks, score_fields


JsonRunner = Callable[..., dict[str, Any]]
BatchRunner = Callable[..., dict[str, Any]]
MODEL_TIER_RANKS = {"low": 1, "medium": 2, "high": 3, "max": 4}
DEFAULT_CROSS_DOMAIN_PROFILES = [
    {
        "profile_id": "forward_transfer",
        "mission": (
            "Treat the first domain card as the source and choose the strongest distinct target from the remaining "
            "cards. Transfer one concrete, mature source method, mechanism, formal structure, or constraint."
        ),
    },
    {
        "profile_id": "reverse_transfer",
        "mission": (
            "Treat the first domain card as the target and choose the strongest distinct source from the remaining "
            "cards. Find a reverse transfer that creates a substantive new target result."
        ),
    },
    {
        "profile_id": "method_transfer",
        "mission": (
            "Compare both directions and choose the strongest method or formalism transfer. State the exact "
            "translation dictionary and the target calculation it newly enables."
        ),
    },
    {
        "profile_id": "observable_or_constraint_transfer",
        "mission": (
            "Compare both directions and transfer an observable, consistency condition, validation strategy, "
            "or constraint that yields a new discriminating target-domain result."
        ),
    },
    {
        "profile_id": "high_upside_wildcard",
        "mission": (
            "Pursue the highest-upside feasible bridge, including a challenge to a standard target assumption. "
            "Require explicit compatibility checks, a bounded first calculation, and a kill criterion."
        ),
    },
]
ARC_PAPER_EVIDENCE_OPERATIONS = [
    {"operation": "paper.metadata", "arguments": "paper_id or paper_ids; optional refresh boolean"},
    {"operation": "paper.section", "arguments": "paper_id or paper_ids, section; optional refresh boolean"},
    {
        "operation": "paper.full_text_search",
        "arguments": "query; optional paper_id or paper_ids, refresh, limit, context, case_sensitive",
    },
    {
        "operation": "paper.references",
        "arguments": "paper_id or paper_ids; optional refresh and enrich booleans",
    },
    {
        "operation": "paper.citers",
        "arguments": "paper_id or paper_ids; optional refresh, limit, sort (mostrecent or mostcited)",
    },
    {"operation": "paper.search", "arguments": "query; optional limit"},
]


@dataclass(frozen=True)
class IdeaPlan:
    idea_id: str
    variant_id: str
    idea_index: int
    loop_id: str
    variant: VariantConfig
    caller_context: dict[str, Any]


class ArcPaperEvidenceResolver:
    """Resolve the bounded evidence vocabulary through deterministic arc-paper services."""

    def __call__(
        self,
        requests: tuple[EvidenceRequest, ...],
        *,
        round_number: int,
    ) -> tuple[EvidenceResponse, ...]:
        return tuple(self._resolve(request, round_number=round_number) for request in requests)

    def _resolve(self, request: EvidenceRequest, *, round_number: int) -> EvidenceResponse:
        try:
            result = _dispatch_arc_paper_evidence(request.operation, request.arguments)
        except Exception as exc:
            return EvidenceResponse(
                request.request_id,
                False,
                error=str(exc) or exc.__class__.__name__,
                provenance={
                    "source": "arc-paper",
                    "operation": request.operation,
                    "evidence_round": round_number,
                    "error_type": exc.__class__.__name__,
                },
            )
        meta = result.get("meta") if isinstance(result, Mapping) else None
        provenance = {
            "source": "arc-paper",
            "operation": request.operation,
            "evidence_round": round_number,
            "service_meta": dict(meta) if isinstance(meta, Mapping) else {},
        }
        if isinstance(result, Mapping) and result.get("ok") is True:
            return EvidenceResponse(request.request_id, True, data=result.get("data"), provenance=provenance)
        error = result.get("error") if isinstance(result, Mapping) else None
        message = error.get("message") if isinstance(error, Mapping) else "arc-paper returned an invalid result"
        return EvidenceResponse(request.request_id, False, error=str(message), provenance=provenance)


def _dispatch_arc_paper_evidence(operation: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    args = dict(arguments)
    refresh = _evidence_bool(args.get("refresh", False), "refresh")
    if operation == "paper.metadata":
        return arc_paper_service.get_metadata(_evidence_paper_ids(args), refresh=refresh)
    if operation == "paper.section":
        return arc_paper_service.get_section(
            _evidence_paper_ids(args),
            _evidence_text(args, "section"),
            refresh=refresh,
        )
    if operation == "paper.full_text_search":
        return arc_paper_service.search_full_text(
            _evidence_paper_ids(args, required=False),
            query=_evidence_text(args, "query"),
            refresh=refresh,
            limit=_evidence_int(args.get("limit", 20), "limit", minimum=1, maximum=100),
            context=_evidence_int(args.get("context", 1), "context", minimum=0, maximum=10),
            case_sensitive=_evidence_bool(args.get("case_sensitive", False), "case_sensitive"),
        )
    if operation == "paper.references":
        return arc_paper_service.get_references(
            _evidence_paper_ids(args),
            refresh=refresh,
            enrich=_evidence_bool(args.get("enrich", True), "enrich"),
        )
    if operation == "paper.citers":
        sort = str(args.get("sort", "mostrecent"))
        if sort not in {"mostrecent", "mostcited"}:
            raise ValueError("sort must be mostrecent or mostcited")
        return arc_paper_service.get_citers(
            _evidence_paper_ids(args),
            refresh=refresh,
            limit=_evidence_int(args.get("limit", 100), "limit", minimum=1, maximum=1000),
            sort=sort,
        )
    if operation == "paper.search":
        return arc_paper_service.search_inspire(
            _evidence_text(args, "query"),
            limit=_evidence_int(args.get("limit", 20), "limit", minimum=1, maximum=100),
        )
    raise ValueError(
        "unsupported evidence operation; use paper.metadata, paper.section, "
        "paper.full_text_search, paper.references, paper.citers, or paper.search"
    )


def _evidence_paper_ids(arguments: Mapping[str, Any], *, required: bool = True) -> str | list[str] | None:
    value = arguments.get("paper_ids", arguments.get("paper_id"))
    if value is None and not required:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list) and value and all(isinstance(item, str) and item.strip() for item in value):
        return [item.strip() for item in value]
    raise ValueError("paper_id or a non-empty paper_ids string array is required")


def _evidence_text(arguments: Mapping[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _evidence_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _evidence_int(value: Any, key: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer between {minimum} and {maximum}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer between {minimum} and {maximum}") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{key} must be an integer between {minimum} and {maximum}")
    return parsed


def run_ideas(
    config: IdeasConfig | Mapping[str, Any],
    *,
    json_runner: JsonRunner | None = None,
    batch_runner: BatchRunner | None = None,
    base_env: Mapping[str, str] | None = None,
    process_chain: list[str] | None = None,
    evidence_controller: EvidenceControllerCallback | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    ideas_config = config if isinstance(config, IdeasConfig) else load_ideas_config(config)
    run_root = ideas_config.run_dir / ideas_config.run_id
    ideas = _materialize_ideas(ideas_config)
    max_concurrent = _max_concurrent_loops(len(ideas))
    batch_config = _loop_batch_config(ideas_config, ideas, run_root=run_root)
    batch_config_path = run_root / "ideas_batch_config.json"
    warnings_path = run_root / "ideas_warnings.txt"
    warnings = [
        _concurrency_warning(ideas_config, len(ideas), max_concurrent=max_concurrent),
        *ideas_config.routing_warnings,
        *_model_tier_warnings(batch_config),
        *_caller_context_warnings(ideas),
    ]

    if dry_run:
        return _result(
            ideas_config,
            run_root=run_root,
            batch_config_path=batch_config_path,
            warnings_path=warnings_path,
            warnings=warnings,
            ideas=ideas,
            batch_result={
                "schema_version": "arc.llm.proposers_reviewer_batch.result.v1",
                "status": "dry_run",
                "run_id": batch_config["run_id"],
                "run_root": str(Path(batch_config["run_dir"]) / str(batch_config["run_id"])),
                "loops": [{"loop_id": loop["loop_id"], "status": "validated"} for loop in batch_config["loops"]],
            },
        )

    atomic_write_json(batch_config_path, batch_config)
    _write_warnings(warnings_path, warnings)
    try:
        sidechannel_callback = _progress_sidechannel_callback(base_env)
        effective_progress_callback = _combined_progress_callback(progress_callback, sidechannel_callback)
        batch_kwargs: dict[str, Any] = {
            "json_runner": json_runner,
            "base_env": base_env,
            "process_chain": process_chain,
            "dry_run": False,
            "max_concurrent_loops": max_concurrent,
        }
        if effective_progress_callback is not None:
            batch_kwargs["progress_callback"] = effective_progress_callback
        if cancel_check is not None:
            batch_kwargs["cancel_check"] = cancel_check
        if evidence_controller is not None:
            batch_kwargs["evidence_controller"] = evidence_controller
        elif batch_runner is None:
            batch_kwargs["evidence_controller"] = ArcPaperEvidenceResolver()
        batch_result = (batch_runner or run_proposers_reviewer_batch)(batch_config, **batch_kwargs)
    except Exception as exc:
        batch_result = {
            "schema_version": "arc.llm.proposers_reviewer_batch.result.v1",
            "status": "failed",
            "run_id": batch_config["run_id"],
            "run_root": str(Path(batch_config["run_dir"]) / str(batch_config["run_id"])),
            "loops": [],
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    return _result(
        ideas_config,
        run_root=run_root,
        batch_config_path=batch_config_path,
        warnings_path=warnings_path,
        warnings=warnings,
        ideas=ideas,
        batch_result=batch_result,
    )


def _result(
    config: IdeasConfig,
    *,
    run_root: Path,
    batch_config_path: Path,
    warnings_path: Path,
    warnings: list[str],
    ideas: list[IdeaPlan],
    batch_result: Mapping[str, Any],
) -> dict[str, Any]:
    batch_run_root = Path(str(batch_result.get("run_root", run_root / "idea_loops")))
    round_score_table = _round_score_table(ideas, batch_run_root=batch_run_root)
    loop_reviewer_call_count = _loop_reviewer_call_count(batch_result, ideas)
    return {
        "schema_version": "arc.workflow.ideas.result.v1",
        "status": str(batch_result.get("status", "failed")),
        "run_id": config.run_id,
        "run_root": str(run_root),
        "research_scope": config.research_scope,
        "domain_manifest_path": str(config.domain_manifest_path),
        "warnings": warnings,
        "warnings_summary": _batch_warnings_summary(batch_result),
        "proposal_count": len(ideas),
        "reviewer_call_count": loop_reviewer_call_count,
        "loop_reviewer_call_count": loop_reviewer_call_count,
        "max_concurrent_loops": _max_concurrent_loops(len(ideas)),
        "max_concurrent_proposal_calls": _max_concurrent_loops(len(ideas)),
        "batch_config_path": str(batch_config_path),
        "warnings_path": str(warnings_path),
        "loops": [_loop_summary(idea, batch_run_root=batch_run_root) for idea in ideas],
        "round_score_table": round_score_table,
        "batch_result": dict(batch_result),
    }


def _batch_warnings_summary(batch_result: Mapping[str, Any]) -> dict[str, Any]:
    summary = batch_result.get("warnings_summary")
    return dict(summary) if isinstance(summary, Mapping) else {
        "structured_output_warning_count": 0,
        "structured_output_warnings_path": "",
        "cache_warning_count": 0,
        "cache_warnings_path": "",
    }


def _materialize_ideas(config: IdeasConfig) -> list[IdeaPlan]:
    ideas: list[IdeaPlan] = []
    for variant in config.variants:
        for idea_index in range(1, config.loops_per_variant + 1):
            idea_id = f"{variant.variant_id}/idea_{idea_index:03d}"
            ideas.append(
                IdeaPlan(
                    idea_id=idea_id,
                    variant_id=variant.variant_id,
                    idea_index=idea_index,
                    loop_id=f"{variant.variant_id}_idea_{idea_index:03d}",
                    variant=variant,
                    caller_context=_caller_context(
                        config,
                        variant=variant,
                        idea_id=idea_id,
                        idea_index=idea_index,
                    ),
                )
            )
    return ideas


def _caller_context(
    config: IdeasConfig,
    *,
    variant: VariantConfig,
    idea_id: str,
    idea_index: int,
) -> dict[str, Any]:
    loop_template = _read_json(variant.loop_template)
    caller_context = copy.deepcopy(loop_template.get("caller_context", {}))
    if not isinstance(caller_context, dict):
        raise ConfigError(f"{variant.loop_template}.caller_context must be an object")
    caller_context = replace_placeholders(caller_context, {"<user_intent>": config.user_intent})
    caller_context["user_intent"] = config.user_intent
    caller_context["variant_id"] = variant.variant_id
    caller_context["idea_id"] = idea_id
    caller_context["marking_scheme"] = marking_scheme_for_context(load_marking_scheme(variant.marking_scheme))
    if variant.research_scope == "cross_domain":
        caller_context["generation_mode"] = "cross_domain"
        domain_cards = _domain_cards(config)
        caller_context["domain_cards"] = domain_cards
        legacy_domain_ids = [
            str(card.get("field_id", ""))
            for card in domain_cards
            if not card.get("summary_capabilities", {}).get("mathematical_opportunities")
        ]
        if legacy_domain_ids:
            caller_context.setdefault("warnings", []).append(
                "legacy_domain_summary_without_mathematical_opportunities: "
                + ", ".join(legacy_domain_ids)
            )
        caller_context["exploration_profile"] = _cross_domain_profile(config, idea_index=idea_index)
    if variant.context_policy.attach_domain_markdown:
        markdown_files = _domain_markdown_files(config.project_dir / "domain")
        if markdown_files:
            caller_context["domain_markdown_files"] = markdown_files
        else:
            if variant.context_policy.require_domain_markdown:
                raise ConfigError(f"{variant.variant_id} requires domain markdown under {config.project_dir / 'domain'}")
            caller_context.pop("domain_markdown_files", None)
            caller_context.setdefault("warnings", []).append(
                "domain_markdown_unavailable: Domain markdown was unavailable; continuing with user intent and ARC paper/tool context only."
            )
    else:
        caller_context.pop("domain_markdown_files", None)
    if not variant.context_policy.attach_arc_paper_tool_notes:
        caller_context.pop("arc_paper_tool_notes", None)
        caller_context.pop("controller_evidence_operations", None)
    else:
        caller_context["controller_evidence_operations"] = copy.deepcopy(ARC_PAPER_EVIDENCE_OPERATIONS)
    return caller_context


def _caller_context_warnings(ideas: list[IdeaPlan]) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    for idea in ideas:
        for warning in idea.caller_context.get("warnings", []):
            text = str(warning)
            if text not in seen:
                seen.add(text)
                warnings.append(text)
    return warnings


def _loop_batch_config(config: IdeasConfig, ideas: list[IdeaPlan], *, run_root: Path) -> dict[str, Any]:
    max_concurrent = _max_concurrent_loops(len(ideas))
    return materialize_batch(
        run_id="idea_loops",
        run_dir=run_root,
        max_concurrent_loops=max_concurrent,
        artifact_options={"save_prompts": config.save_prompts},
        output_recovery=_relaxed_output_recovery_config(),
        session={
            "policy": "stateful",
            "history_mode": "delta",
            "max_concurrent_same_prefix": 12,
            "cache_guard": {
                "enabled": True,
                "mode": "warn",
                "warmup_calls": 1,
                "min_cached_input_ratio": 0.70,
            },
        },
        loops=[_idea_loop_payload(idea) for idea in ideas],
    )


def _relaxed_output_recovery_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": "warn",
        "allow_natural_language": True,
        "schema_violation_policy": "peer_visible",
        "reviewer_validation_retries": 0,
    }


def _idea_loop_payload(idea: IdeaPlan) -> dict[str, Any]:
    if idea.variant.research_scope == "cross_domain":
        static_context_keys = [
            "user_intent",
            "generation_mode",
            "domain_cards",
            "arc_paper_tool_notes",
            "controller_evidence_operations",
            "marking_scheme",
        ]
        volatile_context_keys = ["idea_id", "variant_id", "exploration_profile"]
    else:
        static_context_keys = [
            "user_intent",
            "domain_markdown_files",
            "arc_paper_tool_notes",
            "controller_evidence_operations",
            "marking_scheme",
        ]
        volatile_context_keys = ["idea_id", "variant_id"]
    return materialize_loop(
        _read_json(idea.variant.loop_template),
        loop_id=idea.loop_id,
        caller_context=idea.caller_context,
        proposers=[_proposer_payload(idea.variant)],
        reviewers=[_loop_reviewer_payload(idea.variant)],
        cache_context={
            "static_caller_context_keys": static_context_keys,
            "volatile_caller_context_keys": volatile_context_keys,
        },
        overrides={
            "evidence": {
                "enabled": idea.variant.context_policy.attach_arc_paper_tool_notes,
            }
        },
    )


def _proposer_payload(variant: VariantConfig) -> dict[str, Any]:
    return materialize_worker(_read_json(variant.proposer_template), overrides=variant.proposer_overrides)


def _merged_worker_payload(template: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    return deep_merge(template, overrides)


def _loop_reviewer_payload(variant: VariantConfig) -> dict[str, Any]:
    scheme = load_marking_scheme(variant.marking_scheme)
    payload = _read_json(variant.reviewer_template)
    if not variant.context_policy.attach_arc_paper_tool_notes:
        runtime = dict(payload.get("runtime") or {})
        runtime["arc_paper_cli_access"] = "none"
        runtime["inherit_host_tools"] = False
        payload["runtime"] = runtime
    payload["output_schema"] = _reviewer_output_schema(variant, scheme=scheme)
    return payload


def _reviewer_output_schema(variant: VariantConfig, *, scheme: Mapping[str, Any]) -> dict[str, Any]:
    schema = _read_json(variant.reviewer_output_schema)
    schema["properties"]["review_payload"]["properties"]["marks"] = marks_schema(scheme)
    return schema


def _loop_reviewer_call_count(batch_result: Mapping[str, Any], ideas: list[IdeaPlan]) -> int:
    if batch_result.get("status") == "dry_run":
        return sum(int(_read_json(idea.variant.loop_template).get("max_rounds") or 0) for idea in ideas)
    loops = batch_result.get("loops", [])
    if isinstance(loops, list) and loops:
        return sum(int(item.get("rounds_completed") or 0) for item in loops if isinstance(item, Mapping))
    return 0


def _loop_summary(idea: IdeaPlan, *, batch_run_root: Path) -> dict[str, Any]:
    return {
        "idea_id": idea.idea_id,
        "variant_id": idea.variant_id,
        "idea_index": idea.idea_index,
        "loop_id": idea.loop_id,
        "loop_root": str(batch_run_root / "loops" / idea.loop_id),
    }


def _round_score_table(ideas: list[IdeaPlan], *, batch_run_root: Path) -> dict[str, Any]:
    rows = [_round_score_row(idea, batch_run_root=batch_run_root) for idea in ideas]
    max_round = max(
        (max((int(key) for key in row["total_scores_by_round"]), default=0) for row in rows),
        default=0,
    )
    columns = [
        "Idea",
        "Group",
        "Final Title",
        *[f"R{round_number}" for round_number in range(1, max_round + 1)],
        f"Δ R1→R{max_round}" if max_round else "Δ",
        "Best",
    ]
    return {
        "schema_version": "arc.workflow.ideas.round_score_table.v1",
        "source": "loop_artifacts",
        "columns": columns,
        "rows": rows,
        "markdown": _round_score_markdown(columns, rows, max_round=max_round),
    }


def _round_score_row(idea: IdeaPlan, *, batch_run_root: Path) -> dict[str, Any]:
    loop_root = batch_run_root / "loops" / idea.loop_id
    rounds, final_title = _loop_round_scores(loop_root, idea=idea)
    total_scores = {
        round_number: marks["total_score"]
        for round_number, marks in rounds.items()
        if isinstance(marks.get("total_score"), (int, float))
    }
    first_round = min(total_scores, default=None)
    last_round = max(total_scores, default=None)
    delta_total = (
        total_scores[last_round] - total_scores[first_round]
        if first_round is not None and last_round is not None
        else None
    )
    best_total = max(total_scores.values(), default=None)
    return {
        "idea_id": idea.idea_id,
        "variant_id": idea.variant_id,
        "group": idea.variant_id,
        "loop_id": idea.loop_id,
        "final_title": final_title,
        "rounds": [
            {"round": round_number, "marks": rounds[round_number]}
            for round_number in sorted(rounds)
        ],
        "total_scores_by_round": {str(key): value for key, value in sorted(total_scores.items())},
        "delta_total": delta_total,
        "best_total": best_total,
    }


def _loop_round_scores(loop_root: Path, *, idea: IdeaPlan) -> tuple[dict[int, dict[str, Any]], str]:
    scheme = load_marking_scheme(idea.variant.marking_scheme)
    transcript_rounds, transcript_title = _loop_round_scores_from_transcript(loop_root, scheme=scheme)
    if transcript_rounds:
        return transcript_rounds, transcript_title
    return _loop_round_scores_from_round_dirs(loop_root, scheme=scheme)


def _loop_round_scores_from_transcript(
    loop_root: Path,
    *,
    scheme: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], str]:
    transcript = loop_root / "transcript.jsonl"
    if not transcript.is_file():
        return {}, ""

    rounds: dict[int, dict[str, Any]] = {}
    titles: dict[int, str] = {}
    recovered_proposer_rounds: set[int] = set()
    for line in transcript.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        round_number = _positive_int(event.get("round_number"))
        if round_number is None:
            continue
        event_type = event.get("type")
        if event_type == "proposer_output":
            output = event.get("output")
            if isinstance(output, Mapping):
                if _major_recovered(output):
                    recovered_proposer_rounds.add(round_number)
                title = str(output.get("title", "")).strip()
                if title:
                    titles[round_number] = title
        elif event_type == "review":
            output = event.get("output")
            if isinstance(output, Mapping):
                marks = output.get("review_payload", {}).get("marks", {})
                if isinstance(marks, Mapping) and "total_score" in marks:
                    if _major_recovered(output) or round_number in recovered_proposer_rounds:
                        marks = _zero_marks(scheme)
                    rounds[round_number] = normalized_marks(marks, scheme)

    final_title = titles[max(titles)] if titles else ""
    return rounds, final_title


def _loop_round_scores_from_round_dirs(
    loop_root: Path,
    *,
    scheme: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], str]:
    rounds_root = loop_root / "rounds"
    if not rounds_root.is_dir():
        return {}, ""
    rounds: dict[int, dict[str, Any]] = {}
    titles: dict[int, str] = {}
    for round_root in sorted(path for path in rounds_root.iterdir() if path.is_dir() and path.name.startswith("round_")):
        round_number = _round_dir_number(round_root)
        if round_number is None:
            continue
        proposer_output = _first_json(round_root / "proposer_outputs")
        recovered_proposer = False
        if proposer_output is not None:
            proposer_payload = _read_json(proposer_output)
            recovered_proposer = _major_recovered(proposer_payload)
            title = str(proposer_payload.get("title", "")).strip()
            if title:
                titles[round_number] = title
        review_path = _first_json(round_root / "reviews")
        if review_path is None:
            continue
        review = _read_json(review_path)
        marks = review.get("review_payload", {}).get("marks", {})
        if isinstance(marks, Mapping) and "total_score" in marks:
            if _major_recovered(review) or recovered_proposer:
                marks = _zero_marks(scheme)
            rounds[round_number] = normalized_marks(marks, scheme)
    final_title = titles[max(titles)] if titles else ""
    return rounds, final_title


def _round_score_markdown(columns: list[str], rows: list[dict[str, Any]], *, max_round: int) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---:" if column.startswith("R") or column in {"Best"} or column.startswith("Δ") else "---" for column in columns) + "|",
    ]
    for row in rows:
        total_scores = {int(key): value for key, value in row["total_scores_by_round"].items()}
        values = [
            row["loop_id"],
            row["group"],
            str(row.get("final_title", "")).replace("|", "/"),
            *[_format_score(total_scores.get(round_number)) for round_number in range(1, max_round + 1)],
            _format_delta(row.get("delta_total")),
            _format_score(row.get("best_total")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_score(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return ""


def _format_delta(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:+g}"
    return ""


def _major_recovered(payload: Mapping[str, Any]) -> bool:
    record = payload.get("arc_llm_call_record")
    if not isinstance(record, Mapping):
        return False
    structured = record.get("structured_output")
    if not isinstance(structured, Mapping):
        return False
    return structured.get("mode") == "recovered" and structured.get("severity") in {"major", "fatal"}


def _zero_marks(scheme: Mapping[str, Any]) -> dict[str, int]:
    return {field: 0 for field in score_fields(scheme)}


def _positive_int(value: Any) -> int | None:
    if not isinstance(value, int) or value <= 0:
        return None
    return value


def _round_dir_number(round_root: Path) -> int | None:
    try:
        value = int(round_root.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None
    return value if value > 0 else None


def _first_json(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    return next(iter(sorted(root.glob("*.json"))), None)


def _max_concurrent_loops(proposal_count: int) -> int:
    raw = os.environ.get("ARC_IDEAS_MAX_CONCURRENT_LOOPS", "12")
    try:
        configured = int(raw)
    except ValueError as exc:
        raise ConfigError("ARC_IDEAS_MAX_CONCURRENT_LOOPS must be a positive integer") from exc
    if configured <= 0:
        raise ConfigError("ARC_IDEAS_MAX_CONCURRENT_LOOPS must be a positive integer")
    return min(proposal_count, configured)


def _concurrency_warning(config: IdeasConfig, proposal_count: int, *, max_concurrent: int) -> str:
    round_counts = sorted({int(_read_json(variant.loop_template).get("max_rounds") or 0) for variant in config.variants})
    if len(round_counts) == 1:
        round_text = f"{round_counts[0]} reviewer reports per loop"
    else:
        round_text = f"reviewer report counts {round_counts}"
    return (
        "WARNING: Running "
        f"{len(config.variants)} variants x {config.loops_per_variant} proposer-reviewer loops "
        f"with {round_text} and loop concurrency capped at {max_concurrent} ({proposal_count} loops). "
        "Concurrent artifacts are written only by arc-llm under the batch run root."
    )


def _model_tier_warnings(batch_config: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    for loop in batch_config.get("loops", []):
        if not isinstance(loop, Mapping):
            continue
        reviewers = loop.get("reviewers")
        proposers = loop.get("proposers")
        if not isinstance(reviewers, list) or not reviewers or not isinstance(proposers, list):
            continue
        reviewer = reviewers[0]
        if not isinstance(reviewer, Mapping):
            continue
        reviewer_tier = _model_tier_text(reviewer.get("model_tier"))
        reviewer_rank = MODEL_TIER_RANKS.get(reviewer_tier)
        if reviewer_rank is None:
            continue
        reviewer_id = str(reviewer.get("id") or "reviewer")
        loop_id = str(loop.get("loop_id") or "loop")
        for proposer in proposers:
            if not isinstance(proposer, Mapping):
                continue
            proposer_tier = _model_tier_text(proposer.get("model_tier"))
            proposer_rank = MODEL_TIER_RANKS.get(proposer_tier)
            if proposer_rank is None or reviewer_rank >= proposer_rank:
                continue
            proposer_id = str(proposer.get("id") or "proposer")
            problems.append(f"{loop_id}: {proposer_id}={proposer_tier} > {reviewer_id}={reviewer_tier}")
    if not problems:
        return []
    return [
        "WARNING: REVIEWER MODEL TIER BELOW PROPOSER. "
        "Reviewer feedback may be less useful when the reviewer is configured with a lower model tier than the proposer. "
        "Affected assignments: "
        + "; ".join(problems)
    ]


def _model_tier_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _write_warnings(path: Path, warnings: list[str]) -> None:
    atomic_write_text(path, "\n".join(warnings).rstrip() + "\n")


def _cross_domain_profile(config: IdeasConfig, *, idea_index: int) -> dict[str, str]:
    profiles = config.exploration_profiles or DEFAULT_CROSS_DOMAIN_PROFILES
    try:
        return copy.deepcopy(profiles[idea_index - 1])
    except IndexError as exc:
        raise ConfigError(f"No cross-domain exploration profile is configured for idea {idea_index}") from exc


def _domain_cards(config: IdeasConfig) -> list[dict[str, Any]]:
    manifest = config.domain_manifest
    if not isinstance(manifest, Mapping):
        raise ConfigError("cross-domain ideas require a domain manifest")
    groups = manifest.get("field_groups")
    packages = manifest.get("domain_packages")
    if not isinstance(groups, list) or not isinstance(packages, list):
        raise ConfigError(f"{config.domain_manifest_path}.field_groups must be an array")
    by_id = {str(item.get("domain_package_id", "")): item for item in packages if isinstance(item, Mapping)}
    cards: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ConfigError(f"{config.domain_manifest_path}.field_groups[{index}] must be an object")
        field_id = str(group.get("field_id", "")).strip()
        field_card = group.get("field_card")
        if not field_id or not isinstance(field_card, Mapping):
            raise ConfigError(f"{config.domain_manifest_path}.field_groups[{index}] requires field_id and field_card")
        versions = []
        opportunities: list[Any] = []
        for package_index, package_id in enumerate(group.get("domain_package_ids", [])):
            package = by_id.get(str(package_id))
            if not isinstance(package, Mapping): raise ConfigError(f"field {field_id!r} references unknown package {package_id!r}")
            summary_path = _domain_summary_path(config, entry=package, index=package_index)
            summary = _read_json(summary_path)
            version = str(summary.get("schema_version", "")).strip()
            if version not in {"arc.domain_summary.v4", "arc.domain_summary.v5"}:
                raise ConfigError(f"{summary_path}.schema_version must be arc.domain_summary.v4 or arc.domain_summary.v5")
            if str(summary.get("domain_id", "")).strip() != str(package_id):
                raise ConfigError(f"package {package_id!r} points to summary for another package: {summary_path}")
            versions.append(version)
            if version == "arc.domain_summary.v5":
                raw = summary.get("mathematical_opportunities")
                validation_error = mathematical_opportunities_validation_error(raw)
                if validation_error is not None: raise ConfigError(f"{summary_path}.mathematical_opportunities is invalid for v5: {validation_error}")
                opportunities.extend(copy.deepcopy(raw.get("well_defined_problems", [])))
        supports = isinstance(versions, list) and bool(versions) and all(item == "arc.domain_summary.v5" for item in versions)
        card = copy.deepcopy(dict(field_card))
        card.update({
            "field_id": field_id,
            "domain_package_ids": list(group.get("domain_package_ids", [])),
            "summary_capabilities": {"mathematical_opportunities": supports},
            "mathematical_opportunities": {"well_defined_problems": opportunities},
        })
        cards.append(card)
    if len(cards) < 2:
        raise ConfigError("cross-domain ideas require at least two distinct field cards")
    return cards


def _domain_summary_path(config: IdeasConfig, *, entry: Mapping[str, Any], index: int) -> Path:
    raw = str(
        entry.get("summary_json_path")
        or entry.get("domain_summary_path")
        or entry.get("summary_path")
        or ""
    ).strip()
    if not raw:
        raise ConfigError(
            f"{config.domain_manifest_path}.domains[{index}] requires summary_json_path"
        )
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        path = candidate
    else:
        project_relative = config.project_dir / candidate
        manifest_relative = config.domain_manifest_path.parent / candidate
        path = project_relative if project_relative.is_file() else manifest_relative
    if not path.is_file():
        raise ConfigError(f"domain summary does not exist: {path}")
    return path.resolve()


def _domain_markdown_files(domain_dir: Path) -> list[dict[str, str]]:
    if not domain_dir.exists():
        return []
    return [
        {
            "path": str(path.relative_to(domain_dir.parent)),
            "content": path.read_text(encoding="utf-8", errors="replace"),
        }
        for path in sorted(domain_dir.rglob("*.md"))
        if path.is_file()
    ]


def _replace_placeholders(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result
    if isinstance(value, list):
        return [_replace_placeholders(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_placeholders(item, replacements) for key, item in value.items()}
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError(f"JSON file must contain an object: {path}")
    return payload


def _read_config_file(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError(f"Config file must contain an object: {path}")
    return payload


def _progress_sidechannel_callback(
    base_env: Mapping[str, str] | None,
) -> Callable[[dict[str, Any]], None] | None:
    environment = base_env if base_env is not None else os.environ
    raw = str(environment.get("ARC_JOB_PROGRESS_FILE", "")).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    lock = threading.Lock()

    def append_progress(event: dict[str, Any]) -> None:
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                os.chmod(path, 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

    return append_progress


def _combined_progress_callback(
    first: Callable[[dict[str, Any]], None] | None,
    second: Callable[[dict[str, Any]], None] | None,
) -> Callable[[dict[str, Any]], None] | None:
    callbacks = tuple(item for item in (first, second) if item is not None)
    if not callbacks:
        return None

    def emit(event: dict[str, Any]) -> None:
        for callback in callbacks:
            callback(dict(event))

    return emit


def _foreground_progress_callback() -> Callable[[dict[str, Any]], None] | None:
    """Stream progress to stderr when no owning arc-jobs side channel exists."""
    if str(os.environ.get("ARC_JOB_PROGRESS_FILE", "")).strip():
        return None

    def emit(event: dict[str, Any]) -> None:
        print(
            json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str),
            file=sys.stderr,
            flush=True,
        )

    return emit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARC ideas workflow helper")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cancel_event = threading.Event()
    installed_handlers: dict[int, Any] = {}

    def request_cancel(_signum: int, _frame: Any) -> None:
        cancel_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            installed_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_cancel)
        except (ValueError, OSError):
            pass
    try:
        result = run_ideas(
            _read_config_file(args.config),
            dry_run=args.dry_run,
            progress_callback=_foreground_progress_callback(),
            cancel_check=cancel_event.is_set,
        )
    finally:
        for signum, handler in installed_handlers.items():
            signal.signal(signum, handler)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        for warning in result.get("warnings", []):
            print(warning)
        print(result["status"])
        table = result.get("round_score_table", {}).get("markdown")
        if table:
            print(table)
    return 1 if result.get("status") in {"failed", "cancelled", "needs_llm"} else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
