from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from arc_llm.proposers_reviewer.artifacts import atomic_write_json, atomic_write_text
from arc_llm.proposers_reviewer.runner import run_proposers_reviewer_batch

from ideas_config import ConfigError, IdeasConfig, VariantConfig, load_ideas_config
from ideas_marking import load_marking_scheme, marking_scheme_for_context, marks_schema, normalized_marks


JsonRunner = Callable[..., dict[str, Any]]
BatchRunner = Callable[..., dict[str, Any]]
MODEL_TIER_RANKS = {"low": 1, "medium": 2, "high": 3}


@dataclass(frozen=True)
class IdeaPlan:
    idea_id: str
    variant_id: str
    idea_index: int
    loop_id: str
    variant: VariantConfig
    caller_context: dict[str, Any]


def run_ideas(
    config: IdeasConfig | Mapping[str, Any],
    *,
    json_runner: JsonRunner | None = None,
    batch_runner: BatchRunner | None = None,
    base_env: Mapping[str, str] | None = None,
    process_chain: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    ideas_config = config if isinstance(config, IdeasConfig) else load_ideas_config(config)
    run_root = ideas_config.run_dir / ideas_config.run_id
    ideas = _materialize_ideas(ideas_config)
    batch_config = _loop_batch_config(ideas_config, ideas, run_root=run_root)
    batch_config_path = run_root / "ideas_batch_config.json"
    warnings_path = run_root / "ideas_warnings.txt"
    warnings = [
        _concurrency_warning(ideas_config, len(ideas)),
        *_model_tier_warnings(batch_config),
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
        batch_result = (batch_runner or run_proposers_reviewer_batch)(
            batch_config,
            json_runner=json_runner,
            base_env=base_env,
            process_chain=process_chain,
            dry_run=False,
            max_concurrent_loops=len(ideas),
        )
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
        "warnings": warnings,
        "proposal_count": len(ideas),
        "reviewer_call_count": loop_reviewer_call_count,
        "loop_reviewer_call_count": loop_reviewer_call_count,
        "max_concurrent_loops": len(ideas),
        "max_concurrent_proposal_calls": len(ideas),
        "batch_config_path": str(batch_config_path),
        "warnings_path": str(warnings_path),
        "loops": [_loop_summary(idea, batch_run_root=batch_run_root) for idea in ideas],
        "round_score_table": round_score_table,
        "batch_result": dict(batch_result),
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
                    caller_context=_caller_context(config, variant=variant, idea_id=idea_id),
                )
            )
    return ideas


def _caller_context(config: IdeasConfig, *, variant: VariantConfig, idea_id: str) -> dict[str, Any]:
    loop_template = _read_json(variant.loop_template)
    caller_context = copy.deepcopy(loop_template.get("caller_context", {}))
    if not isinstance(caller_context, dict):
        raise ConfigError(f"{variant.loop_template}.caller_context must be an object")
    caller_context = _replace_placeholders(caller_context, {"<user_intent>": config.user_intent})
    caller_context["user_intent"] = config.user_intent
    caller_context["variant_id"] = variant.variant_id
    caller_context["idea_id"] = idea_id
    caller_context["marking_scheme"] = marking_scheme_for_context(load_marking_scheme(variant.path.parent))
    if variant.context_policy.attach_domain_markdown:
        markdown_files = _domain_markdown_files(config.project_dir / "domain")
        if variant.context_policy.require_domain_markdown and not markdown_files:
            raise ConfigError(f"{variant.variant_id} requires domain markdown under {config.project_dir / 'domain'}")
        caller_context["domain_markdown_files"] = markdown_files
    else:
        caller_context.pop("domain_markdown_files", None)
    if not variant.context_policy.attach_arc_paper_tool_notes:
        caller_context.pop("arc_paper_tool_notes", None)
    return caller_context


def _loop_batch_config(config: IdeasConfig, ideas: list[IdeaPlan], *, run_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "arc.llm.proposers_reviewer_batch.config.v1",
        "run_id": "idea_loops",
        "run_dir": str(run_root),
        "max_concurrent_loops": len(ideas),
        "artifact_options": {"save_prompts": config.save_prompts},
        "loops": [_idea_loop_payload(idea) for idea in ideas],
    }


def _idea_loop_payload(idea: IdeaPlan) -> dict[str, Any]:
    loop = copy.deepcopy(_read_json(idea.variant.loop_template))
    loop["loop_id"] = idea.loop_id
    loop["caller_context"] = copy.deepcopy(idea.caller_context)
    loop["proposers"] = [_proposer_payload(idea.variant)]
    loop["reviewers"] = [_loop_reviewer_payload(idea.variant)]
    return loop


def _proposer_payload(variant: VariantConfig) -> dict[str, Any]:
    return _merged_worker_payload(_read_json(variant.proposer_template), variant.proposer_overrides)


def _merged_worker_payload(template: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(template))
    for key, value in overrides.items():
        if key in {"runtime", "prompt"}:
            target = merged.setdefault(key, {})
            if not isinstance(target, dict):
                target = {}
            target.update(value if isinstance(value, dict) else {})
            merged[key] = target
        else:
            merged[key] = value
    return merged


def _loop_reviewer_payload(variant: VariantConfig) -> dict[str, Any]:
    workflow_dir = variant.path.parent
    scheme = load_marking_scheme(workflow_dir)
    payload = _read_json(workflow_dir / "ideas-reviewer.template.json")
    payload["output_schema"] = _reviewer_output_schema(workflow_dir, scheme=scheme)
    return payload


def _reviewer_output_schema(workflow_dir: Path, *, scheme: Mapping[str, Any]) -> dict[str, Any]:
    schema = _read_json(workflow_dir / "ideas-reviewer-output.schema.json")
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
    scheme = load_marking_scheme(idea.variant.path.parent)
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
                title = str(output.get("title", "")).strip()
                if title:
                    titles[round_number] = title
        elif event_type == "review":
            output = event.get("output")
            if isinstance(output, Mapping):
                marks = output.get("review_payload", {}).get("marks", {})
                if isinstance(marks, Mapping) and "total_score" in marks:
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
        if proposer_output is not None:
            title = str(_read_json(proposer_output).get("title", "")).strip()
            if title:
                titles[round_number] = title
        review_path = _first_json(round_root / "reviews")
        if review_path is None:
            continue
        marks = _read_json(review_path).get("review_payload", {}).get("marks", {})
        if isinstance(marks, Mapping) and "total_score" in marks:
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


def _concurrency_warning(config: IdeasConfig, proposal_count: int) -> str:
    round_counts = sorted({int(_read_json(variant.loop_template).get("max_rounds") or 0) for variant in config.variants})
    if len(round_counts) == 1:
        round_text = f"{round_counts[0]} reviewer reports per loop"
    else:
        round_text = f"reviewer report counts {round_counts}"
    return (
        "WARNING: Running "
        f"{len(config.variants)} variants x {config.loops_per_variant} proposer-reviewer loops "
        f"with {round_text} and unlimited loop concurrency ({proposal_count} loops). "
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARC ideas workflow helper")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_ideas(
        _read_config_file(args.config),
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        for warning in result.get("warnings", []):
            print(warning)
        print(result["status"])
        table = result.get("round_score_table", {}).get("markdown")
        if table:
            print(table)
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
