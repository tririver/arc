from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from arc_llm.proposers_reviewer.runner import run_proposers_reviewer_batch

from research_ideas_config import ConfigError, ResearchIdeasConfig, VariantConfig, load_research_ideas_config
from research_ideas_marking import (
    load_marking_scheme,
    marking_scheme_for_context,
    marks_schema,
    normalized_marks,
    report_columns,
)


JsonRunner = Callable[..., dict[str, Any]]
BatchRunner = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class IdeaPlan:
    idea_id: str
    variant_id: str
    idea_index: int
    loop_id: str
    variant: VariantConfig
    caller_context: dict[str, Any]
    root: Path


def run_research_ideas(
    config: ResearchIdeasConfig | Mapping[str, Any],
    *,
    json_runner: JsonRunner | None = None,
    batch_runner: BatchRunner | None = None,
    base_env: Mapping[str, str] | None = None,
    process_chain: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    research_config = config if isinstance(config, ResearchIdeasConfig) else load_research_ideas_config(config)
    run_root = research_config.run_dir / research_config.run_id
    ideas = _materialize_ideas(research_config, run_root=run_root)
    warnings = [_concurrency_warning(research_config, len(ideas))]

    if dry_run:
        return {
            "schema_version": "arc.workflow.research_ideas.result.v1",
            "status": "dry_run",
            "run_id": research_config.run_id,
            "run_root": str(run_root),
            "warnings": warnings,
            "proposal_count": len(ideas),
            "reviewer_call_count": 0,
            "loop_reviewer_call_count": _planned_loop_reviewer_call_count(ideas),
            "max_concurrent_loops": len(ideas),
            "max_concurrent_proposal_calls": len(ideas),
            "ideas": [_idea_plan_summary(idea) for idea in ideas],
        }

    _prepare_run(research_config, run_root=run_root, warnings=warnings)
    try:
        batch_result = _run_idea_loop_batch(
            research_config,
            ideas,
            run_root=run_root,
            json_runner=json_runner,
            batch_runner=batch_runner,
            base_env=base_env,
            process_chain=process_chain,
            save_prompts=research_config.save_prompts,
        )
    except Exception as exc:
        result = _failed_result(
            research_config,
            run_root=run_root,
            warnings=warnings,
            proposal_results=[],
            error=str(exc),
        )
        result["error_type"] = type(exc).__name__
        result["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        atomic_write_json(run_root / "state.json", result)
        return result

    proposal_results = _proposal_results(ideas, batch_result=batch_result)
    failed = [item for item in proposal_results if item["status"] != "completed"]
    status = "completed" if batch_result.get("status") in {"completed", "stopped"} and not failed else "failed"

    if status == "failed":
        error = f"idea loop batch failed with status {batch_result.get('status')}"
        if failed:
            error = f"{len(failed)} proposal loop(s) failed"
        result = _failed_result(
            research_config,
            run_root=run_root,
            warnings=warnings,
            proposal_results=proposal_results,
            error=error,
        )
        result["loop_batch"] = batch_result
        atomic_write_json(run_root / "state.json", result)
        return result

    report = _write_report(research_config, run_root=run_root, warnings=warnings, proposal_results=proposal_results)
    result = {
        "schema_version": "arc.workflow.research_ideas.result.v1",
        "status": "completed",
        "run_id": research_config.run_id,
        "run_root": str(run_root),
        "warnings": warnings,
        "proposal_count": len(proposal_results),
        "reviewer_call_count": 0,
        "loop_reviewer_call_count": _completed_loop_reviewer_call_count(batch_result),
        "max_concurrent_loops": len(ideas),
        "max_concurrent_proposal_calls": len(ideas),
        "ideas": proposal_results,
        "loop_batch": batch_result,
        "report": str(report),
    }
    atomic_write_json(run_root / "state.json", result)
    return result


def _materialize_ideas(config: ResearchIdeasConfig, *, run_root: Path) -> list[IdeaPlan]:
    ideas: list[IdeaPlan] = []
    for variant in config.variants:
        for idea_index in range(1, config.loops_per_variant + 1):
            idea_id = f"{variant.variant_id}/idea_{idea_index:03d}"
            loop_id = f"{variant.variant_id}_idea_{idea_index:03d}"
            ideas.append(
                IdeaPlan(
                    idea_id=idea_id,
                    variant_id=variant.variant_id,
                    idea_index=idea_index,
                    loop_id=loop_id,
                    variant=variant,
                    caller_context=_caller_context(config, variant=variant, idea_id=idea_id),
                    root=run_root / "variants" / variant.variant_id / f"idea_{idea_index:03d}",
                )
            )
    return ideas


def _caller_context(config: ResearchIdeasConfig, *, variant: VariantConfig, idea_id: str) -> dict[str, Any]:
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


def _prepare_run(config: ResearchIdeasConfig, *, run_root: Path, warnings: list[str]) -> None:
    if run_root.exists():
        if config.existing_run_policy == "fail":
            raise ConfigError(f"run directory already exists: {run_root}")
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        run_root / "config.json",
        {
            "schema_version": config.schema_version,
            "run_id": config.run_id,
            "run_dir": str(config.run_dir),
            "project_dir": str(config.project_dir),
            "user_intent": config.user_intent,
            "variant_config_dir": str(config.variant_config_dir),
            "variant_glob": config.variant_glob,
            "loops_per_variant": config.loops_per_variant,
            "warnings": warnings,
        },
    )
    atomic_write_json(run_root / "state.json", {"status": "running", "run_id": config.run_id, "warnings": warnings})


def _run_idea_loop_batch(
    config: ResearchIdeasConfig,
    ideas: list[IdeaPlan],
    *,
    run_root: Path,
    json_runner: JsonRunner | None,
    batch_runner: BatchRunner | None,
    base_env: Mapping[str, str] | None,
    process_chain: list[str] | None,
    save_prompts: bool,
) -> dict[str, Any]:
    batch_config = _loop_batch_config(config, ideas, run_root=run_root, save_prompts=save_prompts)
    atomic_write_json(run_root / "loop_batch_config.json", batch_config)
    run_batch = batch_runner or run_proposers_reviewer_batch
    return run_batch(
        batch_config,
        json_runner=json_runner,
        base_env=base_env,
        process_chain=process_chain,
        dry_run=False,
        max_concurrent_loops=len(ideas),
    )


def _loop_batch_config(
    config: ResearchIdeasConfig,
    ideas: list[IdeaPlan],
    *,
    run_root: Path,
    save_prompts: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "arc.llm.proposers_reviewer_batch.config.v1",
        "run_id": "idea_loops",
        "run_dir": str(run_root / "loop_batch"),
        "max_concurrent_loops": len(ideas),
        "existing_run_policy": "fail",
        "artifact_options": {"save_prompts": save_prompts},
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
    payload = _read_json(workflow_dir / "suggest-ideas-reviewer.template.json")
    payload["output_schema"] = _reviewer_output_schema(workflow_dir, scheme=scheme)
    return payload


def _reviewer_output_schema(workflow_dir: Path, *, scheme: Mapping[str, Any]) -> dict[str, Any]:
    schema = _read_json(workflow_dir / "suggest-ideas-reviewer-output.schema.json")
    schema["properties"]["review_payload"]["properties"]["marks"] = marks_schema(scheme)
    return schema


def _proposal_results(ideas: list[IdeaPlan], *, batch_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    loop_results = {
        str(item.get("loop_id", "")): item
        for item in batch_result.get("loops", [])
        if isinstance(item, Mapping)
    }
    selected: list[dict[str, Any]] = []
    for idea in ideas:
        loop_result = loop_results.get(idea.loop_id)
        if loop_result is None:
            selected.append(_failed_idea_result(idea, "missing loop result"))
            continue
        if loop_result.get("status") not in {"completed", "stopped"}:
            selected.append(_failed_idea_result(idea, str(loop_result.get("error") or loop_result.get("status"))))
            continue
        round_entry = _latest_scored_round(Path(str(loop_result.get("loop_root", ""))))
        if round_entry is None:
            selected.append(_failed_idea_result(idea, "no scored proposer-reviewer round found"))
            continue
        output = _read_json(round_entry["proposer_output_path"])
        review = _read_json(round_entry["review_path"])
        atomic_write_json(idea.root / "output.json", output)
        atomic_write_json(idea.root / "review.json", review)
        atomic_write_json(
            idea.root / "selection.json",
            {
                "loop_id": idea.loop_id,
                "selected_round": round_entry["round"],
                "marks": round_entry["marks"],
                "proposer_output_path": str(round_entry["proposer_output_path"]),
                "review_path": str(round_entry["review_path"]),
            },
        )
        selected.append(
            {
                "idea_id": idea.idea_id,
                "variant_id": idea.variant_id,
                "idea_index": idea.idea_index,
                "loop_id": idea.loop_id,
                "status": "completed",
                "rounds_completed": int(loop_result.get("rounds_completed") or 0),
                "selected_round": round_entry["round"],
                "loop_root": str(loop_result.get("loop_root", "")),
                "output_path": str(idea.root / "output.json"),
                "selected_round_output_path": str(round_entry["proposer_output_path"]),
                "selected_review_path": str(round_entry["review_path"]),
                "selection_path": str(idea.root / "selection.json"),
                "marks": round_entry["marks"],
                "output": output,
            }
        )
    return sorted(selected, key=lambda item: item["idea_id"])


def _failed_idea_result(idea: IdeaPlan, error: str) -> dict[str, Any]:
    return {
        "idea_id": idea.idea_id,
        "variant_id": idea.variant_id,
        "idea_index": idea.idea_index,
        "loop_id": idea.loop_id,
        "status": "failed",
        "error": error,
    }


def _latest_scored_round(loop_root: Path) -> dict[str, Any] | None:
    rounds_root = loop_root / "rounds"
    if not rounds_root.is_dir():
        return None
    entries = [
        entry
        for entry in (_round_entry(round_root) for round_root in sorted(rounds_root.glob("round_*")))
        if entry is not None
    ]
    return entries[-1] if entries else None


def _round_entry(round_root: Path) -> dict[str, Any] | None:
    proposer_output_path = _first_json(round_root / "proposer_outputs")
    review_path = _first_json(round_root / "reviews")
    if proposer_output_path is None or review_path is None:
        return None
    review = _read_json(review_path)
    marks = normalized_marks(review.get("review_payload", {}).get("marks", {}))
    if marks.get("total_score") is None:
        return None
    return {
        "round": _round_number(round_root),
        "marks": marks,
        "proposer_output_path": proposer_output_path,
        "review_path": review_path,
    }


def _first_json(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    return next(iter(sorted(root.glob("*.json"))), None)


def _round_number(round_root: Path) -> int:
    try:
        return int(round_root.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _planned_loop_reviewer_call_count(ideas: list[IdeaPlan]) -> int:
    return sum(int(_read_json(idea.variant.loop_template).get("max_rounds") or 0) for idea in ideas)


def _completed_loop_reviewer_call_count(batch_result: Mapping[str, Any]) -> int:
    return sum(
        int(item.get("rounds_completed") or 0)
        for item in batch_result.get("loops", [])
        if isinstance(item, Mapping)
    )


def _write_report(
    config: ResearchIdeasConfig,
    *,
    run_root: Path,
    warnings: list[str],
    proposal_results: list[dict[str, Any]],
) -> Path:
    columns = report_columns(load_marking_scheme(config.variant_config_dir))
    mark_header = " | ".join(column["label"] for column in columns)
    mark_separator = "|".join("---:" for _ in columns)
    lines = [
        "# Research Ideas",
        "",
        f"Run: `{config.run_id}`",
        f"User intent: {config.user_intent}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "",
            "## Loop Outputs",
            "",
            f"| Idea ID | Variant | Loop | Selected Round | Title | {mark_header} |",
            f"|---|---|---|---:|---|{mark_separator}|",
        ]
    )
    for item in proposal_results:
        output = item.get("output", {})
        marks = item.get("marks", {})
        if not isinstance(output, Mapping):
            output = {}
        if not isinstance(marks, Mapping):
            marks = {}
        mark_values = " | ".join(str(_mark(marks, column["field"])) for column in columns)
        lines.append(
            f"| `{item['idea_id']}` | `{item['variant_id']}` | `{item['loop_id']}` | "
            f"{item.get('selected_round', '')} | {output.get('title', '')} | {mark_values} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Loop batch: `{run_root / 'loop_batch' / 'idea_loops' / 'loops'}`",
            "- Per-loop summaries: `<run-root>/variants/<variant-id>/idea_<n>/`",
        ]
    )
    path = run_root / "research-ideas.md"
    atomic_write_text(path, "\n".join(lines).rstrip() + "\n")
    try:
        atomic_write_text(config.project_dir / "research-ideas.md", path.read_text(encoding="utf-8"))
    except OSError:
        pass
    return path


def _failed_result(
    config: ResearchIdeasConfig,
    *,
    run_root: Path,
    warnings: list[str],
    proposal_results: list[dict[str, Any]],
    error: str,
) -> dict[str, Any]:
    return {
        "schema_version": "arc.workflow.research_ideas.result.v1",
        "status": "failed",
        "run_id": config.run_id,
        "run_root": str(run_root),
        "warnings": warnings,
        "proposal_count": len(proposal_results),
        "reviewer_call_count": 0,
        "loop_reviewer_call_count": 0,
        "max_concurrent_loops": len(proposal_results),
        "max_concurrent_proposal_calls": len(proposal_results),
        "ideas": proposal_results,
        "error": error,
    }


def _concurrency_warning(config: ResearchIdeasConfig, proposal_count: int) -> str:
    round_counts = sorted({int(_read_json(variant.loop_template).get("max_rounds") or 0) for variant in config.variants})
    if len(round_counts) == 1:
        round_text = f"{round_counts[0]} reviewer reports per loop"
    else:
        round_text = f"reviewer report counts {round_counts}"
    return (
        "WARNING: Running "
        f"{len(config.variants)} variants x {config.loops_per_variant} proposer-reviewer loops "
        f"with {round_text} and unlimited loop concurrency ({proposal_count} loops)."
    )


def _mark(marks: Mapping[str, Any], field: str) -> Any:
    value = marks.get(field, "")
    if isinstance(value, float):
        return f"{value:g}"
    return value


def _idea_plan_summary(idea: IdeaPlan) -> dict[str, Any]:
    return {
        "idea_id": idea.idea_id,
        "variant_id": idea.variant_id,
        "idea_index": idea.idea_index,
        "loop_id": idea.loop_id,
        "output_path": str(idea.root / "output.json"),
    }


def _domain_markdown_files(domain_dir: Path) -> list[dict[str, str]]:
    if not domain_dir.exists():
        return []
    files: list[dict[str, str]] = []
    for path in sorted(domain_dir.rglob("*.md")):
        if path.is_file():
            files.append({"path": str(path.relative_to(domain_dir.parent)), "content": path.read_text(encoding="utf-8")})
    return files


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


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _provider_config_env(provider_config: str | None) -> dict[str, str] | None:
    if not provider_config:
        return None
    env = dict(os.environ)
    env["ARC_LLM_PROVIDER_CONFIG"] = provider_config
    return env


def _read_config_file(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError(f"Config file must contain an object: {path}")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARC research-ideas workflow helper")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--provider-config", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_research_ideas(
        _read_config_file(args.config),
        dry_run=args.dry_run,
        base_env=_provider_config_env(args.provider_config),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str) if args.json else result["status"])
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
