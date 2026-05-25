from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from arc_llm.proposers_reviewer.artifacts import atomic_write_json
from arc_llm.proposers_reviewer.runner import run_proposers_reviewer_batch

from research_ideas_config import ConfigError, ResearchIdeasConfig, VariantConfig, load_research_ideas_config
from research_ideas_marking import load_marking_scheme, marking_scheme_for_context, marks_schema


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
    ideas = _materialize_ideas(research_config)
    batch_config = _loop_batch_config(research_config, ideas, run_root=run_root)
    batch_config_path = run_root / "research_ideas_batch_config.json"
    warnings = [_concurrency_warning(research_config, len(ideas))]

    if dry_run:
        return _result(
            research_config,
            run_root=run_root,
            batch_config_path=batch_config_path,
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
        research_config,
        run_root=run_root,
        batch_config_path=batch_config_path,
        warnings=warnings,
        ideas=ideas,
        batch_result=batch_result,
    )


def _result(
    config: ResearchIdeasConfig,
    *,
    run_root: Path,
    batch_config_path: Path,
    warnings: list[str],
    ideas: list[IdeaPlan],
    batch_result: Mapping[str, Any],
) -> dict[str, Any]:
    batch_run_root = Path(str(batch_result.get("run_root", run_root / "idea_loops")))
    return {
        "schema_version": "arc.workflow.research_ideas.result.v1",
        "status": str(batch_result.get("status", "failed")),
        "run_id": config.run_id,
        "run_root": str(run_root),
        "warnings": warnings,
        "proposal_count": len(ideas),
        "reviewer_call_count": 0,
        "loop_reviewer_call_count": _loop_reviewer_call_count(batch_result, ideas),
        "max_concurrent_loops": len(ideas),
        "max_concurrent_proposal_calls": len(ideas),
        "batch_config_path": str(batch_config_path),
        "loops": [_loop_summary(idea, batch_run_root=batch_run_root) for idea in ideas],
        "batch_result": dict(batch_result),
    }


def _materialize_ideas(config: ResearchIdeasConfig) -> list[IdeaPlan]:
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


def _loop_batch_config(config: ResearchIdeasConfig, ideas: list[IdeaPlan], *, run_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "arc.llm.proposers_reviewer_batch.config.v1",
        "run_id": "idea_loops",
        "run_dir": str(run_root),
        "max_concurrent_loops": len(ideas),
        "existing_run_policy": "fail",
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
    payload = _read_json(workflow_dir / "suggest-ideas-reviewer.template.json")
    payload["output_schema"] = _reviewer_output_schema(workflow_dir, scheme=scheme)
    return payload


def _reviewer_output_schema(workflow_dir: Path, *, scheme: Mapping[str, Any]) -> dict[str, Any]:
    schema = _read_json(workflow_dir / "suggest-ideas-reviewer-output.schema.json")
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


def _concurrency_warning(config: ResearchIdeasConfig, proposal_count: int) -> str:
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
