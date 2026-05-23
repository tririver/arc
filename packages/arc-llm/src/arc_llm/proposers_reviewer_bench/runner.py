from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping

from arc_llm.proposers_reviewer.runner import run_proposers_reviewer_batch
from arc_llm.runner import run_json

from .config import (
    BenchConfig,
    apply_improvement_edits,
    improvement_output_schema,
    load_bench_config,
    materialize_batch_payload,
)


BatchRunner = Callable[..., dict[str, Any]]
ImproverJsonRunner = Callable[..., dict[str, Any]]


def run_proposers_reviewer_bench(
    config: BenchConfig | Mapping[str, Any],
    *,
    batch_runner: BatchRunner | None = None,
    improver_json_runner: ImproverJsonRunner | None = None,
    base_env: Mapping[str, str] | None = None,
    process_chain: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    bench = config if isinstance(config, BenchConfig) else load_bench_config(config)
    run_batch = batch_runner or run_proposers_reviewer_batch
    call_improver = improver_json_runner or run_json

    current_template = bench.batch_payload
    current_payload = materialize_batch_payload(bench, iteration_index=0, candidate_id="current", base_payload_override=current_template)
    if dry_run:
        return {
            "schema_version": "arc.llm.proposers_reviewer_bench.result.v1",
            "status": "dry_run",
            "best_run_id": current_payload["run_id"],
            "batch_payload": current_payload,
        }

    current_result = run_batch(
        current_payload,
        base_env=base_env,
        process_chain=process_chain,
    )
    current_stats = _score_run(current_result, score_path=bench.options.score_path)
    best_run_id = current_payload["run_id"]
    best_stats = current_stats
    no_improvement = 0
    iterations: list[dict[str, Any]] = []
    status = "completed"
    stop_reason = ""

    for iteration_index in range(1, bench.options.max_iterations + 1):
        prompt = _improver_prompt(
            bench=bench,
            current_payload=current_payload,
            current_result=current_result,
            current_stats=current_stats,
            best_run_id=best_run_id,
            best_stats=best_stats,
            iteration_index=iteration_index,
            previous_iterations=iterations,
        )
        improvement = call_improver(
            prompt,
            schema=improvement_output_schema(),
            provider=bench.options.improver_provider,
            model=bench.options.improver_model,
            model_tier=bench.options.improver_model_tier,
            env=base_env,
            process_chain=process_chain,
        )
        candidate_template, applied_edits = apply_improvement_edits(
            current_template,
            improvement,
            allow_reviewer_prompt_edits=bench.options.allow_reviewer_prompt_edits,
        )
        candidate_payload = materialize_batch_payload(
            bench,
            iteration_index=iteration_index,
            candidate_id=f"candidate{iteration_index:03d}",
            base_payload_override=candidate_template,
        )
        candidate_result = run_batch(
            candidate_payload,
            base_env=base_env,
            process_chain=process_chain,
        )
        candidate_stats = _score_run(candidate_result, score_path=bench.options.score_path)
        decision = _decision(current_stats, candidate_stats, min_delta=bench.options.min_delta, min_z=bench.options.min_z)
        iteration_record = {
            "iteration": iteration_index,
            "current_run_id": current_payload["run_id"],
            "candidate_run_id": candidate_payload["run_id"],
            "current_stats": current_stats,
            "candidate_stats": candidate_stats,
            "improvement": improvement,
            "applied_edits": applied_edits,
            "decision": decision,
        }
        iterations.append(iteration_record)

        if decision["accepted"]:
            current_template = candidate_template
            current_payload = candidate_payload
            current_result = candidate_result
            current_stats = candidate_stats
            best_run_id = candidate_payload["run_id"]
            best_stats = candidate_stats
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= bench.options.patience:
                status = "stopped"
                stop_reason = "patience_exhausted"
                break

    return {
        "schema_version": "arc.llm.proposers_reviewer_bench.result.v1",
        "status": status,
        "stop_reason": stop_reason,
        "best_run_id": best_run_id,
        "best_stats": best_stats,
        "baseline_run_id": materialize_batch_payload(
            bench,
            iteration_index=0,
            candidate_id="current",
            base_payload_override=bench.batch_payload,
        )["run_id"],
        "baseline_stats": _score_run(current_result, score_path=bench.options.score_path)
        if best_run_id == current_payload["run_id"] and not iterations
        else iterations[0]["current_stats"]
        if iterations
        else current_stats,
        "iterations": iterations,
    }


def _score_run(batch_result: Mapping[str, Any], *, score_path: str) -> dict[str, Any]:
    run_root = Path(str(batch_result.get("run_root", "")))
    scores: list[float] = []
    files: list[str] = []
    if run_root:
        for path in sorted(run_root.glob("loops/*/rounds/round_*/reviews/*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            value = _dig(payload, score_path)
            if isinstance(value, (int, float)):
                scores.append(float(value))
                files.append(str(path))
    mean = statistics.fmean(scores) if scores else 0.0
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    se = stdev / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
    if not scores:
        raise ValueError(f"No numeric scores found at {score_path} under {run_root}")
    return {
        "score_path": score_path,
        "n": len(scores),
        "mean": mean,
        "stdev": stdev,
        "se": se,
        "files": files,
    }


def _decision(current: Mapping[str, Any], candidate: Mapping[str, Any], *, min_delta: float, min_z: float) -> dict[str, Any]:
    delta = float(candidate["mean"]) - float(current["mean"])
    combined_se = math.sqrt(float(candidate["se"]) ** 2 + float(current["se"]) ** 2)
    if combined_se > 0:
        z = delta / combined_se
    else:
        z = math.inf if delta > 0 else 0.0
    accepted = delta >= min_delta and z >= min_z
    return {
        "accepted": accepted,
        "delta": delta,
        "combined_se": combined_se,
        "z": z,
        "min_delta": min_delta,
        "min_z": min_z,
    }


def _improver_prompt(
    *,
    bench: BenchConfig,
    current_payload: Mapping[str, Any],
    current_result: Mapping[str, Any],
    current_stats: Mapping[str, Any],
    best_run_id: str,
    best_stats: Mapping[str, Any],
    iteration_index: int,
    previous_iterations: list[dict[str, Any]],
) -> str:
    run_root = str(current_result.get("run_root", ""))
    loop_paths = [
        {
            "loop_id": loop.get("loop_id", ""),
            "transcript": str(Path(run_root) / "loops" / str(loop.get("loop_id", "")) / "transcript.jsonl"),
            "loop_root": str(Path(run_root) / "loops" / str(loop.get("loop_id", ""))),
        }
        for loop in current_result.get("loops", [])
        if isinstance(loop, dict)
    ]
    summary = {
        "iteration_index": iteration_index,
        "current_run_id": current_payload["run_id"],
        "current_run_root": run_root,
        "current_stats": current_stats,
        "best_run_id": best_run_id,
        "best_stats": best_stats,
        "current_loop_template": _loop_template_for_prompt(current_payload),
        "previous_decisions": [
            {
                "iteration": item["iteration"],
                "candidate_run_id": item["candidate_run_id"],
                "decision": item["decision"],
                "candidate_stats": item["candidate_stats"],
            }
            for item in previous_iterations
        ],
        "loop_artifacts": loop_paths,
    }
    inline_context = _inline_artifact_context(bench=bench, current_result=current_result)
    return (
        "You are improving prompts for an ARC proposers-reviewer benchmark.\n"
        "Read the artifact files from disk when details are needed. Do not ask for histories to be pasted inline.\n"
        "Worker outputs may include a top-level `suggested_improvement` object. Do not directly follow every "
        "`suggested_improvement`; read and judge those suggestions against scores, transcript histories, reviews, "
        "tool traces, and the current prompt before proposing edits.\n"
        "Focus on changes to proposer prompts unless the benchmark config explicitly allows reviewer prompt edits.\n"
        "Return JSON using schema_version arc.llm.proposers_reviewer_bench.improvement.v1.\n"
        "Allowed edit targets are proposers.*.prompt.template and reviewers.*.prompt.template.\n"
        "Allowed operations are append_paragraph and replace.\n\n"
        "Benchmark summary:\n"
        f"{json.dumps(summary, indent=2, ensure_ascii=False)}\n\n"
        "Inline artifact context:\n"
        f"{json.dumps(inline_context, indent=2, ensure_ascii=False)}\n"
    )


def _dig(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _loop_template_for_prompt(current_payload: Mapping[str, Any]) -> dict[str, Any]:
    loops = current_payload.get("loops", [])
    if not isinstance(loops, list) or not loops:
        return {}
    loop = loops[0]
    if not isinstance(loop, Mapping):
        return {}
    return {
        "loop_id_pattern": "sample loops are materialized from this first loop template",
        "max_rounds": loop.get("max_rounds"),
        "caller_context": loop.get("caller_context", {}),
        "proposers": _workers_for_prompt(loop.get("proposers", [])),
        "reviewers": _workers_for_prompt(loop.get("reviewers", [])),
    }


def _workers_for_prompt(raw_workers: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_workers, list):
        return []
    workers: list[dict[str, Any]] = []
    for worker in raw_workers:
        if not isinstance(worker, Mapping):
            continue
        prompt = worker.get("prompt", {})
        workers.append(
            {
                "id": worker.get("id", ""),
                "prompt": prompt if isinstance(prompt, Mapping) else {},
                "output_schema": worker.get("output_schema", {}),
                "provider": worker.get("provider", ""),
                "model_tier": worker.get("model_tier", ""),
                "runtime": worker.get("runtime", {}),
            }
        )
    return workers


def _inline_artifact_context(*, bench: BenchConfig, current_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not _should_inline_artifacts(bench):
        return []
    run_root = Path(str(current_result.get("run_root", "")))
    if not run_root.exists():
        return []
    remaining = bench.options.improver_context_max_chars
    excerpts: list[dict[str, Any]] = []
    for path in _artifact_paths_for_inline_context(run_root):
        if remaining <= 0:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text:
            continue
        excerpt = text[:remaining]
        remaining -= len(excerpt)
        excerpts.append(
            {
                "path": str(path),
                "truncated": len(excerpt) < len(text),
                "content": excerpt,
            }
        )
    return excerpts


def _should_inline_artifacts(bench: BenchConfig) -> bool:
    mode = bench.options.improver_context_mode
    if mode == "expanded":
        return True
    if mode == "paths":
        return False
    provider = f"{bench.options.improver_provider} {bench.options.default_provider}".lower()
    return "deepseek" in provider


def _artifact_paths_for_inline_context(run_root: Path) -> list[Path]:
    paths: list[Path] = []
    paths.extend(sorted(run_root.glob("loops/*/transcript.jsonl")))
    paths.extend(sorted(run_root.glob("loops/*/rounds/round_*/reviews/*.json")))
    paths.extend(sorted(run_root.glob("loops/*/rounds/round_*/proposer_outputs/*.json")))
    paths.extend(sorted(run_root.glob("loops/*/rounds/round_*/errors/*.json")))
    return paths
