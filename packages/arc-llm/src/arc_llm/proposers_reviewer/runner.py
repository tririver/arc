from __future__ import annotations

import copy
import json
import traceback
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from arc_llm.runner import run_json

from .artifacts import RunPaths, acquire_lock, append_jsonl, atomic_write_json, atomic_write_text
from .config import (
    ArtifactOptions,
    REVIEW_ENVELOPE_SCHEMA,
    BatchConfig,
    ConfigError,
    LoopConfig,
    WorkerConfig,
    load_batch_config,
    worker_env,
)
from .prompts import proposer_context, render_prompt, reviewer_context


JsonRunner = Callable[..., dict[str, Any]]


def run_proposers_reviewer_batch(
    config: BatchConfig | Mapping[str, Any],
    *,
    json_runner: JsonRunner | None = None,
    base_env: Mapping[str, str] | None = None,
    process_chain: list[str] | None = None,
    dry_run: bool = False,
    max_concurrent_loops: int | None = None,
) -> dict[str, Any]:
    batch = config if isinstance(config, BatchConfig) else load_batch_config(config)
    concurrency = max_concurrent_loops or batch.max_concurrent_loops
    if concurrency <= 0:
        raise ConfigError("max_concurrent_loops must be a positive integer")
    paths = RunPaths(run_dir=batch.run_dir, run_id=batch.run_id)
    if dry_run:
        return _dry_run_result(batch, paths)
    _prepare_run(paths, batch)

    loop_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_by_loop = {
            executor.submit(
                _run_loop,
                copy.deepcopy(loop),
                paths.loop(loop.loop_id),
                batch.run_id,
                batch.artifact_options,
                json_runner,
                base_env,
                process_chain,
            ): loop.loop_id
            for loop in batch.loops
        }
        for future in as_completed(future_by_loop):
            try:
                result = future.result()
            except CancelledError:
                result = {
                    "loop_id": future_by_loop[future],
                    "status": "skipped",
                    "rounds_completed": 0,
                    "error": "cancelled by fail_fast",
                    "loop_root": str(paths.loop(future_by_loop[future]).loop_root),
                }
            loop_results.append(result)
            if batch.fail_fast and result["status"] == "failed":
                for pending in future_by_loop:
                    if pending is not future:
                        pending.cancel()

    loop_results.sort(key=lambda item: item["loop_id"])
    status = _batch_status(loop_results)
    run_result = {
        "schema_version": "arc.llm.proposers_reviewer_batch.result.v1",
        "status": status,
        "run_id": batch.run_id,
        "run_root": str(paths.run_root),
        "loops": loop_results,
    }
    _write_run_state(paths, run_result)
    return run_result


def _prepare_run(paths: RunPaths, batch: BatchConfig) -> None:
    if paths.run_root.exists() and batch.existing_run_policy == "fail":
        raise ConfigError(f"run directory already exists: {paths.run_root}")
    paths.run_root.mkdir(parents=True, exist_ok=True)
    with acquire_lock(paths.lock, run_id=batch.run_id):
        atomic_write_json(paths.config, _jsonable(batch))
        atomic_write_json(
            paths.manifest,
            {
                "schema_version": "arc.llm.proposers_reviewer_manifest.v1",
                "run_id": batch.run_id,
                "loops": [
                    {
                        "loop_id": loop.loop_id,
                        "path": str(paths.loop(loop.loop_id).loop_root),
                    }
                    for loop in batch.loops
                ],
            },
        )
        atomic_write_json(paths.state, {"status": "running", "run_id": batch.run_id})


def _write_run_state(paths: RunPaths, run_result: dict[str, Any]) -> None:
    with acquire_lock(paths.lock, run_id=run_result["run_id"]):
        atomic_write_json(paths.state, run_result)


def _run_loop(
    loop: LoopConfig,
    paths,
    run_id: str,
    artifact_options: ArtifactOptions,
    json_runner: JsonRunner | None,
    base_env: Mapping[str, str] | None,
    process_chain: list[str] | None,
) -> dict[str, Any]:
    if paths.loop_root.exists():
        return _loop_failure(loop.loop_id, paths, "loop directory already exists")
    try:
        with acquire_lock(paths.lock, run_id=run_id, loop_id=loop.loop_id):
            atomic_write_json(paths.config, _jsonable(loop))
            atomic_write_json(paths.state, {"status": "running", "loop_id": loop.loop_id, "rounds_completed": 0})
            result = _run_loop_rounds(loop, paths, artifact_options, json_runner, base_env, process_chain)
            atomic_write_json(paths.state, result)
            return result
    except Exception as exc:
        result = _loop_failure(loop.loop_id, paths, str(exc), exc=exc)
        atomic_write_json(paths.state, result)
        return result


def _run_loop_rounds(
    loop: LoopConfig,
    paths,
    artifact_options: ArtifactOptions,
    json_runner: JsonRunner | None,
    base_env: Mapping[str, str] | None,
    process_chain: list[str] | None,
) -> dict[str, Any]:
    correspondence: list[dict[str, Any]] = []
    rounds_completed = 0
    stop_reason = ""
    for round_number in range(1, loop.max_rounds + 1):
        round_paths = paths.round(round_number)
        proposer_outputs = _run_proposers(
            loop,
            round_paths,
            round_number,
            correspondence,
            artifact_options,
            json_runner,
            base_env,
            process_chain,
        )
        reviewer = loop.reviewers[0]
        review_context = reviewer_context(
            loop=loop,
            worker=reviewer,
            round_number=round_number,
            correspondence=copy.deepcopy(correspondence),
            current_proposer_outputs=proposer_outputs,
        )
        review_prompt = render_prompt(reviewer, review_context)
        atomic_write_json(round_paths.reviewer_context(reviewer.id), review_context)
        review_prompt_path = round_paths.prompt(reviewer.id)
        if artifact_options.save_prompts:
            atomic_write_text(review_prompt_path, review_prompt)
        review_output = _call_reviewer_with_validation_retry(
            json_runner,
            review_prompt,
            worker=reviewer,
            loop=loop,
            round_number=round_number,
            error_path=round_paths.worker_error(reviewer.id),
            prompt_path=review_prompt_path if artifact_options.save_prompts else None,
            save_prompt=artifact_options.save_prompts,
            base_env=base_env,
            process_chain=process_chain,
        )
        atomic_write_json(round_paths.review(reviewer.id), review_output)

        round_events = _round_events(
            round_number=round_number,
            proposer_outputs=proposer_outputs,
            review_output=review_output,
            reviewer_id=reviewer.id,
        )
        for event in round_events:
            correspondence.append(event)
            append_jsonl(paths.transcript, event)

        rounds_completed = round_number
        controller = review_output.get("controller", {})
        if controller.get("stop_requested") and loop.early_stop_enabled:
            stop_reason = str(controller.get("stop_reason") or controller.get("message") or "")
            return {
                "loop_id": loop.loop_id,
                "status": "stopped",
                "rounds_completed": rounds_completed,
                "stop_reason": stop_reason,
                "loop_root": str(paths.loop_root),
            }

    return {
        "loop_id": loop.loop_id,
        "status": "completed",
        "rounds_completed": rounds_completed,
        "stop_reason": stop_reason,
        "loop_root": str(paths.loop_root),
    }


def _run_proposers(
    loop: LoopConfig,
    round_paths,
    round_number: int,
    correspondence: list[dict[str, Any]],
    artifact_options: ArtifactOptions,
    json_runner: JsonRunner | None,
    base_env: Mapping[str, str] | None,
    process_chain: list[str] | None,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    prompts: dict[str, str] = {}
    prompt_paths: dict[str, Path | None] = {}
    for proposer in loop.proposers:
        context = proposer_context(
            loop=loop,
            worker=proposer,
            round_number=round_number,
            correspondence=copy.deepcopy(correspondence),
        )
        prompt = render_prompt(proposer, context)
        prompts[proposer.id] = prompt
        prompt_path = round_paths.prompt(proposer.id)
        prompt_paths[proposer.id] = prompt_path if artifact_options.save_prompts else None
        atomic_write_json(round_paths.proposer_context(proposer.id), context)
        if artifact_options.save_prompts:
            atomic_write_text(prompt_path, prompt)

    with ThreadPoolExecutor(max_workers=len(loop.proposers)) as executor:
        future_by_proposer = {
            executor.submit(
                _call_json_runner_with_error_artifact,
                json_runner,
                prompts[proposer.id],
                worker=proposer,
                round_number=round_number,
                error_path=round_paths.worker_error(proposer.id),
                prompt_path=prompt_paths[proposer.id],
                base_env=base_env,
                process_chain=process_chain,
            ): proposer
            for proposer in loop.proposers
        }
        for future in as_completed(future_by_proposer):
            proposer = future_by_proposer[future]
            output = future.result()
            outputs[proposer.id] = output
            atomic_write_json(round_paths.proposer_output(proposer.id), output)
    return dict(sorted(outputs.items()))


def _call_reviewer_with_validation_retry(
    json_runner: JsonRunner | None,
    prompt: str,
    *,
    worker: WorkerConfig,
    loop: LoopConfig,
    round_number: int,
    error_path: Path,
    prompt_path: Path | None,
    save_prompt: bool,
    base_env: Mapping[str, str] | None,
    process_chain: list[str] | None,
) -> dict[str, Any]:
    review_output = _call_json_runner_with_error_artifact(
        json_runner,
        prompt,
        worker=worker,
        round_number=round_number,
        error_path=error_path,
        prompt_path=prompt_path,
        base_env=base_env,
        process_chain=process_chain,
    )
    try:
        _validate_review_envelope(review_output, loop)
        return review_output
    except Exception as exc:
        retry_prompt = _review_validation_retry_prompt(prompt, exc)
        if save_prompt and prompt_path is not None:
            atomic_write_text(prompt_path, retry_prompt)
        review_output = _call_json_runner_with_error_artifact(
            json_runner,
            retry_prompt,
            worker=worker,
            round_number=round_number,
            error_path=error_path,
            prompt_path=prompt_path,
            base_env=base_env,
            process_chain=process_chain,
        )
        _validate_review_envelope(review_output, loop)
        return review_output


def _review_validation_retry_prompt(prompt: str, exc: Exception) -> str:
    return (
        f"{prompt.rstrip()}\n\n"
        "## Reviewer Output Retry\n"
        f"Previous reviewer response failed validation: {exc}\n"
        "Retry the review. Return exactly one JSON object using schema_version "
        f"{REVIEW_ENVELOPE_SCHEMA}, with controller, proposer_messages for every proposer, "
        "and review_payload. Do not return a proposer idea or any non-envelope object.\n"
    )


def _call_json_runner_with_error_artifact(
    json_runner: JsonRunner | None,
    prompt: str,
    *,
    worker: WorkerConfig,
    round_number: int,
    error_path: Path,
    prompt_path: Path | None,
    base_env: Mapping[str, str] | None,
    process_chain: list[str] | None,
) -> dict[str, Any]:
    try:
        return _call_json_runner(
            json_runner,
            prompt,
            worker=worker,
            base_env=base_env,
            process_chain=process_chain,
        )
    except Exception as exc:
        atomic_write_json(
            error_path,
            {
                "worker_id": worker.id,
                "round_number": round_number,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "prompt_path": str(prompt_path) if prompt_path is not None else "",
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            },
        )
        raise


def _call_json_runner(
    json_runner: JsonRunner | None,
    prompt: str,
    *,
    worker: WorkerConfig,
    base_env: Mapping[str, str] | None,
    process_chain: list[str] | None,
) -> dict[str, Any]:
    env = worker_env(worker, base_env=base_env)
    if json_runner is not None:
        return json_runner(
            prompt,
            schema=worker.output_schema,
            provider=worker.provider,
            model=worker.model,
            env=env,
        )
    return run_json(
        prompt,
        schema=worker.output_schema,
        provider=worker.provider,
        model=worker.model,
        model_tier=worker.model_tier,
        env=env,
        process_chain=process_chain,
    )


def _validate_review_envelope(review: dict[str, Any], loop: LoopConfig) -> None:
    if review.get("schema_version") != REVIEW_ENVELOPE_SCHEMA:
        raise ValueError(f"review schema_version must be {REVIEW_ENVELOPE_SCHEMA}")
    controller = review.get("controller")
    if not isinstance(controller, dict):
        raise ValueError("review.controller must be an object")
    if not isinstance(controller.get("stop_requested", False), bool):
        raise ValueError("review.controller.stop_requested must be a boolean")
    proposer_messages = review.get("proposer_messages")
    if not isinstance(proposer_messages, dict):
        raise ValueError("review.proposer_messages must be an object")
    missing = [proposer.id for proposer in loop.proposers if proposer.id not in proposer_messages]
    if missing:
        raise ValueError(f"review.proposer_messages missing: {', '.join(missing)}")
    if not isinstance(review.get("review_payload"), dict):
        raise ValueError("review.review_payload must be an object")


def _round_events(
    *,
    round_number: int,
    proposer_outputs: dict[str, Any],
    review_output: dict[str, Any],
    reviewer_id: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for proposer_id, output in proposer_outputs.items():
        events.append(
            {
                "type": "proposer_output",
                "round_number": round_number,
                "worker_id": proposer_id,
                "output": output,
            }
        )
    events.append(
        {
            "type": "review",
            "round_number": round_number,
            "worker_id": reviewer_id,
            "output": review_output,
        }
    )
    controller = review_output.get("controller", {})
    if controller:
        events.append({"type": "controller_message", "round_number": round_number, "message": controller})
    for proposer_id, message in sorted(review_output.get("proposer_messages", {}).items()):
        events.append(
            {
                "type": "proposer_message",
                "round_number": round_number,
                "worker_id": proposer_id,
                "message": message,
            }
        )
    return events


def _loop_failure(loop_id: str, paths, message: str, *, exc: BaseException | None = None) -> dict[str, Any]:
    result = {
        "loop_id": loop_id,
        "status": "failed",
        "rounds_completed": 0,
        "error": message,
        "loop_root": str(paths.loop_root),
    }
    if exc is not None:
        result["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return result


def _batch_status(loop_results: list[dict[str, Any]]) -> str:
    if any(item["status"] == "failed" for item in loop_results):
        return "failed"
    if any(item["status"] == "skipped" for item in loop_results):
        return "failed"
    if loop_results and all(item["status"] == "stopped" for item in loop_results):
        return "stopped"
    return "completed"


def _dry_run_result(batch: BatchConfig, paths: RunPaths) -> dict[str, Any]:
    return {
        "schema_version": "arc.llm.proposers_reviewer_batch.result.v1",
        "status": "dry_run",
        "run_id": batch.run_id,
        "run_root": str(paths.run_root),
        "loops": [{"loop_id": loop.loop_id, "status": "validated"} for loop in batch.loops],
    }


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
