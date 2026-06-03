from __future__ import annotations

import copy
import inspect
import json
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, CancelledError, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from threading import Lock, Semaphore
from typing import Any, Callable, Mapping

from arc_llm.call_record import ARC_LLM_CALL_RECORD_FIELD, ARC_LLM_CALL_RECORD_SCHEMA_VERSION
from arc_llm.call_record import strip_arc_llm_call_records
from arc_llm.runner import run_json
from arc_llm.schema_cache import schema_hash, sha256_text
from arc_llm.sessions import LLMSessionManager, runtime_fingerprint
from arc_llm.structured_recovery import structured_metadata

from .artifacts import LockConflictError, RunPaths, acquire_lock, append_jsonl, atomic_write_json, atomic_write_text
from .config import (
    ArtifactOptions,
    REVIEW_ENVELOPE_SCHEMA,
    BatchConfig,
    CacheGuardOptions,
    ConfigError,
    LoopConfig,
    OutputRecoveryOptions,
    WorkerConfig,
    load_batch_config,
    worker_env,
)
from .dialogue import (
    render_initial_worker_prompt,
    render_legacy_full_prompt,
    render_proposer_delta_prompt,
    render_reviewer_delta_prompt,
)
from .prompts import proposer_context, reviewer_context


JsonRunner = Callable[..., Any]


@dataclass(frozen=True)
class WorkerPromptOption:
    turn_kind: str
    prompt: str
    context: dict[str, Any]
    static_prefix: str | None
    prompt_path: Path


@dataclass(frozen=True)
class WorkerCallResult:
    output: Any
    turn_kind: str
    prompt: str
    prompt_path: Path | None
    static_prefix: str | None


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
    session_manager = LLMSessionManager(_session_root(batch, paths))
    prefix_limiter = PrefixConcurrencyLimiter(batch.session.max_concurrent_same_prefix)

    loops = list(batch.loops)
    next_loop_index = 0
    stop_scheduling = False
    loop_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_by_loop = {}
        while True:
            while not stop_scheduling and len(future_by_loop) < concurrency and next_loop_index < len(loops):
                loop = loops[next_loop_index]
                next_loop_index += 1
                future = executor.submit(
                    _run_loop,
                    copy.deepcopy(loop),
                    paths.loop(loop.loop_id),
                    batch.run_id,
                    batch.artifact_options,
                    batch,
                    session_manager,
                    prefix_limiter,
                    json_runner,
                    base_env,
                    process_chain,
                )
                future_by_loop[future] = loop.loop_id
            if not future_by_loop:
                break
            done, _pending = wait(future_by_loop, return_when=FIRST_COMPLETED)
            for future in done:
                loop_id = future_by_loop.pop(future)
                try:
                    result = future.result()
                except CancelledError:
                    result = _skipped_loop_result(paths, loop_id, "cancelled by fail_fast")
                loop_results.append(result)
                if batch.fail_fast and result["status"] == "failed":
                    stop_scheduling = True
            if stop_scheduling and not future_by_loop:
                break
        if stop_scheduling:
            for loop in loops[next_loop_index:]:
                loop_results.append(_skipped_loop_result(paths, loop.loop_id, "skipped by fail_fast"))

    loop_results.sort(key=lambda item: item["loop_id"])
    status = _batch_status(loop_results)
    run_result = {
        "schema_version": "arc.llm.proposers_reviewer_batch.result.v1",
        "status": status,
        "run_id": batch.run_id,
        "run_root": str(paths.run_root),
        "warnings_summary": _warnings_summary(paths),
        "loops": loop_results,
    }
    _write_run_state(paths, run_result)
    return run_result


def _prepare_run(paths: RunPaths, batch: BatchConfig) -> None:
    if paths.run_root.exists():
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


def _warnings_summary(paths: RunPaths) -> dict[str, Any]:
    structured_path = paths.run_root / "structured_output_warnings.jsonl"
    cache_path = paths.run_root / "cache_warnings.jsonl"
    return {
        "structured_output_warning_count": _count_jsonl(structured_path),
        "structured_output_warnings_path": str(structured_path),
        "cache_warning_count": _count_jsonl(cache_path),
        "cache_warnings_path": str(cache_path),
    }


def _count_jsonl(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except FileNotFoundError:
        return 0


def _run_loop(
    loop: LoopConfig,
    paths,
    run_id: str,
    artifact_options: ArtifactOptions,
    batch: BatchConfig,
    session_manager: LLMSessionManager,
    prefix_limiter: "PrefixConcurrencyLimiter",
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
            result = _run_loop_rounds(
                loop,
                paths,
                artifact_options,
                batch,
                session_manager,
                prefix_limiter,
                json_runner,
                base_env,
                process_chain,
            )
            atomic_write_json(paths.state, result)
            return result
    except Exception as exc:
        result = _loop_failure(loop.loop_id, paths, str(exc), exc=exc)
        _write_loop_failure_state(paths, run_id=run_id, result=result)
        return result


def _run_loop_rounds(
    loop: LoopConfig,
    paths,
    artifact_options: ArtifactOptions,
    batch: BatchConfig,
    session_manager: LLMSessionManager,
    prefix_limiter: "PrefixConcurrencyLimiter",
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
            batch,
            session_manager,
            prefix_limiter,
            json_runner,
            base_env,
            process_chain,
            cache_warnings_path=paths.run_root / "cache_warnings.jsonl",
        )
        reviewer = loop.reviewers[0]
        session_key = _worker_session_key(batch.run_id, loop, reviewer, "reviewer")
        prompt_options = _reviewer_prompt_options(
            loop=loop,
            reviewer=reviewer,
            round_number=round_number,
            correspondence=correspondence,
            proposer_outputs=proposer_outputs,
            round_paths=round_paths,
            stateful_supported=_stateful_supported(json_runner),
        )
        try:
            review_output = _call_reviewer(
                json_runner,
                prompt_options,
                worker=reviewer,
                loop=loop,
                batch=batch,
                round_number=round_number,
                error_path=round_paths.worker_error(reviewer.id),
                context_path=round_paths.reviewer_context(reviewer.id),
                save_prompt=artifact_options.save_prompts,
                base_env=base_env,
                process_chain=process_chain,
                session_manager=session_manager,
                session_key=session_key,
                prefix_limiter=prefix_limiter,
                artifact_dir=round_paths.round_root / "llm_calls" / reviewer.id,
                cache_guard=loop.session.cache_guard,
                cache_warnings_path=paths.run_root / "cache_warnings.jsonl",
            )
        except Exception as exc:
            if not _warn_continue_policy(batch.output_recovery) or _is_fatal_provider_failure_exception(exc):
                raise
            review_output = _worker_exception_output(
                worker=reviewer,
                role="reviewer",
                exc=exc,
                call_label=f"loop/{loop.loop_id}/round_{round_number:03d}/{reviewer.id}",
            )
            _record_structured_output_warning(
                review_output[ARC_LLM_CALL_RECORD_FIELD]["structured_output"],
                warnings_path=paths.run_root / "structured_output_warnings.jsonl",
                worker=reviewer,
                call_label=f"loop/{loop.loop_id}/round_{round_number:03d}/{reviewer.id}",
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
    batch: BatchConfig,
    session_manager: LLMSessionManager,
    prefix_limiter: "PrefixConcurrencyLimiter",
    json_runner: JsonRunner | None,
    base_env: Mapping[str, str] | None,
    process_chain: list[str] | None,
    cache_warnings_path: Path,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    session_keys: dict[str, str] = {}
    prompt_options_by_proposer: dict[str, dict[str, WorkerPromptOption]] = {}
    for proposer in loop.proposers:
        session_key = _worker_session_key(batch.run_id, loop, proposer, "proposer")
        prompt_options = _proposer_prompt_options(
            loop=loop,
            proposer=proposer,
            round_number=round_number,
            correspondence=correspondence,
            round_paths=round_paths,
            stateful_supported=_stateful_supported(json_runner),
        )
        session_keys[proposer.id] = session_key
        prompt_options_by_proposer[proposer.id] = prompt_options

    with ThreadPoolExecutor(max_workers=len(loop.proposers)) as executor:
        future_by_proposer = {
            executor.submit(
                _call_json_runner_with_prompt_options,
                json_runner,
                prompt_options_by_proposer[proposer.id],
                worker=proposer,
                loop=loop,
                round_number=round_number,
                error_path=round_paths.worker_error(proposer.id),
                context_path=round_paths.proposer_context(proposer.id),
                save_prompt=artifact_options.save_prompts,
                base_env=base_env,
                process_chain=process_chain,
                session_manager=session_manager,
                session_key=session_keys[proposer.id],
                call_label=f"loop/{loop.loop_id}/round_{round_number:03d}/{proposer.id}",
                artifact_dir=round_paths.round_root / "llm_calls" / proposer.id,
                prefix_limiter=prefix_limiter,
                cache_guard=loop.session.cache_guard,
                cache_warnings_path=cache_warnings_path,
                output_recovery=batch.output_recovery,
                validate_schema=True,
            ): proposer
            for proposer in loop.proposers
        }
        for future in as_completed(future_by_proposer):
            proposer = future_by_proposer[future]
            call_label = f"loop/{loop.loop_id}/round_{round_number:03d}/{proposer.id}"
            try:
                output = future.result().output
                output = _prepare_peer_visible_proposer_output(
                    output,
                    worker=proposer,
                    output_recovery=batch.output_recovery,
                    call_label=call_label,
                    warnings_path=cache_warnings_path.with_name("structured_output_warnings.jsonl"),
                )
            except Exception as exc:
                if not _warn_continue_policy(batch.output_recovery) or _is_fatal_provider_failure_exception(exc):
                    raise
                output = _worker_exception_output(worker=proposer, role="proposer", exc=exc, call_label=call_label)
                _record_structured_output_warning(
                    output[ARC_LLM_CALL_RECORD_FIELD]["structured_output"],
                    warnings_path=cache_warnings_path.with_name("structured_output_warnings.jsonl"),
                    worker=proposer,
                    call_label=call_label,
                )
            outputs[proposer.id] = output
            atomic_write_json(round_paths.proposer_output(proposer.id), output)
    return dict(sorted(outputs.items()))


def _call_reviewer(
    json_runner: JsonRunner | None,
    prompt_options: dict[str, WorkerPromptOption],
    *,
    worker: WorkerConfig,
    loop: LoopConfig,
    batch: BatchConfig,
    round_number: int,
    error_path: Path,
    context_path: Path,
    save_prompt: bool,
    base_env: Mapping[str, str] | None,
    process_chain: list[str] | None,
    session_manager: LLMSessionManager,
    session_key: str,
    prefix_limiter: "PrefixConcurrencyLimiter",
    artifact_dir: Path | None,
    cache_guard: CacheGuardOptions,
    cache_warnings_path: Path,
) -> dict[str, Any]:
    first_call = _call_json_runner_with_prompt_options(
        json_runner,
        prompt_options,
        worker=worker,
        loop=loop,
        round_number=round_number,
        error_path=error_path,
        context_path=context_path,
        save_prompt=save_prompt,
        base_env=base_env,
        process_chain=process_chain,
        session_manager=session_manager,
        session_key=session_key,
        call_label=f"loop/{loop.loop_id}/round_{round_number:03d}/{worker.id}",
        artifact_dir=artifact_dir,
        prefix_limiter=prefix_limiter,
        cache_guard=cache_guard,
        cache_warnings_path=cache_warnings_path,
        output_recovery=batch.output_recovery,
        validate_schema=True,
    )
    review_output = first_call.output
    _validate_reviewer_output(review_output, worker=worker, loop=loop)
    return review_output


def _worker_exception_output(
    *,
    worker: WorkerConfig,
    role: str,
    exc: BaseException,
    call_label: str,
) -> dict[str, Any]:
    raw_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    structured_output = structured_metadata(
        severity="major",
        warnings=[
            f"{role} worker call failed; ARC is continuing with a warning artifact instead of aborting the loop.",
            raw_text,
        ],
        raw_text=raw_text,
        strategy="worker_exception_continue",
        provider_error_type=type(exc).__name__,
    )
    return _unstructured_output_wrapper(
        raw_text=raw_text,
        worker=worker,
        role=role,
        structured_output=structured_output,
        call_label=call_label,
    )


def _prepare_peer_visible_proposer_output(
    output: Any,
    *,
    worker: WorkerConfig,
    output_recovery: OutputRecoveryOptions,
    call_label: str,
    warnings_path: Path,
) -> Any:
    if not _peer_visible_schema_policy(output_recovery):
        return output
    if not isinstance(output, dict):
        raw_text = _raw_output_text(output)
        structured_output = structured_metadata(
            severity="major",
            warnings=["Proposer output was not a JSON object; forwarding as unstructured peer-visible text."],
            raw_text=raw_text,
            strategy="peer_visible_unstructured_output",
            provider_error_type=type(output).__name__,
        )
        wrapped = _unstructured_output_wrapper(
            raw_text=raw_text,
            worker=worker,
            role="proposer",
            structured_output=structured_output,
            call_label=call_label,
        )
        _record_structured_output_warning(
            structured_output,
            warnings_path=warnings_path,
            worker=worker,
            call_label=call_label,
        )
        return wrapped

    structured_from_provider = _structured_output_from_payload(output)
    if (
        isinstance(structured_from_provider, Mapping)
        and structured_from_provider.get("mode") == "recovered"
        and str(structured_from_provider.get("recovery_strategy") or "") == "natural_language_fallback"
        and str(structured_from_provider.get("raw_text_excerpt") or "").strip()
    ):
        raw_text = str(structured_from_provider.get("raw_text_excerpt") or "")
        structured_output = structured_metadata(
            severity="major",
            warnings=["Provider returned natural language; forwarding as unstructured peer-visible text."],
            raw_text=raw_text,
            strategy="peer_visible_unstructured_output",
            provider_error_type=str(structured_from_provider.get("provider_error_type") or type(output).__name__),
        )
        wrapped = _unstructured_output_wrapper(
            raw_text=raw_text,
            worker=worker,
            role="proposer",
            structured_output=structured_output,
            call_label=call_label,
        )
        _record_structured_output_warning(
            structured_output,
            warnings_path=warnings_path,
            worker=worker,
            call_label=call_label,
        )
        return wrapped

    schema_error = _schema_validation_error(output, worker.output_schema)
    if schema_error is None:
        return output
    structured_output = structured_metadata(
        severity="minor",
        warnings=["Proposer output did not match its schema; forwarding original object without retry.", schema_error],
        raw_text=json.dumps(strip_arc_llm_call_records(output), ensure_ascii=False, sort_keys=True, default=str),
        strategy="peer_visible_schema_violation",
        provider_error_type="JsonSchemaValidationError",
    )
    _attach_structured_output(output, structured_output)
    _record_structured_output_warning(
        structured_output,
        warnings_path=warnings_path,
        worker=worker,
        call_label=call_label,
    )
    return output


def _unstructured_output_wrapper(
    *,
    raw_text: str,
    worker: WorkerConfig,
    role: str,
    structured_output: dict[str, Any],
    call_label: str | None = None,
) -> dict[str, Any]:
    excerpt = raw_text[:4000]
    return {
        "schema_version": "arc.llm.unstructured_output.v1",
        "worker_id": worker.id,
        "role": role,
        "raw_text": raw_text,
        "recovered_unstructured_text": excerpt,
        "structured_output_warning": "This output was recovered from malformed or natural-language LLM output.",
        "idea_title": _first_nonempty_line(raw_text)[:120] or "Recovered unstructured idea",
        "idea_summary": excerpt[:2000],
        "description": excerpt[:2000],
        "warning": "Output did not satisfy the requested JSON schema and was forwarded as unstructured text.",
        ARC_LLM_CALL_RECORD_FIELD: _local_recovery_call_record(
            worker=worker,
            schema=worker.output_schema,
            signal="peer_visible_unstructured_output",
            structured_output=structured_output,
            call_label=call_label,
        ),
    }


def _local_recovery_call_record(
    *,
    worker: WorkerConfig,
    schema: Mapping[str, Any] | None,
    signal: str,
    structured_output: dict[str, Any],
    call_label: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": ARC_LLM_CALL_RECORD_SCHEMA_VERSION,
        "provider_requested": worker.provider,
        "model_requested": worker.model,
        "model_tier_requested": worker.model_tier,
        "provider_used": worker.provider,
        "model_used": worker.model,
        "fallback_index": 0,
        "attempt": 1,
        "host": "local-recovery",
        "signals": [signal],
        "attempts": [],
        "session_policy": "local-recovery",
        "session_key": None,
        "native_session_id": None,
        "call_label": call_label,
        "prompt_sha256": None,
        "static_prefix_sha256": None,
        "schema_sha256": schema_hash(schema),
        "runtime_fingerprint": None,
        "usage": {},
        "structured_output": structured_output,
    }


def _call_json_runner_with_prompt_options(
    json_runner: JsonRunner | None,
    prompt_options: dict[str, WorkerPromptOption],
    *,
    worker: WorkerConfig,
    loop: LoopConfig,
    round_number: int,
    error_path: Path,
    context_path: Path,
    save_prompt: bool,
    base_env: Mapping[str, str] | None,
    process_chain: list[str] | None,
    session_manager: LLMSessionManager,
    session_key: str,
    call_label: str,
    artifact_dir: Path | None,
    prefix_limiter: "PrefixConcurrencyLimiter",
    cache_guard: CacheGuardOptions,
    cache_warnings_path: Path,
    output_recovery: OutputRecoveryOptions,
    validate_schema: bool = True,
) -> WorkerCallResult:
    selected: WorkerPromptOption | None = None

    def invoke_locked() -> WorkerCallResult:
        nonlocal selected
        turn_kind = _locked_turn_kind(loop, session_manager, session_key, prompt_options)
        selected = prompt_options[turn_kind]
        context = copy.deepcopy(selected.context)
        context.update({"session_key": session_key, "history_mode": loop.session.history_mode})
        atomic_write_json(context_path, context)
        prompt_path = selected.prompt_path if save_prompt else None
        if prompt_path is not None:
            atomic_write_text(prompt_path, selected.prompt)
        session_policy = loop.session.policy if selected.turn_kind != "legacy_full" else "stateless"
        output = _call_json_runner(
            json_runner,
            selected.prompt,
            worker=worker,
            base_env=base_env,
            process_chain=process_chain,
            session_policy=session_policy,
            session_manager=session_manager,
            session_key=session_key,
            call_label=call_label,
            artifact_dir=artifact_dir,
            prefix_limiter=prefix_limiter,
            static_prefix=selected.static_prefix,
            cache_guard=cache_guard,
            cache_warnings_path=cache_warnings_path,
            output_recovery=output_recovery,
            validate_schema=validate_schema,
        )
        return WorkerCallResult(
            output=output,
            turn_kind=selected.turn_kind,
            prompt=selected.prompt,
            prompt_path=prompt_path,
            static_prefix=selected.static_prefix,
        )

    try:
        if _uses_stateful_delta(loop, prompt_options):
            with session_manager.lock(session_key):
                return invoke_locked()
        return invoke_locked()
    except Exception as exc:
        prompt_path = selected.prompt_path if selected is not None and save_prompt else None
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
    session_policy: str,
    session_manager: LLMSessionManager,
    session_key: str,
    call_label: str,
    artifact_dir: Path | None,
    prefix_limiter: "PrefixConcurrencyLimiter",
    static_prefix: str | None,
    cache_guard: CacheGuardOptions,
    cache_warnings_path: Path,
    output_recovery: OutputRecoveryOptions,
    validate_schema: bool = True,
) -> dict[str, Any]:
    env = worker_env(worker, base_env=base_env)
    effective_session_policy = session_policy
    if json_runner is not None and session_policy == "stateful" and not _declares_keyword(json_runner, "session_policy"):
        effective_session_policy = "stateless"
    prefix_key = _prefix_key(worker, env, process_chain, prompt, static_prefix)
    with prefix_limiter.acquire(prefix_key):
        if json_runner is not None:
            before_count = session_manager.turn_count(session_key) if effective_session_policy == "stateful" else None
            kwargs = {
                "schema": worker.output_schema,
                "provider": worker.provider,
                "model": worker.model,
                "model_tier": worker.model_tier,
                "env": env,
            }
            optional = {
                "process_chain": process_chain,
                "session_policy": effective_session_policy,
                "session_manager": session_manager,
                "session_key": session_key,
                "call_label": call_label,
                "artifact_dir": artifact_dir,
                "static_prefix": static_prefix,
                "validate_schema": validate_schema,
                "output_recovery": _output_recovery_mode(output_recovery),
                "schema_formatter_enabled": output_recovery.schema_formatter_enabled,
                "role_hint": _role_hint(worker),
            }
            for key, value in optional.items():
                if _accepts_keyword(json_runner, key):
                    kwargs[key] = value
            result = json_runner(prompt, **kwargs)
            after_count = session_manager.turn_count(session_key) if effective_session_policy == "stateful" else None
            if before_count == after_count:
                _record_custom_session_turn(
                    result,
                    session_policy=effective_session_policy,
                    session_manager=session_manager,
                    session_key=session_key,
                    call_label=call_label,
                    prompt=prompt,
                    worker=worker,
                    env=env,
                    process_chain=process_chain,
                    static_prefix=static_prefix,
                )
            _maybe_record_cache_warning(
                result,
                session_policy=effective_session_policy,
                session_manager=session_manager,
                session_key=session_key,
                call_label=call_label,
                cache_guard=cache_guard,
                cache_warnings_path=cache_warnings_path,
            )
            _maybe_record_structured_output_warning(
                result,
                warnings_path=cache_warnings_path.with_name("structured_output_warnings.jsonl"),
                worker=worker,
                call_label=call_label,
            )
            return result
        result = run_json(
            prompt,
            schema=worker.output_schema,
            validate_schema=validate_schema,
            provider=worker.provider,
            model=worker.model,
            model_tier=worker.model_tier,
            env=env,
            process_chain=process_chain,
            session_policy=effective_session_policy,
            session_manager=session_manager if effective_session_policy == "stateful" else None,
            session_key=session_key if effective_session_policy == "stateful" else None,
            session_name=worker.id,
            session_metadata={"worker_id": worker.id},
            artifact_dir=artifact_dir,
            call_label=call_label,
            static_prefix=static_prefix,
            output_recovery=_output_recovery_mode(output_recovery),
            schema_formatter_enabled=output_recovery.schema_formatter_enabled,
            role_hint=_role_hint(worker),
        )
        _maybe_record_cache_warning(
            result,
            session_policy=effective_session_policy,
            session_manager=session_manager,
            session_key=session_key,
            call_label=call_label,
            cache_guard=cache_guard,
            cache_warnings_path=cache_warnings_path,
        )
        _maybe_record_structured_output_warning(
            result,
            warnings_path=cache_warnings_path.with_name("structured_output_warnings.jsonl"),
            worker=worker,
            call_label=call_label,
        )
        return result


def _accepts_keyword(callable_obj: JsonRunner, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.kind in {inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
            if parameter.name == name:
                return True
    return False


def _declares_keyword(callable_obj: JsonRunner, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind in {inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
            if parameter.name == name:
                return True
    return False


class PrefixConcurrencyLimiter:
    """Process-local limiter for calls sharing the same cache prefix."""

    def __init__(self, default_limit: int) -> None:
        self.default_limit = max(1, int(default_limit))
        self._semaphores: dict[str, Semaphore] = {}
        self._lock = Lock()

    def acquire(self, key: str):
        limiter = self

        class _Guard:
            def __enter__(self_inner):
                with limiter._lock:
                    semaphore = limiter._semaphores.setdefault(key, Semaphore(limiter.default_limit))
                semaphore.acquire()
                self_inner._semaphore = semaphore
                return None

            def __exit__(self_inner, exc_type, exc, tb):
                self_inner._semaphore.release()
                return False

        return _Guard()


def _session_root(batch: BatchConfig, paths: RunPaths) -> Path:
    roots = {str(loop.session.root) for loop in batch.loops if loop.session.root is not None}
    if batch.session.root is not None:
        roots.add(str(batch.session.root))
    if len(roots) > 1:
        raise ConfigError("all loop session.root values must match batch session.root")
    if roots:
        return Path(next(iter(roots))).expanduser()
    if batch.session.reuse_across_batch_calls or any(loop.session.reuse_across_batch_calls for loop in batch.loops):
        return batch.run_dir / "_sessions"
    return paths.sessions_root


def _worker_session_key(batch_run_id: str, loop: LoopConfig, worker: WorkerConfig, role: str) -> str:
    if loop.session.reuse_across_batch_calls:
        if not loop.session.scope_id:
            raise ConfigError("reuse_across_batch_calls requires session.scope_id")
        scope = loop.session.scope_id
    else:
        scope = f"{loop.session.scope_id or batch_run_id}/{loop.loop_id}"
    return f"{scope}/{role}/{worker.id}"


def _turn_kind(
    loop: LoopConfig,
    session_manager: LLMSessionManager,
    session_key: str,
    *,
    stateful_supported: bool,
) -> str:
    if loop.session.policy != "stateful" or loop.session.history_mode == "full" or not stateful_supported:
        return "legacy_full"
    return "initial" if session_manager.turn_count(session_key) == 0 else "delta"


def _stateful_supported(json_runner: JsonRunner | None) -> bool:
    return json_runner is None or _declares_keyword(json_runner, "session_policy")


def _uses_stateful_delta(loop: LoopConfig, prompt_options: Mapping[str, WorkerPromptOption]) -> bool:
    return loop.session.policy == "stateful" and "initial" in prompt_options and "delta" in prompt_options


def _locked_turn_kind(
    loop: LoopConfig,
    session_manager: LLMSessionManager,
    session_key: str,
    prompt_options: Mapping[str, WorkerPromptOption],
) -> str:
    if not _uses_stateful_delta(loop, prompt_options):
        return "legacy_full"
    if session_manager.turn_count(session_key) == 0:
        return "initial"
    if not session_manager.has_native_session(session_key):
        return "initial"
    return "delta"


def _proposer_prompt_options(
    *,
    loop: LoopConfig,
    proposer: WorkerConfig,
    round_number: int,
    correspondence: list[dict[str, Any]],
    round_paths,
    stateful_supported: bool,
) -> dict[str, WorkerPromptOption]:
    if loop.session.policy == "stateful" and loop.session.history_mode == "delta" and stateful_supported:
        return {
            turn_kind: _prompt_option(
                worker_id=proposer.id,
                round_paths=round_paths,
                turn_kind=turn_kind,
                rendered=_proposer_prompt_and_context(
                    loop=loop,
                    proposer=proposer,
                    round_number=round_number,
                    correspondence=correspondence,
                    turn_kind=turn_kind,
                ),
            )
            for turn_kind in ("initial", "delta")
        }
    return {
        "legacy_full": _prompt_option(
            worker_id=proposer.id,
            round_paths=round_paths,
            turn_kind="legacy_full",
            rendered=_proposer_prompt_and_context(
                loop=loop,
                proposer=proposer,
                round_number=round_number,
                correspondence=correspondence,
                turn_kind="legacy_full",
            ),
        )
    }


def _reviewer_prompt_options(
    *,
    loop: LoopConfig,
    reviewer: WorkerConfig,
    round_number: int,
    correspondence: list[dict[str, Any]],
    proposer_outputs: dict[str, Any],
    round_paths,
    stateful_supported: bool,
) -> dict[str, WorkerPromptOption]:
    if loop.session.policy == "stateful" and loop.session.history_mode == "delta" and stateful_supported:
        return {
            turn_kind: _prompt_option(
                worker_id=reviewer.id,
                round_paths=round_paths,
                turn_kind=turn_kind,
                rendered=_reviewer_prompt_and_context(
                    loop=loop,
                    reviewer=reviewer,
                    round_number=round_number,
                    correspondence=correspondence,
                    proposer_outputs=proposer_outputs,
                    turn_kind=turn_kind,
                ),
            )
            for turn_kind in ("initial", "delta")
        }
    return {
        "legacy_full": _prompt_option(
            worker_id=reviewer.id,
            round_paths=round_paths,
            turn_kind="legacy_full",
            rendered=_reviewer_prompt_and_context(
                loop=loop,
                reviewer=reviewer,
                round_number=round_number,
                correspondence=correspondence,
                proposer_outputs=proposer_outputs,
                turn_kind="legacy_full",
            ),
        )
    }


def _prompt_option(
    *,
    worker_id: str,
    round_paths,
    turn_kind: str,
    rendered: tuple[str, dict[str, Any], str | None],
) -> WorkerPromptOption:
    prompt, context, static_prefix = rendered
    prompt_kind = "initial" if turn_kind == "initial" else "delta" if turn_kind == "delta" else "prompt"
    return WorkerPromptOption(
        turn_kind=turn_kind,
        prompt=prompt,
        context=context,
        static_prefix=static_prefix,
        prompt_path=round_paths.prompt(worker_id, kind=prompt_kind),
    )


def _proposer_prompt_and_context(
    *,
    loop: LoopConfig,
    proposer: WorkerConfig,
    round_number: int,
    correspondence: list[dict[str, Any]],
    turn_kind: str,
) -> tuple[str, dict[str, Any], str | None]:
    if turn_kind == "initial":
        return render_initial_worker_prompt(loop=loop, worker=proposer, role="proposer", round_number=round_number)
    if turn_kind == "delta":
        prompt, context = render_proposer_delta_prompt(
            loop=loop,
            worker=proposer,
            round_number=round_number,
            correspondence=copy.deepcopy(correspondence),
        )
        return prompt, context, None
    context = proposer_context(
        loop=loop,
        worker=proposer,
        round_number=round_number,
        correspondence=copy.deepcopy(correspondence),
    )
    context["turn_kind"] = "legacy_full"
    return render_legacy_full_prompt(proposer, context), context, None


def _reviewer_prompt_and_context(
    *,
    loop: LoopConfig,
    reviewer: WorkerConfig,
    round_number: int,
    correspondence: list[dict[str, Any]],
    proposer_outputs: dict[str, Any],
    turn_kind: str,
) -> tuple[str, dict[str, Any], str | None]:
    if turn_kind == "initial":
        prompt, context, static_prefix = render_initial_worker_prompt(
            loop=loop,
            worker=reviewer,
            role="reviewer",
            round_number=round_number,
        )
        context["current_proposer_outputs"] = copy.deepcopy(proposer_outputs)
        prompt = (
            prompt.rstrip()
            + "\n\n## Current Proposer Outputs To Review\n"
            + json.dumps(proposer_outputs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        return prompt, context, static_prefix
    if turn_kind == "delta":
        prompt, context = render_reviewer_delta_prompt(
            loop=loop,
            worker=reviewer,
            round_number=round_number,
            current_proposer_outputs=proposer_outputs,
        )
        return prompt, context, None
    context = reviewer_context(
        loop=loop,
        worker=reviewer,
        round_number=round_number,
        correspondence=copy.deepcopy(correspondence),
        current_proposer_outputs=proposer_outputs,
    )
    context["turn_kind"] = "legacy_full"
    return render_legacy_full_prompt(reviewer, context), context, None


def _prefix_key(
    worker: WorkerConfig,
    env: Mapping[str, str],
    process_chain: list[str] | None,
    prompt: str,
    static_prefix: str | None,
) -> str:
    fp = runtime_fingerprint(
        provider=worker.provider,
        model=worker.model,
        model_tier=worker.model_tier,
        env=env,
        process_chain=process_chain,
    )
    prefix_hash = sha256_text(static_prefix or prompt[:4096])
    return "|".join([worker.provider, str(worker.model), str(worker.model_tier), fp, prefix_hash, str(schema_hash(worker.output_schema))])


def _record_custom_session_turn(
    result: dict[str, Any],
    *,
    session_policy: str,
    session_manager: LLMSessionManager,
    session_key: str,
    call_label: str,
    prompt: str,
    worker: WorkerConfig,
    env: Mapping[str, str],
    process_chain: list[str] | None,
    static_prefix: str | None,
) -> None:
    if session_policy != "stateful":
        return
    fp = runtime_fingerprint(
        provider=worker.provider,
        model=worker.model,
        model_tier=worker.model_tier,
        env=env,
        process_chain=process_chain,
    )
    session_manager.get_or_create(
        key=session_key,
        provider=worker.provider,
        model=worker.model,
        runtime_fingerprint=fp,
        name=worker.id,
        metadata={"worker_id": worker.id},
    )
    record = result.get(ARC_LLM_CALL_RECORD_FIELD) if isinstance(result, dict) else None
    usage = record.get("usage", {}) if isinstance(record, dict) and isinstance(record.get("usage"), dict) else {}
    native_session_id = record.get("native_session_id") if isinstance(record, dict) else None
    if native_session_id:
        session_manager.update_native_session_id(session_key, str(native_session_id))
    session_manager.record_turn(
        session_key,
        call_label=call_label,
        prompt_sha256=sha256_text(prompt),
        static_prefix_sha256=sha256_text(static_prefix) if static_prefix else None,
        schema_sha256=schema_hash(worker.output_schema),
        usage=usage,
        provider_used=worker.provider,
        model_used=worker.model,
        native_session_id=str(native_session_id) if native_session_id else None,
        extra={"runtime_fingerprint": fp},
    )


def _maybe_record_cache_warning(
    result: dict[str, Any],
    *,
    session_policy: str,
    session_manager: LLMSessionManager,
    session_key: str,
    call_label: str,
    cache_guard: CacheGuardOptions,
    cache_warnings_path: Path,
) -> None:
    if session_policy != "stateful" or not cache_guard.enabled:
        return
    ratio = _cached_input_ratio(result)
    if ratio is None:
        return
    turn_count = session_manager.turn_count(session_key)
    if turn_count <= cache_guard.warmup_calls or ratio >= cache_guard.min_cached_input_ratio:
        return
    warning = {
        "schema_version": "arc.llm.cache_warning.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_key": session_key,
        "call_label": call_label,
        "turn_count": turn_count,
        "cached_input_ratio": ratio,
        "min_cached_input_ratio": cache_guard.min_cached_input_ratio,
        "warmup_calls": cache_guard.warmup_calls,
        "mode": cache_guard.mode,
    }
    append_jsonl(cache_warnings_path, warning)
    if cache_guard.mode == "abort":
        raise ConfigError(
            "cache guard failed for "
            f"{call_label}: cached_input_ratio={ratio:.3f} "
            f"< {cache_guard.min_cached_input_ratio:.3f}"
        )


def _maybe_record_structured_output_warning(
    result: dict[str, Any],
    *,
    warnings_path: Path,
    worker: WorkerConfig,
    call_label: str,
) -> None:
    record = result.get(ARC_LLM_CALL_RECORD_FIELD) if isinstance(result, dict) else None
    structured = record.get("structured_output") if isinstance(record, Mapping) else None
    if not isinstance(structured, Mapping) or structured.get("mode") != "recovered":
        return
    _record_structured_output_warning(
        structured,
        warnings_path=warnings_path,
        worker=worker,
        call_label=call_label,
    )


def _record_structured_output_warning(
    structured: Mapping[str, Any],
    *,
    warnings_path: Path,
    worker: WorkerConfig,
    call_label: str,
) -> None:
    append_jsonl(
        warnings_path,
        {
            "schema_version": "arc.llm.structured_output_warning.v1",
            "worker_id": worker.id,
            "call_label": call_label,
            "severity": structured.get("severity"),
            "warnings": list(structured.get("warnings", []))
            if isinstance(structured.get("warnings"), list)
            else [],
            "raw_text_excerpt": str(structured.get("raw_text_excerpt") or ""),
            "recovery_strategy": structured.get("recovery_strategy"),
            "provider_error_type": structured.get("provider_error_type"),
        },
    )


def _peer_visible_schema_policy(options: OutputRecoveryOptions) -> bool:
    return options.schema_violation_policy == "peer_visible" and _output_recovery_mode(options) == "warn"


def _warn_continue_policy(options: OutputRecoveryOptions) -> bool:
    return _output_recovery_mode(options) == "warn" and options.allow_natural_language


def _output_recovery_mode(options: OutputRecoveryOptions) -> str:
    if not options.enabled or not options.allow_natural_language:
        return "strict"
    return options.mode


def _is_fatal_provider_failure_exception(exc: BaseException) -> bool:
    text = str(exc).lower()
    fatal_markers = [
        "fatal provider failure text",
        "mcp server failed",
        "authentication",
        "permission denied",
        "invalid api key",
        "not authorized",
        "command not found",
        "no such file or directory",
        "arc-only mcp",
    ]
    return any(marker in text for marker in fatal_markers)


def _raw_output_text(output: Any) -> str:
    return _raw_excerpt_from_any(output)


def _raw_excerpt_from_any(output: Any) -> str:
    if isinstance(output, Mapping):
        structured = _structured_output_from_payload(output)
        if isinstance(structured, Mapping):
            raw_excerpt = structured.get("raw_text_excerpt")
            if isinstance(raw_excerpt, str) and raw_excerpt.strip():
                return raw_excerpt
        return json.dumps(strip_arc_llm_call_records(dict(output)), ensure_ascii=False, sort_keys=True, default=str)[:4000]
    if isinstance(output, str):
        return output[:4000]
    return str(output or "")[:4000]


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().strip("#*- ")
        if stripped:
            return stripped
    return ""


def _structured_output_from_payload(output: Mapping[str, Any]) -> Mapping[str, Any] | None:
    record = output.get(ARC_LLM_CALL_RECORD_FIELD)
    if not isinstance(record, Mapping):
        return None
    structured = record.get("structured_output")
    return structured if isinstance(structured, Mapping) else None


def _schema_validation_error(output: Mapping[str, Any], schema: Mapping[str, Any] | None) -> str | None:
    if schema is None:
        return None
    from jsonschema import ValidationError as JsonSchemaValidationError
    from jsonschema import validate as validate_json_schema
    from jsonschema.exceptions import SchemaError as JsonSchemaError

    try:
        validate_json_schema(instance=strip_arc_llm_call_records(dict(output)), schema=schema)
    except JsonSchemaValidationError as exc:
        return str(exc.message)
    except JsonSchemaError as exc:
        return f"schema invalid: {exc.message}"
    return None


def _attach_structured_output(output: dict[str, Any], structured_output: dict[str, Any]) -> None:
    record = output.get(ARC_LLM_CALL_RECORD_FIELD)
    if not isinstance(record, dict):
        record = {}
        output[ARC_LLM_CALL_RECORD_FIELD] = record
    record["structured_output"] = structured_output


def _role_hint(worker: WorkerConfig) -> str:
    if worker.id.startswith("reviewer"):
        return "reviewer"
    if worker.id.startswith("proposer"):
        return "proposer"
    return "generic"


def _cached_input_ratio(result: dict[str, Any]) -> float | None:
    record = result.get(ARC_LLM_CALL_RECORD_FIELD) if isinstance(result, dict) else None
    usage = record.get("usage") if isinstance(record, dict) else None
    if not isinstance(usage, Mapping):
        return None
    raw_ratio = usage.get("cached_input_ratio")
    if raw_ratio is not None:
        try:
            return float(raw_ratio)
        except (TypeError, ValueError):
            return None
    if usage.get("total_input_tokens") is not None:
        input_tokens = usage.get("total_input_tokens")
    elif usage.get("cache_creation_input_tokens") is not None or usage.get("cache_read_input_tokens") is not None:
        input_tokens = (
            _int(usage.get("input_tokens"))
            + _int(usage.get("cache_creation_input_tokens"))
            + _int(usage.get("cache_read_input_tokens"))
        )
    else:
        input_tokens = usage.get("input_tokens")

    if usage.get("effective_cached_input_tokens") is not None:
        cached_input_tokens = usage.get("effective_cached_input_tokens")
    elif usage.get("cache_creation_input_tokens") is not None or usage.get("cache_read_input_tokens") is not None:
        cached_input_tokens = _int(usage.get("cache_read_input_tokens"))
    else:
        cached_input_tokens = usage.get("cached_input_tokens")
    if input_tokens is None or cached_input_tokens is None:
        return None
    try:
        input_count = float(input_tokens)
        cached_count = float(cached_input_tokens)
    except (TypeError, ValueError):
        return None
    if input_count <= 0:
        return None
    return cached_count / input_count


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _validate_reviewer_output(review_output: dict[str, Any], *, worker: WorkerConfig, loop: LoopConfig) -> None:
    """Validate reviewer output against full schema plus envelope rules."""
    payload = strip_arc_llm_call_records(review_output)
    if worker.output_schema is not None:
        from jsonschema import ValidationError as JsonSchemaValidationError
        from jsonschema import validate as validate_json_schema
        from jsonschema.exceptions import SchemaError as JsonSchemaError

        try:
            validate_json_schema(instance=payload, schema=worker.output_schema)
        except JsonSchemaValidationError as exc:
            raise ValueError(f"reviewer output failed JSON schema validation: {exc.message}") from exc
        except JsonSchemaError as exc:
            raise ValueError(f"reviewer output schema is invalid: {exc.message}") from exc
    _validate_review_envelope(payload, loop)


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
    expected_ids = {proposer.id for proposer in loop.proposers}
    actual_ids = {str(proposer_id) for proposer_id in proposer_messages}
    missing = [proposer.id for proposer in loop.proposers if proposer.id not in actual_ids]
    if missing:
        raise ValueError(f"review.proposer_messages missing: {', '.join(missing)}")
    extra = sorted(actual_ids - expected_ids)
    if extra:
        raise ValueError(f"review.proposer_messages unexpected: {', '.join(extra)}")
    invalid_messages = sorted(
        str(proposer_id)
        for proposer_id, message in proposer_messages.items()
        if not isinstance(message, dict)
    )
    if invalid_messages:
        raise ValueError(f"review.proposer_messages entries must be objects: {', '.join(invalid_messages)}")
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


def _write_loop_failure_state(paths, *, run_id: str, result: dict[str, Any]) -> None:
    try:
        with acquire_lock(paths.lock, run_id=run_id, loop_id=str(result.get("loop_id") or "")):
            atomic_write_json(paths.state, result)
    except LockConflictError as exc:
        atomic_write_json(
            _loop_failure_diagnostic_path(paths),
            {
                "schema_version": "arc.llm.loop_failure_diagnostic.v1",
                "reason": "failed_to_reacquire_loop_lock",
                "lock_error": str(exc),
                "failure_result": result,
            },
        )


def _loop_failure_diagnostic_path(paths) -> Path:
    return paths.loop_root / "errors" / f"failure_after_lock_lost.{os.getpid()}.{time.time_ns()}.json"


def _skipped_loop_result(paths: RunPaths, loop_id: str, error: str) -> dict[str, Any]:
    return {
        "loop_id": loop_id,
        "status": "skipped",
        "rounds_completed": 0,
        "error": error,
        "loop_root": str(paths.loop(loop_id).loop_root),
    }


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
