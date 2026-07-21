from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, replace
import inspect
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .call_record import ARC_LLM_CALL_RECORD_SCHEMA_VERSION, attach_arc_llm_call_record, strip_arc_llm_call_records
from .call_checkpoint import (
    LLMCallNeedsSupervision,
    LLMCallRetryExhausted,
    checkpoint_path,
    prepare_call,
    record_failure,
    record_response,
    record_validated,
)
from .host import HostDetection, select_llm_provider
from .json_schema import to_provider_json_schema
from .model import reasoning_effort_for_model_tier, resolve_model_with_warnings
from .providers.activity import resolve_idle_timeout_seconds
from .providers.base import (
    LLMConfigurationError,
    LLMSchemaError,
    LLMWorkerCancelled,
    LLMWorkerError,
    LLMWorkerTimeout,
)
from .progress_journal import ProgressJournal
from .progress_prompt import apply_runtime_progress_contract
from .providers.registry import get_provider_spec
from .providers.select import select_provider
from .schema_cache import schema_hash, sha256_text
from .safety import LLMSafetyController
from .sessions import LLMSessionManager, LLMSessionRef, runtime_fingerprint
from .structured_recovery import recover_json_output, structured_metadata
from .usage import LLMProviderResponse, LLMUsage


MAX_ATTEMPTS_PER_PROVIDER = 1
RETRY_INTERVAL_SECONDS = 10
LOW_CONTENT_TOKEN_THRESHOLD = 10
NATIVE_RESUME_RECONCILIATION_PROMPT = (
    "Supervised native-session recovery: the preceding turn may already have run. "
    "Do not repeat its work. Reconcile the native session and return the final answer "
    "for that preceding request in the required format."
)


class LLMTaskError(RuntimeError):
    pass


class LLMNeedsLLM(LLMTaskError):
    """Automatic provider selection found no runnable host LLM."""

    def __init__(self, config: "LLMConfig") -> None:
        super().__init__(
            "provider=auto resolved to manual; run from a supported agent host "
            "or select an explicit provider"
        )
        self.config = config


class LLMOutputValidationError(RuntimeError):
    pass


class LLMRetryableProviderOutputError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str | None
    host: HostDetection
    signals: list[str]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMAttemptFailure:
    provider: str
    attempt: int
    error: str


@dataclass(frozen=True)
class LLMCallOutcome:
    value: Any
    usage: LLMUsage
    native_session_id: str | None
    session_policy: str
    session_key: str | None
    call_label: str | None
    prompt_sha256: str | None
    static_prefix_sha256: str | None
    schema_sha256: str | None
    runtime_fingerprint: str | None
    idempotency_key: str | None = None
    generation: int | None = None
    prompt_bytes: int | None = None
    logical_receipt: dict[str, Any] | None = None
    call_record: dict[str, Any] | None = None
    structured_output: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()


def resolve_llm_config(
    *,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
) -> LLMConfig:
    if provider == "auto" and model:
        raise ValueError("Exact model requires explicit provider; use provider=<provider> or model_tier=<low|medium|high|max>.")
    selected = select_llm_provider(
        env=env,
        process_chain=process_chain,
        explicit_provider=None if provider == "auto" else provider,
    )
    model_resolution = resolve_model_with_warnings(
        selected.provider,
        model,
        model_tier=model_tier,
        env=env,
    )
    spec = get_provider_spec(selected.provider)
    return LLMConfig(
        provider=selected.provider,
        model=model_resolution.model,
        host=selected.host,
        signals=selected.signals,
        warnings=(*spec.warning_codes, *model_resolution.warnings),
    )


def resolve_llm_configs(
    *,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
) -> list[LLMConfig]:
    return [
        resolve_llm_config(
            provider=provider,
            model=model,
            model_tier=model_tier,
            env=env,
            process_chain=process_chain,
        )
    ]


def run_json(
    prompt: str,
    *,
    schema: dict[str, Any] | None = None,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    validate_schema: bool = True,
    output_recovery: str = "strict",
    schema_formatter_enabled: bool = True,
    role_hint: str | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
    session_policy: str = "stateless",
    session_manager: LLMSessionManager | None = None,
    session_key: str | None = None,
    session_name: str | None = None,
    session_metadata: Mapping[str, Any] | None = None,
    artifact_dir: Path | str | None = None,
    call_label: str | None = None,
    static_prefix: str | None = None,
    idle_timeout_seconds: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    idempotency_key: str | None = None,
    progress_contract_scope: str = "call",
    supervised_native_resume: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper returning the JSON value with its audit record."""

    outcome = run_json_result(
        prompt,
        schema=schema,
        provider=provider,
        model=model,
        model_tier=model_tier,
        validate_schema=validate_schema,
        output_recovery=output_recovery,
        schema_formatter_enabled=schema_formatter_enabled,
        role_hint=role_hint,
        env=env,
        process_chain=process_chain,
        session_policy=session_policy,
        session_manager=session_manager,
        session_key=session_key,
        session_name=session_name,
        session_metadata=session_metadata,
        artifact_dir=artifact_dir,
        call_label=call_label,
        static_prefix=static_prefix,
        idle_timeout_seconds=idle_timeout_seconds,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        idempotency_key=idempotency_key,
        progress_contract_scope=progress_contract_scope,
        supervised_native_resume=supervised_native_resume,
    )
    return attach_arc_llm_call_record(outcome.value, dict(outcome.call_record or {}))


def run_json_result(
    prompt: str,
    *,
    schema: dict[str, Any] | None = None,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    validate_schema: bool = True,
    output_recovery: str = "strict",
    schema_formatter_enabled: bool = True,
    role_hint: str | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
    session_policy: str = "stateless",
    session_manager: LLMSessionManager | None = None,
    session_key: str | None = None,
    session_name: str | None = None,
    session_metadata: Mapping[str, Any] | None = None,
    artifact_dir: Path | str | None = None,
    call_label: str | None = None,
    static_prefix: str | None = None,
    idle_timeout_seconds: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    idempotency_key: str | None = None,
    progress_contract_scope: str = "call",
    supervised_native_resume: bool = False,
) -> LLMCallOutcome:
    if session_policy not in {"stateless", "stateful"}:
        raise ValueError("session_policy must be stateless or stateful")
    if output_recovery not in {"strict", "warn"}:
        raise ValueError("output_recovery must be strict or warn")
    if session_policy == "stateful" and (session_manager is None or not session_key):
        raise ValueError("stateful run_json requires session_manager and session_key")
    if progress_contract_scope not in {"call", "session"}:
        raise ValueError("progress_contract_scope must be call or session")
    if session_policy == "stateful" and (not idempotency_key or artifact_dir is None):
        raise ValueError("stateful run_json requires idempotency_key and artifact_dir")
    if idempotency_key and artifact_dir is None:
        raise ValueError("idempotency_key requires artifact_dir")
    if supervised_native_resume and session_policy != "stateful":
        raise ValueError("supervised_native_resume requires a stateful session")
    configs = resolve_llm_configs(
        provider=provider,
        model=model,
        model_tier=model_tier,
        env=env,
        process_chain=process_chain,
    )
    _raise_if_auto_resolved_manual(provider, configs)
    return _run_with_retries(
        configs,
        provider_requested=provider,
        model_requested=model,
        model_tier_requested=model_tier,
        attach_call_record=False,
        env=env,
        process_chain=process_chain,
        max_attempts=MAX_ATTEMPTS_PER_PROVIDER,
        return_outcome=True,
        idle_timeout_seconds=idle_timeout_seconds,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        call=lambda selected, config, effective_idle_timeout_seconds: _generate_json(
            selected,
            prompt,
            schema=schema,
            model=config.model,
            validate_schema=validate_schema,
            output_recovery=output_recovery,
            schema_formatter_enabled=schema_formatter_enabled,
            role_hint=role_hint,
            provider_used=config.provider,
            model_tier=model_tier,
            env=env,
            process_chain=process_chain,
            session_policy=session_policy,
            session_manager=session_manager,
            session_key=session_key,
            session_name=session_name,
            session_metadata=session_metadata,
            artifact_dir=Path(artifact_dir) if artifact_dir else None,
            call_label=call_label,
            static_prefix=static_prefix,
            idle_timeout_seconds=effective_idle_timeout_seconds,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            idempotency_key=idempotency_key,
            progress_contract_scope=progress_contract_scope,
            supervised_native_resume=supervised_native_resume,
        ),
    )


def run_text(
    prompt: str,
    *,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
    session_policy: str = "stateless",
    session_manager: LLMSessionManager | None = None,
    session_key: str | None = None,
    session_name: str | None = None,
    session_metadata: Mapping[str, Any] | None = None,
    artifact_dir: Path | str | None = None,
    call_label: str | None = None,
    static_prefix: str | None = None,
    idle_timeout_seconds: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    idempotency_key: str | None = None,
    progress_contract_scope: str = "call",
    supervised_native_resume: bool = False,
) -> str:
    return run_text_result(
        prompt,
        provider=provider,
        model=model,
        model_tier=model_tier,
        env=env,
        process_chain=process_chain,
        session_policy=session_policy,
        session_manager=session_manager,
        session_key=session_key,
        session_name=session_name,
        session_metadata=session_metadata,
        artifact_dir=artifact_dir,
        call_label=call_label,
        static_prefix=static_prefix,
        idle_timeout_seconds=idle_timeout_seconds,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        idempotency_key=idempotency_key,
        progress_contract_scope=progress_contract_scope,
        supervised_native_resume=supervised_native_resume,
    ).value


def run_text_result(
    prompt: str,
    *,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
    session_policy: str = "stateless",
    session_manager: LLMSessionManager | None = None,
    session_key: str | None = None,
    session_name: str | None = None,
    session_metadata: Mapping[str, Any] | None = None,
    artifact_dir: Path | str | None = None,
    call_label: str | None = None,
    static_prefix: str | None = None,
    idle_timeout_seconds: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    idempotency_key: str | None = None,
    progress_contract_scope: str = "call",
    supervised_native_resume: bool = False,
) -> LLMCallOutcome:
    if session_policy not in {"stateless", "stateful"}:
        raise ValueError("session_policy must be stateless or stateful")
    if session_policy == "stateful" and (session_manager is None or not session_key):
        raise ValueError("stateful run_text requires session_manager and session_key")
    if progress_contract_scope not in {"call", "session"}:
        raise ValueError("progress_contract_scope must be call or session")
    if session_policy == "stateful" and (not idempotency_key or artifact_dir is None):
        raise ValueError("stateful run_text requires idempotency_key and artifact_dir")
    if idempotency_key and artifact_dir is None:
        raise ValueError("idempotency_key requires artifact_dir")
    if supervised_native_resume and session_policy != "stateful":
        raise ValueError("supervised_native_resume requires a stateful session")
    configs = resolve_llm_configs(
        provider=provider,
        model=model,
        model_tier=model_tier,
        env=env,
        process_chain=process_chain,
    )
    _raise_if_auto_resolved_manual(provider, configs)
    return _run_with_retries(
        configs,
        provider_requested=provider,
        model_requested=model,
        model_tier_requested=model_tier,
        attach_call_record=False,
        env=env,
        process_chain=process_chain,
        max_attempts=1 if session_policy == "stateful" else MAX_ATTEMPTS_PER_PROVIDER,
        return_outcome=True,
        idle_timeout_seconds=idle_timeout_seconds,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        call=lambda selected, config, effective_idle_timeout_seconds: _generate_text(
            selected,
            prompt,
            model=config.model,
            provider_used=config.provider,
            model_tier=model_tier,
            env=env,
            process_chain=process_chain,
            session_policy=session_policy,
            session_manager=session_manager,
            session_key=session_key,
            session_name=session_name,
            session_metadata=session_metadata,
            artifact_dir=Path(artifact_dir) if artifact_dir else None,
            call_label=call_label,
            static_prefix=static_prefix,
            idle_timeout_seconds=effective_idle_timeout_seconds,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            idempotency_key=idempotency_key,
            progress_contract_scope=progress_contract_scope,
            supervised_native_resume=supervised_native_resume,
        ),
    )


def _raise_if_auto_resolved_manual(provider_requested: str, configs: Sequence[LLMConfig]) -> None:
    if provider_requested == "auto" and configs and configs[0].provider == "manual":
        raise LLMNeedsLLM(configs[0])


def _run_with_retries(
    configs: Sequence[LLMConfig],
    *,
    provider_requested: str,
    model_requested: str | None,
    model_tier_requested: str | None,
    attach_call_record: bool,
    env: Mapping[str, str] | None,
    process_chain: Sequence[str] | None,
    max_attempts: int = MAX_ATTEMPTS_PER_PROVIDER,
    return_outcome: bool = False,
    idle_timeout_seconds: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    call: Callable[[Any, LLMConfig, float], Any],
) -> Any:
    del progress_callback
    failures: list[LLMAttemptFailure] = []
    attempt_records: list[dict[str, Any]] = []
    for fallback_index, config in enumerate(configs):
        _check_cancel(cancel_check)
        provider_env = _env_with_tier_reasoning_default(env, config.provider, model_tier_requested)
        try:
            effective_idle_timeout_seconds = resolve_idle_timeout_seconds(
                idle_timeout_seconds,
                env=provider_env,
                provider=config.provider,
            )
        except (TypeError, ValueError) as exc:
            raise LLMConfigurationError(str(exc)) from exc
        selected = select_provider(config.provider, env=provider_env, process_chain=process_chain)
        for attempt in range(1, max_attempts + 1):
            try:
                _check_cancel(cancel_check)
                result = call(selected, config, effective_idle_timeout_seconds)
                value = result.value if isinstance(result, LLMCallOutcome) else result
                attempt_record = _attempt_record(
                    config,
                    fallback_index=fallback_index,
                    attempt=attempt,
                    status="success",
                )
                attempt_records.append(attempt_record)
                if isinstance(result, LLMCallOutcome):
                    result = replace(
                        result,
                        call_record=_call_record(
                            config,
                            provider_requested=provider_requested,
                            model_requested=model_requested,
                            model_tier_requested=model_tier_requested,
                            fallback_index=fallback_index,
                            attempt=attempt,
                            attempts=attempt_records,
                            outcome=result,
                        ),
                    )
                    value = result.value
                if attach_call_record and isinstance(value, dict):
                    outcome = result if isinstance(result, LLMCallOutcome) else None
                    return attach_arc_llm_call_record(
                        value,
                        _call_record(
                            config,
                            provider_requested=provider_requested,
                            model_requested=model_requested,
                            model_tier_requested=model_tier_requested,
                            fallback_index=fallback_index,
                            attempt=attempt,
                            attempts=attempt_records,
                            outcome=outcome,
                        ),
                    )
                return result if return_outcome and isinstance(result, LLMCallOutcome) else value
            except Exception as exc:
                failures.append(LLMAttemptFailure(provider=config.provider, attempt=attempt, error=str(exc)))
                attempt_records.append(
                    _attempt_record(
                        config,
                        fallback_index=fallback_index,
                        attempt=attempt,
                        status="failed",
                        error=exc,
                    )
                )
                if isinstance(
                    exc,
                    (
                        LLMSchemaError,
                        LLMWorkerCancelled,
                        LLMWorkerTimeout,
                        LLMCallNeedsSupervision,
                        LLMCallRetryExhausted,
                    ),
                ):
                    raise
                if isinstance(exc, LLMOutputValidationError) or (
                    isinstance(exc, LLMWorkerError) and not exc.retryable
                ):
                    raise LLMTaskError(_failure_message(failures, max_attempts=max_attempts)) from exc
                if _has_remaining_attempt(configs, fallback_index=fallback_index, attempt=attempt, max_attempts=max_attempts):
                    time.sleep(RETRY_INTERVAL_SECONDS)
                    _check_cancel(cancel_check)
    raise LLMTaskError(_failure_message(failures, max_attempts=max_attempts))


def _check_cancel(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise LLMWorkerCancelled("LLM worker call was cancelled")


def _env_with_tier_reasoning_default(
    env: Mapping[str, str] | None,
    provider: str,
    model_tier: str | None,
) -> Mapping[str, str] | None:
    effort = reasoning_effort_for_model_tier(provider, model_tier)
    if effort is None:
        return env
    resolved = dict(env) if env is not None else dict(os.environ)
    if provider == "codex-cli":
        resolved.setdefault("ARC_CODEX_REASONING_EFFORT", effort)
    elif provider == "claude-cli":
        resolved.setdefault("ARC_CLAUDE_EFFORT", effort)
    return resolved


def _has_remaining_attempt(
    configs: Sequence[LLMConfig],
    *,
    fallback_index: int,
    attempt: int,
    max_attempts: int,
) -> bool:
    return attempt < max_attempts or fallback_index < len(configs) - 1


def _generate_json(
    selected: Any,
    prompt: str,
    *,
    schema: dict[str, Any] | None,
    model: str | None,
    validate_schema: bool,
    output_recovery: str,
    schema_formatter_enabled: bool,
    role_hint: str | None,
    provider_used: str,
    model_tier: str | None,
    env: Mapping[str, str] | None,
    process_chain: Sequence[str] | None,
    session_policy: str,
    session_manager: LLMSessionManager | None,
    session_key: str | None,
    session_name: str | None,
    session_metadata: Mapping[str, Any] | None,
    artifact_dir: Path | None,
    call_label: str | None,
    static_prefix: str | None,
    idle_timeout_seconds: float | None,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    cancel_check: Callable[[], bool] | None,
    idempotency_key: str | None,
    progress_contract_scope: str,
    supervised_native_resume: bool,
) -> LLMCallOutcome:
    runtime_fp = _runtime_fp(
        provider_used=provider_used,
        model=model,
        model_tier=model_tier,
        env=env,
        process_chain=process_chain,
    )
    if session_policy == "stateful" and not hasattr(selected, "generate_json_result"):
        raise LLMTaskError(f"Provider {provider_used} does not support stateful sessions")
    provider_schema = to_provider_json_schema(schema)
    prepared_checkpoint = None
    effective_prompt = apply_runtime_progress_contract(
        prompt, scope=progress_contract_scope, generation_bootstrap=True
    )
    generation: int | None = None
    progress = _progress_journal(
        artifact_dir=artifact_dir,
        call_label=call_label,
        provider=provider_used,
        callback=progress_callback,
    )

    def call_provider(
        session: LLMSessionRef | None, session_turn: int | None = None
    ) -> LLMProviderResponse[dict[str, Any]]:
        nonlocal prepared_checkpoint
        if artifact_dir is not None and (call_label or idempotency_key):
            path, identity = checkpoint_path(
                artifact_dir,
                prompt=prompt,
                schema=provider_schema,
                provider=provider_used,
                model=model,
                call_label=call_label,
                session_policy=session_policy,
                session_key=session.key if session is not None else None,
                session_turn=session_turn,
                runtime_fingerprint=runtime_fp,
                idempotency_key=idempotency_key,
                generation=session.generation if session is not None else None,
                progress_contract_scope=progress_contract_scope,
            )
            prepared_checkpoint = prepare_call(
                path,
                identity=identity,
                cancel_check=cancel_check,
                supervised_native_resume=supervised_native_resume,
                native_session_available=bool(session and session.native_session_id),
            )
            if prepared_checkpoint.replay_response is not None:
                return prepared_checkpoint.replay_response
            if supervised_native_resume and (session is None or not session.native_session_id):
                prepared_checkpoint.release_lock()
                raise LLMTaskError(
                    "supervised native resume requires an existing provider session id"
                )
        provider_prompt = (
            NATIVE_RESUME_RECONCILIATION_PROMPT
            if supervised_native_resume
            else effective_prompt
        )
        def invoke() -> LLMProviderResponse[dict[str, Any]]:
            if hasattr(selected, "generate_json_result"):
                kwargs = {
                    "schema": provider_schema,
                    "model": model,
                    "session": session,
                    "session_policy": session_policy,
                    "schema_cache_dir": _schema_cache_dir(artifact_dir),
                    "artifact_dir": artifact_dir,
                }
                if _accepts_keyword(selected.generate_json_result, "output_recovery"):
                    kwargs["output_recovery"] = output_recovery
                if _accepts_keyword(selected.generate_json_result, "idle_timeout_seconds"):
                    kwargs["idle_timeout_seconds"] = idle_timeout_seconds
                if _accepts_keyword(selected.generate_json_result, "progress_callback"):
                    kwargs["progress_callback"] = progress
                if _accepts_keyword(selected.generate_json_result, "cancel_check"):
                    kwargs["cancel_check"] = cancel_check
                return selected.generate_json_result(provider_prompt, **kwargs)
            return LLMProviderResponse(selected.generate_json(provider_prompt, schema=provider_schema, model=model))

        try:
            response = _controlled_provider_call(
                provider_used,
                env=env,
                cancel_check=cancel_check,
                call_label=call_label,
                invoke=invoke,
            )
        except BaseException as exc:
            if prepared_checkpoint is not None:
                record_failure(prepared_checkpoint, exc)
            raise
        if response.prompt_sent_bytes is None or response.prompt_sent_sha256 is None:
            response = replace(
                response,
                prompt_sent_bytes=response.prompt_sent_bytes or len(provider_prompt.encode("utf-8")),
                prompt_sent_sha256=response.prompt_sent_sha256 or sha256_text(provider_prompt),
            )
        if prepared_checkpoint is not None:
            record_response(prepared_checkpoint, response)
        progress({"event": "call_finished", "substantive": False, "resumable": False})
        return response

    if session_policy == "stateful":
        assert session_manager is not None
        assert session_key is not None
        with session_manager.locked_turn(
            key=session_key,
            provider=provider_used,
            model=model,
            runtime_fingerprint=runtime_fp,
            name=session_name,
            metadata=session_metadata,
        ) as (session, turn_count):
            generation = session.generation
            effective_prompt = apply_runtime_progress_contract(
                prompt,
                scope=progress_contract_scope,
                generation_bootstrap=turn_count == 0,
            )
            response = call_provider(session, turn_count)
            result = response.value
            prompt_sha = response.prompt_sent_sha256 or sha256_text(effective_prompt)
            native_session_id = response.native_session_id
            session_warnings: list[str] = []
            if native_session_id:
                warning = _update_native_session_id_with_self_heal(
                    session_manager,
                    session=session,
                    native_session_id=native_session_id,
                    provider=provider_used,
                    model=model,
                    runtime_fingerprint=runtime_fp,
                    name=session_name,
                    metadata=session_metadata,
                )
                if warning:
                    session_warnings.append(warning)
            recorded_native_session_id = native_session_id or session.native_session_id

            def record_turn(structured_output: dict[str, Any] | None = None) -> None:
                extra = {"runtime_fingerprint": runtime_fp}
                if structured_output:
                    extra["structured_output"] = structured_output
                if session_warnings:
                    extra["session_warnings"] = list(session_warnings)
                session_manager.record_turn(
                    session.key,
                    call_label=call_label or "",
                    prompt_sha256=prompt_sha,
                    static_prefix_sha256=sha256_text(static_prefix) if static_prefix else None,
                    schema_sha256=schema_hash(schema),
                    usage=response.usage.to_json(),
                    provider_used=provider_used,
                    model_used=model,
                    native_session_id=recorded_native_session_id,
                    idempotency_key=idempotency_key,
                    generation=session.generation,
                    extra=extra,
                )

            try:
                result, structured_output = _recover_or_validate_json_output(
                    result,
                    schema=schema,
                    validate_schema=validate_schema,
                    output_recovery=output_recovery,
                    schema_formatter_enabled=schema_formatter_enabled,
                    role_hint=role_hint,
                    response=response,
                    provider=provider_used,
                    model=model,
                    model_tier=model_tier,
                    env=env,
                    process_chain=process_chain,
                    artifact_dir=artifact_dir,
                    call_label=call_label,
                    idle_timeout_seconds=idle_timeout_seconds,
                    progress_callback=progress,
                    cancel_check=cancel_check,
                    idempotency_key=idempotency_key,
                )
            except Exception:
                record_turn(response.structured_output)
                raise
            record_turn(structured_output)
    else:
        session = None
        response = call_provider(None)
        result = response.value
        result, structured_output = _recover_or_validate_json_output(
            result,
            schema=schema,
            validate_schema=validate_schema,
            output_recovery=output_recovery,
            schema_formatter_enabled=schema_formatter_enabled,
            role_hint=role_hint,
            response=response,
            provider=provider_used,
            model=model,
            model_tier=model_tier,
            env=env,
            process_chain=process_chain,
            artifact_dir=artifact_dir,
            call_label=call_label,
            idle_timeout_seconds=idle_timeout_seconds,
            progress_callback=progress,
            cancel_check=cancel_check,
            idempotency_key=idempotency_key,
        )
        native_session_id = response.native_session_id
        prompt_sha = response.prompt_sent_sha256 or sha256_text(effective_prompt)
    if prepared_checkpoint is not None:
        record_validated(prepared_checkpoint)
    return LLMCallOutcome(
        value=result,
        usage=response.usage,
        native_session_id=(response.native_session_id or session.native_session_id) if session else response.native_session_id,
        session_policy=session_policy,
        session_key=session.key if session else None,
        call_label=call_label,
        prompt_sha256=prompt_sha,
        static_prefix_sha256=sha256_text(static_prefix) if static_prefix else None,
        schema_sha256=schema_hash(schema),
        runtime_fingerprint=runtime_fp,
        idempotency_key=idempotency_key,
        generation=generation,
        prompt_bytes=response.prompt_sent_bytes,
        logical_receipt=_logical_receipt(
            idempotency_key=idempotency_key,
            generation=generation,
            prepared=prepared_checkpoint,
            response=response,
        ),
        structured_output=structured_output,
    )


def _generate_text(
    selected: Any,
    prompt: str,
    *,
    model: str | None,
    provider_used: str,
    model_tier: str | None,
    env: Mapping[str, str] | None,
    process_chain: Sequence[str] | None,
    session_policy: str,
    session_manager: LLMSessionManager | None,
    session_key: str | None,
    session_name: str | None,
    session_metadata: Mapping[str, Any] | None,
    artifact_dir: Path | None,
    call_label: str | None,
    static_prefix: str | None,
    idle_timeout_seconds: float | None,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    cancel_check: Callable[[], bool] | None,
    idempotency_key: str | None,
    progress_contract_scope: str,
    supervised_native_resume: bool,
) -> LLMCallOutcome:
    runtime_fp = _runtime_fp(
        provider_used=provider_used,
        model=model,
        model_tier=model_tier,
        env=env,
        process_chain=process_chain,
    )
    if session_policy == "stateful" and not hasattr(selected, "generate_text_result"):
        raise LLMTaskError(f"Provider {provider_used} does not support stateful sessions")

    prepared_checkpoint = None
    effective_prompt = apply_runtime_progress_contract(
        prompt, scope=progress_contract_scope, generation_bootstrap=True
    )
    generation: int | None = None
    progress = _progress_journal(
        artifact_dir=artifact_dir,
        call_label=call_label,
        provider=provider_used,
        callback=progress_callback,
    )

    def call_provider(
        session: LLMSessionRef | None, session_turn: int | None = None
    ) -> LLMProviderResponse[str]:
        nonlocal prepared_checkpoint
        if artifact_dir is not None and (call_label or idempotency_key):
            path, identity = checkpoint_path(
                artifact_dir,
                prompt=prompt,
                schema=None,
                provider=provider_used,
                model=model,
                call_label=call_label,
                session_policy=session_policy,
                session_key=session.key if session is not None else None,
                session_turn=session_turn,
                runtime_fingerprint=runtime_fp,
                idempotency_key=idempotency_key,
                generation=session.generation if session is not None else None,
                progress_contract_scope=progress_contract_scope,
            )
            prepared_checkpoint = prepare_call(
                path,
                identity=identity,
                cancel_check=cancel_check,
                supervised_native_resume=supervised_native_resume,
                native_session_available=bool(session and session.native_session_id),
            )
            if prepared_checkpoint.replay_response is not None:
                return prepared_checkpoint.replay_response
            if supervised_native_resume and (session is None or not session.native_session_id):
                prepared_checkpoint.release_lock()
                raise LLMTaskError(
                    "supervised native resume requires an existing provider session id"
                )
        provider_prompt = (
            NATIVE_RESUME_RECONCILIATION_PROMPT
            if supervised_native_resume
            else effective_prompt
        )
        def invoke() -> LLMProviderResponse[str]:
            if hasattr(selected, "generate_text_result"):
                kwargs = {
                    "model": model,
                    "session": session,
                    "session_policy": session_policy,
                    "artifact_dir": artifact_dir,
                }
                if _accepts_keyword(selected.generate_text_result, "idle_timeout_seconds"):
                    kwargs["idle_timeout_seconds"] = idle_timeout_seconds
                if _accepts_keyword(selected.generate_text_result, "progress_callback"):
                    kwargs["progress_callback"] = progress
                if _accepts_keyword(selected.generate_text_result, "cancel_check"):
                    kwargs["cancel_check"] = cancel_check
                return selected.generate_text_result(provider_prompt, **kwargs)
            return LLMProviderResponse(selected.generate_text(provider_prompt, model=model))

        try:
            response = _controlled_provider_call(
                provider_used,
                env=env,
                cancel_check=cancel_check,
                call_label=call_label,
                invoke=invoke,
            )
        except BaseException as exc:
            if prepared_checkpoint is not None:
                record_failure(prepared_checkpoint, exc)
            raise
        if response.prompt_sent_bytes is None or response.prompt_sent_sha256 is None:
            response = replace(
                response,
                prompt_sent_bytes=response.prompt_sent_bytes or len(provider_prompt.encode("utf-8")),
                prompt_sent_sha256=response.prompt_sent_sha256 or sha256_text(provider_prompt),
            )
        if prepared_checkpoint is not None:
            record_response(prepared_checkpoint, response)
        progress({"event": "call_finished", "substantive": False, "resumable": False})
        return response

    if session_policy == "stateful":
        assert session_manager is not None
        assert session_key is not None
        with session_manager.locked_turn(
            key=session_key,
            provider=provider_used,
            model=model,
            runtime_fingerprint=runtime_fp,
            name=session_name,
            metadata=session_metadata,
        ) as (session, turn_count):
            generation = session.generation
            effective_prompt = apply_runtime_progress_contract(
                prompt,
                scope=progress_contract_scope,
                generation_bootstrap=turn_count == 0,
            )
            response = call_provider(session, turn_count)
            prompt_sha = response.prompt_sent_sha256 or sha256_text(effective_prompt)
            session_warnings: list[str] = []
            if response.native_session_id:
                warning = _update_native_session_id_with_self_heal(
                    session_manager,
                    session=session,
                    native_session_id=response.native_session_id,
                    provider=provider_used,
                    model=model,
                    runtime_fingerprint=runtime_fp,
                    name=session_name,
                    metadata=session_metadata,
                )
                if warning:
                    session_warnings.append(warning)
            recorded_native_session_id = response.native_session_id or session.native_session_id
            extra = {"runtime_fingerprint": runtime_fp}
            if session_warnings:
                extra["session_warnings"] = list(session_warnings)
            session_manager.record_turn(
                session.key,
                call_label=call_label or "",
                prompt_sha256=prompt_sha,
                static_prefix_sha256=sha256_text(static_prefix) if static_prefix else None,
                schema_sha256=None,
                usage=response.usage.to_json(),
                provider_used=provider_used,
                model_used=model,
                native_session_id=recorded_native_session_id,
                idempotency_key=idempotency_key,
                generation=session.generation,
                extra=extra,
            )
    else:
        session = None
        response = call_provider(None)
        prompt_sha = response.prompt_sent_sha256 or sha256_text(effective_prompt)
    if not str(response.value or "").strip():
        raise LLMOutputValidationError("LLM text output was empty")
    if prepared_checkpoint is not None:
        record_validated(prepared_checkpoint)
    return LLMCallOutcome(
        value=response.value,
        usage=response.usage,
        native_session_id=(response.native_session_id or session.native_session_id) if session else response.native_session_id,
        session_policy=session_policy,
        session_key=session.key if session else None,
        call_label=call_label,
        prompt_sha256=prompt_sha,
        static_prefix_sha256=sha256_text(static_prefix) if static_prefix else None,
        schema_sha256=None,
        runtime_fingerprint=runtime_fp,
        idempotency_key=idempotency_key,
        generation=generation,
        prompt_bytes=response.prompt_sent_bytes,
        logical_receipt=_logical_receipt(
            idempotency_key=idempotency_key,
            generation=generation,
            prepared=prepared_checkpoint,
            response=response,
        ),
    )


def _runtime_fp(
    *,
    provider_used: str,
    model: str | None,
    model_tier: str | None,
    env: Mapping[str, str] | None,
    process_chain: Sequence[str] | None,
) -> str:
    return runtime_fingerprint(
        provider=provider_used,
        model=model,
        model_tier=model_tier,
        env=env,
        process_chain=process_chain,
    )


def _logical_receipt(
    *,
    idempotency_key: str | None,
    generation: int | None,
    prepared: Any,
    response: LLMProviderResponse[Any],
) -> dict[str, Any] | None:
    if idempotency_key is None:
        return None
    value = response.value
    if isinstance(value, str):
        response_sha = sha256_text(value)
    else:
        from .schema_cache import canonical_json

        response_sha = sha256_text(canonical_json(value))
    return {
        "schema_version": "arc.llm.logical_receipt.v1",
        "idempotency_key": idempotency_key,
        "generation": generation,
        "checkpoint_state": "validated",
        "replayed": bool(prepared and prepared.replayed),
        "response_sha256": response_sha,
    }


def _progress_journal(
    *,
    artifact_dir: Path | None,
    call_label: str | None,
    provider: str,
    callback: Callable[[dict[str, Any]], None] | None,
) -> ProgressJournal:
    if isinstance(callback, ProgressJournal):
        return callback
    return ProgressJournal(
        artifact_dir=artifact_dir,
        call_label=call_label,
        provider=provider,
        callback=callback,
    )


def _controlled_provider_call(
    provider: str,
    *,
    env: Mapping[str, str] | None,
    cancel_check: Callable[[], bool] | None,
    call_label: str | None,
    invoke: Callable[[], LLMProviderResponse[Any]],
) -> LLMProviderResponse[Any]:
    if provider == "manual":
        return invoke()
    safety_env = dict(os.environ)
    if env is not None:
        safety_env.update(env)
    controller = LLMSafetyController(env=safety_env)
    with controller.acquire_call(
        provider,
        timeout_seconds=None,
        cancel_check=cancel_check,
        call_label=call_label,
    ) as permit:
        response = invoke()
        permit.report_success()
        return response


def _schema_cache_dir(artifact_dir: Path | None) -> Path | None:
    if artifact_dir is None:
        return None
    return artifact_dir / "schemas"


def _update_native_session_id_with_self_heal(
    session_manager: LLMSessionManager,
    *,
    session: LLMSessionRef,
    native_session_id: str,
    provider: str,
    model: str | None,
    runtime_fingerprint: str,
    name: str | None,
    metadata: Mapping[str, Any] | None,
) -> str | None:
    try:
        session_manager.update_native_session_id(session.key, native_session_id)
        return None
    except KeyError:
        session_manager.get_or_create(
            key=session.key,
            provider=provider,
            model=model,
            runtime_fingerprint=runtime_fingerprint,
            name=name,
            metadata=metadata,
        )
        session_manager.update_native_session_id(session.key, native_session_id)
        return f"self_healed_missing_session_record:{session.key}"


def _validate_json_output(result: dict[str, Any], schema: dict[str, Any]) -> None:
    from jsonschema import ValidationError as JsonSchemaValidationError
    from jsonschema import validate as validate_json_schema
    from jsonschema.exceptions import SchemaError as JsonSchemaError

    try:
        validate_json_schema(instance=result, schema=schema)
    except JsonSchemaValidationError as exc:
        raise LLMOutputValidationError(f"JSON output failed schema validation: {exc.message}") from exc
    except JsonSchemaError as exc:
        raise LLMOutputValidationError(f"JSON schema is invalid: {exc.message}") from exc


def format_to_schema_or_retry(*args: Any, **kwargs: Any):
    from .schema_formatter import format_to_schema_or_retry as formatter

    return formatter(*args, **kwargs)


def _recover_or_validate_json_output(
    result: Any,
    *,
    schema: dict[str, Any] | None,
    validate_schema: bool,
    output_recovery: str,
    schema_formatter_enabled: bool = True,
    role_hint: str | None,
    response: LLMProviderResponse[dict[str, Any]],
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
    artifact_dir: Path | None = None,
    call_label: str | None = None,
    idle_timeout_seconds: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    idempotency_key: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    structured_output = response.structured_output
    provider_recovered = isinstance(structured_output, Mapping) and structured_output.get("mode") == "recovered"
    if not validate_schema:
        if isinstance(result, dict):
            if schema is not None and output_recovery == "warn":
                try:
                    _validate_json_output(result, schema)
                except Exception as exc:
                    structured_output = _merge_structured_output_warning(
                        structured_output,
                        severity="minor",
                        warnings=[
                            "JSON object did not satisfy schema, but validate_schema=False allowed continuation.",
                            str(exc),
                        ],
                        raw_text=response.raw_output,
                        strategy="schema_warning_no_validation",
                        provider_error_type=type(exc).__name__,
                    )
            return result, structured_output
        if output_recovery != "warn":
            raise LLMOutputValidationError("JSON output was not an object")
        return {}, _recovered_natural_language_metadata(result, response)
    if not isinstance(result, dict):
        if output_recovery != "warn":
            raise LLMOutputValidationError("JSON output was not an object")
        if schema is None:
            return {}, _recovered_natural_language_metadata(result, response)
        result, structured_output = _recover_warn_schema_output(
            result,
            schema=schema,
            role_hint=role_hint,
            response=response,
            error=LLMOutputValidationError("JSON output was not an object"),
            provider_metadata=structured_output,
            schema_formatter_enabled=schema_formatter_enabled,
            provider=provider,
            model=model,
            model_tier=model_tier,
            env=env,
            process_chain=process_chain,
            artifact_dir=artifact_dir,
            call_label=call_label,
            idle_timeout_seconds=idle_timeout_seconds,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            idempotency_key=idempotency_key,
        )
        if schema is not None:
            _validate_json_output(result, schema)
        return result, structured_output
    if schema is None:
        return result, structured_output
    if validate_schema:
        try:
            _validate_json_output(result, schema)
            return result, structured_output
        except Exception as exc:
            if output_recovery != "warn":
                raise
            result, structured_output = _recover_warn_schema_output(
                result,
                schema=schema,
                role_hint=role_hint,
                response=response,
                error=exc,
                provider_metadata=structured_output,
                schema_formatter_enabled=schema_formatter_enabled,
                provider=provider,
                model=model,
                model_tier=model_tier,
                env=env,
                process_chain=process_chain,
                artifact_dir=artifact_dir,
                call_label=call_label,
                idle_timeout_seconds=idle_timeout_seconds,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                idempotency_key=idempotency_key,
            )
            _validate_json_output(result, schema)
            return result, structured_output
    if provider_recovered and output_recovery == "warn":
        recovered = recover_json_output(
            value=result,
            schema=schema,
            raw_text=response.raw_output,
            role_hint=role_hint,
            provider_metadata=structured_output,
        )
        result = recovered.value
        structured_output = recovered.structured_output or structured_output
    return result, structured_output


def _recover_warn_schema_output(
    result: Any,
    *,
    schema: dict[str, Any] | None,
    role_hint: str | None,
    response: LLMProviderResponse[dict[str, Any]],
    error: Exception,
    provider_metadata: Any,
    schema_formatter_enabled: bool,
    provider: str,
    model: str | None,
    model_tier: str | None,
    env: Mapping[str, str] | None,
    process_chain: Sequence[str] | None,
    artifact_dir: Path | None,
    call_label: str | None,
    idle_timeout_seconds: float | None,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    cancel_check: Callable[[], bool] | None,
    idempotency_key: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    source = _schema_recovery_source_text(result, response=response, provider_metadata=provider_metadata)
    if _is_low_content_source(source):
        raise LLMRetryableProviderOutputError(
            "JSON output failed schema validation: empty or low-content output; retry original worker"
        )
    try:
        recovered = recover_json_output(
            value=result,
            schema=schema,
            raw_text=source,
            error=error,
            role_hint=role_hint,
            provider_metadata=provider_metadata if isinstance(provider_metadata, Mapping) else None,
            allow_schema_fallback=False,
        )
        if schema is None or _json_output_validates(recovered.value, schema):
            return recovered.value, recovered.structured_output or provider_metadata
    except Exception:
        pass
    if not schema_formatter_enabled:
        raise LLMOutputValidationError("JSON output failed schema validation and schema formatter is disabled")
    try:
        def formatter_runner(format_prompt: str, **formatter_kwargs: Any) -> dict[str, Any]:
            formatter_kwargs["cancel_check"] = cancel_check
            formatter_kwargs["idle_timeout_seconds"] = idle_timeout_seconds
            formatter_kwargs["progress_callback"] = progress_callback
            # A formatter is the final paid recovery step. It must never invoke
            # another formatter or cause the original worker to be replayed.
            formatter_kwargs["schema_formatter_enabled"] = False
            if artifact_dir is not None and call_label:
                formatter_kwargs["artifact_dir"] = artifact_dir
                formatter_kwargs["call_label"] = f"{call_label}/schema_formatter"
            if idempotency_key:
                formatter_kwargs["idempotency_key"] = f"{idempotency_key}/schema_formatter"
            return run_json(format_prompt, **formatter_kwargs)

        formatted = format_to_schema_or_retry(
            raw_text=source,
            schema=schema or {"type": "object"},
            role_hint=role_hint,
            json_runner=formatter_runner,
            provider=provider,
            model=model,
            model_tier=model_tier,
            env=env,
            process_chain=list(process_chain) if process_chain is not None else None,
        )
    except (LLMWorkerCancelled, LLMWorkerTimeout, LLMSchemaError):
        raise
    except Exception as exc:
        raise LLMOutputValidationError(
            f"JSON output failed schema validation: schema formatter failed: {exc}"
        ) from exc
    if formatted.action == "retry":
        reason = getattr(formatted, "reason", "")
        raise LLMOutputValidationError(
            f"JSON output failed schema validation: schema formatter could not repair output: {reason}"
        )
    if not isinstance(formatted.value, dict):
        raise LLMOutputValidationError(
            "JSON output failed schema validation: schema formatter returned no formatted object"
        )
    return formatted.value, formatted.structured_output


def _json_output_validates(value: Any, schema: Mapping[str, Any]) -> bool:
    try:
        _validate_json_output(value, dict(schema))
        return True
    except Exception:
        return False


def _schema_recovery_source_text(
    result: Any,
    *,
    response: LLMProviderResponse[dict[str, Any]],
    provider_metadata: Any,
) -> str:
    metadata = provider_metadata if isinstance(provider_metadata, Mapping) else {}
    raw_model_output = str(response.raw_model_output or "")
    if raw_model_output.strip():
        return raw_model_output
    raw_output = response.raw_output or ""
    if metadata.get("provider_error_type") == "error_max_structured_output_retries":
        return _model_text_from_raw_output(raw_output)
    model_text = _model_text_from_raw_output(raw_output)
    if model_text.strip():
        return model_text
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return json.dumps(strip_arc_llm_call_records(result), ensure_ascii=False, sort_keys=True, default=str)
    raw_excerpt = metadata.get("raw_text_excerpt")
    if isinstance(raw_excerpt, str) and raw_excerpt.strip():
        return raw_excerpt
    return str(result or "")


def _model_text_from_raw_output(raw_output: str | None) -> str:
    raw = str(raw_output or "").strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(payload, Mapping):
        return raw
    result = payload.get("result")
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        return json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
    return ""


def _is_low_content_source(source: str) -> bool:
    return _content_token_count(source) < LOW_CONTENT_TOKEN_THRESHOLD


def _content_token_count(source: str) -> int:
    text = str(source or "").strip()
    if not text or text in {"{}", "[]"}:
        return 0
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, Mapping):
        parsed = strip_arc_llm_call_records(dict(parsed))
        text = json.dumps(parsed, ensure_ascii=False, sort_keys=True, default=str)
    return sum(1 for character in text if character.isalnum())


def _recovered_natural_language_metadata(result: Any, response: LLMProviderResponse[Any]) -> dict[str, Any]:
    existing = response.structured_output if isinstance(response.structured_output, Mapping) else None
    if isinstance(existing, Mapping):
        return dict(existing)
    raw_text = response.raw_output or (result if isinstance(result, str) else repr(result))
    return structured_metadata(
        severity="major",
        warnings=["Provider returned non-object output; accepted because schema validation is disabled for this call."],
        raw_text=str(raw_text),
        strategy="natural_language_fallback",
        provider_error_type=type(result).__name__,
    )


def _merge_structured_output_warning(
    existing: Any,
    *,
    severity: str,
    warnings: list[str],
    raw_text: str | None,
    strategy: str,
    provider_error_type: str | None,
) -> dict[str, Any]:
    if isinstance(existing, Mapping):
        merged = dict(existing)
        old_warnings = merged.get("warnings") if isinstance(merged.get("warnings"), list) else []
        merged["warnings"] = [*old_warnings, *warnings]
        if not merged.get("raw_text_excerpt") and raw_text:
            merged["raw_text_excerpt"] = str(raw_text)[:4000]
        merged.setdefault("mode", "recovered")
        merged.setdefault("severity", severity)
        merged.setdefault("recovery_strategy", strategy)
        merged.setdefault("provider_error_type", provider_error_type)
        return merged
    return structured_metadata(
        severity=severity,
        warnings=warnings,
        raw_text=raw_text,
        strategy=strategy,
        provider_error_type=provider_error_type,
    )


def _call_record(
    config: LLMConfig,
    *,
    provider_requested: str,
    model_requested: str | None,
    model_tier_requested: str | None,
    fallback_index: int,
    attempt: int,
    attempts: Sequence[dict[str, Any]],
    outcome: LLMCallOutcome | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": ARC_LLM_CALL_RECORD_SCHEMA_VERSION,
        "provider_requested": provider_requested,
        "model_requested": model_requested,
        "model_tier_requested": model_tier_requested,
        "provider_used": config.provider,
        "model_used": config.model,
        "fallback_index": fallback_index,
        "attempt": attempt,
        "host": config.host.host,
        "signals": list(config.signals),
        "attempts": [dict(item) for item in attempts],
        "session_policy": outcome.session_policy if outcome else "stateless",
        "session_key": outcome.session_key if outcome else None,
        "native_session_id": outcome.native_session_id if outcome else None,
        "call_label": outcome.call_label if outcome else None,
        "prompt_sha256": outcome.prompt_sha256 if outcome else None,
        "static_prefix_sha256": outcome.static_prefix_sha256 if outcome else None,
        "schema_sha256": outcome.schema_sha256 if outcome else None,
        "runtime_fingerprint": outcome.runtime_fingerprint if outcome else None,
        "idempotency_key": outcome.idempotency_key if outcome else None,
        "generation": outcome.generation if outcome else None,
        "prompt_bytes": outcome.prompt_bytes if outcome else None,
        "logical_receipt": outcome.logical_receipt if outcome else None,
        "usage": outcome.usage.to_json() if outcome else LLMUsage().to_json(),
        "structured_output": outcome.structured_output if outcome else None,
        "warnings": list(
            dict.fromkeys((*config.warnings, *(outcome.warnings if outcome else ())))
        ),
        "call_status": (
            "recovered"
            if outcome and isinstance(outcome.structured_output, Mapping)
            and outcome.structured_output.get("mode") == "recovered"
            else "valid"
        ),
    }


def _attempt_record(
    config: LLMConfig,
    *,
    fallback_index: int,
    attempt: int,
    status: str,
    error: Exception | None = None,
) -> dict[str, Any]:
    record = {
        "provider": config.provider,
        "model": config.model,
        "fallback_index": fallback_index,
        "attempt": attempt,
        "status": status,
        "error_type": None,
        "message": None,
    }
    if error is not None:
        record["error_type"] = type(error).__name__
        record["message"] = str(error)
    return record


def _failure_message(failures: Sequence[LLMAttemptFailure], *, max_attempts: int = MAX_ATTEMPTS_PER_PROVIDER) -> str:
    provider_count = len({failure.provider for failure in failures})
    lines = [
        f"LLM task failed after {len(failures)} attempt(s) across {provider_count} provider(s).",
        "Failures:",
    ]
    for failure in failures:
        lines.append(f"- {failure.provider} attempt {failure.attempt}/{max_attempts}: {failure.error}")
    return "\n".join(lines)


def _accepts_keyword(callable_obj: Any, name: str) -> bool:
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
