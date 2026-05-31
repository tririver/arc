from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from jsonschema.exceptions import SchemaError as JsonSchemaError

from .call_record import ARC_LLM_CALL_RECORD_SCHEMA_VERSION, attach_arc_llm_call_record
from .host import HostDetection, select_llm_provider
from .model import resolve_model
from .providers.select import select_provider
from .schema_cache import schema_hash, sha256_text
from .sessions import LLMSessionManager, LLMSessionRef, runtime_fingerprint
from .usage import LLMProviderResponse, LLMUsage


MAX_ATTEMPTS_PER_PROVIDER = 3
RETRY_INTERVAL_SECONDS = 10


class LLMTaskError(RuntimeError):
    pass


class LLMOutputValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str | None
    host: HostDetection
    signals: list[str]


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


def resolve_llm_config(
    *,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
) -> LLMConfig:
    if provider == "auto" and model:
        raise ValueError("Exact model requires explicit provider; use provider=<provider> or model_tier=<low|medium|high>.")
    selected = select_llm_provider(
        env=env,
        process_chain=process_chain,
        explicit_provider=None if provider == "auto" else provider,
    )
    return LLMConfig(
        provider=selected.provider,
        model=resolve_model(selected.provider, model, model_tier=model_tier, env=env),
        host=selected.host,
        signals=selected.signals,
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
) -> dict[str, Any]:
    if session_policy not in {"stateless", "stateful"}:
        raise ValueError("session_policy must be stateless or stateful")
    if session_policy == "stateful" and (session_manager is None or not session_key):
        raise ValueError("stateful run_json requires session_manager and session_key")
    configs = resolve_llm_configs(
        provider=provider,
        model=model,
        model_tier=model_tier,
        env=env,
        process_chain=process_chain,
    )
    return _run_with_retries(
        configs,
        provider_requested=provider,
        model_requested=model,
        model_tier_requested=model_tier,
        attach_call_record=True,
        env=env,
        process_chain=process_chain,
        max_attempts=1 if session_policy == "stateful" else MAX_ATTEMPTS_PER_PROVIDER,
        call=lambda selected, config: _generate_json(
            selected,
            prompt,
            schema=schema,
            model=config.model,
            validate_schema=validate_schema,
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
) -> LLMCallOutcome:
    if session_policy not in {"stateless", "stateful"}:
        raise ValueError("session_policy must be stateless or stateful")
    if session_policy == "stateful" and (session_manager is None or not session_key):
        raise ValueError("stateful run_text requires session_manager and session_key")
    configs = resolve_llm_configs(
        provider=provider,
        model=model,
        model_tier=model_tier,
        env=env,
        process_chain=process_chain,
    )
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
        call=lambda selected, config: _generate_text(
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
        ),
    )


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
    call: Callable[[Any, LLMConfig], Any],
) -> Any:
    failures: list[LLMAttemptFailure] = []
    attempt_records: list[dict[str, Any]] = []
    for fallback_index, config in enumerate(configs):
        selected = select_provider(config.provider, env=env, process_chain=process_chain)
        for attempt in range(1, max_attempts + 1):
            try:
                result = call(selected, config)
                value = result.value if isinstance(result, LLMCallOutcome) else result
                attempt_record = _attempt_record(
                    config,
                    fallback_index=fallback_index,
                    attempt=attempt,
                    status="success",
                )
                attempt_records.append(attempt_record)
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
                if _has_remaining_attempt(configs, fallback_index=fallback_index, attempt=attempt, max_attempts=max_attempts):
                    time.sleep(RETRY_INTERVAL_SECONDS)
    raise LLMTaskError(_failure_message(failures, max_attempts=max_attempts))


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

    def call_provider(session: LLMSessionRef | None) -> LLMProviderResponse[dict[str, Any]]:
        if hasattr(selected, "generate_json_result"):
            return selected.generate_json_result(
                prompt,
                schema=schema,
                model=model,
                session=session,
                session_policy=session_policy,
                schema_cache_dir=_schema_cache_dir(artifact_dir),
                artifact_dir=artifact_dir,
            )
        return LLMProviderResponse(selected.generate_json(prompt, schema=schema, model=model))

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
        ) as (session, _turn_count):
            response = call_provider(session)
            result = response.value
            prompt_sha = response.prompt_sent_sha256 or sha256_text(prompt)
            native_session_id = response.native_session_id
            if native_session_id:
                session_manager.update_native_session_id(session.key, native_session_id)
            recorded_native_session_id = native_session_id or session.native_session_id
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
                extra={"runtime_fingerprint": runtime_fp},
            )
            if schema is not None and validate_schema:
                _validate_json_output(result, schema)
    else:
        session = None
        response = call_provider(None)
        result = response.value
        if schema is not None and validate_schema:
            _validate_json_output(result, schema)
        native_session_id = response.native_session_id
        prompt_sha = response.prompt_sent_sha256 or sha256_text(prompt)
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

    def call_provider(session: LLMSessionRef | None) -> LLMProviderResponse[str]:
        if hasattr(selected, "generate_text_result"):
            return selected.generate_text_result(
                prompt,
                model=model,
                session=session,
                session_policy=session_policy,
                artifact_dir=artifact_dir,
            )
        return LLMProviderResponse(selected.generate_text(prompt, model=model))

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
        ) as (session, _turn_count):
            response = call_provider(session)
            prompt_sha = response.prompt_sent_sha256 or sha256_text(prompt)
            if response.native_session_id:
                session_manager.update_native_session_id(session.key, response.native_session_id)
            recorded_native_session_id = response.native_session_id or session.native_session_id
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
                extra={"runtime_fingerprint": runtime_fp},
            )
    else:
        session = None
        response = call_provider(None)
        prompt_sha = response.prompt_sent_sha256 or sha256_text(prompt)
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


def _schema_cache_dir(artifact_dir: Path | None) -> Path | None:
    if artifact_dir is None:
        return None
    return artifact_dir / "schemas"


def _validate_json_output(result: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        validate_json_schema(instance=result, schema=schema)
    except JsonSchemaValidationError as exc:
        raise LLMOutputValidationError(f"JSON output failed schema validation: {exc.message}") from exc
    except JsonSchemaError as exc:
        raise LLMOutputValidationError(f"JSON schema is invalid: {exc.message}") from exc


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
        "usage": outcome.usage.to_json() if outcome else LLMUsage().to_json(),
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
