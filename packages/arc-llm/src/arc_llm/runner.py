from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from jsonschema.exceptions import SchemaError as JsonSchemaError

from .call_record import ARC_LLM_CALL_RECORD_SCHEMA_VERSION, attach_arc_llm_call_record
from .host import HostDetection, select_llm_provider
from .model import resolve_model
from .providers.select import select_provider


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
) -> dict[str, Any]:
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
        call=lambda selected, config: _generate_json(
            selected,
            prompt,
            schema=schema,
            model=config.model,
            validate_schema=validate_schema,
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
) -> str:
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
        call=lambda selected, config: selected.generate_text(prompt, model=config.model),
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
    call: Callable[[Any, LLMConfig], Any],
) -> Any:
    failures: list[LLMAttemptFailure] = []
    attempt_records: list[dict[str, Any]] = []
    for fallback_index, config in enumerate(configs):
        selected = select_provider(config.provider, env=env, process_chain=process_chain)
        for attempt in range(1, MAX_ATTEMPTS_PER_PROVIDER + 1):
            try:
                result = call(selected, config)
                attempt_record = _attempt_record(
                    config,
                    fallback_index=fallback_index,
                    attempt=attempt,
                    status="success",
                )
                attempt_records.append(attempt_record)
                if attach_call_record and isinstance(result, dict):
                    return attach_arc_llm_call_record(
                        result,
                        _call_record(
                            config,
                            provider_requested=provider_requested,
                            model_requested=model_requested,
                            model_tier_requested=model_tier_requested,
                            fallback_index=fallback_index,
                            attempt=attempt,
                            attempts=attempt_records,
                        ),
                    )
                return result
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
                if _has_remaining_attempt(configs, fallback_index=fallback_index, attempt=attempt):
                    time.sleep(RETRY_INTERVAL_SECONDS)
    raise LLMTaskError(_failure_message(failures))


def _has_remaining_attempt(
    configs: Sequence[LLMConfig],
    *,
    fallback_index: int,
    attempt: int,
) -> bool:
    return attempt < MAX_ATTEMPTS_PER_PROVIDER or fallback_index < len(configs) - 1


def _generate_json(
    selected: Any,
    prompt: str,
    *,
    schema: dict[str, Any] | None,
    model: str | None,
    validate_schema: bool,
) -> dict[str, Any]:
    result = selected.generate_json(prompt, schema=schema, model=model)
    if schema is not None and validate_schema:
        _validate_json_output(result, schema)
    return result


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


def _failure_message(failures: Sequence[LLMAttemptFailure]) -> str:
    provider_count = len({failure.provider for failure in failures})
    lines = [
        f"LLM task failed after {len(failures)} attempt(s) across {provider_count} provider(s).",
        "Failures:",
    ]
    for failure in failures:
        lines.append(f"- {failure.provider} attempt {failure.attempt}/{MAX_ATTEMPTS_PER_PROVIDER}: {failure.error}")
    return "\n".join(lines)
