from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .host import HostDetection, select_llm_provider
from .model import resolve_model
from .providers.config import ProviderConfigError, usable_configured_providers
from .providers.select import select_provider


MAX_ATTEMPTS_PER_PROVIDER = 3


class LLMTaskError(RuntimeError):
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
    primary = resolve_llm_config(
        provider=provider,
        model=model,
        model_tier=model_tier,
        env=env,
        process_chain=process_chain,
    )
    if provider != "auto":
        return [primary]

    configs = [primary]
    seen = {primary.provider}
    try:
        configured = usable_configured_providers(env=env)
    except ProviderConfigError:
        configured = []
    for candidate in configured:
        if candidate.id in seen:
            continue
        seen.add(candidate.id)
        configs.append(
            LLMConfig(
                provider=candidate.id,
                model=resolve_model(candidate.id, model, model_tier=model_tier, env=env),
                host=primary.host,
                signals=[*primary.signals, f"provider-fallback:{candidate.id}"],
            )
        )
    return configs


def run_json(
    prompt: str,
    *,
    schema: dict[str, Any] | None = None,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
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
        env=env,
        process_chain=process_chain,
        call=lambda selected, config: selected.generate_json(prompt, schema=schema, model=config.model),
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
        env=env,
        process_chain=process_chain,
        call=lambda selected, config: selected.generate_text(prompt, model=config.model),
    )


def _run_with_retries(
    configs: Sequence[LLMConfig],
    *,
    env: Mapping[str, str] | None,
    process_chain: Sequence[str] | None,
    call: Callable[[Any, LLMConfig], Any],
) -> Any:
    failures: list[LLMAttemptFailure] = []
    for config in configs:
        selected = select_provider(config.provider, env=env, process_chain=process_chain)
        for attempt in range(1, MAX_ATTEMPTS_PER_PROVIDER + 1):
            try:
                return call(selected, config)
            except Exception as exc:
                failures.append(LLMAttemptFailure(provider=config.provider, attempt=attempt, error=str(exc)))
    raise LLMTaskError(_failure_message(failures))


def _failure_message(failures: Sequence[LLMAttemptFailure]) -> str:
    provider_count = len({failure.provider for failure in failures})
    lines = [
        f"LLM task failed after {len(failures)} attempt(s) across {provider_count} provider(s).",
        "Failures:",
    ]
    for failure in failures:
        lines.append(f"- {failure.provider} attempt {failure.attempt}/{MAX_ATTEMPTS_PER_PROVIDER}: {failure.error}")
    return "\n".join(lines)
