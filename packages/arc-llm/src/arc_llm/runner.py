from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .host import HostDetection, select_llm_provider
from .model import resolve_model
from .providers.select import select_provider


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str | None
    host: HostDetection
    signals: list[str]


def resolve_llm_config(
    *,
    provider: str = "auto",
    model: str | None = None,
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
        model=resolve_model(selected.provider, model, env=env),
        host=selected.host,
        signals=selected.signals,
    )


def run_json(
    prompt: str,
    *,
    schema: dict[str, Any] | None = None,
    provider: str = "auto",
    model: str | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
) -> dict[str, Any]:
    config = resolve_llm_config(provider=provider, model=model, env=env, process_chain=process_chain)
    runner = select_provider(config.provider, env=env, process_chain=process_chain)
    return runner.generate_json(prompt, schema=schema, model=config.model)


def run_text(
    prompt: str,
    *,
    provider: str = "auto",
    model: str | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
) -> str:
    config = resolve_llm_config(provider=provider, model=model, env=env, process_chain=process_chain)
    runner = select_provider(config.provider, env=env, process_chain=process_chain)
    return runner.generate_text(prompt, model=config.model)
