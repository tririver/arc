from __future__ import annotations

import os
from typing import Mapping

from .providers.config import ProviderConfigError, configured_provider


PROVIDER_MODEL_ENV = {
    "codex-cli": "ARC_CODEX_MODEL",
    "claude-cli": "ARC_CLAUDE_MODEL",
}

PROVIDER_MODEL_TIER_ENV = {
    "codex-cli": "ARC_CODEX_MODEL_TIER",
    "claude-cli": "ARC_CLAUDE_MODEL_TIER",
}

DEFAULT_PROVIDER_MODELS = {
    "codex-cli": "gpt-5.4-mini",
    "claude-cli": "haiku",
}

PROVIDER_MODEL_TIERS = {
    "codex-cli": {
        "low": "gpt-5.3-codex-spark",
        "medium": "gpt-5.4",
        "high": "gpt-5.5",
    },
    "claude-cli": {
        "low": "haiku",
        "medium": "sonnet",
        "high": "opus",
    },
}

VALID_MODEL_TIERS = frozenset({"low", "medium", "high"})


class ModelTierError(ValueError):
    pass


def resolve_model(
    provider_name: str,
    explicit_model: str | None = None,
    *,
    model_tier: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    if explicit_model:
        return explicit_model
    env = env if env is not None else os.environ
    if provider_env := PROVIDER_MODEL_ENV.get(provider_name):
        if model := env.get(provider_env):
            return model
    if model := env.get("ARC_LLM_MODEL"):
        return model
    tier = _resolve_tier(provider_name, model_tier, env=env)
    if configured := _configured_provider(provider_name, env=env):
        if tier:
            if model := configured.model_for_tier(tier):
                return model
        if model := configured.default_model():
            return model
    if tier:
        return _model_for_tier(provider_name, tier)
    return DEFAULT_PROVIDER_MODELS.get(provider_name)


def _configured_provider(provider_name: str, *, env: Mapping[str, str]) :
    try:
        return configured_provider(provider_name, env=env)
    except ProviderConfigError:
        return None


def _resolve_tier(provider_name: str, explicit_tier: str | None, *, env: Mapping[str, str]) -> str | None:
    tier = explicit_tier
    if tier is None:
        if provider_env := PROVIDER_MODEL_TIER_ENV.get(provider_name):
            tier = env.get(provider_env)
    if tier is None:
        tier = env.get("ARC_LLM_MODEL_TIER")
    if tier is None or not str(tier).strip():
        return None
    normalized = str(tier).strip().lower()
    if normalized not in VALID_MODEL_TIERS:
        raise ModelTierError("model_tier must be one of: high, medium, low")
    return normalized


def _model_for_tier(provider_name: str, tier: str) -> str | None:
    models = PROVIDER_MODEL_TIERS.get(provider_name)
    if not models:
        return DEFAULT_PROVIDER_MODELS.get(provider_name)
    return models[tier]
