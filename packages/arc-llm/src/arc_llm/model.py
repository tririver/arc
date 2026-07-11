from __future__ import annotations

import os
from typing import Mapping

PROVIDER_MODEL_TIERS = {
    "codex-cli": {
        "low": "gpt-5.6-luna",
        "medium": "gpt-5.6-luna",
        "high": "gpt-5.6-sol",
        "xhigh": "gpt-5.6-sol",
    },
    "claude-cli": {
        "low": "haiku",
        "medium": "sonnet",
        "high": "opus",
        "xhigh": "opus",
    },
}

PROVIDER_REASONING_EFFORT_TIERS = {
    "codex-cli": {
        "low": "medium",
        "medium": "xhigh",
        "high": "high",
        "xhigh": "max",
    },
    "claude-cli": {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
    },
}

PROVIDER_MODEL_TIER_ENV_PREFIXES = {
    "codex-cli": "ARC_LLM_CODEX",
    "claude-cli": "ARC_LLM_CLAUDE",
}

DEFAULT_MODEL_TIER = "medium"

DEFAULT_PROVIDER_MODELS = {
    "codex-cli": PROVIDER_MODEL_TIERS["codex-cli"][DEFAULT_MODEL_TIER],
    "claude-cli": PROVIDER_MODEL_TIERS["claude-cli"][DEFAULT_MODEL_TIER],
}

VALID_MODEL_TIERS = frozenset({"low", "medium", "high", "xhigh"})


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
    tier = resolve_model_tier(model_tier)
    if tier:
        return _model_for_tier(provider_name, tier, env=env)
    return DEFAULT_PROVIDER_MODELS.get(provider_name)


def resolve_model_tier(explicit_tier: str | None) -> str:
    tier = explicit_tier
    if tier is None or not str(tier).strip():
        return DEFAULT_MODEL_TIER
    normalized = str(tier).strip().lower()
    if normalized not in VALID_MODEL_TIERS:
        raise ModelTierError("model_tier must be one of: low, medium, high, xhigh")
    return normalized


def reasoning_effort_for_model_tier(provider_name: str, model_tier: str | None) -> str | None:
    tier = resolve_model_tier(model_tier)
    return PROVIDER_REASONING_EFFORT_TIERS.get(provider_name, {}).get(tier)


def _model_for_tier(provider_name: str, tier: str, *, env: Mapping[str, str] | None = None) -> str | None:
    override = _model_tier_env_override(provider_name, tier, env=env)
    if override:
        return override
    models = PROVIDER_MODEL_TIERS.get(provider_name)
    if not models:
        return DEFAULT_PROVIDER_MODELS.get(provider_name)
    return models[tier]


def _model_tier_env_override(provider_name: str, tier: str, *, env: Mapping[str, str] | None) -> str | None:
    env = os.environ if env is None else env
    prefix = PROVIDER_MODEL_TIER_ENV_PREFIXES.get(provider_name)
    if not prefix:
        return None
    value = env.get(f"{prefix}_{tier.upper()}_MODEL")
    if value is None or not value.strip():
        return None
    return value.strip()
