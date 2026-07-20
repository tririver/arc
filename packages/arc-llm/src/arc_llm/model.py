from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

KIMI_TIER_UNMAPPED_WARNING = "kimi_code_cli.model_tier_unmapped"

PROVIDER_MODEL_TIERS = {
    "codex-cli": {
        "low": "gpt-5.6-luna",
        "medium": "gpt-5.6-luna",
        "high": "gpt-5.6-sol",
        "max": "gpt-5.6-sol",
    },
    "claude-cli": {
        "low": "haiku",
        "medium": "sonnet",
        "high": "opus",
        "max": "opus",
    },
}

PROVIDER_REASONING_EFFORT_TIERS = {
    "codex-cli": {
        "low": "medium",
        "medium": "high",
        "high": "high",
        "max": "max",
    },
    "claude-cli": {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "max": "high",
    },
}

PROVIDER_MODEL_TIER_ENV_PREFIXES = {
    "codex-cli": "ARC_LLM_CODEX",
    "claude-cli": "ARC_LLM_CLAUDE",
    "kimi-code-cli": "ARC_LLM_KIMI",
}

DEFAULT_MODEL_TIER = "medium"

DEFAULT_PROVIDER_MODELS = {
    "codex-cli": PROVIDER_MODEL_TIERS["codex-cli"][DEFAULT_MODEL_TIER],
    "claude-cli": PROVIDER_MODEL_TIERS["claude-cli"][DEFAULT_MODEL_TIER],
    "kimi-code-cli": "default_model",
}

VALID_MODEL_TIERS = frozenset({"low", "medium", "high", "max"})


class ModelTierError(ValueError):
    pass


@dataclass(frozen=True)
class ModelResolution:
    model: str | None
    warnings: tuple[str, ...] = ()


def resolve_model(
    provider_name: str,
    explicit_model: str | None = None,
    *,
    model_tier: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    return resolve_model_with_warnings(
        provider_name,
        explicit_model,
        model_tier=model_tier,
        env=env,
    ).model


def resolve_model_with_warnings(
    provider_name: str,
    explicit_model: str | None = None,
    *,
    model_tier: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ModelResolution:
    if explicit_model:
        return ModelResolution(explicit_model)
    tier = resolve_model_tier(model_tier)
    if tier:
        model = _model_for_tier(provider_name, tier, env=env)
        warnings = ()
        if provider_name == "kimi-code-cli" and _model_tier_env_override(provider_name, tier, env=env) is None:
            warnings = (KIMI_TIER_UNMAPPED_WARNING,)
        return ModelResolution(model, warnings)
    return ModelResolution(DEFAULT_PROVIDER_MODELS.get(provider_name))


def resolve_model_tier(explicit_tier: str | None) -> str:
    tier = explicit_tier
    if tier is None or not str(tier).strip():
        return DEFAULT_MODEL_TIER
    normalized = str(tier).strip().lower()
    if normalized not in VALID_MODEL_TIERS:
        raise ModelTierError("model_tier must be one of: low, medium, high, max")
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
