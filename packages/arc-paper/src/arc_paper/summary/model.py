from __future__ import annotations

from typing import Mapping

from arc_llm.model import resolve_model

DEFAULT_SUMMARY_MODEL_TIER = "low"


def resolve_summary_model(
    provider_name: str,
    explicit_model: str | None = None,
    *,
    model_tier: str | None = DEFAULT_SUMMARY_MODEL_TIER,
    env: Mapping[str, str] | None = None,
) -> str | None:
    return resolve_model(provider_name, explicit_model, model_tier=model_tier, env=env)
