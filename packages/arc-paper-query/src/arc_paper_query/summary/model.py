from __future__ import annotations

import os
from typing import Mapping


PROVIDER_MODEL_ENV = {
    "codex-cli": "ARC_CODEX_MODEL",
    "claude-cli": "ARC_CLAUDE_MODEL",
}

DEFAULT_PROVIDER_MODELS = {
    "codex-cli": "gpt-5.4-mini",
    "claude-cli": "haiku",
}


def resolve_summary_model(
    provider_name: str,
    explicit_model: str | None = None,
    *,
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
    return DEFAULT_PROVIDER_MODELS.get(provider_name)
