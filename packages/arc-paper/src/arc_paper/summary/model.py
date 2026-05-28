from __future__ import annotations

from typing import Mapping

from arc_llm.model import resolve_model


def resolve_summary_model(
    provider_name: str,
    explicit_model: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    return resolve_model(provider_name, explicit_model, env=env)
