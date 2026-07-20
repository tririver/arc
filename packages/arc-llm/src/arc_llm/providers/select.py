from __future__ import annotations

from typing import Mapping, Sequence

from ..host import select_llm_provider
from .registry import create_provider


def select_provider(
    provider: str = "auto",
    *,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
):
    name = provider
    if provider == "auto":
        name = select_llm_provider(env=env, process_chain=process_chain).provider
    return create_provider(name, env=env)
