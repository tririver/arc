from __future__ import annotations

from typing import Mapping, Sequence

from ..host import select_llm_provider
from .claude_cli import ClaudeCliProvider
from .codex_cli import CodexCliProvider
from .config import configured_provider, select_default_configured_provider
from .manual import ManualProvider
from .openai_compatible import OpenAICompatibleProvider


def select_provider(
    provider: str = "auto",
    *,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
):
    name = provider
    if provider == "auto":
        name = select_llm_provider(env=env, process_chain=process_chain).provider
    if name == "openai-compatible":
        configured = select_default_configured_provider(env=env)
        if configured:
            return OpenAICompatibleProvider(configured, env=env)
    if configured := configured_provider(name, env=env):
        return OpenAICompatibleProvider(configured, env=env)
    if name == "codex-cli":
        return CodexCliProvider(env=env)
    if name == "claude-cli":
        return ClaudeCliProvider(env=env)
    if name == "manual":
        return ManualProvider()
    raise ValueError(f"Unknown LLM provider: {name}")
