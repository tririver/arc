from __future__ import annotations

from typing import Mapping, Sequence

from ...host import select_llm_provider
from arc_llm.providers.select import select_provider as select_prompt_provider
from .claude_cli import ClaudeCliProvider
from .codex_cli import CodexCliProvider
from .manual import ManualProvider
from .prompt import PromptProviderSummaryAdapter


def select_summary_provider(
    provider: str = "auto",
    *,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
):
    name = provider
    if provider == "auto":
        name = select_llm_provider(env=env, process_chain=process_chain).provider
    if name == "codex-cli":
        return CodexCliProvider()
    if name == "claude-cli":
        return ClaudeCliProvider()
    if name == "manual":
        return ManualProvider()
    try:
        return PromptProviderSummaryAdapter(select_prompt_provider(name, env=env, process_chain=process_chain), env=env)
    except ValueError as exc:
        raise ValueError(f"Unknown summary provider: {name}") from exc
