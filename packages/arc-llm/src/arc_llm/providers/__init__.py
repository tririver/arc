from __future__ import annotations

from .base import LLMWorkerError
from .claude_cli import ClaudeCliProvider
from .codex_cli import CodexCliProvider
from .kimi_code_cli import KimiCodeCliProvider
from .manual import ManualProvider
from .registry import PROVIDER_SPECS, ProviderSpec, create_provider, get_provider_spec, provider_diagnostic


def select_provider(*args, **kwargs):
    from .select import select_provider as _select_provider

    return _select_provider(*args, **kwargs)

__all__ = [
    "ClaudeCliProvider",
    "CodexCliProvider",
    "KimiCodeCliProvider",
    "LLMWorkerError",
    "ManualProvider",
    "PROVIDER_SPECS",
    "ProviderSpec",
    "create_provider",
    "get_provider_spec",
    "provider_diagnostic",
    "select_provider",
]
