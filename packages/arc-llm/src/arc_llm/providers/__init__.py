from __future__ import annotations

from importlib import import_module

from .base import LLMWorkerError


_LAZY_EXPORTS = {
    "ClaudeCliProvider": ("arc_llm.providers.claude_cli", "ClaudeCliProvider"),
    "CodexCliProvider": ("arc_llm.providers.codex_cli", "CodexCliProvider"),
    "KimiCodeCliProvider": ("arc_llm.providers.kimi_code_cli", "KimiCodeCliProvider"),
    "ManualProvider": ("arc_llm.providers.manual", "ManualProvider"),
    "PROVIDER_SPECS": ("arc_llm.providers.registry", "PROVIDER_SPECS"),
    "ProviderSpec": ("arc_llm.providers.registry", "ProviderSpec"),
    "create_provider": ("arc_llm.providers.registry", "create_provider"),
    "get_provider_spec": ("arc_llm.providers.registry", "get_provider_spec"),
    "provider_diagnostic": ("arc_llm.providers.registry", "provider_diagnostic"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


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
