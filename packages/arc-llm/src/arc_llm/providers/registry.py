from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from arc_llm.nested_shell_capability import (
    NESTED_SHELL_CAPABILITY_SCHEMA_VERSION,
    NESTED_SHELL_PROBE_ID,
    NestedShellCapability,
    resolve_nested_shell_capability,
)

from .claude_cli import ClaudeCliProvider
from .codex_cli import CodexCliProvider
from .manual import ManualProvider


KIMI_EXPERIMENTAL_WARNING = "kimi_code_cli.experimental"
KIMI_PERSISTENCE_WARNING = "kimi_code_cli.provider_side_persistence"
KIMI_INHERITED_CONFIG_WARNING = "kimi_code_cli.inherits_user_configuration"


ProviderFactory = Callable[[Mapping[str, str] | None], Any]


@dataclass(frozen=True)
class ProviderSpec:
    provider_id: str
    factory: ProviderFactory
    experimental: bool = False
    supports_sessions: bool = False
    supports_usage: bool = False
    supports_native_schema: bool = False
    provider_side_persistence: bool = False
    warning_codes: tuple[str, ...] = ()
    risk_warning: str | None = None

    def diagnostic_metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider_id,
            "experimental": self.experimental,
            "supports_sessions": self.supports_sessions,
            "supports_usage": self.supports_usage,
            "supports_native_schema": self.supports_native_schema,
            "provider_side_persistence": self.provider_side_persistence,
            "warning_codes": list(self.warning_codes),
            "risk_warning": self.risk_warning,
        }


def _codex_factory(env: Mapping[str, str] | None) -> CodexCliProvider:
    return CodexCliProvider(env=env)


def _claude_factory(env: Mapping[str, str] | None) -> ClaudeCliProvider:
    return ClaudeCliProvider(env=env)


def _manual_factory(env: Mapping[str, str] | None) -> ManualProvider:
    del env
    return ManualProvider()


def _kimi_factory(env: Mapping[str, str] | None) -> Any:
    # Keep the transport optional at import time so host detection and doctor
    # remain usable even when only the core provider metadata is installed.
    from .kimi_code_cli import KimiCodeCliProvider

    return KimiCodeCliProvider(env=env)


PROVIDER_SPECS: Mapping[str, ProviderSpec] = MappingProxyType(
    {
        "codex-cli": ProviderSpec(
            provider_id="codex-cli",
            factory=_codex_factory,
            supports_sessions=True,
            supports_usage=True,
            supports_native_schema=True,
        ),
        "claude-cli": ProviderSpec(
            provider_id="claude-cli",
            factory=_claude_factory,
            supports_sessions=True,
            supports_usage=True,
            supports_native_schema=True,
        ),
        "kimi-code-cli": ProviderSpec(
            provider_id="kimi-code-cli",
            factory=_kimi_factory,
            experimental=True,
            supports_sessions=True,
            supports_usage=False,
            supports_native_schema=False,
            provider_side_persistence=True,
            warning_codes=(
                KIMI_EXPERIMENTAL_WARNING,
                KIMI_PERSISTENCE_WARNING,
                KIMI_INHERITED_CONFIG_WARNING,
            ),
            risk_warning=(
                "kimi-code-cli is experimental and inherits Kimi Code configuration, instructions, skills, "
                "hooks, plugins, MCP, tool permissions, and persistent sessions; it may access the network, "
                "run commands, and modify files."
            ),
        ),
        "manual": ProviderSpec(provider_id="manual", factory=_manual_factory),
    }
)


def get_provider_spec(provider_id: str) -> ProviderSpec:
    try:
        return PROVIDER_SPECS[provider_id]
    except KeyError as exc:
        raise ValueError(f"Unknown LLM provider: {provider_id}") from exc


def create_provider(provider_id: str, *, env: Mapping[str, str] | None = None) -> Any:
    return get_provider_spec(provider_id).factory(env)


def provider_diagnostic(
    provider_id: str,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    *,
    probe_nested_shell: bool = False,
) -> dict[str, Any]:
    spec = get_provider_spec(provider_id)
    result = spec.diagnostic_metadata()
    result["risks"] = _kimi_risk_paths(env=env, cwd=cwd) if provider_id == "kimi-code-cli" else []
    capability = (
        resolve_nested_shell_capability(
            provider=provider_id, env=env, cwd=cwd
        )
        if probe_nested_shell
        else NestedShellCapability(
            schema_version=NESTED_SHELL_CAPABILITY_SCHEMA_VERSION,
            provider=provider_id,
            nested_sandboxed_shell=False,
            status="not_requested",
            probe_kind="none",
            probe_identity=NESTED_SHELL_PROBE_ID,
            warning="nested_shell.not_requested",
        )
    )
    result["nested_shell_capability"] = capability.doctor_json()
    return result


def _kimi_risk_paths(
    *,
    env: Mapping[str, str] | None,
    cwd: Path | str | None,
) -> list[dict[str, str]]:
    resolved_env = os.environ if env is None else env
    user_home = _user_home(resolved_env)
    kimi_home = _expand_path(resolved_env.get("KIMI_CODE_HOME") or ".kimi-code", base=user_home)
    work_dir = Path(cwd or os.getcwd()).expanduser().resolve(strict=False)
    project_root = _nearest_project_root(work_dir)

    candidates = (
        ("configuration", kimi_home / "config.toml"),
        ("mcp", kimi_home / "mcp.json"),
        ("hooks", kimi_home / "hooks"),
        ("plugins", kimi_home / "plugins"),
        ("instructions", kimi_home / "AGENTS.md"),
        ("skills", kimi_home / "skills"),
        ("instructions", user_home / ".agents" / "AGENTS.md"),
        ("skills", user_home / ".agents" / "skills"),
        ("configuration", project_root / ".kimi-code" / "local.toml"),
        ("mcp", project_root / ".kimi-code" / "mcp.json"),
        ("instructions", project_root / "AGENTS.md"),
        ("instructions", project_root / ".kimi-code" / "AGENTS.md"),
        ("instructions", project_root / ".agents" / "AGENTS.md"),
        ("skills", project_root / ".kimi-code" / "skills"),
        ("skills", project_root / ".agents" / "skills"),
    )
    risks: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for category, path in candidates:
        resolved_path = path.resolve(strict=False)
        key = (category, str(resolved_path))
        if key in seen or not resolved_path.exists():
            continue
        seen.add(key)
        risks.append({"category": category, "path": str(resolved_path)})
    return risks


def _user_home(env: Mapping[str, str]) -> Path:
    raw = env.get("HOME") or env.get("USERPROFILE")
    return Path(raw).expanduser().resolve(strict=False) if raw else Path.home().resolve(strict=False)


def _expand_path(value: str, *, base: Path) -> Path:
    if value == "~":
        return base
    if value.startswith("~/") or value.startswith("~\\"):
        return (base / value[2:]).resolve(strict=False)
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def _nearest_project_root(work_dir: Path) -> Path:
    for candidate in (work_dir, *work_dir.parents):
        if (candidate / ".git").exists():
            return candidate
    return work_dir
