from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .host import detect_host, select_llm_provider
from .proposers_reviewer.consensus import run_proposers_reviewer_consensus
from .proposers_reviewer.runner import run_proposers_reviewer_batch
from .proposers_reviewer_bench.runner import run_proposers_reviewer_bench
from .providers.config import PROVIDER_CONFIG_SCHEMA, load_provider_config, parse_provider_config, provider_config_path
from .runner import resolve_llm_config, run_json, run_text


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = _dispatch(args)
    if isinstance(result, str):
        print(result, end="" if result.endswith("\n") else "\n")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reusable ARC host LLM worker")
    sub = parser.add_subparsers(dest="command", required=True)

    run_json_parser = sub.add_parser("run-json")
    run_json_parser.add_argument("--prompt", default="-")
    run_json_parser.add_argument("--schema", default=None)
    run_json_parser.add_argument("--provider", default="auto")
    run_json_parser.add_argument("--model", default=None)
    run_json_parser.add_argument("--model-tier", choices=["high", "medium", "low"], default=None)
    run_json_parser.add_argument("--json", action="store_true")
    _shared_runtime_args(run_json_parser)
    _llm_runtime_args(run_json_parser)

    run_text_parser = sub.add_parser("run-text")
    run_text_parser.add_argument("--prompt", default="-")
    run_text_parser.add_argument("--provider", default="auto")
    run_text_parser.add_argument("--model", default=None)
    run_text_parser.add_argument("--model-tier", choices=["high", "medium", "low"], default=None)
    _shared_runtime_args(run_text_parser)
    _llm_runtime_args(run_text_parser)

    loop_parser = sub.add_parser("proposers-reviewer-loop")
    loop_parser.add_argument("--config", required=True)
    loop_parser.add_argument("--json", action="store_true")
    loop_parser.add_argument("--dry-run", action="store_true")
    loop_parser.add_argument("--max-concurrent-loops", type=int, default=None)
    loop_parser.add_argument("--provider-config", default=None)

    bench_parser = sub.add_parser("proposers-reviewer-bench")
    bench_parser.add_argument("--config", required=True)
    bench_parser.add_argument("--json", action="store_true")
    bench_parser.add_argument("--dry-run", action="store_true")
    bench_parser.add_argument("--provider-config", default=None)

    consensus_parser = sub.add_parser("proposers-reviewer-consensus")
    consensus_parser.add_argument("--config", required=True)
    consensus_parser.add_argument("--json", action="store_true")
    consensus_parser.add_argument("--dry-run", action="store_true")
    consensus_parser.add_argument("--provider-config", default=None)

    doctor = sub.add_parser("doctor")
    doctor_sub = doctor.add_subparsers(dest="doctor_command", required=True)
    doctor_sub.add_parser("host")
    doctor_sub.add_parser("provider")
    doctor_sub.add_parser("config")

    providers = sub.add_parser("providers")
    providers_sub = providers.add_subparsers(dest="providers_command", required=True)
    providers_list = providers_sub.add_parser("list")
    providers_list.add_argument("--provider-config", default=None)
    providers_doctor = providers_sub.add_parser("doctor")
    providers_doctor.add_argument("--provider-config", default=None)
    providers_init = providers_sub.add_parser("init")
    providers_init.add_argument("--provider-config", default=None)
    providers_init.add_argument("--force", action="store_true")
    providers_add = providers_sub.add_parser("add")
    providers_add_sub = providers_add.add_subparsers(dest="provider_type", required=True)
    openai_add = providers_add_sub.add_parser("openai-compatible")
    openai_add.add_argument("--provider-config", default=None)
    openai_add.add_argument("--id", required=True)
    openai_add.add_argument("--base-url", required=True)
    openai_add.add_argument("--api-key", default=None)
    openai_add.add_argument("--api-key-optional", action="store_true")
    openai_add.add_argument("--low-model", default=None)
    openai_add.add_argument("--medium-model", default=None)
    openai_add.add_argument("--high-model", default=None)
    openai_add.add_argument("--json-mode", choices=["json_schema", "json_object", "none"], default="json_schema")
    return parser


def _dispatch(args: argparse.Namespace) -> Any:
    if args.command == "doctor":
        if args.doctor_command == "host":
            return detect_host().__dict__
        if args.doctor_command == "provider":
            selected = select_llm_provider()
            return {
                "provider": selected.provider,
                "host": selected.host.host,
                "signals": selected.signals,
            }
        config = resolve_llm_config()
        return {
            "provider": config.provider,
            "model": config.model,
            "host": config.host.host,
            "signals": config.signals,
        }
    if args.command == "run-json":
        return run_json(
            _read_prompt(args.prompt),
            schema=_read_schema(args.schema),
            provider=args.provider,
            model=args.model,
            model_tier=args.model_tier,
            env=_runtime_env(args),
        )
    if args.command == "run-text":
        return run_text(
            _read_prompt(args.prompt),
            provider=args.provider,
            model=args.model,
            model_tier=args.model_tier,
            env=_runtime_env(args),
        )
    if args.command == "proposers-reviewer-loop":
        return run_proposers_reviewer_batch(
            _read_json_file(args.config),
            base_env=_provider_config_env(args) if args.provider_config else None,
            dry_run=args.dry_run,
            max_concurrent_loops=args.max_concurrent_loops,
        )
    if args.command == "proposers-reviewer-bench":
        return run_proposers_reviewer_bench(
            _read_json_file(args.config),
            base_env=_provider_config_env(args) if args.provider_config else None,
            dry_run=args.dry_run,
        )
    if args.command == "proposers-reviewer-consensus":
        return run_proposers_reviewer_consensus(
            _read_json_file(args.config),
            base_env=_provider_config_env(args) if args.provider_config else None,
            dry_run=args.dry_run,
        )
    if args.command == "providers":
        if args.providers_command == "list":
            return _providers_list(args)
        if args.providers_command == "doctor":
            payload = _providers_list(args)
            selected = select_llm_provider(env=_provider_config_env(args))
            payload["auto_selection"] = {
                "provider": selected.provider,
                "host": selected.host.host,
                "signals": selected.signals,
            }
            return payload
        if args.providers_command == "init":
            return _providers_init(args)
        if args.providers_command == "add":
            return _providers_add(args)
    raise AssertionError(f"Unhandled command: {args.command}")


def _shared_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider-config", default=None)
    parser.add_argument("--allow-internet", action="store_true")
    parser.add_argument("--allow-mcp", action="store_true")
    parser.add_argument("--mcp-mode", choices=["user-config", "arc-only"], default=None)


def _llm_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--codex-sandbox", default=None)
    parser.add_argument("--codex-profile", default=None)
    parser.add_argument("--codex-profile-v2", default=None)
    parser.add_argument("--codex-work-dir", default=None)
    parser.add_argument("--codex-add-dir", action="append", default=[])
    parser.add_argument("--arc-mcp-command", default=None)
    parser.add_argument("--arc-mcp-env", action="append", default=[])
    parser.add_argument("--codex-config", action="append", default=[])
    parser.add_argument("--codex-reasoning-effort", default=None)
    parser.add_argument("--codex-reasoning-summary", default=None)
    parser.add_argument("--codex-model-verbosity", default=None)
    parser.add_argument("--codex-web-search", default=None)
    parser.add_argument("--codex-network-access", choices=["true", "false"], default=None)
    parser.add_argument("--no-codex-ephemeral", action="store_true")
    parser.add_argument("--codex-use-user-config", action="store_true")
    parser.add_argument("--codex-ignore-user-config", action="store_true")
    parser.add_argument("--codex-use-rules", action="store_true")
    parser.add_argument("--codex-ignore-rules", action="store_true")
    parser.add_argument("--claude-effort", default=None)
    parser.add_argument("--claude-tools", default=None)
    parser.add_argument("--claude-mcp-config", action="append", default=[])
    parser.add_argument("--claude-strict-mcp-config", action="store_true")
    parser.add_argument("--no-claude-strict-mcp-config", action="store_true")
    parser.add_argument("--no-claude-bare", action="store_true")
    parser.add_argument("--no-claude-session-persistence", action="store_true")
    parser.add_argument("--no-claude-exclude-dynamic-system-prompt-sections", action="store_true")
    parser.add_argument("--claude-max-budget-usd", default=None)
    parser.add_argument("--claude-fallback-model", default=None)


def _runtime_env(args: argparse.Namespace) -> dict[str, str] | None:
    overrides: dict[str, str] = {}
    _put(overrides, "ARC_LLM_PROVIDER_CONFIG", getattr(args, "provider_config", None))
    if args.allow_internet:
        overrides["ARC_CODEX_ALLOW_INTERNET"] = "true"
        overrides["ARC_CLAUDE_ALLOW_INTERNET"] = "true"
    if args.allow_mcp:
        overrides["ARC_CODEX_ENABLE_MCP"] = "true"
        overrides["ARC_CLAUDE_ALLOW_MCP"] = "true"
    _put(overrides, "ARC_CODEX_MCP_MODE", args.mcp_mode)
    _put(overrides, "ARC_CLAUDE_MCP_MODE", args.mcp_mode)
    _put(overrides, "ARC_CODEX_SANDBOX", args.codex_sandbox)
    _put(overrides, "ARC_CODEX_PROFILE", args.codex_profile)
    _put(overrides, "ARC_CODEX_PROFILE_V2", args.codex_profile_v2)
    _put(overrides, "ARC_CODEX_WORK_DIR", args.codex_work_dir)
    if args.codex_add_dir:
        overrides["ARC_CODEX_ADD_DIRS"] = json.dumps(args.codex_add_dir, ensure_ascii=False)
    _put(overrides, "ARC_CODEX_ARC_MCP_COMMAND", args.arc_mcp_command)
    if args.arc_mcp_env:
        overrides["ARC_CODEX_ARC_MCP_ENV_JSON"] = json.dumps(_parse_key_value_items(args.arc_mcp_env), ensure_ascii=False)
    if args.codex_config:
        overrides["ARC_CODEX_CONFIG"] = "\n".join(args.codex_config)
    _put(overrides, "ARC_CODEX_REASONING_EFFORT", args.codex_reasoning_effort)
    _put(overrides, "ARC_CODEX_REASONING_SUMMARY", args.codex_reasoning_summary)
    _put(overrides, "ARC_CODEX_MODEL_VERBOSITY", args.codex_model_verbosity)
    _put(overrides, "ARC_CODEX_WEB_SEARCH", args.codex_web_search)
    _put(overrides, "ARC_CODEX_NETWORK_ACCESS", args.codex_network_access)
    if args.no_codex_ephemeral:
        overrides["ARC_CODEX_EPHEMERAL"] = "false"
    if args.codex_use_user_config:
        overrides["ARC_CODEX_IGNORE_USER_CONFIG"] = "false"
    if args.codex_ignore_user_config:
        overrides["ARC_CODEX_IGNORE_USER_CONFIG"] = "true"
    if args.codex_use_rules:
        overrides["ARC_CODEX_IGNORE_RULES"] = "false"
    if args.codex_ignore_rules:
        overrides["ARC_CODEX_IGNORE_RULES"] = "true"
    _put(overrides, "ARC_CLAUDE_EFFORT", args.claude_effort)
    _put(overrides, "ARC_CLAUDE_TOOLS", args.claude_tools)
    if args.claude_mcp_config:
        overrides["ARC_CLAUDE_MCP_CONFIG"] = "\n".join(args.claude_mcp_config)
    if args.claude_strict_mcp_config:
        overrides["ARC_CLAUDE_STRICT_MCP_CONFIG"] = "true"
    if args.no_claude_strict_mcp_config:
        overrides["ARC_CLAUDE_STRICT_MCP_CONFIG"] = "false"
    if args.no_claude_bare:
        overrides["ARC_CLAUDE_BARE"] = "false"
    if args.no_claude_session_persistence:
        overrides["ARC_CLAUDE_NO_SESSION_PERSISTENCE"] = "false"
    if args.no_claude_exclude_dynamic_system_prompt_sections:
        overrides["ARC_CLAUDE_EXCLUDE_DYNAMIC_SYSTEM_PROMPT_SECTIONS"] = "false"
    _put(overrides, "ARC_CLAUDE_MAX_BUDGET_USD", args.claude_max_budget_usd)
    _put(overrides, "ARC_CLAUDE_FALLBACK_MODEL", args.claude_fallback_model)
    if not overrides:
        return None
    env = dict(os.environ)
    env.update(overrides)
    return env


def _put(env: dict[str, str], key: str, value: str | None) -> None:
    if value is not None:
        env[key] = value


def _parse_key_value_items(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Expected KEY=VALUE for --arc-mcp-env, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit("Expected non-empty KEY for --arc-mcp-env")
        parsed[key] = value
    return parsed


def _read_prompt(value: str) -> str:
    if value == "-":
        return sys.stdin.read()
    return Path(value).read_text(encoding="utf-8")


def _read_schema(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _read_json_file(value: str) -> dict[str, Any]:
    payload = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {value}")
    return payload


def _provider_config_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    if provider_config := getattr(args, "provider_config", None):
        env["ARC_LLM_PROVIDER_CONFIG"] = provider_config
    return env


def _providers_list(args: argparse.Namespace) -> dict[str, Any]:
    env = _provider_config_env(args)
    config = load_provider_config(env=env)
    configured = []
    for provider in config.providers:
        configured.append(
            {
                "id": provider.id,
                "type": provider.type,
                "base_url": provider.base_url,
                "api_key_optional": provider.api_key_optional,
                "has_api_key": provider.has_api_key(env=env),
                "usable": provider.is_usable(env=env),
                "models": provider.models or {},
                "json_mode": provider.json_mode,
            }
        )
    return {
        "schema_version": "arc.llm.providers.list.v1",
        "config_path": config.path,
        "selection_policy": "provider/model_tier selected per run; provider=auto uses host-native provider, then usable configured providers",
        "builtins": ["codex-cli", "claude-cli", "manual"],
        "configured": configured,
    }


def _providers_init(args: argparse.Namespace) -> dict[str, Any]:
    env = _provider_config_env(args)
    path = provider_config_path(env=env)
    if path.exists() and not args.force:
        return {"status": "exists", "config_path": str(path)}
    payload = {
        "_comment": _provider_config_comment(path),
        "schema_version": PROVIDER_CONFIG_SCHEMA,
        "providers": [
            {
                "id": "deepseek",
                "type": "openai-compatible",
                "base_url": "https://api.deepseek.com/v1",
                "api_key": "replace-with-your-deepseek-api-key",
                "models": {"medium": "deepseek-chat", "high": "deepseek-reasoner"},
                "json_mode": "json_object",
            },
            {
                "id": "ollama",
                "type": "openai-compatible",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key_optional": True,
                "models": {"medium": "llama3.1"},
                "json_mode": "json_object",
            },
        ],
    }
    _write_provider_config(path, payload)
    return {"status": "written", "config_path": str(path)}


def _providers_add(args: argparse.Namespace) -> dict[str, Any]:
    env = _provider_config_env(args)
    path = provider_config_path(env=env)
    payload = _read_provider_config_payload(path)
    providers = payload.setdefault("providers", [])
    if not isinstance(providers, list):
        raise ValueError("providers must be a list")
    provider = {
        "id": args.id,
        "type": "openai-compatible",
        "base_url": args.base_url,
        "json_mode": args.json_mode,
    }
    if args.api_key:
        provider["api_key"] = args.api_key
    if args.api_key_optional:
        provider["api_key_optional"] = True
    models = {
        key: value
        for key, value in {
            "low": args.low_model,
            "medium": args.medium_model,
            "high": args.high_model,
        }.items()
        if value
    }
    if models:
        provider["models"] = models
    payload.setdefault("_comment", _provider_config_comment(path))
    replaced = False
    for index, item in enumerate(providers):
        if isinstance(item, dict) and item.get("id") == args.id:
            providers[index] = provider
            replaced = True
            break
    if not replaced:
        providers.append(provider)
    parse_provider_config(payload, path=str(path))
    _write_provider_config(path, payload)
    return {"status": "written", "config_path": str(path), "provider": args.id}


def _read_provider_config_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "_comment": _provider_config_comment(path),
            "schema_version": PROVIDER_CONFIG_SCHEMA,
            "providers": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Provider config must contain an object: {path}")
    return payload


def _write_provider_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _provider_config_comment(path: Path) -> list[str]:
    project_path = Path.cwd() / "llm-providers.json"
    return [
        "ARC LLM provider config. JSON does not support comments; this _comment field is informational.",
        "This file was generated by `arc-llm providers init`; edit providers with `arc-llm providers add openai-compatible`.",
        f"Project-local config path: {project_path}",
        "Linux user config: ~/.config/arc/llm-providers.json",
        "macOS user config: ~/.config/arc/llm-providers.json",
        "Windows user config: %USERPROFILE%\\.config\\arc\\llm-providers.json",
        f"This generated file is currently at: {path}",
        "This file may contain API keys. Keep real llm-providers.json files out of git.",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
