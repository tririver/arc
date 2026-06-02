from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .cache_audit import audit_run
from .host import detect_host, select_llm_provider
from .proposers_reviewer.template_materializer import materialize_batch
from .proposers_reviewer.runner import run_proposers_reviewer_batch
from .proposers_reviewer_bench.runner import run_proposers_reviewer_bench
from .runner import resolve_llm_config, run_json, run_text
from .sessions import LLMSessionManager


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
    _session_args(run_json_parser)
    _shared_runtime_args(run_json_parser)
    _llm_runtime_args(run_json_parser)

    run_text_parser = sub.add_parser("run-text")
    run_text_parser.add_argument("--prompt", default="-")
    run_text_parser.add_argument("--provider", default="auto")
    run_text_parser.add_argument("--model", default=None)
    run_text_parser.add_argument("--model-tier", choices=["high", "medium", "low"], default=None)
    _session_args(run_text_parser)
    _shared_runtime_args(run_text_parser)
    _llm_runtime_args(run_text_parser)

    loop_parser = sub.add_parser("proposers-reviewer-loop")
    loop_parser.add_argument("--config", required=True)
    loop_parser.add_argument("--json", action="store_true")
    loop_parser.add_argument("--dry-run", action="store_true")
    loop_parser.add_argument("--max-concurrent-loops", type=int, default=None)
    loop_parser.add_argument("--session-policy", choices=["stateful", "stateless"], default=None)
    loop_parser.add_argument("--history-mode", choices=["auto", "delta", "full"], default=None)
    loop_parser.add_argument("--session-scope-id", default=None)
    loop_parser.add_argument("--max-concurrent-same-prefix", type=int, default=None)

    bench_parser = sub.add_parser("proposers-reviewer-bench")
    bench_parser.add_argument("--config", required=True)
    bench_parser.add_argument("--json", action="store_true")
    bench_parser.add_argument("--dry-run", action="store_true")

    doctor = sub.add_parser("doctor")
    doctor_sub = doctor.add_subparsers(dest="doctor_command", required=True)
    doctor_sub.add_parser("host")
    doctor_sub.add_parser("provider")
    doctor_sub.add_parser("config")
    cache_audit = sub.add_parser("cache-audit")
    cache_audit.add_argument("run_root")
    materialize = sub.add_parser("materialize-proposers-reviewer")
    materialize.add_argument("--spec", required=True)
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
        session_manager = _session_manager(args)
        return run_json(
            _read_prompt(args.prompt),
            schema=_read_schema(args.schema),
            provider=args.provider,
            model=args.model,
            model_tier=args.model_tier,
            env=_runtime_env(args),
            session_policy=args.session_policy,
            session_manager=session_manager,
            session_key=args.session_key,
            session_name=args.session_name,
            call_label=args.call_label,
            artifact_dir=args.session_root,
        )
    if args.command == "run-text":
        session_manager = _session_manager(args)
        return run_text(
            _read_prompt(args.prompt),
            provider=args.provider,
            model=args.model,
            model_tier=args.model_tier,
            env=_runtime_env(args),
            session_policy=args.session_policy,
            session_manager=session_manager,
            session_key=args.session_key,
            session_name=args.session_name,
            call_label=args.call_label,
            artifact_dir=args.session_root,
        )
    if args.command == "proposers-reviewer-loop":
        config = _read_json_file(args.config)
        _apply_loop_session_overrides(config, args)
        return run_proposers_reviewer_batch(
            config,
            dry_run=args.dry_run,
            max_concurrent_loops=args.max_concurrent_loops,
        )
    if args.command == "proposers-reviewer-bench":
        return run_proposers_reviewer_bench(
            _read_json_file(args.config),
            dry_run=args.dry_run,
        )
    if args.command == "cache-audit":
        return audit_run(args.run_root)
    if args.command == "materialize-proposers-reviewer":
        spec = _read_json_file(args.spec)
        return materialize_batch(**spec)
    raise AssertionError(f"Unhandled command: {args.command}")


def _session_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session-policy", choices=["stateless", "stateful"], default="stateless")
    parser.add_argument("--session-root", default=None)
    parser.add_argument("--session-key", default=None)
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--call-label", default=None)


def _session_manager(args: argparse.Namespace):
    if getattr(args, "session_policy", "stateless") != "stateful":
        return None
    root = Path(args.session_root or ".arc-llm/sessions")
    return LLMSessionManager(root)


def _apply_loop_session_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    overrides = {}
    if args.session_policy is not None:
        overrides["policy"] = args.session_policy
    if args.history_mode is not None:
        overrides["history_mode"] = args.history_mode
    if args.session_scope_id is not None:
        overrides["scope_id"] = args.session_scope_id
    if args.max_concurrent_same_prefix is not None:
        overrides["max_concurrent_same_prefix"] = args.max_concurrent_same_prefix
    if overrides:
        session = dict(config.get("session") or {})
        session.update(overrides)
        config["session"] = session


def _shared_runtime_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--claude-allowed-tools", default=None)
    parser.add_argument("--claude-mcp-config", action="append", default=[])
    parser.add_argument("--claude-strict-mcp-config", action="store_true")
    parser.add_argument("--no-claude-strict-mcp-config", action="store_true")
    parser.add_argument("--no-claude-bare", action="store_true")
    parser.add_argument("--no-claude-session-persistence", action="store_true")
    parser.add_argument("--claude-session-persistence", action="store_true")
    parser.add_argument("--claude-no-session-persistence", action="store_true")
    parser.add_argument("--no-claude-exclude-dynamic-system-prompt-sections", action="store_true")
    parser.add_argument("--claude-max-budget-usd", default=None)
    parser.add_argument("--claude-fallback-model", default=None)


def _runtime_env(args: argparse.Namespace) -> dict[str, str] | None:
    overrides: dict[str, str] = {}
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
    _put(overrides, "ARC_CLAUDE_ARC_MCP_COMMAND", args.arc_mcp_command)
    if args.arc_mcp_env:
        parsed_arc_mcp_env = _parse_key_value_items(args.arc_mcp_env)
        overrides["ARC_CODEX_ARC_MCP_ENV_JSON"] = json.dumps(parsed_arc_mcp_env, ensure_ascii=False)
        overrides["ARC_CLAUDE_ARC_MCP_ENV_JSON"] = json.dumps(parsed_arc_mcp_env, ensure_ascii=False)
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
    _put(overrides, "ARC_CLAUDE_ALLOWED_TOOLS", args.claude_allowed_tools)
    if args.claude_mcp_config:
        overrides["ARC_CLAUDE_MCP_CONFIG"] = "\n".join(args.claude_mcp_config)
    if args.claude_strict_mcp_config:
        overrides["ARC_CLAUDE_STRICT_MCP_CONFIG"] = "true"
    if args.no_claude_strict_mcp_config:
        overrides["ARC_CLAUDE_STRICT_MCP_CONFIG"] = "false"
    if args.no_claude_bare:
        overrides["ARC_CLAUDE_BARE"] = "false"
    if args.no_claude_session_persistence:
        overrides["ARC_CLAUDE_NO_SESSION_PERSISTENCE"] = "true"
    if args.claude_session_persistence:
        overrides["ARC_CLAUDE_NO_SESSION_PERSISTENCE"] = "false"
    if args.claude_no_session_persistence:
        overrides["ARC_CLAUDE_NO_SESSION_PERSISTENCE"] = "true"
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


if __name__ == "__main__":
    raise SystemExit(main())
