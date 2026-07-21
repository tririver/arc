from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

from .cache_audit import audit_run
from .host import detect_host, select_llm_provider
from .paths import llm_cache_root
from .proposers_reviewer.template_materializer import materialize_batch
from .proposers_reviewer.runner import run_proposers_reviewer_batch
from .proposers_reviewer_bench.runner import run_proposers_reviewer_bench
from .providers.registry import provider_diagnostic
from .providers.kimi_safety import kimi_retry_safety_diagnostic
from .runner import LLMNeedsLLM, resolve_llm_config, run_json, run_text
from .schema_formatter import format_to_schema
from .safety import LLMSafetyController
from .sessions import LLMSessionManager


_SIGNAL_CANCEL_REQUESTED = threading.Event()
_PROGRESS_FILE_LOCK = threading.Lock()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _SIGNAL_CANCEL_REQUESTED.clear()
    previous_handlers = _install_cancel_signal_handlers()
    try:
        try:
            result = _dispatch(args)
        except Exception as exc:
            if not isinstance(exc, LLMNeedsLLM) and not getattr(args, "json", False):
                raise
            result = _exception_result(exc)
    finally:
        _restore_signal_handlers(previous_handlers)
    if isinstance(result, str):
        print(result, end="" if result.endswith("\n") else "\n")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if _result_succeeded(result) else 1


def _result_succeeded(result: Any) -> bool:
    if not isinstance(result, Mapping):
        return True
    status = str(result.get("status") or "").strip().lower()
    if status in {"done", "completed", "degraded", "stopped", "needs_llm"}:
        return True
    if status in {"cancelled", "error", "failed", "failure"}:
        return False
    if result.get("ok") is False:
        return False
    return True


def _exception_result(exc: Exception) -> dict[str, Any]:
    """Return the stable failure envelope promised by ``--json`` commands."""
    if isinstance(exc, LLMNeedsLLM):
        return {
            "ok": False,
            "status": "needs_llm",
            "llm_task": {
                "provider_requested": "auto",
                "provider_resolved": exc.config.provider,
                "host": exc.config.host.host,
                "signals": list(exc.config.signals),
                "message": str(exc),
            },
            "errors": [],
            "meta": {},
        }
    return {
        "ok": False,
        "status": "error",
        "error": {
            "code": "command_failed",
            "message": str(exc) or exc.__class__.__name__,
            "type": exc.__class__.__name__,
        },
        "errors": [],
        "meta": {},
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reusable ARC host LLM worker")
    sub = parser.add_subparsers(dest="command", required=True)

    run_json_parser = sub.add_parser("run-json")
    _prompt_args(run_json_parser)
    run_json_parser.add_argument("--schema", default=None)
    run_json_parser.add_argument("--provider", default="auto")
    run_json_parser.add_argument("--model", default=None)
    run_json_parser.add_argument("--model-tier", choices=["max", "high", "medium", "low"], default=None)
    run_json_parser.add_argument("--json", action="store_true")
    _session_args(run_json_parser)
    _shared_runtime_args(run_json_parser)
    _llm_runtime_args(run_json_parser)

    run_text_parser = sub.add_parser("run-text")
    _prompt_args(run_text_parser)
    run_text_parser.add_argument("--provider", default="auto")
    run_text_parser.add_argument("--model", default=None)
    run_text_parser.add_argument("--model-tier", choices=["max", "high", "medium", "low"], default=None)
    _session_args(run_text_parser)
    _shared_runtime_args(run_text_parser)
    _llm_runtime_args(run_text_parser)

    schema_format_parser = sub.add_parser("schema-format")
    schema_format_parser.add_argument("--input", default="-")
    schema_format_parser.add_argument("--schema", required=True)
    schema_format_parser.add_argument("--provider", default="auto")
    schema_format_parser.add_argument("--model", default=None)
    schema_format_parser.add_argument("--model-tier", choices=["max", "high", "medium", "low"], default=None)
    schema_format_parser.add_argument("--role-hint", default=None)
    schema_format_parser.add_argument("--json", action="store_true")
    _shared_runtime_args(schema_format_parser)
    _llm_runtime_args(schema_format_parser)

    loop_parser = sub.add_parser("proposers-reviewer-loop")
    loop_parser.add_argument("--config", required=True)
    loop_parser.add_argument("--json", action="store_true")
    loop_parser.add_argument("--dry-run", action="store_true")
    loop_parser.add_argument("--max-concurrent-loops", type=int, default=None)
    loop_parser.add_argument("--session-policy", choices=["stateful", "stateless"], default=None)
    loop_parser.add_argument("--history-mode", choices=["auto", "delta", "full"], default=None)
    loop_parser.add_argument("--session-scope-id", default=None)
    loop_parser.add_argument("--max-concurrent-same-prefix", type=int, default=None)
    loop_parser.add_argument(
        "--idle-timeout-seconds",
        type=float,
        default=None,
        help="Cancel a worker after this many seconds without substantive provider progress.",
    )
    _worker_capability_args(loop_parser)

    bench_parser = sub.add_parser("proposers-reviewer-bench")
    bench_parser.add_argument("--config", required=True)
    bench_parser.add_argument("--json", action="store_true")
    bench_parser.add_argument("--dry-run", action="store_true")

    doctor = sub.add_parser("doctor")
    doctor_sub = doctor.add_subparsers(dest="doctor_command", required=True)
    for doctor_command in ("host", "provider", "config"):
        doctor_parser = doctor_sub.add_parser(doctor_command)
        doctor_parser.add_argument("--json", action="store_true")
    circuit = sub.add_parser("circuit", help="Inspect or reset shared provider safety circuits")
    circuit_sub = circuit.add_subparsers(dest="circuit_command", required=True)
    circuit_status = circuit_sub.add_parser("status")
    circuit_status.add_argument("--provider", default=None)
    circuit_status.add_argument("--json", action="store_true")
    circuit_reset = circuit_sub.add_parser("reset")
    circuit_reset.add_argument("--provider", default=None)
    circuit_reset.add_argument("--endpoint", default=None)
    circuit_reset.add_argument("--json", action="store_true")
    cache_audit = sub.add_parser("cache-audit")
    cache_audit.add_argument("run_root")
    materialize = sub.add_parser("materialize-proposers-reviewer")
    materialize.add_argument("--spec", required=True)
    return parser


def _dispatch(args: argparse.Namespace) -> Any:
    if args.command == "doctor":
        safety = LLMSafetyController().status()
        if args.doctor_command == "host":
            return {**detect_host().__dict__, "llm_safety": safety}
        if args.doctor_command == "provider":
            selected = select_llm_provider()
            result = {
                **provider_diagnostic(selected.provider),
                "host": selected.host.host,
                "signals": selected.signals,
                "llm_safety": safety,
            }
            if selected.provider == "kimi-code-cli":
                result["kimi_retry_safety"] = kimi_retry_safety_diagnostic()
            return result
        config = resolve_llm_config()
        result = {
            **provider_diagnostic(config.provider),
            "model": config.model,
            "host": config.host.host,
            "signals": config.signals,
            "warnings": list(config.warnings),
            "llm_safety": safety,
        }
        if config.provider == "kimi-code-cli":
            result["kimi_retry_safety"] = kimi_retry_safety_diagnostic()
        return result
    if args.command == "circuit":
        controller = LLMSafetyController()
        if args.circuit_command == "status":
            status = controller.status()
            if args.provider:
                status["circuits"] = [
                    item for item in status["circuits"] if item.get("provider") == args.provider
                ]
            return {"ok": True, "status": "completed", **status}
        reset_count = controller.reset_circuit(args.provider, endpoint=args.endpoint)
        return {
            "ok": True,
            "status": "completed",
            "provider": args.provider,
            "endpoint": args.endpoint,
            "reset_count": reset_count,
        }
    if args.command == "run-json":
        session_manager = _session_manager(args)
        return run_json(
            _read_prompt_argument(args),
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
            idle_timeout_seconds=args.idle_timeout_seconds,
            progress_callback=_job_progress_callback(),
            cancel_check=_job_cancel_check,
            idempotency_key=args.idempotency_key,
            progress_contract_scope="session" if args.session_policy == "stateful" else "call",
        )
    if args.command == "run-text":
        session_manager = _session_manager(args)
        return run_text(
            _read_prompt_argument(args),
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
            idle_timeout_seconds=args.idle_timeout_seconds,
            progress_callback=_job_progress_callback(),
            cancel_check=_job_cancel_check,
            idempotency_key=args.idempotency_key,
            progress_contract_scope="session" if args.session_policy == "stateful" else "call",
        )
    if args.command == "schema-format":
        return format_to_schema(
            raw_text=_read_prompt(args.input),
            schema=_read_schema(args.schema) or {},
            role_hint=args.role_hint,
            json_runner=run_json,
            provider=args.provider,
            model=args.model,
            model_tier=args.model_tier,
            env=_runtime_env(args),
            idle_timeout_seconds=args.idle_timeout_seconds,
            progress_callback=_job_progress_callback(),
            cancel_check=_job_cancel_check,
        ).value
    if args.command == "proposers-reviewer-loop":
        config = _read_json_file(args.config)
        _apply_loop_session_overrides(config, args)
        _apply_loop_runtime_overrides(config, args)
        if args.idle_timeout_seconds is not None:
            config["worker_idle_timeout_seconds"] = args.idle_timeout_seconds
        return run_proposers_reviewer_batch(
            config,
            dry_run=args.dry_run,
            max_concurrent_loops=args.max_concurrent_loops,
            progress_callback=_job_progress_callback(),
            cancel_check=_job_cancel_check,
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
    parser.add_argument("--idempotency-key", default=None)


def _prompt_args(parser: argparse.ArgumentParser) -> None:
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument(
        "--prompt",
        default=None,
        help="Legacy alias for --prompt-file; pass '-' to read stdin.",
    )
    prompts.add_argument(
        "--prompt-file",
        default=None,
        help="Read the prompt from this UTF-8 file, or '-' for stdin.",
    )
    prompts.add_argument(
        "--prompt-text",
        default=None,
        help="Use this argument verbatim as the prompt text.",
    )


def _session_manager(args: argparse.Namespace):
    if getattr(args, "session_policy", "stateless") != "stateful":
        return None
    root = Path(args.session_root) if args.session_root else llm_cache_root() / "sessions"
    return LLMSessionManager(root)


def _install_cancel_signal_handlers() -> dict[int, Any]:
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _request_signal_cancel)
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _request_signal_cancel(signum: int, frame: Any) -> None:
    del signum, frame
    _SIGNAL_CANCEL_REQUESTED.set()


def _job_cancel_check() -> bool:
    if _SIGNAL_CANCEL_REQUESTED.is_set():
        return True
    explicit = os.environ.get("ARC_JOB_CANCEL_FILE")
    if explicit:
        return Path(explicit).expanduser().exists()
    progress = os.environ.get("ARC_JOB_PROGRESS_FILE")
    return bool(progress) and Path(progress).expanduser().with_name("cancel.request").exists()


def _job_progress_callback():
    raw_path = os.environ.get("ARC_JOB_PROGRESS_FILE")
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()

    def append(event: dict[str, Any]) -> None:
        sidechannel_event = {
            key: value
            for key, value in event.items()
            if key not in {"schema_version", "job_id", "environment", "argv", "command"}
        }
        if "status" in sidechannel_event:
            sidechannel_event["run_status"] = sidechannel_event.pop("status")
        encoded = json.dumps(
            sidechannel_event,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ) + "\n"
        with _PROGRESS_FILE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                os.chmod(path, 0o600)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())

    return append


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


def _apply_loop_runtime_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    access = getattr(args, "arc_paper_cli_access", None)
    inherit = bool(getattr(args, "inherit_host_tools", False))
    if access is None and not inherit:
        return
    for loop in config.get("loops", []):
        if not isinstance(loop, dict):
            continue
        for collection in ("proposers", "reviewers"):
            for worker in loop.get(collection, []):
                if not isinstance(worker, dict):
                    continue
                runtime = dict(worker.get("runtime") or {})
                if access is not None:
                    runtime["arc_paper_cli_access"] = access
                if inherit:
                    runtime["inherit_host_tools"] = True
                worker["runtime"] = runtime


def _shared_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--allow-internet", action="store_true")
    parser.add_argument("--allow-mcp", action="store_true")
    parser.add_argument("--mcp-mode", choices=["user-config", "arc-only"], default=None)


def _llm_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--idle-timeout-seconds",
        type=float,
        default=None,
        help="Cancel after this many seconds without substantive provider progress.",
    )
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
    _worker_capability_args(parser)
    parser.add_argument("--codex-use-user-config", action="store_true")
    _remaining_llm_runtime_args(parser)


def _worker_capability_args(parser: argparse.ArgumentParser) -> None:
    paper_cli = parser.add_mutually_exclusive_group()
    paper_cli.add_argument(
        "--arc-paper-cli",
        dest="arc_paper_cli_access",
        action="store_const",
        const="full",
        default=None,
        help="Allow the non-LLM arc-paper-worker CLI (the default for new ordinary calls).",
    )
    paper_cli.add_argument(
        "--no-arc-paper-cli",
        dest="arc_paper_cli_access",
        action="store_const",
        const="none",
        help="Disable arc-paper-worker for this call.",
    )
    parser.add_argument(
        "--inherit-host-tools",
        action="store_true",
        help="HIGH RISK: inherit host user configuration, rules, skills, plugins, and MCP tools.",
    )


def _remaining_llm_runtime_args(parser: argparse.ArgumentParser) -> None:
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
    # Ordinary newly submitted workers receive the paper-only CLI. Isolated
    # internal stages override this below or at their call site.
    overrides["ARC_PAPER_CLI_ACCESS"] = (
        "none" if args.command == "schema-format" else (args.arc_paper_cli_access or "full")
    )
    overrides["ARC_LLM_INHERIT_HOST_TOOLS"] = "true" if args.inherit_host_tools else "false"
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
    if args.inherit_host_tools:
        overrides.update(
            {
                "ARC_LLM_HOST_TOOLS_RISK": "high",
                "ARC_CODEX_ENABLE_MCP": "true",
                "ARC_CODEX_IGNORE_USER_CONFIG": "false",
                "ARC_CODEX_IGNORE_RULES": "false",
                "ARC_CLAUDE_ALLOW_MCP": "true",
                "ARC_CLAUDE_BARE": "false",
            }
        )
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


def _read_prompt_argument(args: argparse.Namespace) -> str:
    if args.prompt_text is not None:
        return args.prompt_text
    return _read_prompt(args.prompt_file or args.prompt or "-")


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
