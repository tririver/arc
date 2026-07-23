from __future__ import annotations

import json
import signal
from types import SimpleNamespace

import pytest

from arc_llm import cli
from arc_llm.providers.base import LLMWorkerCancelled


def test_main_returns_nonzero_for_error_envelope(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_dispatch",
        lambda _args: {"ok": False, "error": {"code": "failed", "message": "boom"}},
    )

    assert cli.main(["doctor", "host"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_main_returns_nonzero_for_failed_status(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_dispatch", lambda _args: {"status": "failed"})

    assert cli.main(["doctor", "host"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_terminal_status_exit_code_matrix(monkeypatch, capsys):
    for status in ("completed", "degraded", "stopped"):
        monkeypatch.setattr(cli, "_dispatch", lambda _args, status=status: {"ok": False, "status": status})
        assert cli.main(["doctor", "host"]) == 0
        assert json.loads(capsys.readouterr().out)["status"] == status

    monkeypatch.setattr(cli, "_dispatch", lambda _args: {"status": "cancelled"})
    assert cli.main(["doctor", "host"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "cancelled"


def test_main_treats_needs_llm_as_successful_handoff(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_dispatch",
        lambda _args: {"ok": False, "status": "needs_llm", "llm_task": {"prompt": "..."}},
    )

    assert cli.main(["doctor", "host"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "needs_llm"


def test_main_json_wraps_dispatch_exception(monkeypatch, capsys):
    def fail(_args):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(cli, "_dispatch", fail)

    assert cli.main(["run-json", "--json"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["error"] == {
        "code": "command_failed",
        "message": "provider unavailable",
        "type": "RuntimeError",
    }


def test_main_json_returns_needs_llm_for_auto_manual_without_provider_call(monkeypatch, capsys):
    monkeypatch.delenv("ARC_AGENT_HOST", raising=False)
    monkeypatch.setattr("arc_llm.host._parent_process_chain", lambda: [])
    monkeypatch.setattr(cli, "_read_prompt", lambda _value: "prompt")

    assert cli.main(["run-json", "--json", "--prompt", "-"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "needs_llm"
    assert output["llm_task"]["provider_resolved"] == "manual"


def test_run_text_returns_needs_llm_handoff_without_json_flag(monkeypatch, capsys):
    monkeypatch.delenv("ARC_AGENT_HOST", raising=False)
    monkeypatch.setattr("arc_llm.host._parent_process_chain", lambda: [])
    monkeypatch.setattr(cli, "_read_prompt", lambda _value: "prompt")

    assert cli.main(["run-text", "--prompt", "-"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "needs_llm"


def test_job_progress_callback_writes_arc_jobs_compatible_jsonl(tmp_path, monkeypatch):
    path = tmp_path / "progress.jsonl"
    monkeypatch.setenv("ARC_JOB_PROGRESS_FILE", str(path))
    callback = cli._job_progress_callback()

    callback({"schema_version": "internal", "event": "run_finished", "status": "degraded", "completed_workers": 2})

    event = json.loads(path.read_text(encoding="utf-8"))
    assert event == {"event": "run_finished", "completed_workers": 2, "run_status": "degraded"}


def test_job_cancel_check_accepts_explicit_and_progress_sibling_files(tmp_path, monkeypatch):
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("ARC_JOB_PROGRESS_FILE", str(progress))
    monkeypatch.delenv("ARC_JOB_CANCEL_FILE", raising=False)
    assert cli._job_cancel_check() is False
    (tmp_path / "cancel.request").write_text("cancel", encoding="utf-8")
    assert cli._job_cancel_check() is True

    (tmp_path / "cancel.request").unlink()
    explicit = tmp_path / "explicit.cancel"
    monkeypatch.setenv("ARC_JOB_CANCEL_FILE", str(explicit))
    explicit.write_text("cancel", encoding="utf-8")
    assert cli._job_cancel_check() is True


def test_signal_handler_sets_cli_cancellation_flag():
    cli._SIGNAL_CANCEL_REQUESTED.clear()
    cli._request_signal_cancel(signal.SIGTERM, None)
    assert cli._job_cancel_check() is True
    cli._SIGNAL_CANCEL_REQUESTED.clear()


def test_doctor_provider_and_config_include_kimi_risk_metadata(tmp_path, monkeypatch):
    kimi_home = tmp_path / "kimi-home"
    kimi_home.mkdir()
    (kimi_home / "mcp.json").write_text("do-not-report-this-value", encoding="utf-8")
    monkeypatch.setenv("ARC_AGENT_HOST", "kimi-code")
    monkeypatch.setenv("KIMI_CODE_HOME", str(kimi_home))

    provider = cli._dispatch(cli._build_parser().parse_args(["doctor", "provider"]))
    config = cli._dispatch(cli._build_parser().parse_args(["doctor", "config"]))

    assert provider["provider"] == "kimi-code-cli"
    assert provider["experimental"] is True
    assert provider["provider_side_persistence"] is True
    assert any(item["category"] == "mcp" for item in provider["risks"])
    assert "do-not-report-this-value" not in repr(provider)
    assert provider["kimi_retry_safety"]["safe"] is False
    assert "do-not-report-this-value" not in repr(provider["kimi_retry_safety"])
    assert "kimi_code_cli.experimental" in config["warnings"]
    assert config["kimi_retry_safety"]["safe"] is False


def test_doctor_json_is_accepted_after_each_doctor_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(cli, "detect_host", lambda: SimpleNamespace(host="unknown", confidence=0.0, signals=[]))

    assert cli.main(["doctor", "host", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["host"] == "unknown"

    parser = cli._build_parser()
    assert parser.parse_args(["doctor", "provider", "--json"]).json is True
    assert parser.parse_args(["doctor", "config", "--json"]).json is True


def test_circuit_status_and_reset_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARC_HOME", str(tmp_path))
    from arc_llm.providers.base import LLMWorkerError
    from arc_llm.safety import LLMSafetyController

    LLMSafetyController().report_failure(
        "codex-cli",
        LLMWorkerError("quota exhausted", category="quota", abort_scope="provider"),
    )

    assert cli.main(["circuit", "status", "--provider", "codex-cli", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["circuits"][0]["category"] == "quota"

    assert cli.main(["circuit", "reset", "--provider", "codex-cli", "--json"]) == 0
    reset = json.loads(capsys.readouterr().out)
    assert reset["reset_count"] == 1


@pytest.mark.parametrize("command", ["run-text", "run-json", "proposers-reviewer-loop"])
def test_timeout_help_documents_idle_semantics(command, capsys):
    with pytest.raises(SystemExit) as caught:
        cli._build_parser().parse_args([command, "--help"])

    assert caught.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "without substantive provider progress" in help_text
    assert "total" not in help_text.lower()


def test_runtime_env_merges_cli_llm_options(monkeypatch):
    monkeypatch.setenv("ARC_AGENT_HOST", "codex")
    parser_args = cli._build_parser().parse_args(
        [
            "run-text",
            "--allow-internet",
            "--allow-mcp",
            "--mcp-mode",
            "arc-only",
            "--codex-reasoning-effort",
            "minimal",
            "--codex-model-verbosity",
            "low",
            "--codex-work-dir",
            "/tmp/project",
            "--codex-add-dir",
            "/tmp/project/skills",
            "--arc-mcp-command",
            "/tmp/arc-mcp",
            "--arc-mcp-env",
            "ARC_PAPER_CACHE=/tmp/arc-paper",
            "--codex-config",
            'mcp_servers.arc.command="arc-mcp"',
            "--claude-effort",
            "medium",
            "--claude-allowed-tools",
            "mcp__arc__get_title",
            "--claude-mcp-config",
            "/tmp/arc-mcp.json",
            "--prompt",
            "-",
        ]
    )

    env = cli._runtime_env(parser_args)

    assert env is not None
    assert env["ARC_AGENT_HOST"] == "codex"
    assert env["ARC_CODEX_ALLOW_INTERNET"] == "true"
    assert env["ARC_CODEX_ENABLE_MCP"] == "true"
    assert env["ARC_CODEX_MCP_MODE"] == "arc-only"
    assert env["ARC_CODEX_REASONING_EFFORT"] == "minimal"
    assert env["ARC_CODEX_MODEL_VERBOSITY"] == "low"
    assert env["ARC_CODEX_WORK_DIR"] == "/tmp/project"
    assert json.loads(env["ARC_CODEX_ADD_DIRS"]) == ["/tmp/project/skills"]
    assert env["ARC_CODEX_ARC_MCP_COMMAND"] == "/tmp/arc-mcp"
    assert env["ARC_CLAUDE_ARC_MCP_COMMAND"] == "/tmp/arc-mcp"
    assert json.loads(env["ARC_CODEX_ARC_MCP_ENV_JSON"]) == {"ARC_PAPER_CACHE": "/tmp/arc-paper"}
    assert json.loads(env["ARC_CLAUDE_ARC_MCP_ENV_JSON"]) == {"ARC_PAPER_CACHE": "/tmp/arc-paper"}
    assert env["ARC_CODEX_CONFIG"] == 'mcp_servers.arc.command="arc-mcp"'
    assert env["ARC_CLAUDE_EFFORT"] == "medium"
    assert env["ARC_CLAUDE_ALLOWED_TOOLS"] == "mcp__arc__get_title"
    assert env["ARC_CLAUDE_MCP_CONFIG"] == "/tmp/arc-mcp.json"
    assert env["ARC_PAPER_CLI_ACCESS"] == "full"
    assert env["ARC_LLM_INHERIT_HOST_TOOLS"] == "false"


def test_runtime_env_paper_cli_defaults_and_isolation_overrides():
    parser = cli._build_parser()

    ordinary = cli._runtime_env(parser.parse_args(["run-text", "--prompt-text", "hello"]))
    disabled = cli._runtime_env(
        parser.parse_args(["run-text", "--prompt-text", "hello", "--no-arc-paper-cli"])
    )
    isolated = cli._runtime_env(
        parser.parse_args(["schema-format", "--schema", "schema.json"])
    )

    assert ordinary["ARC_PAPER_CLI_ACCESS"] == "full"
    assert disabled["ARC_PAPER_CLI_ACCESS"] == "none"
    assert isolated["ARC_PAPER_CLI_ACCESS"] == "none"


def test_inherit_host_tools_is_explicit_and_restores_host_surface():
    parser = cli._build_parser()

    ordinary = cli._runtime_env(parser.parse_args(["run-text", "--prompt-text", "hello"]))
    inherited = cli._runtime_env(
        parser.parse_args(["run-text", "--prompt-text", "hello", "--inherit-host-tools"])
    )

    assert ordinary["ARC_LLM_INHERIT_HOST_TOOLS"] == "false"
    assert inherited["ARC_LLM_INHERIT_HOST_TOOLS"] == "true"
    assert inherited["ARC_CODEX_ENABLE_MCP"] == "true"
    assert inherited["ARC_CODEX_IGNORE_USER_CONFIG"] == "false"
    assert inherited["ARC_CODEX_IGNORE_RULES"] == "false"
    assert inherited["ARC_CLAUDE_ALLOW_MCP"] == "true"
    assert inherited["ARC_CLAUDE_BARE"] == "false"


def test_loop_worker_capability_flags_override_all_workers():
    config = {
        "loops": [
            {
                "proposers": [{"runtime": {"arc_paper_cli_access": "full"}}],
                "reviewers": [{"runtime": {}}],
            }
        ]
    }
    args = cli._build_parser().parse_args(
        [
            "proposers-reviewer-loop",
            "--config",
            "config.json",
            "--no-arc-paper-cli",
            "--inherit-host-tools",
        ]
    )

    cli._apply_loop_runtime_overrides(config, args)

    for collection in ("proposers", "reviewers"):
        runtime = config["loops"][0][collection][0]["runtime"]
        assert runtime["arc_paper_access"] == "none"
        assert "arc_paper_cli_access" not in runtime
        assert runtime["inherit_host_tools"] is True


def test_run_text_cli_passes_model_tier(monkeypatch):
    captured = {}

    def fake_run_text(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(cli, "_read_prompt", lambda value: "prompt text")
    monkeypatch.setattr(cli, "run_text", fake_run_text)

    args = cli._build_parser().parse_args(
        ["run-text", "--prompt", "-", "--provider", "auto", "--model-tier", "high"]
    )

    result = cli._dispatch(args)

    assert result == "ok"
    assert captured["prompt"] == "prompt text"
    assert captured["model_tier"] == "high"
    assert captured["model"] is None


def test_prompt_text_is_literal_and_does_not_read_a_file(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cli,
        "_read_prompt",
        lambda _value: (_ for _ in ()).throw(AssertionError("literal prompt must not be opened")),
    )
    monkeypatch.setattr(cli, "run_text", lambda prompt, **_kwargs: captured.setdefault("prompt", prompt) or "ok")

    args = cli._build_parser().parse_args(["run-text", "--prompt-text", "Say hello", "--provider", "manual"])

    assert cli._dispatch(args) == "Say hello"
    assert captured["prompt"] == "Say hello"


def test_prompt_file_and_legacy_prompt_keep_file_semantics(tmp_path, monkeypatch):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("from file", encoding="utf-8")
    seen = []
    monkeypatch.setattr(cli, "run_text", lambda prompt, **_kwargs: seen.append(prompt) or "ok")
    parser = cli._build_parser()

    assert cli._dispatch(parser.parse_args(["run-text", "--prompt-file", str(prompt_file)])) == "ok"
    assert cli._dispatch(parser.parse_args(["run-text", "--prompt", str(prompt_file)])) == "ok"
    assert seen == ["from file", "from file"]


def test_prompt_sources_are_mutually_exclusive():
    parser = cli._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["run-text", "--prompt-text", "literal", "--prompt-file", "prompt.txt"])


def test_run_json_cli_passes_stateful_session_args(tmp_path, monkeypatch):
    captured = {}

    def fake_run_json(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli, "_read_prompt", lambda value: "prompt text")
    monkeypatch.setattr(cli, "run_json", fake_run_json)

    args = cli._build_parser().parse_args(
        [
            "run-json",
            "--prompt",
            "-",
            "--provider",
            "codex-cli",
            "--session-policy",
            "stateful",
            "--session-root",
            str(tmp_path / "sessions"),
            "--session-key",
            "scope/proposer/proposer_001",
            "--session-name",
            "proposer_001",
            "--call-label",
            "round_001/proposer_001",
        ]
    )

    result = cli._dispatch(args)

    assert result == {"ok": True}
    assert captured["session_policy"] == "stateful"
    assert captured["session_manager"].root == tmp_path / "sessions"
    assert captured["session_key"] == "scope/proposer/proposer_001"
    assert captured["session_name"] == "proposer_001"
    assert captured["call_label"] == "round_001/proposer_001"


def test_schema_format_cli_passes_schema_and_model_tier(monkeypatch):
    captured = {}

    def fake_format_to_schema(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(value={"ok": True})

    monkeypatch.setattr(cli, "_read_prompt", lambda value: "raw text")
    monkeypatch.setattr(cli, "_read_schema", lambda value: {"type": "object"})
    monkeypatch.setattr(cli, "format_to_schema", fake_format_to_schema)

    args = cli._build_parser().parse_args(
        [
            "schema-format",
            "--input",
            "-",
            "--schema",
            "schema.json",
            "--model-tier",
            "medium",
            "--role-hint",
            "reviewer",
            "--idle-timeout-seconds",
            "12.5",
        ]
    )

    result = cli._dispatch(args)

    assert result == {"ok": True}
    assert captured["raw_text"] == "raw text"
    assert captured["schema"] == {"type": "object"}
    assert captured["model_tier"] == "medium"
    assert captured["role_hint"] == "reviewer"
    assert captured["idle_timeout_seconds"] == 12.5
    assert captured["cancel_check"] is cli._job_cancel_check


def test_schema_format_cli_idle_timeout_and_cancellation_reach_json_runner(monkeypatch):
    captured = {}

    def fake_run_json(prompt, *, idle_timeout_seconds=None, cancel_check=None, **_kwargs):
        captured["idle_timeout_seconds"] = idle_timeout_seconds
        captured["cancel_check"] = cancel_check
        if cancel_check is not None and cancel_check():
            raise LLMWorkerCancelled("cancelled by test")
        raise AssertionError("cancellation check was not forwarded")

    cancel_check = lambda: True
    monkeypatch.setattr(cli, "_read_prompt", lambda _value: "raw text")
    monkeypatch.setattr(cli, "_read_schema", lambda _value: {"type": "object"})
    monkeypatch.setattr(cli, "_job_cancel_check", cancel_check)
    monkeypatch.setattr(cli, "run_json", fake_run_json)
    args = cli._build_parser().parse_args(
        [
            "schema-format",
            "--input",
            "-",
            "--schema",
            "schema.json",
            "--idle-timeout-seconds",
            "0.25",
        ]
    )

    with pytest.raises(LLMWorkerCancelled, match="cancelled by test"):
        cli._dispatch(args)

    assert captured == {"idle_timeout_seconds": 0.25, "cancel_check": cancel_check}


def test_removed_total_timeout_cli_flag_fails_before_dispatch():
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(
            ["run-text", "--prompt-text", "hello", "--timeout-seconds", "12"]
        )


def test_claude_session_persistence_flags_are_consistent():
    parser = cli._build_parser()

    disabled = cli._runtime_env(
        parser.parse_args(["run-text", "--prompt", "-", "--claude-no-session-persistence"])
    )
    legacy_disabled = cli._runtime_env(
        parser.parse_args(["run-text", "--prompt", "-", "--no-claude-session-persistence"])
    )
    enabled = cli._runtime_env(
        parser.parse_args(["run-text", "--prompt", "-", "--claude-session-persistence"])
    )

    assert disabled["ARC_CLAUDE_NO_SESSION_PERSISTENCE"] == "true"
    assert legacy_disabled["ARC_CLAUDE_NO_SESSION_PERSISTENCE"] == "true"
    assert enabled["ARC_CLAUDE_NO_SESSION_PERSISTENCE"] == "false"


def test_run_text_cli_rejects_provider_config_option():
    parser = cli._build_parser()

    try:
        parser.parse_args(["run-text", "--provider-config", "/tmp/llm-providers.json"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("arc-llm run commands must not accept provider config files")


def test_cli_does_not_expose_provider_config_commands():
    parser = cli._build_parser()

    try:
        parser.parse_args(["providers", "list"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("arc-llm must not expose provider-config commands")
