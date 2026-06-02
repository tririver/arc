from __future__ import annotations

import json
from types import SimpleNamespace

from arc_llm import cli


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
        ["schema-format", "--input", "-", "--schema", "schema.json", "--model-tier", "medium", "--role-hint", "reviewer"]
    )

    result = cli._dispatch(args)

    assert result == {"ok": True}
    assert captured["raw_text"] == "raw text"
    assert captured["schema"] == {"type": "object"}
    assert captured["model_tier"] == "medium"
    assert captured["role_hint"] == "reviewer"


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
