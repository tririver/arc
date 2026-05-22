from __future__ import annotations

from arc_llm import cli


def test_runtime_env_merges_cli_llm_options(monkeypatch):
    monkeypatch.setenv("ARC_AGENT_HOST", "codex")
    parser_args = cli._build_parser().parse_args(
        [
            "run-text",
            "--allow-internet",
            "--codex-reasoning-effort",
            "minimal",
            "--codex-model-verbosity",
            "low",
            "--codex-config",
            'mcp_servers.arc.command="arc-mcp"',
            "--claude-effort",
            "medium",
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
    assert env["ARC_CODEX_REASONING_EFFORT"] == "minimal"
    assert env["ARC_CODEX_MODEL_VERBOSITY"] == "low"
    assert env["ARC_CODEX_CONFIG"] == 'mcp_servers.arc.command="arc-mcp"'
    assert env["ARC_CLAUDE_EFFORT"] == "medium"
    assert env["ARC_CLAUDE_MCP_CONFIG"] == "/tmp/arc-mcp.json"
