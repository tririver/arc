from __future__ import annotations

import json

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
    assert json.loads(env["ARC_CODEX_ARC_MCP_ENV_JSON"]) == {"ARC_PAPER_CACHE": "/tmp/arc-paper"}
    assert env["ARC_CODEX_CONFIG"] == 'mcp_servers.arc.command="arc-mcp"'
    assert env["ARC_CLAUDE_EFFORT"] == "medium"
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


def test_runtime_env_merges_provider_config_path(monkeypatch):
    monkeypatch.setenv("ARC_AGENT_HOST", "codex")
    parser_args = cli._build_parser().parse_args(
        [
            "run-text",
            "--provider-config",
            "/tmp/llm-providers.json",
            "--prompt",
            "-",
        ]
    )

    env = cli._runtime_env(parser_args)

    assert env is not None
    assert env["ARC_LLM_PROVIDER_CONFIG"] == "/tmp/llm-providers.json"


def test_providers_list_reports_builtins_and_configured_providers(tmp_path, monkeypatch):
    config_path = tmp_path / "llm-providers.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "arc.llm.providers.v1",
                "providers": [
                        {
                            "id": "deepseek",
                            "type": "openai-compatible",
                            "base_url": "https://api.deepseek.example/v1",
                            "api_key": "secret-key",
                            "models": {"medium": "deepseek-chat"},
                        }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = cli._build_parser().parse_args(
        ["providers", "list", "--provider-config", str(config_path)]
    )
    result = cli._dispatch(args)

    assert result["config_path"] == str(config_path)
    assert "codex-cli" in result["builtins"]
    assert result["configured"][0]["id"] == "deepseek"
    assert result["configured"][0]["has_api_key"] is True
    assert "secret-key" not in json.dumps(result)


def test_providers_add_can_write_inline_api_key_to_local_config_without_echoing_secret(tmp_path):
    config_path = tmp_path / "llm-providers.json"

    args = cli._build_parser().parse_args(
        [
            "providers",
            "add",
            "openai-compatible",
            "--provider-config",
            str(config_path),
            "--id",
            "deepseek",
            "--base-url",
            "https://api.deepseek.example/v1",
            "--api-key",
            "secret-key",
            "--medium-model",
            "deepseek-chat",
            "--high-model",
            "deepseek-reasoner",
        ]
    )
    result = cli._dispatch(args)

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert result["status"] == "written"
    assert payload["providers"][0]["api_key"] == "secret-key"
    assert "secret-key" not in json.dumps(result)
    assert payload["providers"][0]["models"]["medium"] == "deepseek-chat"
    assert payload["providers"][0]["models"]["high"] == "deepseek-reasoner"


def test_providers_init_writes_project_local_config_with_location_comment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = cli._build_parser().parse_args(["providers", "init"])

    result = cli._dispatch(args)

    config_path = tmp_path / "llm-providers.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    comment = "\n".join(payload["_comment"])
    assert result["config_path"] == str(config_path)
    assert "arc-llm providers init" in comment
    assert "arc-llm providers add openai-compatible" in comment
    assert "Linux" in comment
    assert "macOS" in comment
    assert "Windows" in comment
    assert str(config_path) in comment
