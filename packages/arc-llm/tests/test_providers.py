import json
import subprocess

from arc_llm.providers.base import LLMWorkerError
from arc_llm.providers.claude_cli import ClaudeCliProvider
from arc_llm.providers.codex_cli import CodexCliProvider
from arc_llm.sessions import LLMSessionRef


def test_codex_generate_json_writes_prompt_to_stdin_and_reads_output(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CodexCliProvider().generate_json("prompt text", schema={"type": "object"}, model="test-model")

    assert result == {"ok": True}
    assert captured["cmd"][:2] == ["codex", "exec"]
    assert "--ephemeral" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--sandbox") + 1] == "read-only"
    assert "--ignore-user-config" in captured["cmd"]
    assert "--ignore-rules" in captured["cmd"]
    assert "-c" in captured["cmd"]
    assert 'model_reasoning_effort="low"' in captured["cmd"]
    assert 'model_reasoning_summary="none"' in captured["cmd"]
    assert 'model_verbosity="low"' in captured["cmd"]
    assert "hide_agent_reasoning=true" in captured["cmd"]
    assert 'history.persistence="none"' in captured["cmd"]
    assert 'web_search="disabled"' in captured["cmd"]
    assert "--output-schema" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("-m") + 1] == "test-model"
    assert captured["cmd"][-1] == "-"
    assert captured["input"] == "prompt text"
    assert "prompt text" not in captured["cmd"]


def test_codex_generate_text_reads_last_message(monkeypatch):
    def fake_run(cmd, **kwargs):
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("plain text")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert CodexCliProvider().generate_text("prompt") == "plain text"


def test_codex_stateful_first_call_uses_json_events_and_keeps_history(monkeypatch, tmp_path):
    captured = {}
    session = LLMSessionRef(
        key="scope/proposer/proposer_001",
        provider="codex-cli",
        model="test-model",
        runtime_fingerprint="fp",
    )

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        stdout = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread-123"}),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 70,
                            "output_tokens": 5,
                        },
                    }
                ),
            ]
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    response = CodexCliProvider().generate_json_result(
        "prompt text",
        schema={"type": "object"},
        model="test-model",
        session=session,
        session_policy="stateful",
        schema_cache_dir=tmp_path / "schemas",
    )

    assert response.value == {"ok": True}
    assert response.native_session_id == "thread-123"
    assert response.usage.input_tokens == 100
    assert response.usage.cached_input_ratio == 0.7
    assert "--json" in captured["cmd"]
    assert "--ephemeral" not in captured["cmd"]
    assert 'history.persistence="none"' not in captured["cmd"]
    assert "--output-schema" in captured["cmd"]
    assert captured["input"] == "prompt text"


def test_codex_stateful_resume_keeps_session_when_schema_probe_fails(monkeypatch, tmp_path):
    captured = {}
    session = LLMSessionRef(
        key="scope/proposer/proposer_001",
        provider="codex-cli",
        model="test-model",
        runtime_fingerprint="fp",
        native_session_id="thread-123",
    )

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write('{"ok": true}')
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("arc_llm.providers.codex_cli._codex_resume_supports_output_schema", lambda env: False)

    response = CodexCliProvider().generate_json_result(
        "delta prompt",
        schema={"type": "object", "required": ["ok"]},
        model="test-model",
        session=session,
        session_policy="stateful",
        schema_cache_dir=tmp_path / "schemas",
    )

    resume_index = captured["cmd"].index("resume")
    assert response.value == {"ok": True}
    assert captured["cmd"][resume_index + 1] == "thread-123"
    assert "--output-schema" not in captured["cmd"]
    assert "JSON output contract for this turn" in captured["input"]
    assert "delta prompt" in captured["input"]


def test_codex_passes_provider_env_to_subprocess(monkeypatch):
    captured = {}
    provider_env = {"ARC_CODEX_SANDBOX": "read-only", "CUSTOM_SETTING": "value"}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("plain text")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert CodexCliProvider(env=provider_env).generate_text("prompt") == "plain text"
    assert captured["env"] == provider_env
    assert captured["env"] is not provider_env


def test_codex_timeout_uses_provider_specific_env(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("plain text")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert CodexCliProvider(env={"ARC_LLM_TIMEOUT_SECONDS": "30", "ARC_CODEX_TIMEOUT_SECONDS": "12.5"}).generate_text(
        "prompt"
    ) == "plain text"
    assert captured["timeout"] == 12.5


def test_codex_options_can_be_overridden_by_env(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("plain text")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    provider = CodexCliProvider(
        env={
            "ARC_CODEX_SANDBOX": "workspace-write",
            "ARC_CODEX_EPHEMERAL": "false",
            "ARC_CODEX_REASONING_EFFORT": "minimal",
            "ARC_CODEX_NETWORK_ACCESS": "false",
            "ARC_CODEX_IGNORE_USER_CONFIG": "false",
            "ARC_CODEX_IGNORE_RULES": "true",
        }
    )

    assert provider.generate_text("prompt") == "plain text"
    assert "--ephemeral" not in captured["cmd"]
    assert "--ignore-user-config" not in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--sandbox") + 1] == "workspace-write"
    assert "--ignore-rules" in captured["cmd"]
    assert 'model_reasoning_effort="minimal"' in captured["cmd"]
    assert "sandbox_workspace_write.network_access=false" in captured["cmd"]


def test_codex_can_opt_into_internet_and_selected_config(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("plain text")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    provider = CodexCliProvider(
        env={
            "ARC_CODEX_ALLOW_INTERNET": "true",
            "ARC_CODEX_CONFIG": 'mcp_servers.arc.command="arc-mcp"\nmcp_servers.arc.args=["stdio"]',
        }
    )

    assert provider.generate_text("prompt") == "plain text"
    assert "--ignore-user-config" in captured["cmd"]
    assert 'web_search="live"' in captured["cmd"]
    assert "sandbox_workspace_write.network_access=true" in captured["cmd"]
    assert 'mcp_servers.arc.command="arc-mcp"' in captured["cmd"]
    assert 'mcp_servers.arc.args=["stdio"]' in captured["cmd"]


def test_codex_arc_only_mcp_keeps_user_config_ignored_and_injects_arc_server(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("plain text")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    provider = CodexCliProvider(
        env={
            "ARC_CODEX_ENABLE_MCP": "true",
            "ARC_CODEX_MCP_MODE": "arc-only",
            "ARC_CODEX_ARC_MCP_COMMAND": "/tmp/arc-mcp",
            "ARC_CODEX_WORK_DIR": "/tmp/project",
            "ARC_CODEX_ADD_DIRS": json.dumps(["/tmp/project/skills", "/tmp/arc-skills"]),
            "ARC_PAPER_CACHE": "/tmp/cache/arc-paper",
            "ARC_CODEX_ARC_MCP_ENV_JSON": json.dumps({"ARC_MCP_CACHE": "/tmp/cache/arc-mcp"}),
        }
    )

    assert provider.generate_text("prompt") == "plain text"
    assert "--ignore-user-config" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--cd") + 1] == "/tmp/project"
    assert captured["cmd"].count("--add-dir") == 2
    assert "/tmp/project/skills" in captured["cmd"]
    assert "/tmp/arc-skills" in captured["cmd"]
    assert 'mcp_servers.arc.command="/tmp/arc-mcp"' in captured["cmd"]
    assert 'mcp_servers.arc.default_tools_approval_mode="approve"' in captured["cmd"]
    assert 'mcp_servers.arc.env.ARC_AGENT_HOST="codex"' in captured["cmd"]
    assert 'mcp_servers.arc.env.ARC_PAPER_CACHE="/tmp/cache/arc-paper"' in captured["cmd"]
    assert 'mcp_servers.arc.env.ARC_MCP_CACHE="/tmp/cache/arc-mcp"' in captured["cmd"]


def test_codex_invalid_mcp_mode_fails_closed(monkeypatch):
    def fake_run(*args, **kwargs):
        raise AssertionError("subprocess should not run for invalid MCP mode")

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CodexCliProvider(env={"ARC_CODEX_ENABLE_MCP": "true", "ARC_CODEX_MCP_MODE": "broad"})

    try:
        provider.generate_text("prompt")
    except LLMWorkerError as exc:
        assert "ARC_CODEX_MCP_MODE" in str(exc)
    else:
        raise AssertionError("expected LLMWorkerError")


def test_codex_profile_loads_user_config(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("plain text")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    provider = CodexCliProvider(env={"ARC_CODEX_PROFILE_V2": "arc-paper-check"})

    assert provider.generate_text("prompt") == "plain text"
    assert captured["cmd"][captured["cmd"].index("--profile-v2") + 1] == "arc-paper-check"
    assert "--ignore-user-config" not in captured["cmd"]


def test_claude_generate_json_parses_direct_json(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"ok": True}), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ClaudeCliProvider().generate_json("prompt text", schema={"type": "object"}, model="test-model")

    assert result == {"ok": True}
    assert captured["cmd"][:2] == ["claude", "-p"]
    assert "--bare" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--tools") + 1] == ""
    assert captured["cmd"][captured["cmd"].index("--effort") + 1] == "low"
    assert "--no-session-persistence" in captured["cmd"]
    assert "--exclude-dynamic-system-prompt-sections" in captured["cmd"]
    assert "--json-schema" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--model") + 1] == "test-model"
    assert captured["input"] == "prompt text"
    assert "prompt text" not in captured["cmd"]


def test_claude_generate_json_parses_result_wrapper(monkeypatch):
    def fake_run(cmd, **kwargs):
        wrapped = {"type": "result", "result": json.dumps({"ok": True})}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(wrapped), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert ClaudeCliProvider().generate_json("prompt") == {"ok": True}


def test_claude_generate_text_returns_stdout(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="plain text", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert ClaudeCliProvider().generate_text("prompt") == "plain text"


def test_claude_stateful_first_call_uses_session_id(monkeypatch):
    captured = {}
    session = LLMSessionRef(key="scope/reviewer/reviewer_001", provider="claude-cli", model="m", runtime_fingerprint="fp")

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        payload = {
            "session_id": "00000000-0000-4000-8000-000000000001",
            "usage": {"input_tokens": 11, "output_tokens": 2, "cache_read_input_tokens": 8},
            "result": json.dumps({"ok": True}),
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("arc_llm.providers.claude_cli.uuid.uuid4", lambda: "00000000-0000-4000-8000-000000000001")

    response = ClaudeCliProvider().generate_json_result(
        "prompt",
        schema={"type": "object"},
        model="m",
        session=session,
        session_policy="stateful",
    )

    assert response.value == {"ok": True}
    assert response.native_session_id == "00000000-0000-4000-8000-000000000001"
    assert response.usage.cache_read_input_tokens == 8
    assert "--session-id" in captured["cmd"]
    assert "--resume" not in captured["cmd"]
    assert "--no-session-persistence" not in captured["cmd"]
    assert "--exclude-dynamic-system-prompt-sections" in captured["cmd"]


def test_claude_stateful_resume_uses_resume_id(monkeypatch):
    captured = {}
    session = LLMSessionRef(
        key="scope/reviewer/reviewer_001",
        provider="claude-cli",
        model="m",
        runtime_fingerprint="fp",
        native_session_id="00000000-0000-4000-8000-000000000001",
    )

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"result": {"ok": True}}), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    response = ClaudeCliProvider().generate_json_result(
        "prompt",
        schema={"type": "object"},
        model="m",
        session=session,
        session_policy="stateful",
    )

    assert response.value == {"ok": True}
    assert captured["cmd"][captured["cmd"].index("--resume") + 1] == "00000000-0000-4000-8000-000000000001"
    assert "--session-id" not in captured["cmd"]
    assert "--no-session-persistence" not in captured["cmd"]


def test_claude_passes_provider_env_to_subprocess(monkeypatch):
    captured = {}
    provider_env = {"ARC_CLAUDE_EFFORT": "low", "CUSTOM_SETTING": "value"}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(cmd, 0, stdout="plain text", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert ClaudeCliProvider(env=provider_env).generate_text("prompt") == "plain text"
    assert captured["env"] == provider_env
    assert captured["env"] is not provider_env


def test_claude_timeout_uses_generic_env(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(cmd, 0, stdout="plain text", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert ClaudeCliProvider(env={"ARC_LLM_TIMEOUT_SECONDS": "30"}).generate_text("prompt") == "plain text"
    assert captured["timeout"] == 30.0


def test_claude_options_can_be_overridden_by_env(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="plain text", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    provider = ClaudeCliProvider(
        env={
            "ARC_CLAUDE_BARE": "false",
            "ARC_CLAUDE_TOOLS": "Read",
            "ARC_CLAUDE_EFFORT": "medium",
            "ARC_CLAUDE_NO_SESSION_PERSISTENCE": "false",
            "ARC_CLAUDE_MAX_BUDGET_USD": "0.25",
        }
    )

    assert provider.generate_text("prompt") == "plain text"
    assert "--bare" not in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--tools") + 1] == "Read"
    assert captured["cmd"][captured["cmd"].index("--effort") + 1] == "medium"
    assert "--no-session-persistence" not in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--max-budget-usd") + 1] == "0.25"


def test_claude_can_opt_into_internet(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="plain text", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    provider = ClaudeCliProvider(env={"ARC_CLAUDE_ALLOW_INTERNET": "true"})

    assert provider.generate_text("prompt") == "plain text"
    assert "--bare" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--tools") + 1] == "WebSearch,WebFetch"


def test_claude_can_use_selected_mcp_config(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="plain text", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    provider = ClaudeCliProvider(env={"ARC_CLAUDE_MCP_CONFIG": "/tmp/arc-mcp.json"})

    assert provider.generate_text("prompt") == "plain text"
    assert "--bare" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--tools") + 1] == "default"
    assert captured["cmd"][captured["cmd"].index("--mcp-config") + 1] == "/tmp/arc-mcp.json"
    assert "--strict-mcp-config" in captured["cmd"]
