import json
import subprocess

from arc_llm_worker.providers.claude_cli import ClaudeCliProvider
from arc_llm_worker.providers.codex_cli import CodexCliProvider


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
