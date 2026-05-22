import json
import subprocess

from arc_paper_query.summary.providers.claude_cli import ClaudeCliProvider
from arc_paper_query.summary.providers.codex_cli import CodexCliProvider


def valid_summary():
    return {
        "schema_version": "arc.paper_llm_summary.v1",
        "paper_id": "arXiv:0911.3380",
        "title": "A Test Paper",
        "authors_short": "Alice and Bob",
        "high_value_summary": ["The paper computes a useful result."],
        "toc": [
            {
                "section_id": "S1",
                "title": "1 Introduction",
                "level": 2,
            }
        ],
        "section_summaries": [
            {
                "section_id": "S1",
                "title": "1 Introduction",
                "summary": "Introduces the problem.",
                "warnings": [],
            }
        ],
        "reading_guide": [
            {
                "purpose": "Understand the main result",
                "sections": ["S1"],
                "reason": "This section defines the setup.",
            }
        ],
        "warnings": [],
        "provenance": {
            "created_at": "2026-05-22T00:00:00Z",
            "method": "manual",
            "model": "test-model",
            "prompt_version": "paper-summary-v1",
            "source_hash": "a" * 64,
        },
    }


def llm_task():
    return {
        "system_prompt": "system",
        "user_prompt": "user",
        "input_pack": {"paper_id": "arXiv:0911.3380"},
        "output_schema": {},
    }


def test_codex_cli_provider_writes_prompt_and_reads_output(monkeypatch):
    summary = valid_summary()
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CodexCliProvider().generate_summary(llm_task(), model="test-model")

    assert result["title"] == "A Test Paper"
    assert captured["cmd"][:2] == ["codex", "exec"]
    assert "--output-schema" in captured["cmd"]
    assert "-m" in captured["cmd"]
    assert captured["cmd"][-1] == "-"
    assert captured["input"]
    assert captured["input"] not in captured["cmd"]


def test_codex_cli_provider_uses_fast_default_model(monkeypatch):
    summary = valid_summary()
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    CodexCliProvider().generate_summary(llm_task())

    assert captured["cmd"][captured["cmd"].index("-m") + 1] == "gpt-5.4-mini"
    assert captured["input"]


def test_claude_cli_provider_parses_json_stdout(monkeypatch):
    summary = valid_summary()
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(summary), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ClaudeCliProvider().generate_summary(llm_task(), model="test-model")

    assert result["title"] == "A Test Paper"
    assert captured["cmd"][:2] == ["claude", "-p"]
    assert "--json-schema" in captured["cmd"]
    assert "--model" in captured["cmd"]
    assert captured["input"]
    assert captured["input"] not in captured["cmd"]


def test_claude_cli_provider_uses_haiku_default_model(monkeypatch):
    summary = valid_summary()
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(summary), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    ClaudeCliProvider().generate_summary(llm_task())

    assert captured["cmd"][captured["cmd"].index("--model") + 1] == "haiku"
    assert captured["input"]
