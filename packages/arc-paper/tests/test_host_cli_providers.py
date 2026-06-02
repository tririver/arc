import json

from arc_paper.summary.providers import claude_cli as claude_module
from arc_paper.summary.providers import codex_cli as codex_module
from arc_paper.summary.providers.claude_cli import ClaudeCliProvider
from arc_paper.summary.providers.codex_cli import CodexCliProvider


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


def test_codex_summary_provider_uses_arc_llm_run_json(monkeypatch):
    summary = valid_summary()
    calls = []

    def fake_run_json(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return dict(summary)

    monkeypatch.setattr(codex_module, "run_json", fake_run_json)

    result = CodexCliProvider(env={}).generate_summary(llm_task(), model="test-model")

    assert result["title"] == "A Test Paper"
    assert calls
    assert calls[0]["provider"] == "codex-cli"
    assert calls[0]["session_policy"] == "stateless"
    assert calls[0]["call_label"] == "arc-paper/summary"
    assert calls[0]["env"] == {}


def test_codex_cli_provider_uses_medium_default_model(monkeypatch):
    summary = valid_summary()
    calls = []

    def fake_run_json(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return dict(summary)

    monkeypatch.setattr(codex_module, "run_json", fake_run_json)

    CodexCliProvider(env={}).generate_summary(llm_task())

    assert calls[0]["model"] == "gpt-5.4"


def test_claude_summary_provider_uses_arc_llm_run_json(monkeypatch):
    summary = valid_summary()
    calls = []

    def fake_run_json(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return dict(summary)

    monkeypatch.setattr(claude_module, "run_json", fake_run_json)

    result = ClaudeCliProvider(env={}).generate_summary(llm_task(), model="test-model")

    assert result["title"] == "A Test Paper"
    assert calls
    assert calls[0]["provider"] == "claude-cli"
    assert calls[0]["session_policy"] == "stateless"
    assert calls[0]["call_label"] == "arc-paper/summary"
    assert calls[0]["env"] == {}


def test_claude_cli_provider_uses_medium_default_model(monkeypatch):
    summary = valid_summary()
    calls = []

    def fake_run_json(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return dict(summary)

    monkeypatch.setattr(claude_module, "run_json", fake_run_json)

    ClaudeCliProvider(env={}).generate_summary(llm_task())

    assert calls[0]["model"] == "sonnet"


def test_codex_summary_provider_keeps_test_prompt_provider():
    class FakePromptProvider:
        def generate_json(self, prompt, *, schema, model):
            return valid_summary()

    result = CodexCliProvider(_test_prompt_provider=FakePromptProvider()).generate_summary(llm_task(), model="test-model")

    assert result["title"] == "A Test Paper"


def test_claude_summary_provider_keeps_test_prompt_provider():
    class FakePromptProvider:
        def generate_json(self, prompt, *, schema, model):
            return valid_summary()

    result = ClaudeCliProvider(_test_prompt_provider=FakePromptProvider()).generate_summary(llm_task(), model="test-model")

    assert result["title"] == "A Test Paper"
