from __future__ import annotations

import json

import pytest

from arc_llm.providers.config import ConfiguredProvider
from arc_llm.providers.openai_compatible import OpenAICompatibleProvider
from arc_llm.providers.base import LLMWorkerError


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChoice:
    def __init__(self, content: str):
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, response: str = "plain text", error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return FakeCompletion(self.response)


class SequencedCompletions:
    def __init__(self, events):
        self.events = list(events)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return FakeCompletion(event)


class FakeChat:
    def __init__(self, completions: FakeCompletions):
        self.completions = completions


class FakeClient:
    def __init__(self, completions: FakeCompletions):
        self.chat = FakeChat(completions)


def provider_config(**overrides) -> ConfiguredProvider:
    values = {
        "id": "deepseek",
        "type": "openai-compatible",
        "base_url": "https://api.deepseek.example/v1",
        "api_key": "secret-key",
        "api_key_optional": False,
        "models": {"default": "deepseek-chat"},
        "json_mode": "json_schema",
    }
    values.update(overrides)
    return ConfiguredProvider(**values)


def test_generate_text_uses_openai_client_with_base_url_and_file_api_key():
    completions = FakeCompletions("hello")
    captured = {}

    def client_factory(**kwargs):
        captured.update(kwargs)
        return FakeClient(completions)

    provider = OpenAICompatibleProvider(
        provider_config(),
        env={},
        client_factory=client_factory,
    )

    assert provider.generate_text("Say hello", model="deepseek-chat") == "hello"
    assert captured == {"api_key": "secret-key", "base_url": "https://api.deepseek.example/v1"}
    assert completions.calls[0]["model"] == "deepseek-chat"
    assert completions.calls[0]["messages"] == [{"role": "user", "content": "Say hello"}]


def test_generate_json_uses_json_schema_response_format_and_parses_content():
    completions = FakeCompletions(json.dumps({"ok": True}))

    provider = OpenAICompatibleProvider(
        provider_config(),
        env={},
        client_factory=lambda **kwargs: FakeClient(completions),
    )

    result = provider.generate_json("Return JSON", schema={"type": "object"}, model="deepseek-chat")

    assert result == {"ok": True}
    response_format = completions.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "arc_deepseek_output"
    assert response_format["json_schema"]["schema"] == {"type": "object"}
    assert response_format["json_schema"]["strict"] is True


def test_generate_json_can_use_json_object_mode_for_local_compatibility():
    completions = FakeCompletions(json.dumps({"ok": True}))
    config = provider_config(
        id="ollama",
        base_url="http://127.0.0.1:11434/v1",
        api_key_optional=True,
        json_mode="json_object",
    )

    provider = OpenAICompatibleProvider(
        config,
        env={},
        client_factory=lambda **kwargs: FakeClient(completions),
    )

    assert provider.generate_json("Return JSON", schema={"type": "object"}, model="llama3.1") == {"ok": True}
    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    assert completions.calls[0]["messages"][0]["role"] == "system"
    assert "JSON" in completions.calls[0]["messages"][0]["content"]


def test_generate_json_recovers_first_object_from_trailing_model_text():
    completions = FakeCompletions('{"ok": true, "mode": "first"}\n{"ignored": true}')
    config = provider_config(json_mode="json_object")
    provider = OpenAICompatibleProvider(
        config,
        env={},
        client_factory=lambda **kwargs: FakeClient(completions),
    )

    assert provider.generate_json("Return JSON", schema={"type": "object"}, model="deepseek-chat") == {
        "ok": True,
        "mode": "first",
    }


def test_generate_json_falls_back_to_json_object_when_json_schema_is_unavailable():
    completions = SequencedCompletions(
        [
            RuntimeError("This response_format type is unavailable now secret-key"),
            json.dumps({"ok": True, "mode": "fallback"}),
        ]
    )
    provider = OpenAICompatibleProvider(
        provider_config(),
        env={},
        client_factory=lambda **kwargs: FakeClient(completions),
    )

    result = provider.generate_json("Return JSON", schema={"type": "object"}, model="deepseek-chat")

    assert result == {"ok": True, "mode": "fallback"}
    assert completions.calls[0]["response_format"]["type"] == "json_schema"
    assert completions.calls[1]["response_format"] == {"type": "json_object"}
    assert completions.calls[1]["messages"][0]["role"] == "system"


def test_missing_required_api_key_fails_before_calling_client():
    called = False

    def client_factory(**kwargs):
        nonlocal called
        called = True
        return FakeClient(FakeCompletions())

    provider = OpenAICompatibleProvider(provider_config(api_key=None), env={}, client_factory=client_factory)

    with pytest.raises(LLMWorkerError, match="api_key"):
        provider.generate_text("prompt", model="deepseek-chat")

    assert called is False


def test_provider_error_redacts_api_key_values():
    provider = OpenAICompatibleProvider(
        provider_config(),
        env={},
        client_factory=lambda **kwargs: FakeClient(FakeCompletions(error=RuntimeError("bad secret-key"))),
    )

    with pytest.raises(LLMWorkerError) as exc:
        provider.generate_text("prompt", model="deepseek-chat")

    assert "secret-key" not in str(exc.value)
    assert "[redacted]" in str(exc.value)


def test_generate_text_can_use_inline_api_key_from_local_config():
    completions = FakeCompletions("hello")
    captured = {}

    def client_factory(**kwargs):
        captured.update(kwargs)
        return FakeClient(completions)

    provider = OpenAICompatibleProvider(
        provider_config(api_key="inline-secret"),
        env={},
        client_factory=client_factory,
    )

    assert provider.generate_text("Say hello", model="deepseek-chat") == "hello"
    assert captured["api_key"] == "inline-secret"
