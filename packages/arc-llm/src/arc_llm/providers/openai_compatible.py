from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Mapping

from .base import LLMWorkerError
from .config import ConfiguredProvider


ClientFactory = Callable[..., Any]


class OpenAICompatibleProvider:
    def __init__(
        self,
        provider: ConfiguredProvider,
        *,
        env: Mapping[str, str] | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.config = provider
        self.name = provider.id
        self.env = os.environ if env is None else env
        self._client_factory = client_factory

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        resolved_model = self._require_model(model)
        request: dict[str, Any] = {
            "model": resolved_model,
            "messages": _json_messages(prompt, self.config.json_mode),
        }
        response_format = _response_format(self.config, schema or {"type": "object"})
        if response_format:
            request["response_format"] = response_format
        content = self._create(request)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMWorkerError(f"{self.name} JSON output was not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise LLMWorkerError(f"{self.name} JSON output was not an object")
        return payload

    def generate_text(self, prompt: str, *, model: str | None = None) -> str:
        resolved_model = self._require_model(model)
        return self._create(
            {
                "model": resolved_model,
                "messages": [{"role": "user", "content": prompt}],
            }
        )

    def _create(self, request: dict[str, Any]) -> str:
        api_key = self.config.resolved_api_key(env=self.env)
        if not api_key:
            raise LLMWorkerError(f"{self.name} requires api_key or api_key_optional=true")
        try:
            client = self._client(api_key)
            completion = client.chat.completions.create(**request)
            return _first_message_content(completion)
        except LLMWorkerError:
            raise
        except Exception as exc:
            raise LLMWorkerError(_redact(str(exc), api_key)) from exc

    def _client(self, api_key: str) -> Any:
        factory = self._client_factory or _openai_client_factory()
        return factory(api_key=api_key, base_url=self.config.base_url)

    def _require_model(self, model: str | None) -> str:
        resolved = model or self.config.default_model()
        if not resolved:
            raise LLMWorkerError(f"{self.name} requires a model from --model, ARC_LLM_MODEL, or provider config")
        return resolved


def _openai_client_factory() -> ClientFactory:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMWorkerError("openai-compatible providers require the openai Python package") from exc
    return OpenAI


def _response_format(provider: ConfiguredProvider, schema: dict[str, Any]) -> dict[str, Any] | None:
    if provider.json_mode == "none":
        return None
    if provider.json_mode == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"arc_{_schema_name_part(provider.id)}_output",
            "schema": schema,
            "strict": True,
        },
    }


def _json_messages(prompt: str, json_mode: str) -> list[dict[str, str]]:
    if json_mode == "json_object":
        return [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ]
    return [{"role": "user", "content": prompt}]


def _first_message_content(completion: Any) -> str:
    choices = _field(completion, "choices")
    if not choices:
        raise LLMWorkerError("OpenAI-compatible response did not include choices")
    message = _field(choices[0], "message")
    content = _field(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = _field(item, "text")
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "".join(parts)
    raise LLMWorkerError("OpenAI-compatible response did not include text content")


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _schema_name_part(provider_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", provider_id).strip("_") or "provider"


def _redact(message: str, secret: str) -> str:
    if secret:
        return message.replace(secret, "[redacted]")
    return message
