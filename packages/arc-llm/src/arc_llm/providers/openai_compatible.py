from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Mapping

from .base import LLMWorkerError
from .config import ConfiguredProvider


ClientFactory = Callable[..., Any]
MAX_JSON_PARSE_ATTEMPTS = 3


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
        json_mode = _effective_json_mode(self.config)
        schema_payload = schema or {"type": "object"}
        request: dict[str, Any] = {
            "model": resolved_model,
            "messages": _json_messages(prompt, json_mode, schema=schema_payload),
        }
        response_format = _response_format(self.config, schema_payload)
        if response_format:
            request["response_format"] = response_format
        active_json_mode = json_mode
        try:
            content = self._create(request)
        except LLMWorkerError as exc:
            fallback = _response_format_fallback_mode(json_mode, exc)
            if fallback is None:
                raise
            active_json_mode = fallback
            content = self._create(_json_request(resolved_model, prompt, fallback, schema=schema_payload))
        payload = self._parse_with_retries(
            content,
            prompt=prompt,
            model=resolved_model,
            json_mode=active_json_mode,
            schema=schema_payload,
        )
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

    def _parse_with_retries(
        self,
        content: str,
        *,
        prompt: str,
        model: str,
        json_mode: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        error = ""
        for attempt in range(1, MAX_JSON_PARSE_ATTEMPTS + 1):
            try:
                return _parse_json_object(content, provider_name=self.name)
            except LLMWorkerError as exc:
                error = str(exc)
                if attempt == MAX_JSON_PARSE_ATTEMPTS:
                    raise
                content = self._create(
                    _json_retry_request(
                        model,
                        prompt,
                        json_mode,
                        schema=schema,
                        error=error,
                        attempt=attempt + 1,
                    )
                )
        raise LLMWorkerError(error or f"{self.name} JSON output was not valid JSON")


def _openai_client_factory() -> ClientFactory:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMWorkerError("openai-compatible providers require the openai Python package") from exc
    return OpenAI


def _response_format(provider: ConfiguredProvider, schema: dict[str, Any]) -> dict[str, Any] | None:
    json_mode = _effective_json_mode(provider)
    if json_mode == "none":
        return None
    if json_mode == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"arc_{_schema_name_part(provider.id)}_output",
            "schema": schema,
            "strict": True,
        },
    }


def _effective_json_mode(provider: ConfiguredProvider) -> str:
    if provider.json_mode == "json_schema" and _is_deepseek_provider(provider):
        return "json_object"
    return provider.json_mode


def _is_deepseek_provider(provider: ConfiguredProvider) -> bool:
    text = f"{provider.id} {provider.base_url}".lower()
    return "deepseek" in text


def _json_request(model: str, prompt: str, json_mode: str, *, schema: dict[str, Any]) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": _json_messages(prompt, json_mode, schema=schema),
    }
    response_format = _response_format_for_mode(json_mode)
    if response_format:
        request["response_format"] = response_format
    return request


def _json_retry_request(
    model: str,
    prompt: str,
    json_mode: str,
    *,
    schema: dict[str, Any],
    error: str,
    attempt: int,
) -> dict[str, Any]:
    request = _json_request(model, prompt, json_mode, schema=schema)
    retry_note = (
        f"Previous response was invalid JSON. This is repair attempt {attempt}. "
        "Retry the same task and return exactly one valid JSON object. "
        "Do not include markdown, comments, a second JSON object, or unescaped quotes/newlines inside strings. "
        "Escape every backslash that appears inside a JSON string, for example use \\\\mu rather than \\mu. "
        f"Error: {error}"
    )
    if request["messages"] and request["messages"][0]["role"] == "system":
        request["messages"][0]["content"] = f"{request['messages'][0]['content']}\n\n{retry_note}"
    else:
        request["messages"].insert(0, {"role": "system", "content": retry_note})
    return request


def _response_format_for_mode(json_mode: str) -> dict[str, Any] | None:
    if json_mode == "json_object":
        return {"type": "json_object"}
    if json_mode == "none":
        return None
    raise LLMWorkerError(f"unsupported fallback json_mode: {json_mode}")


def _response_format_fallback_mode(json_mode: str, exc: LLMWorkerError) -> str | None:
    if not _is_response_format_unavailable(str(exc)):
        return None
    if json_mode == "json_schema":
        return "json_object"
    if json_mode == "json_object":
        return "none"
    return None


def _is_response_format_unavailable(message: str) -> bool:
    normalized = message.lower()
    return "response_format" in normalized and (
        "unavailable" in normalized
        or "unsupported" in normalized
        or "not support" in normalized
        or "not supported" in normalized
    )


def _parse_json_object(content: str, *, provider_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as first_exc:
        decoder = json.JSONDecoder()
        start = content.find("{")
        if start < 0:
            raise LLMWorkerError(f"{provider_name} JSON output was not valid JSON: {first_exc}") from first_exc
        try:
            payload, _ = decoder.raw_decode(content[start:])
        except json.JSONDecodeError:
            raise LLMWorkerError(f"{provider_name} JSON output was not valid JSON: {first_exc}") from first_exc
    if not isinstance(payload, dict):
        raise LLMWorkerError(f"{provider_name} JSON output was not an object")
    return payload


def _json_messages(prompt: str, json_mode: str, *, schema: dict[str, Any] | None = None) -> list[dict[str, str]]:
    if json_mode == "json_object":
        return [
            {"role": "system", "content": _json_only_system_message(schema)},
            {"role": "user", "content": prompt},
        ]
    if json_mode == "none":
        return [
            {"role": "system", "content": _json_only_system_message(schema)},
            {"role": "user", "content": prompt},
        ]
    return [{"role": "user", "content": prompt}]


def _json_only_system_message(schema: dict[str, Any] | None) -> str:
    if not schema:
        return "Return exactly one JSON object and no other text."
    return (
        "Return exactly one JSON object and no other text. The object must match this JSON Schema:\n"
        f"{json.dumps(schema, indent=2, ensure_ascii=False)}"
    )


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
