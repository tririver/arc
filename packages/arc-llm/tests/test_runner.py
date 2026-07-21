import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from arc_llm.call_checkpoint import LLMCallNeedsSupervision, LLMCallRetryExhausted

from arc_llm.call_record import (
    ARC_LLM_CALL_RECORD_FIELD,
    ARC_LLM_CALL_RECORD_SCHEMA,
    ARC_LLM_CALL_RECORD_SCHEMA_VERSION,
    allow_arc_llm_call_record,
)
from arc_llm.json_schema import to_provider_json_schema
from arc_llm.providers.base import (
    LLMConfigurationError,
    LLMFailureCategory,
    LLMSubmissionState,
    LLMWorkerCancelled,
    LLMWorkerError,
)
from arc_llm.progress_prompt import ensure_runtime_progress_contract
from arc_llm.sessions import LLMSessionManager, runtime_fingerprint
from arc_llm.schema_cache import sha256_text
from arc_llm.structured_recovery import structured_metadata
from arc_llm import runner
from arc_llm.runner import LLMTaskError, resolve_llm_config, run_json, run_text, run_text_result
from arc_llm.usage import LLMProviderResponse, LLMUsage


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)


@pytest.mark.parametrize(
    "error",
    [
        LLMCallNeedsSupervision(checkpoint_path=Path("/tmp/uncertain.json")),
        LLMCallRetryExhausted(
            "terminal checkpoint",
            checkpoint_path=Path("/tmp/terminal.json"),
        ),
    ],
)
def test_runner_preserves_checkpoint_supervision_errors(monkeypatch, tmp_path, error):
    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: FakeResultProvider())
    monkeypatch.setattr(runner, "prepare_call", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))

    with pytest.raises(type(error)) as caught:
        run_json(
            "prompt",
            provider="codex-cli",
            env={},
            process_chain=[],
            artifact_dir=tmp_path,
            call_label="supervised",
        )

    assert caught.value is error
    assert caught.value.checkpoint_path == error.checkpoint_path


def without_call_record(result):
    return {key: value for key, value in result.items() if key != ARC_LLM_CALL_RECORD_FIELD}


def test_call_record_is_not_added_to_provider_schema():
    schema = allow_arc_llm_call_record(
        {
            "type": "object",
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
            "additionalProperties": False,
        }
    )

    assert_strict_objects(schema)
    assert ARC_LLM_CALL_RECORD_FIELD not in schema["properties"]


def test_provider_schema_strips_arc_llm_call_record_and_required_entry():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok", ARC_LLM_CALL_RECORD_FIELD],
        "properties": {
            "ok": {"type": "boolean"},
            ARC_LLM_CALL_RECORD_FIELD: {"type": "object"},
        },
    }

    provider_schema = to_provider_json_schema(schema)

    assert provider_schema["required"] == ["ok"]
    assert ARC_LLM_CALL_RECORD_FIELD not in provider_schema["properties"]


def test_provider_schema_makes_nested_objects_strict():
    schema = {
        "type": "object",
        "required": ["payload"],
        "properties": {
            "payload": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"name": {"type": "string"}}},
                    }
                },
            }
        },
    }

    provider_schema = to_provider_json_schema(schema)

    assert provider_schema["additionalProperties"] is False
    assert provider_schema["properties"]["payload"]["additionalProperties"] is False
    item_schema = provider_schema["properties"]["payload"]["properties"]["items"]["items"]
    assert item_schema["additionalProperties"] is False


def test_provider_schema_preserves_explicit_empty_schema():
    assert to_provider_json_schema({}) == {}


def test_provider_schema_overrides_additional_properties_true():
    schema = {
        "type": "object",
        "additionalProperties": True,
        "properties": {"ok": {"type": "boolean"}},
    }

    provider_schema = to_provider_json_schema(schema)

    assert provider_schema["additionalProperties"] is False


def test_provider_schema_strips_call_record_from_required_without_properties():
    schema = {
        "type": "object",
        "required": ["ok", ARC_LLM_CALL_RECORD_FIELD],
    }

    provider_schema = to_provider_json_schema(schema)

    assert provider_schema["required"] == ["ok"]


def test_provider_schema_preserves_absent_schema():
    assert to_provider_json_schema(None) is None


def test_provider_schema_does_not_mutate_input_schema():
    schema = {
        "type": "object",
        "additionalProperties": True,
        "required": ["ok", ARC_LLM_CALL_RECORD_FIELD],
        "properties": {
            "ok": {"type": "boolean"},
            ARC_LLM_CALL_RECORD_FIELD: {"type": "object"},
            "payload": {"type": "object", "properties": {"name": {"type": "string"}}},
        },
    }
    original = deepcopy(schema)

    provider_schema = to_provider_json_schema(schema)

    assert schema == original
    assert provider_schema != schema


def test_provider_schema_treats_properties_without_type_as_strict_object():
    schema = {"properties": {"ok": {"type": "boolean"}}}

    provider_schema = to_provider_json_schema(schema)

    assert provider_schema["additionalProperties"] is False


def test_provider_schema_does_not_normalize_default_annotation_payloads():
    default_payload = {
        "properties": {"x": 1},
        "nested": {"type": "object", "properties": {"y": 2}},
    }
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "default": default_payload,
                "properties": {"name": {"type": "string"}},
            }
        },
    }

    provider_schema = to_provider_json_schema(schema)

    assert provider_schema["properties"]["payload"]["default"] == default_payload
    assert "additionalProperties" not in provider_schema["properties"]["payload"]["default"]
    assert "additionalProperties" not in provider_schema["properties"]["payload"]["default"]["nested"]


def test_provider_schema_makes_list_valued_items_strict():
    schema = {
        "type": "array",
        "items": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "object", "properties": {"ok": {"type": "boolean"}}},
        ],
    }

    provider_schema = to_provider_json_schema(schema)

    assert provider_schema["items"][0]["additionalProperties"] is False
    assert provider_schema["items"][1]["additionalProperties"] is False


def test_provider_schema_makes_additional_items_schema_strict():
    schema = {
        "type": "array",
        "items": [{"type": "string"}],
        "additionalItems": {"type": "object", "properties": {"name": {"type": "string"}}},
    }

    provider_schema = to_provider_json_schema(schema)

    assert provider_schema["additionalItems"]["additionalProperties"] is False


def test_warn_recovery_does_not_force_unknown_schema_when_validation_disabled():
    schema = {
        "type": "object",
        "required": ["foo"],
        "properties": {"foo": {"type": "string"}},
        "additionalProperties": False,
    }
    response = LLMProviderResponse(
        {},
        raw_output="plain text",
        structured_output=structured_metadata(
            severity="major",
            warnings=["natural language fallback"],
            raw_text="plain text",
            strategy="natural_language_fallback",
            provider_error_type="text",
        ),
    )

    result, structured = runner._recover_or_validate_json_output(  # noqa: SLF001
        {},
        schema=schema,
        validate_schema=False,
        output_recovery="warn",
        role_hint="proposer",
        response=response,
    )

    assert result == {}
    assert structured["mode"] == "recovered"
    assert structured["recovery_strategy"] == "natural_language_fallback"


def test_warn_validate_false_valid_object_no_structured_warning():
    schema = {
        "type": "object",
        "required": ["foo"],
        "properties": {"foo": {"type": "string"}},
        "additionalProperties": False,
    }
    payload = {"foo": "bar"}
    response = LLMProviderResponse(payload, raw_output='{"foo":"bar"}')

    result, structured = runner._recover_or_validate_json_output(  # noqa: SLF001
        payload,
        schema=schema,
        validate_schema=False,
        output_recovery="warn",
        role_hint="domain",
        response=response,
    )

    assert result == payload
    assert structured is None


def test_warn_validate_false_invalid_object_records_schema_warning():
    schema = {
        "type": "object",
        "required": ["foo"],
        "properties": {"foo": {"type": "string"}},
        "additionalProperties": False,
    }
    payload = {"bar": "extra"}
    response = LLMProviderResponse(payload, raw_output='{"bar":"extra"}')

    result, structured = runner._recover_or_validate_json_output(  # noqa: SLF001
        payload,
        schema=schema,
        validate_schema=False,
        output_recovery="warn",
        role_hint="domain",
        response=response,
    )

    assert result == payload
    assert structured["mode"] == "recovered"
    assert structured["severity"] == "minor"
    assert structured["recovery_strategy"] == "schema_warning_no_validation"
    assert "validate_schema=False allowed continuation" in "\n".join(structured["warnings"])


def test_valid_schema_output_unchanged_when_validation_enabled():
    schema = {
        "type": "object",
        "required": ["foo"],
        "properties": {"foo": {"type": "string"}},
        "additionalProperties": False,
    }
    payload = {"foo": "bar"}
    response = LLMProviderResponse(payload, raw_output='{"foo":"bar"}')

    result, structured = runner._recover_or_validate_json_output(  # noqa: SLF001
        payload,
        schema=schema,
        validate_schema=True,
        output_recovery="warn",
        role_hint="proposer",
        response=response,
    )

    assert result == {"foo": "bar"}
    assert structured is None


def test_stateful_run_json_self_heals_missing_session_on_native_id_update(monkeypatch, tmp_path):
    manager = LLMSessionManager(tmp_path / "sessions")
    provider = FakeResultProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: provider)
    original_update = manager.update_native_session_id
    calls = 0

    def flaky_update(key, native_session_id, *, allow_overwrite=False):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyError(f"unknown LLM session key: {key}")
        return original_update(key, native_session_id, allow_overwrite=allow_overwrite)

    monkeypatch.setattr(manager, "update_native_session_id", flaky_update)

    result = run_json(
        "prompt",
        provider="codex-cli",
        model="m",
        env={},
        process_chain=[],
        session_policy="stateful",
        session_manager=manager,
        session_key="idea_loops/loop/proposer/proposer_001",
        artifact_dir=tmp_path / "artifacts",
        idempotency_key="self-heal",
    )

    assert result["ok"] is True
    assert calls == 2
    assert manager.has_native_session("idea_loops/loop/proposer/proposer_001") is True


def assert_strict_objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object" or "object" in schema.get("type", []):
            assert schema.get("additionalProperties") is False
        for value in schema.values():
            assert_strict_objects(value)
    elif isinstance(schema, list):
        for item in schema:
            assert_strict_objects(item)


class FakeProvider:
    name = "codex-cli"

    def generate_json(self, prompt, *, schema=None, model=None):
        return {"prompt": prompt, "schema": schema, "model": model}

    def generate_text(self, prompt, *, model=None):
        return f"{model}:{prompt}"


class FakeResultProvider:
    name = "codex-cli"

    def generate_json_result(
        self,
        prompt,
        *,
        schema=None,
        model=None,
        session=None,
        session_policy="stateless",
        schema_cache_dir=None,
        artifact_dir=None,
    ):
        return LLMProviderResponse(
            {"ok": True, "model": model, "session_key": session.key if session else None},
            usage=LLMUsage(input_tokens=10, cached_input_tokens=8, output_tokens=2),
            native_session_id="native-123" if session_policy == "stateful" else None,
        )

    def generate_json(self, prompt, *, schema=None, model=None):
        return self.generate_json_result(prompt, schema=schema, model=model).value


class CapturingSchemaResultProvider:
    name = "codex-cli"

    def __init__(self):
        self.schema = None

    def generate_json_result(
        self,
        prompt,
        *,
        schema=None,
        model=None,
        session=None,
        session_policy="stateless",
        schema_cache_dir=None,
        artifact_dir=None,
    ):
        self.schema = deepcopy(schema)
        return LLMProviderResponse({"ok": True})


class FakeTextResultProvider:
    name = "claude-cli"

    def generate_text_result(
        self,
        prompt,
        *,
        model=None,
        session=None,
        session_policy="stateless",
        artifact_dir=None,
    ):
        return LLMProviderResponse(
            f"{model}:{prompt}",
            usage=LLMUsage(input_tokens=12, cached_input_tokens=9, output_tokens=3),
            native_session_id="native-text" if session_policy == "stateful" else None,
        )


class IdleTimeoutCapturingProvider:
    name = "codex-cli"

    def __init__(self):
        self.idle_timeouts = []

    def generate_text_result(
        self,
        prompt,
        *,
        model=None,
        session=None,
        session_policy="stateless",
        artifact_dir=None,
        idle_timeout_seconds=None,
    ):
        self.idle_timeouts.append(idle_timeout_seconds)
        return LLMProviderResponse("ok")


class PromptHashResultProvider:
    name = "codex-cli"

    def generate_json_result(
        self,
        prompt,
        *,
        schema=None,
        model=None,
        session=None,
        session_policy="stateless",
        schema_cache_dir=None,
        artifact_dir=None,
    ):
        return LLMProviderResponse(
            {"ok": True},
            native_session_id="native-123" if session_policy == "stateful" else None,
            prompt_sent_sha256=sha256_text(f"{prompt}\nprovider contract"),
        )


class InvalidThenValidResultProvider:
    name = "codex-cli"

    def __init__(self):
        self.attempts = 0

    def generate_json_result(
        self,
        prompt,
        *,
        schema=None,
        model=None,
        session=None,
        session_policy="stateless",
        schema_cache_dir=None,
        artifact_dir=None,
    ):
        self.attempts += 1
        return LLMProviderResponse(
            {"ok": "not-a-boolean"} if self.attempts == 1 else {"ok": True},
            usage=LLMUsage(input_tokens=10, cached_input_tokens=0, output_tokens=2),
            native_session_id="native-123" if session_policy == "stateful" else None,
        )


class RecoverableTextResultProvider:
    name = "claude-cli"

    def __init__(self):
        self.attempts = 0

    def generate_json_result(
        self,
        prompt,
        *,
        schema=None,
        model=None,
        session=None,
        session_policy="stateless",
        schema_cache_dir=None,
        artifact_dir=None,
        output_recovery="strict",
    ):
        self.attempts += 1
        return LLMProviderResponse(
            {},
            raw_output=json.dumps({"type": "result", "result": "Too short"}),
            structured_output={
                "schema_version": "arc.llm.structured_output.v1",
                "mode": "recovered",
                "severity": "major",
                "warnings": ["provider returned natural language"],
                "raw_text_excerpt": "Too short",
                "provider_error_type": None,
                "recovery_strategy": "natural_language_fallback",
            },
        )


class ServiceUnavailableThenOkProvider:
    name = "codex-cli"

    def __init__(self) -> None:
        self.attempts = 0

    def generate_json_result(
        self,
        prompt,
        *,
        schema=None,
        model=None,
        session=None,
        session_policy="stateless",
        schema_cache_dir=None,
        artifact_dir=None,
    ):
        self.attempts += 1
        if self.attempts == 1:
            return LLMProviderResponse("Service unavailable")
        return LLMProviderResponse({"ok": True})


class FatalBadRequestProvider:
    name = "codex-cli"

    def __init__(self) -> None:
        self.attempts = 0

    def generate_json_result(
        self,
        prompt,
        *,
        schema=None,
        model=None,
        session=None,
        session_policy="stateless",
        schema_cache_dir=None,
        artifact_dir=None,
    ):
        self.attempts += 1
        return LLMProviderResponse("HTTP 400 Bad Request: invalid schema")


class EmptyThenValidResultProvider:
    name = "codex-cli"

    def __init__(self):
        self.attempts = 0

    def generate_json_result(
        self,
        prompt,
        *,
        schema=None,
        model=None,
        session=None,
        session_policy="stateless",
        schema_cache_dir=None,
        artifact_dir=None,
    ):
        self.attempts += 1
        if self.attempts == 1:
            return LLMProviderResponse({}, raw_output="")
        return LLMProviderResponse({"ok": True})


class RichInvalidResultProvider:
    name = "codex-cli"

    def __init__(self):
        self.attempts = 0

    def generate_json_result(
        self,
        prompt,
        *,
        schema=None,
        model=None,
        session=None,
        session_policy="stateless",
        schema_cache_dir=None,
        artifact_dir=None,
    ):
        self.attempts += 1
        if self.attempts == 1:
            raw = "Detailed proposal text with enough scientific content to reformat into the required schema."
            return LLMProviderResponse({"notes": raw}, raw_output=raw)
        return LLMProviderResponse({"ok": True})


class FlakyJsonProvider:
    def __init__(self, *, name, failures_before_success=None, result=None):
        self.name = name
        self.failures_before_success = failures_before_success
        self.result = result or {"ok": True}
        self.attempts = 0

    def generate_json(self, prompt, *, schema=None, model=None):
        self.attempts += 1
        if self.failures_before_success is None or self.attempts <= self.failures_before_success:
            raise RuntimeError(f"{self.name} failed")
        return {**self.result, "model": model}


class InvalidJsonProvider:
    name = "codex-cli"

    def __init__(self):
        self.attempts = 0

    def generate_json(self, prompt, *, schema=None, model=None):
        self.attempts += 1
        return {"ok": "not-a-boolean"}


class FlakyTextProvider:
    def __init__(self, *, name, failures_before_success=None):
        self.name = name
        self.failures_before_success = failures_before_success
        self.attempts = 0

    def generate_text(self, prompt, *, model=None):
        self.attempts += 1
        if self.failures_before_success is None or self.attempts <= self.failures_before_success:
            raise RuntimeError(f"{self.name} failed")
        return f"{self.name}:{model}:{prompt}"


class NonRetryableProvider:
    name = "codex-cli"

    def __init__(self):
        self.attempts = 0

    def generate_json(self, prompt, *, schema=None, model=None):
        self.attempts += 1
        raise LLMWorkerError("invalid native session", retryable=False)


def test_resolve_llm_config_uses_host_and_default_model(tmp_path):
    config = resolve_llm_config(env={"ARC_AGENT_HOST": "codex"}, process_chain=[])
    assert config.provider == "codex-cli"
    assert config.model == "gpt-5.6-luna"
    assert config.host.host == "codex"


def test_run_json_uses_selected_provider_and_model(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: FakeProvider())

    result = run_json(
        "prompt",
        schema={"type": "object"},
        model="fast",
        provider="codex-cli",
        env={"ARC_AGENT_HOST": "codex"},
        process_chain=[],
    )

    assert result["prompt"] == ensure_runtime_progress_contract("prompt")
    assert result["schema"] == {"type": "object", "additionalProperties": False}
    assert result["model"] == "fast"
    assert result[ARC_LLM_CALL_RECORD_FIELD]["provider_used"] == "codex-cli"
    assert result[ARC_LLM_CALL_RECORD_FIELD]["model_used"] == "fast"


def test_run_json_passes_provider_safe_schema_but_preserves_local_validation(monkeypatch):
    provider = CapturingSchemaResultProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: provider)
    schema = {
        "type": "object",
        "required": ["ok"],
        "properties": {
            "ok": {"type": "boolean"},
            ARC_LLM_CALL_RECORD_FIELD: {"type": "object"},
        },
        "additionalProperties": False,
    }
    original = deepcopy(schema)

    result = run_json(
        "prompt",
        schema=schema,
        provider="codex-cli",
        model="m",
        env={},
        process_chain=[],
    )

    assert result["ok"] is True
    assert ARC_LLM_CALL_RECORD_FIELD in result
    assert provider.schema["properties"] == {"ok": {"type": "boolean"}}
    assert provider.schema["additionalProperties"] is False
    assert schema == original


def test_run_json_recovery_warn_valid_json_is_unchanged(monkeypatch):
    provider = CapturingSchemaResultProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: provider)
    schema = {
        "type": "object",
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
        "additionalProperties": False,
    }

    result = run_json(
        "prompt",
        schema=schema,
        provider="codex-cli",
        model="m",
        env={},
        process_chain=[],
        output_recovery="warn",
    )

    assert without_call_record(result) == {"ok": True}
    assert result[ARC_LLM_CALL_RECORD_FIELD].get("structured_output") is None


def test_run_json_does_not_replay_provider_failure_text(monkeypatch):
    provider = ServiceUnavailableThenOkProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: provider)

    with pytest.raises(LLMTaskError):
        run_json(
            "prompt",
            schema={
                "type": "object",
                "required": ["ok"],
                "properties": {"ok": {"type": "boolean"}},
                "additionalProperties": False,
            },
            provider="codex-cli",
            env={},
            process_chain=[],
            output_recovery="warn",
        )

    # One original provider response plus one formatter submission; the
    # original worker itself was not replayed.
    assert provider.attempts == 2


def test_run_json_warn_retries_short_provider_text_without_status_code_policy(monkeypatch):
    provider = FatalBadRequestProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: provider)

    def formatter_should_not_run(**kwargs):
        raise AssertionError("short provider text should retry before schema formatter")

    monkeypatch.setattr(runner, "format_to_schema_or_retry", formatter_should_not_run, raising=False)

    with pytest.raises(LLMTaskError, match="JSON output failed schema validation"):
        run_json(
            "prompt",
            schema={
                "type": "object",
                "required": ["ok"],
                "properties": {"ok": {"type": "boolean"}},
                "additionalProperties": False,
            },
            provider="codex-cli",
            env={},
            process_chain=[],
            output_recovery="warn",
        )

    assert provider.attempts == 1


def test_run_json_warn_retries_empty_schema_failure_before_formatter(monkeypatch):
    provider = EmptyThenValidResultProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: provider)

    def formatter_should_not_run(**kwargs):
        raise AssertionError("empty output should retry before schema formatter")

    monkeypatch.setattr(runner, "format_to_schema_or_retry", formatter_should_not_run, raising=False)

    with pytest.raises(LLMTaskError, match="empty or low-content output"):
        run_json(
            "prompt",
            schema={
                "type": "object",
                "required": ["ok"],
                "properties": {"ok": {"type": "boolean"}},
                "additionalProperties": False,
            },
            provider="codex-cli",
            env={},
            process_chain=[],
            output_recovery="warn",
        )

    assert provider.attempts == 1


def test_run_json_warn_uses_schema_formatter_for_rich_schema_failure(monkeypatch):
    provider = RichInvalidResultProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: provider)

    def fake_formatter(**kwargs):
        return SimpleNamespace(
            action="format",
            value={"ok": True},
            reason="rich text had enough information",
            structured_output=structured_metadata(
                severity="minor",
                warnings=["formatted"],
                raw_text=kwargs["raw_text"],
                strategy="schema_formatter",
            ),
        )

    monkeypatch.setattr(runner, "format_to_schema_or_retry", fake_formatter, raising=False)

    result = run_json(
        "prompt",
        schema={
            "type": "object",
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
            "additionalProperties": False,
        },
        provider="codex-cli",
        env={},
        process_chain=[],
        output_recovery="warn",
    )

    assert provider.attempts == 1
    assert result["ok"] is True
    assert result[ARC_LLM_CALL_RECORD_FIELD]["structured_output"]["recovery_strategy"] == "schema_formatter"


def test_run_json_warn_can_disable_schema_formatter(monkeypatch):
    provider = RichInvalidResultProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: provider)

    def formatter_should_not_run(**kwargs):
        raise AssertionError("schema formatter disabled")

    monkeypatch.setattr(runner, "format_to_schema_or_retry", formatter_should_not_run, raising=False)

    with pytest.raises(LLMTaskError, match="schema formatter is disabled"):
        run_json(
            "prompt",
            schema={
                "type": "object",
                "required": ["ok"],
                "properties": {"ok": {"type": "boolean"}},
                "additionalProperties": False,
            },
            provider="codex-cli",
            env={},
            process_chain=[],
            output_recovery="warn",
            schema_formatter_enabled=False,
        )

    assert provider.attempts == 1


def test_run_json_warn_does_not_replay_when_schema_formatter_requests_retry(monkeypatch):
    provider = RichInvalidResultProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: provider)

    def fake_formatter(**kwargs):
        return SimpleNamespace(
            action="retry",
            value=None,
            reason="formatter judged source unrecoverable",
            structured_output=structured_metadata(
                severity="major",
                warnings=["retry requested"],
                raw_text=kwargs["raw_text"],
                strategy="schema_formatter_retry",
            ),
        )

    monkeypatch.setattr(runner, "format_to_schema_or_retry", fake_formatter, raising=False)

    with pytest.raises(LLMTaskError, match="could not repair output"):
        run_json(
            "prompt",
            schema={
                "type": "object",
                "required": ["ok"],
                "properties": {"ok": {"type": "boolean"}},
                "additionalProperties": False,
            },
            provider="codex-cli",
            env={},
            process_chain=[],
            output_recovery="warn",
        )

    assert provider.attempts == 1


def test_run_json_recovery_warn_retries_low_content_natural_language(monkeypatch):
    provider = RecoverableTextResultProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: provider)
    schema = {
        "type": "object",
        "required": ["title", "idea_summary", "motivation", "novelty_checks", "calculation_plan", "validation_checks", "risks"],
        "properties": {
            "title": {"type": "string"},
            "idea_summary": {"type": "string"},
            "motivation": {"type": "string"},
            "novelty_checks": {"type": "array", "items": {"type": "string"}},
            "calculation_plan": {"type": "string"},
            "validation_checks": {"type": "array", "items": {"type": "string"}},
            "risks": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }

    with pytest.raises(LLMTaskError, match="empty or low-content output"):
        run_json(
            "prompt",
            schema=schema,
            provider="claude-cli",
            model="deepseek-v4-flash",
            env={},
            process_chain=[],
            output_recovery="warn",
            role_hint="proposer",
        )

    assert provider.attempts == 1


def test_low_content_detection_counts_unicode_letters() -> None:
    assert runner._is_low_content_source("这是一个包含充分信息的中文模型响应") is False  # noqa: SLF001
    assert runner._is_low_content_source("错误") is True  # noqa: SLF001


def test_schema_recovery_prefers_full_model_output_over_excerpt() -> None:
    response = LLMProviderResponse(
        {},
        raw_output='{"result":"wrapper output"}',
        raw_model_output="完整模型输出，包含足够的上下文与结论。",
        structured_output={"raw_text_excerpt": "short excerpt"},
    )
    source = runner._schema_recovery_source_text(  # noqa: SLF001
        {}, response=response, provider_metadata=response.structured_output
    )
    assert source == "完整模型输出，包含足够的上下文与结论。"


def test_run_json_without_schema_passes_none_and_adds_call_record(monkeypatch):
    provider = CapturingSchemaResultProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: provider)

    result = run_json(
        "prompt",
        schema=None,
        provider="codex-cli",
        model="m",
        env={},
        process_chain=[],
    )

    assert result["ok"] is True
    assert provider.schema is None
    assert ARC_LLM_CALL_RECORD_FIELD in result


def test_call_record_v3_requires_warnings_and_existing_provider_emits_empty_list(monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: FakeProvider())

    result = run_json("prompt", provider="codex-cli", env={}, process_chain=[])
    record = result[ARC_LLM_CALL_RECORD_FIELD]

    assert ARC_LLM_CALL_RECORD_SCHEMA_VERSION == "arc.llm.call_record.v4"
    assert "warnings" in ARC_LLM_CALL_RECORD_SCHEMA["required"]
    assert record["warnings"] == []


def test_kimi_config_warnings_propagate_to_call_record(monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: FakeProvider())

    result = run_json("prompt", provider="kimi-code-cli", model_tier="high", env={}, process_chain=[])
    warnings = result[ARC_LLM_CALL_RECORD_FIELD]["warnings"]

    assert "kimi_code_cli.experimental" in warnings
    assert "kimi_code_cli.provider_side_persistence" in warnings
    assert "kimi_code_cli.inherits_user_configuration" in warnings
    assert "kimi_code_cli.model_tier_unmapped" in warnings


def test_non_retryable_worker_error_stops_after_first_attempt(monkeypatch):
    provider = NonRetryableProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: provider)

    with pytest.raises(LLMTaskError, match="invalid native session"):
        run_json("prompt", provider="codex-cli", env={}, process_chain=[])

    assert provider.attempts == 1


def test_worker_error_is_non_retryable_by_default():
    assert LLMWorkerError("transient provider failure").retryable is False
    assert LLMWorkerError("transient provider failure").abort_batch is False


def test_run_json_stateful_records_session_usage_and_call_record(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: FakeResultProvider())
    manager = LLMSessionManager(tmp_path / "sessions")

    result = run_json(
        "prompt",
        schema={"type": "object"},
        model="fast",
        provider="codex-cli",
        env={"ARC_AGENT_HOST": "codex"},
        process_chain=[],
        session_policy="stateful",
        session_manager=manager,
        session_key="scope/proposer/proposer_001",
        call_label="round_001/proposer_001",
        artifact_dir=tmp_path / "artifacts",
        idempotency_key="usage-call",
    )

    call_record = result[ARC_LLM_CALL_RECORD_FIELD]
    assert result["session_key"] == "scope/proposer/proposer_001"
    assert call_record["session_policy"] == "stateful"
    assert call_record["session_key"] == "scope/proposer/proposer_001"
    assert call_record["native_session_id"] == "native-123"
    assert call_record["usage"]["cached_input_ratio"] == 0.8
    assert manager.turn_count("scope/proposer/proposer_001") == 1


def test_run_json_stateful_records_static_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: FakeResultProvider())
    manager = LLMSessionManager(tmp_path / "sessions")

    result = run_json(
        "prompt",
        schema={"type": "object"},
        model="fast",
        provider="codex-cli",
        env={"ARC_AGENT_HOST": "codex"},
        process_chain=[],
        session_policy="stateful",
        session_manager=manager,
        session_key="scope/proposer/proposer_001",
        call_label="round_001/proposer_001",
        artifact_dir=tmp_path / "artifacts",
        idempotency_key="prefix-call",
        static_prefix="stable prefix",
    )

    line = (tmp_path / "sessions" / "calls.jsonl").read_text(encoding="utf-8").splitlines()[0]
    call = json.loads(line)

    effective_prefix = "stable prefix"
    assert result[ARC_LLM_CALL_RECORD_FIELD]["static_prefix_sha256"] == sha256_text(effective_prefix)
    assert call["static_prefix_sha256"] == sha256_text(effective_prefix)


def test_run_json_records_provider_prompt_hash_when_prompt_is_rewritten(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: PromptHashResultProvider())
    manager = LLMSessionManager(tmp_path / "sessions")

    result = run_json(
        "prompt",
        schema={"type": "object"},
        model="fast",
        provider="codex-cli",
        env={},
        process_chain=[],
        session_policy="stateful",
        session_manager=manager,
        session_key="scope/proposer/proposer_001",
        call_label="round_001/proposer_001",
        artifact_dir=tmp_path / "artifacts",
        idempotency_key="prompt-hash-call",
    )

    expected = sha256_text(f"{ensure_runtime_progress_contract('prompt')}\nprovider contract")
    call = json.loads((tmp_path / "sessions" / "calls.jsonl").read_text(encoding="utf-8"))

    assert result[ARC_LLM_CALL_RECORD_FIELD]["prompt_sha256"] == expected
    assert call["prompt_sha256"] == expected


def test_run_json_stateful_does_not_retry_after_invalid_output(tmp_path, monkeypatch):
    invalid = InvalidThenValidResultProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: invalid)
    manager = LLMSessionManager(tmp_path / "sessions")

    with pytest.raises(runner.LLMTaskError, match="JSON output failed schema validation"):
        run_json(
            "prompt",
            provider="codex-cli",
            schema={
                "type": "object",
                "required": ["ok"],
                "properties": {"ok": {"type": "boolean"}},
                "additionalProperties": False,
            },
            env={},
            process_chain=[],
            session_policy="stateful",
            session_manager=manager,
            session_key="scope/proposer/proposer_001",
            artifact_dir=tmp_path / "artifacts",
            idempotency_key="invalid-output-call",
        )

    assert invalid.attempts == 1
    assert manager.turn_count("scope/proposer/proposer_001") == 1


def test_run_text_result_returns_usage_and_records_stateful_call(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: FakeTextResultProvider())
    manager = LLMSessionManager(tmp_path / "sessions")

    outcome = run_text_result(
        "prompt",
        provider="claude-cli",
        model="m",
        env={},
        process_chain=[],
        session_policy="stateful",
        session_manager=manager,
        session_key="scope/reviewer/reviewer_001",
        static_prefix="stable text",
        artifact_dir=tmp_path / "artifacts",
        idempotency_key="text-call",
    )

    call = json.loads((tmp_path / "sessions" / "calls.jsonl").read_text(encoding="utf-8"))

    assert outcome.value == f"m:{ensure_runtime_progress_contract('prompt')}"
    assert outcome.usage.cached_input_ratio == 0.75
    assert outcome.static_prefix_sha256 == sha256_text("stable text")
    assert call["usage"]["cached_input_ratio"] == 0.75


def test_run_json_stateful_requires_provider_result_support(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: FakeProvider())

    with pytest.raises(runner.LLMTaskError, match="does not support stateful sessions"):
        run_json(
            "prompt",
            provider="codex-cli",
            env={},
            process_chain=[],
            session_policy="stateful",
            session_manager=LLMSessionManager(tmp_path / "sessions"),
            session_key="scope/proposer/proposer_001",
            artifact_dir=tmp_path / "artifacts",
            idempotency_key="unsupported-provider",
        )


def test_run_json_stateful_requires_session_key_and_manager():
    with pytest.raises(ValueError, match="requires session_manager and session_key"):
        run_json("prompt", provider="codex-cli", env={}, process_chain=[], session_policy="stateful")


def test_auto_provider_rejects_exact_model():
    with pytest.raises(ValueError, match="Exact model requires explicit provider"):
        run_json("prompt", provider="auto", model="gpt-5.5", env={}, process_chain=[])


def test_resolve_llm_config_rejects_auto_provider_with_exact_model():
    with pytest.raises(ValueError, match="Exact model requires explicit provider"):
        resolve_llm_config(provider="auto", model="gpt-5.5", env={}, process_chain=[])


def test_env_model_does_not_override_model_tier(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: FakeProvider())

    result = run_json(
        "prompt",
        schema={"type": "object"},
        model_tier="high",
        env={"ARC_AGENT_HOST": "codex", "ARC_CODEX_MODEL": "fast"},
        process_chain=[],
    )

    assert result["model"] == "gpt-5.6-sol"


def test_run_json_uses_model_tier_when_exact_model_is_not_set(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: FakeProvider())

    result = run_json(
        "prompt",
        schema={"type": "object"},
        model_tier="high",
        env={"ARC_AGENT_HOST": "codex"},
        process_chain=[],
    )

    assert result["model"] == "gpt-5.6-sol"


def test_run_json_passes_model_tier_reasoning_effort_to_codex(monkeypatch):
    captured = {}

    def fake_select_provider(provider, **kwargs):
        captured.update(kwargs["env"])
        return FakeProvider()

    monkeypatch.setattr(runner, "select_provider", fake_select_provider)

    run_json(
        "prompt",
        schema={"type": "object"},
        model_tier="max",
        env={"ARC_AGENT_HOST": "codex"},
        process_chain=[],
    )

    assert captured["ARC_CODEX_REASONING_EFFORT"] == "max"


def test_explicit_codex_reasoning_effort_overrides_model_tier_default(monkeypatch):
    captured = {}

    def fake_select_provider(provider, **kwargs):
        captured.update(kwargs["env"])
        return FakeProvider()

    monkeypatch.setattr(runner, "select_provider", fake_select_provider)

    run_json(
        "prompt",
        schema={"type": "object"},
        model_tier="low",
        env={"ARC_AGENT_HOST": "codex", "ARC_CODEX_REASONING_EFFORT": "high"},
        process_chain=[],
    )

    assert captured["ARC_CODEX_REASONING_EFFORT"] == "high"


def test_run_json_validates_provider_output_against_schema(monkeypatch):
    invalid = InvalidJsonProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: invalid)

    with pytest.raises(runner.LLMTaskError, match="JSON output failed schema validation"):
        run_json(
            "prompt",
            provider="codex-cli",
            schema={
                "type": "object",
                "required": ["ok"],
                "properties": {"ok": {"type": "boolean"}},
                "additionalProperties": False,
            },
            env={},
            process_chain=[],
        )

    assert invalid.attempts == 1


def test_run_text_uses_selected_provider_and_model(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: FakeProvider())

    result = run_text("prompt", env={"ARC_AGENT_HOST": "codex"}, process_chain=[])

    assert result == f"gpt-5.6-luna:{ensure_runtime_progress_contract('prompt')}"


@pytest.mark.parametrize(
    ("provider", "expected_status", "error_type"),
    [
        (FakeProvider(), "success", None),
        (FlakyTextProvider(name="codex-cli"), "failed", RuntimeError),
    ],
)
def test_controller_finalizes_paper_overlay_after_success_and_failure(
    monkeypatch, provider, expected_status, error_type
):
    finalized = []
    monkeypatch.setattr(runner, "select_provider", lambda _name, **_kwargs: provider)
    monkeypatch.setattr(
        runner,
        "_finalize_paper_worker_call",
        lambda _env, **kwargs: finalized.append(kwargs),
    )

    if error_type is None:
        run_text("prompt", provider="codex-cli", env={}, process_chain=[], call_label="call-1")
    else:
        with pytest.raises(error_type):
            run_text("prompt", provider="codex-cli", env={}, process_chain=[], call_label="call-1")

    assert finalized == [{"status": expected_status, "worker_id": None, "call_id": "call-1"}]


def test_controller_finalizes_paper_overlay_after_cancel(monkeypatch):
    class CancelledProvider:
        name = "codex-cli"

        def generate_text(self, _prompt, *, model=None):
            raise LLMWorkerCancelled("cancelled")

    finalized = []
    monkeypatch.setattr(runner, "select_provider", lambda _name, **_kwargs: CancelledProvider())
    monkeypatch.setattr(
        runner,
        "_finalize_paper_worker_call",
        lambda _env, **kwargs: finalized.append(kwargs),
    )

    with pytest.raises(LLMWorkerCancelled):
        run_text("prompt", provider="codex-cli", env={}, process_chain=[], call_label="call-2")

    assert finalized == [{"status": "cancelled", "worker_id": None, "call_id": "call-2"}]


@pytest.mark.parametrize(
    ("explicit", "env", "expected"),
    [
        (7, {"ARC_CODEX_IDLE_TIMEOUT_SECONDS": "11", "ARC_LLM_IDLE_TIMEOUT_SECONDS": "22"}, 7),
        (None, {"ARC_CODEX_IDLE_TIMEOUT_SECONDS": "11", "ARC_LLM_IDLE_TIMEOUT_SECONDS": "22"}, 11),
        (None, {"ARC_LLM_IDLE_TIMEOUT_SECONDS": "22"}, 22),
        (None, {}, 1800),
    ],
)
def test_runner_resolves_idle_timeout_before_passing_it_to_provider(
    monkeypatch, explicit, env, expected
):
    provider = IdleTimeoutCapturingProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: provider)

    assert run_text(
        "prompt",
        provider="codex-cli",
        env=env,
        process_chain=[],
        idle_timeout_seconds=explicit,
    ) == "ok"
    assert provider.idle_timeouts == [expected]


@pytest.mark.parametrize(
    "env",
    [
        {"ARC_LLM_IDLE_TIMEOUT_SECONDS": "not-a-number"},
        {"ARC_LLM_IDLE_TIMEOUT_SECONDS": "nan"},
        {"ARC_LLM_IDLE_TIMEOUT_SECONDS": "inf"},
        {"ARC_CODEX_IDLE_TIMEOUT_SECONDS": "not-a-number"},
        {"ARC_LLM_TIMEOUT_SECONDS": "60"},
        {"ARC_CODEX_TIMEOUT_SECONDS": "60"},
    ],
)
def test_invalid_idle_timeout_fails_not_submitted_without_provider_or_checkpoint(
    tmp_path, monkeypatch, env
):
    selected = []
    monkeypatch.setattr(
        runner,
        "select_provider",
        lambda provider_name, **kwargs: selected.append(provider_name),
    )
    artifact_dir = tmp_path / "artifacts"

    with pytest.raises(LLMConfigurationError) as caught:
        run_text(
            "prompt",
            provider="codex-cli",
            env=env,
            process_chain=[],
            artifact_dir=artifact_dir,
            call_label="worker",
        )

    assert caught.value.category == LLMFailureCategory.INVALID_REQUEST
    assert caught.value.submission_state == LLMSubmissionState.NOT_SUBMITTED
    assert selected == []
    assert not artifact_dir.exists()


def test_run_text_does_not_replay_unknown_provider_failure(tmp_path, monkeypatch):
    flaky = FlakyTextProvider(name="codex-cli", failures_before_success=2)
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: flaky)

    with pytest.raises(LLMTaskError):
        run_text(
            "prompt",
            env={"ARC_AGENT_HOST": "codex"},
            process_chain=[],
        )

    assert flaky.attempts == 1


def test_run_text_does_not_sleep_to_replay_unknown_failure(monkeypatch):
    flaky = FlakyTextProvider(name="codex-cli", failures_before_success=2)
    sleep_calls = []
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: flaky)
    monkeypatch.setattr("time.sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(LLMTaskError):
        run_text("prompt", provider="codex-cli", env={}, process_chain=[])

    assert sleep_calls == []
    assert flaky.attempts == 1


def test_run_json_does_not_replay_unknown_provider_failure(tmp_path, monkeypatch):
    flaky = FlakyJsonProvider(name="codex-cli", failures_before_success=2, result={"provider": "codex-cli"})
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: flaky)

    with pytest.raises(LLMTaskError):
        run_json(
            "prompt",
            env={"ARC_AGENT_HOST": "codex"},
            process_chain=[],
        )

    assert flaky.attempts == 1


def test_run_json_auto_does_not_replay_selected_provider(monkeypatch):
    codex = FlakyJsonProvider(name="codex-cli")
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: codex)

    with pytest.raises(RuntimeError, match="LLM task failed after 1 attempt\\(s\\) across 1 provider\\(s\\)"):
        run_json(
            "prompt",
            env={"ARC_AGENT_HOST": "codex"},
            process_chain=[],
        )

    assert codex.attempts == 1


def test_run_json_explicit_provider_does_not_replay_without_proof(tmp_path, monkeypatch):
    codex = FlakyJsonProvider(name="codex-cli")
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: codex)

    with pytest.raises(RuntimeError, match="LLM task failed after 1 attempt\\(s\\) across 1 provider\\(s\\)"):
        run_json(
            "prompt",
            provider="codex-cli",
            env={"ARC_AGENT_HOST": "codex"},
            process_chain=[],
        )

    assert codex.attempts == 1


def test_legacy_session_without_capability_metadata_resumes_with_paper_cli_disabled(tmp_path):
    manager = LLMSessionManager(tmp_path / "sessions")
    legacy_env = {"ARC_AGENT_HOST": "codex"}
    legacy_fp = runtime_fingerprint(
        provider="codex-cli",
        model="m",
        model_tier=None,
        env=legacy_env,
    )
    manager.get_or_create(
        key="legacy",
        provider="codex-cli",
        model="m",
        runtime_fingerprint=legacy_fp,
    )

    env, metadata = runner._runtime_compatibility_policy(
        {**legacy_env, "ARC_PAPER_CLI_ACCESS": "full"},
        session_policy="stateful",
        session_manager=manager,
        session_key="legacy",
        session_metadata=None,
        artifact_dir=None,
        idempotency_key=None,
    )
    assert env["ARC_PAPER_CLI_ACCESS"] == "none"
    assert metadata["arc_runtime_capabilities"]["arc_paper_cli_access"] == "none"
    assert runtime_fingerprint(
        provider="codex-cli", model="m", model_tier=None, env=env
    ) == legacy_fp


def test_legacy_checkpoint_without_capability_metadata_resumes_with_paper_cli_disabled(tmp_path):
    key = "legacy-call"
    checkpoint_dir = tmp_path / "call-checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / f"idempotency-{sha256_text(key)}.json"
    checkpoint.write_text('{"schema_version":"arc.llm.call_checkpoint.v2"}\n', encoding="utf-8")

    env, _metadata = runner._runtime_compatibility_policy(
        {"ARC_PAPER_CLI_ACCESS": "full"},
        session_policy="stateless",
        session_manager=None,
        session_key=None,
        session_metadata=None,
        artifact_dir=tmp_path,
        idempotency_key=key,
    )

    assert env["ARC_PAPER_CLI_ACCESS"] == "none"


def test_new_call_constructs_shared_paper_overlay_before_submission(tmp_path):
    run_root = tmp_path / "run"
    artifact_dir = run_root / "loops" / "loop_1" / "rounds" / "round_001"
    artifact_dir.mkdir(parents=True)
    (run_root / "config.json").write_text("{}\n", encoding="utf-8")
    (run_root / "manifest.json").write_text("{}\n", encoding="utf-8")
    base_cache = tmp_path / "global-paper-cache"

    env, metadata = runner._runtime_compatibility_policy(
        {
            "ARC_PAPER_CLI_ACCESS": "full",
            "ARC_PAPER_CACHE": str(base_cache),
            "ARC_CODEX_ADD_DIRS": '["/unrelated"]',
        },
        session_policy="stateless",
        session_manager=None,
        session_key=None,
        session_metadata=None,
        artifact_dir=artifact_dir,
        idempotency_key=None,
    )
    runner._configure_paper_worker_session(
        env, artifact_dir=artifact_dir, session_manager=None
    )
    metadata["arc_runtime_capabilities"] = runner._runtime_capabilities(env)

    overlay = run_root / "paper-cache-overlay"
    state = overlay / ".arc-paper-worker"
    assert env["ARC_PAPER_WORKER_BASE_CACHE"] == str(base_cache)
    assert env["ARC_PAPER_CACHE"] == str(overlay)
    assert env["ARC_PAPER_WORKER_SESSION_DIR"] == str(run_root)
    assert env["ARC_PAPER_WORKER_TOMBSTONE_DIR"] == str(state / "tombstones")
    assert env["ARC_PAPER_WORKER_SESSION_ID"].startswith("arc-llm-")
    assert env["ARC_LLM_WORKER_CONTEXT"] == "true"
    assert Path(env["ARC_PAPER_WORKER_GUARD"]).is_file()
    assert env["ARC_PAPER_WORKER_TOKEN"]
    guard = json.loads(Path(env["ARC_PAPER_WORKER_GUARD"]).read_text(encoding="utf-8"))
    assert guard["schema_version"] == "arc.paper.worker-guard.v1"
    assert guard["token_sha256"] == sha256_text(env["ARC_PAPER_WORKER_TOKEN"])
    controller_token = runner._PAPER_CONTROLLER_SESSIONS[env["ARC_PAPER_WORKER_SESSION_ID"]]["controller_token"]
    assert controller_token not in json.dumps(guard)
    controller_guard_path = Path(
        runner._PAPER_CONTROLLER_SESSIONS[env["ARC_PAPER_WORKER_SESSION_ID"]]["controller_guard"]
    )
    controller_guard = json.loads(controller_guard_path.read_text(encoding="utf-8"))
    assert controller_guard == {
        "schema_version": "arc.paper.controller-guard.v1",
        "session_id": env["ARC_PAPER_WORKER_SESSION_ID"],
        "run_root": str(run_root),
        "base_root": str(base_cache),
        "token_sha256": sha256_text(controller_token),
    }
    assert controller_guard_path.stat().st_mode & 0o077 == 0
    assert state.is_dir()
    assert env["ARC_CODEX_SANDBOX"] == "workspace-write"
    assert env["ARC_CODEX_WORK_DIR"] == str(run_root)
    assert "ARC_CODEX_ADD_DIRS" not in env
    assert metadata["arc_runtime_capabilities"]["arc_paper_cli_access"] == "full"


def test_stateful_calls_share_paper_overlay_across_artifact_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: FakeResultProvider())
    manager = LLMSessionManager(tmp_path / "sessions")
    env = {
        "ARC_PAPER_CLI_ACCESS": "full",
        "ARC_PAPER_CACHE": str(tmp_path / "global-paper-cache"),
    }

    for index in (1, 2):
        result = run_json(
            f"prompt {index}",
            schema={"type": "object"},
            model="fast",
            provider="codex-cli",
            env=env,
            process_chain=[],
            session_policy="stateful",
            session_manager=manager,
            session_key="chapter/companion",
            call_label=f"segment-{index}",
            artifact_dir=tmp_path / "artifacts" / f"segment-{index}",
            idempotency_key=f"segment-{index}",
        )

        assert result["ok"] is True

    calls = [
        json.loads(line)
        for line in (manager.root / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manager.turn_count("chapter/companion") == 2
    assert calls[0]["runtime_fingerprint"] == calls[1]["runtime_fingerprint"]
    assert (manager.root / "paper-cache-overlay").is_dir()
    assert not (tmp_path / "artifacts" / "segment-1" / "paper-cache-overlay").exists()
    assert not (tmp_path / "artifacts" / "segment-2" / "paper-cache-overlay").exists()


def test_stateful_rotated_unused_generation_rebinds_runtime_before_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: FakeResultProvider())
    manager = LLMSessionManager(tmp_path / "sessions")
    manager.get_or_create(
        key="chapter/companion",
        provider="codex-cli",
        model="fast",
        runtime_fingerprint="pre-fix-runtime",
        metadata={
            "arc_runtime_capabilities": {
                "arc_paper_cli_access": "full",
                "inherit_host_tools": False,
            }
        },
    )
    manager.rotate("chapter/companion", reason="restart generation")

    result = run_json(
        "prompt",
        schema={"type": "object"},
        model="fast",
        provider="codex-cli",
        env={
            "ARC_PAPER_CLI_ACCESS": "full",
            "ARC_PAPER_CACHE": str(tmp_path / "global-paper-cache"),
        },
        process_chain=[],
        session_policy="stateful",
        session_manager=manager,
        session_key="chapter/companion",
        artifact_dir=tmp_path / "artifacts" / "segment-2",
        idempotency_key="segment-2",
    )

    rebound = manager.get_existing("chapter/companion")
    assert result["ok"] is True
    assert rebound is not None
    assert rebound.generation == 2
    assert rebound.runtime_fingerprint != "pre-fix-runtime"
    assert rebound.native_session_id == "native-123"
    assert manager.turn_count("chapter/companion", generation=2) == 1


def test_new_call_without_artifact_or_session_root_uses_managed_tmp_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_LLM_TMP_DIR", str(tmp_path / "llm-tmp"))
    env, metadata = runner._runtime_compatibility_policy(
        {"ARC_PAPER_CLI_ACCESS": "full", "ARC_LLM_TMP_DIR": str(tmp_path / "llm-tmp")},
        session_policy="stateless",
        session_manager=None,
        session_key=None,
        session_metadata=None,
        artifact_dir=None,
        idempotency_key=None,
    )
    runner._configure_paper_worker_session(env, artifact_dir=None, session_manager=None)
    metadata["arc_runtime_capabilities"] = runner._runtime_capabilities(env)

    assert env["ARC_PAPER_CLI_ACCESS"] == "full"
    assert "paper-worker-isolation/paper-cache-overlay" in env["ARC_PAPER_CACHE"]
    assert metadata["arc_runtime_capabilities"]["arc_paper_cli_access"] == "full"


def test_disabled_paper_cli_hides_inherited_global_cache(tmp_path):
    artifact_dir = tmp_path / "isolated-stage"
    global_cache = tmp_path / "global-paper-cache"
    env = {
        "ARC_PAPER_CLI_ACCESS": "none",
        "ARC_PAPER_CACHE": str(global_cache),
        "ARC_PAPER_WORKER_BASE_CACHE": str(global_cache),
        "ARC_PAPER_WORKER_GUARD": str(tmp_path / "forged-guard"),
        "ARC_PAPER_WORKER_TOKEN": "forged",
    }

    runner._configure_paper_worker_session(
        env, artifact_dir=artifact_dir, session_manager=None
    )

    assert env["ARC_PAPER_CACHE"] == str(artifact_dir / "paper-cache-disabled")
    assert env["ARC_LLM_WORKER_CONTEXT"] == "true"
    assert "ARC_PAPER_WORKER_BASE_CACHE" not in env
    assert "ARC_PAPER_WORKER_SESSION_DIR" not in env
    assert "ARC_PAPER_WORKER_GUARD" not in env
    assert "ARC_PAPER_WORKER_TOKEN" not in env


def test_controller_finalizer_uses_secret_not_exposed_to_worker(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    env = {
        "ARC_PAPER_CLI_ACCESS": "full",
        "ARC_PAPER_CACHE": str(tmp_path / "base"),
        "ARC_LLM_CACHE": str(tmp_path / "llm-cache"),
    }
    runner._configure_paper_worker_session(env, artifact_dir=run_root, session_manager=None)
    staged = Path(env["ARC_PAPER_CACHE"]) / "sources" / "paper.json"
    staged.parent.mkdir(parents=True)
    staged.write_text("{}\n", encoding="utf-8")
    captured = {}

    def fake_run(command, **kwargs):
        captured.update({"command": command, **kwargs})
        return SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    runner._finalize_paper_worker_call(
        env, status="failed", worker_id="worker-1", call_id="call-1"
    )

    assert captured["command"][1:4] == ["-m", "arc_paper.worker_controller", "finalize"]
    assert captured["command"][-2:] == ["--status", "failed"]
    controller_env = captured["env"]
    assert controller_env["ARC_PAPER_CONTROLLER_MODE"] == "trusted"
    assert controller_env["ARC_PAPER_CONTROLLER_TOKEN"]
    assert controller_env["ARC_PAPER_CONTROLLER_TOKEN"] != env["ARC_PAPER_WORKER_TOKEN"]
    assert "ARC_PAPER_WORKER_TOKEN" not in controller_env
