import json
from copy import deepcopy

import pytest

from arc_llm.call_record import ARC_LLM_CALL_RECORD_FIELD, allow_arc_llm_call_record
from arc_llm.json_schema import to_provider_json_schema
from arc_llm.schema_cache import sha256_text
from arc_llm.sessions import LLMSessionManager
from arc_llm.structured_recovery import structured_metadata
from arc_llm.usage import LLMProviderResponse, LLMUsage
from arc_llm import runner
from arc_llm.runner import LLMTaskError, resolve_llm_config, run_json, run_text, run_text_result


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)


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
        return LLMProviderResponse(
            {},
            raw_output=json.dumps({"type": "result", "result": "Recovered idea text"}),
            structured_output={
                "schema_version": "arc.llm.structured_output.v1",
                "mode": "recovered",
                "severity": "major",
                "warnings": ["provider returned natural language"],
                "raw_text_excerpt": "Recovered idea text",
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


def test_resolve_llm_config_uses_host_and_default_model(tmp_path):
    config = resolve_llm_config(env={"ARC_AGENT_HOST": "codex"}, process_chain=[])
    assert config.provider == "codex-cli"
    assert config.model == "gpt-5.4"
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

    assert result["prompt"] == "prompt"
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


def test_run_json_retries_provider_failure_text_before_recovery(monkeypatch):
    provider = ServiceUnavailableThenOkProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: provider)

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

    assert provider.attempts == 2
    assert result["ok"] is True


def test_run_json_fails_fast_for_fatal_provider_failure_text(monkeypatch):
    provider = FatalBadRequestProvider()
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: provider)

    with pytest.raises(LLMTaskError, match="fatal provider failure text"):
        run_json(
            "prompt",
            schema={"type": "object"},
            provider="codex-cli",
            env={},
            process_chain=[],
            output_recovery="warn",
        )

    assert provider.attempts == 1


def test_run_json_recovery_warn_builds_known_proposer_fallback(monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider_name, **kwargs: RecoverableTextResultProvider())
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

    result = run_json(
        "prompt",
        schema=schema,
        provider="claude-cli",
        model="deepseek-v4-flash",
        env={},
        process_chain=[],
        output_recovery="warn",
        role_hint="proposer",
    )

    assert result["title"] == "Recovered idea text"
    assert result["idea_summary"] == "Recovered idea text"
    assert result["risks"]
    structured = result[ARC_LLM_CALL_RECORD_FIELD]["structured_output"]
    assert structured["severity"] == "major"
    assert structured["recovery_strategy"] == "natural_language_fallback"


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
        static_prefix="stable prefix",
    )

    line = (tmp_path / "sessions" / "calls.jsonl").read_text(encoding="utf-8").splitlines()[0]
    call = json.loads(line)

    assert result[ARC_LLM_CALL_RECORD_FIELD]["static_prefix_sha256"] == sha256_text("stable prefix")
    assert call["static_prefix_sha256"] == sha256_text("stable prefix")


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
    )

    expected = sha256_text("prompt\nprovider contract")
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
    )

    call = json.loads((tmp_path / "sessions" / "calls.jsonl").read_text(encoding="utf-8"))

    assert outcome.value == "m:prompt"
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

    assert result["model"] == "gpt-5.5"


def test_run_json_uses_model_tier_when_exact_model_is_not_set(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: FakeProvider())

    result = run_json(
        "prompt",
        schema={"type": "object"},
        model_tier="high",
        env={"ARC_AGENT_HOST": "codex"},
        process_chain=[],
    )

    assert result["model"] == "gpt-5.5"


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

    assert result == "gpt-5.4:prompt"


def test_run_text_retries_selected_provider_twice_before_success(tmp_path, monkeypatch):
    flaky = FlakyTextProvider(name="codex-cli", failures_before_success=2)
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: flaky)

    result = run_text(
        "prompt",
        env={"ARC_AGENT_HOST": "codex"},
        process_chain=[],
    )

    assert result == "codex-cli:gpt-5.4:prompt"
    assert flaky.attempts == 3


def test_run_text_waits_ten_seconds_between_retry_attempts(monkeypatch):
    flaky = FlakyTextProvider(name="codex-cli", failures_before_success=2)
    sleep_calls = []
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: flaky)
    monkeypatch.setattr("time.sleep", lambda seconds: sleep_calls.append(seconds))

    result = run_text("prompt", provider="codex-cli", env={}, process_chain=[])

    assert result == "codex-cli:gpt-5.4:prompt"
    assert sleep_calls == [10, 10]


def test_run_json_retries_selected_provider_twice_before_success(tmp_path, monkeypatch):
    flaky = FlakyJsonProvider(name="codex-cli", failures_before_success=2, result={"provider": "codex-cli"})
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: flaky)

    result = run_json(
        "prompt",
        env={"ARC_AGENT_HOST": "codex"},
        process_chain=[],
    )

    assert result["provider"] == "codex-cli"
    assert result[ARC_LLM_CALL_RECORD_FIELD]["attempt"] == 3
    assert [item["status"] for item in result[ARC_LLM_CALL_RECORD_FIELD]["attempts"]] == [
        "failed",
        "failed",
        "success",
    ]
    assert flaky.attempts == 3


def test_run_json_auto_retries_only_selected_provider(monkeypatch):
    codex = FlakyJsonProvider(name="codex-cli")
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: codex)

    with pytest.raises(RuntimeError, match="LLM task failed after 3 attempt\\(s\\) across 1 provider\\(s\\)"):
        run_json(
            "prompt",
            env={"ARC_AGENT_HOST": "codex"},
            process_chain=[],
        )

    assert codex.attempts == 3


def test_run_json_explicit_provider_retries_without_fallback(tmp_path, monkeypatch):
    codex = FlakyJsonProvider(name="codex-cli")
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: codex)

    with pytest.raises(RuntimeError, match="LLM task failed after 3 attempt\\(s\\) across 1 provider\\(s\\)"):
        run_json(
            "prompt",
            provider="codex-cli",
            env={"ARC_AGENT_HOST": "codex"},
            process_chain=[],
        )

    assert codex.attempts == 3
