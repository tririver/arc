from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import pytest

from arc_llm.attempt_diagnostics import AttemptDiagnostics, bind_attempt_diagnostics
from arc_llm.providers import kimi_code_cli as kimi_module
from arc_llm.providers.base import (
    LLMConfigurationError,
    LLMFailureCategory,
    LLMWorkerError,
    LLMWorkerTimeout,
)
from arc_llm.providers.kimi_code_cli import EXPERIMENTAL_WARNING, KimiCodeCliProvider
from arc_llm.providers.activity import resolve_idle_timeout_seconds
from arc_llm.runner import run_text, run_text_result
from arc_llm.schema_cache import canonical_json, sha256_text
from arc_llm.sessions import LLMSessionManager, LLMSessionRef


FAKE_KIMI = Path(__file__).parent / "fixtures" / "fake_kimi_acp.py"
pytestmark = pytest.mark.filterwarnings("ignore:kimi-code-cli is experimental.*:RuntimeWarning")


@pytest.fixture(autouse=True)
def reset_experimental_warning(monkeypatch):
    monkeypatch.setattr(kimi_module, "_WARNING_EMITTED", False)


def test_kimi_idle_timeout_defaults_to_half_hour_when_environment_is_unset():
    assert resolve_idle_timeout_seconds(None, env={}, provider="kimi-code-cli") == 1800


def fake_env(tmp_path: Path, *, scenario: str = "happy", output: str = "hello") -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "ARC_LLM_TIMEOUT_SECONDS",
        "ARC_CODEX_TIMEOUT_SECONDS",
        "ARC_CLAUDE_TIMEOUT_SECONDS",
        "ARC_KIMI_TIMEOUT_SECONDS",
    ):
        env.pop(key, None)
    env.update(
        {
            "ARC_KIMI_BIN": str(FAKE_KIMI),
            "ARC_HOME": str(tmp_path / "arc-home"),
            "ARC_LLM_CACHE": str(tmp_path / "arc-home" / "cache" / "arc-llm"),
            "ARC_KIMI_WORK_DIR": str(tmp_path),
            "ARC_KIMI_IDLE_TIMEOUT_SECONDS": "5",
            "FAKE_KIMI_RECORD": str(tmp_path / "fake-kimi.jsonl"),
            "FAKE_KIMI_SCENARIO": scenario,
            "FAKE_KIMI_OUTPUT": output,
        }
    )
    return env


def test_removed_kimi_total_timeout_env_fails_before_model_call(tmp_path):
    env = fake_env(tmp_path)
    env["ARC_KIMI_TIMEOUT_SECONDS"] = "42"

    with pytest.raises(LLMConfigurationError, match="total-timeout.*removed.*ARC_KIMI_IDLE_TIMEOUT_SECONDS"):
        run_text("must not be submitted", provider="kimi-code-cli", env=env)

    assert records(env) == []


def records(env: dict[str, str]) -> list[dict]:
    path = Path(env["FAKE_KIMI_RECORD"])
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def client_messages(env: dict[str, str]) -> list[dict]:
    return [item["message"] for item in records(env) if item["kind"] == "client_message"]


def client_requests(env: dict[str, str]) -> list[dict]:
    return [message for message in client_messages(env) if "method" in message]


def test_text_happy_path_uses_official_wire_order_and_aggregates_chunks(tmp_path):
    env = fake_env(tmp_path, output="hello")

    response = KimiCodeCliProvider(env=env).generate_text_result("say hello")

    assert response.value == "hello"
    assert response.native_session_id == "fake-kimi-session-1"
    assert response.prompt_sent_sha256 == sha256_text("say hello")
    requests = client_requests(env)
    assert [request["method"] for request in requests] == [
        "initialize",
        "authenticate",
        "session/new",
        "session/prompt",
    ]
    assert requests[0]["params"] == {
        "protocolVersion": 1,
        "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}},
    }
    assert requests[1]["params"] == {"methodId": "login"}
    assert requests[2]["params"] == {"cwd": str(tmp_path.resolve()), "mcpServers": []}
    assert requests[3]["params"] == {
        "sessionId": "fake-kimi-session-1",
        "prompt": [{"type": "text", "text": "say hello"}],
    }
    chunks = [
        event["params"]["update"]["content"]["text"]
        for event in response.raw_events
        if event.get("method") == "session/update"
    ]
    assert chunks == ["he", "llo"]


def test_stateful_resume_uses_native_session_id_and_does_not_create(tmp_path):
    env = fake_env(tmp_path, output="resumed")
    session = LLMSessionRef(
        key="scope/worker",
        provider="kimi-code-cli",
        model=None,
        runtime_fingerprint="fp",
        native_session_id="persisted-kimi-session",
    )

    response = KimiCodeCliProvider(env=env).generate_text_result(
        "continue",
        session=session,
        session_policy="stateful",
    )

    assert response.value == "resumed"
    assert response.native_session_id == "persisted-kimi-session"
    requests = client_requests(env)
    assert [request["method"] for request in requests] == [
        "initialize",
        "authenticate",
        "session/resume",
        "session/prompt",
    ]
    assert requests[2]["params"] == {
        "sessionId": "persisted-kimi-session",
        "cwd": str(tmp_path.resolve()),
        "mcpServers": [],
    }
    assert requests[3]["params"]["sessionId"] == "persisted-kimi-session"


def test_runner_stateful_second_turn_resumes_provider_session(tmp_path):
    env = fake_env(tmp_path, output="continued")
    manager = LLMSessionManager(tmp_path / "arc-sessions")

    first = run_text_result(
        "first turn",
        provider="kimi-code-cli",
        env=env,
        session_policy="stateful",
        session_manager=manager,
        session_key="scope/worker",
        artifact_dir=tmp_path / "artifacts",
        idempotency_key="turn-1",
    )
    second = run_text_result(
        "second turn",
        provider="kimi-code-cli",
        env=env,
        session_policy="stateful",
        session_manager=manager,
        session_key="scope/worker",
        artifact_dir=tmp_path / "artifacts",
        idempotency_key="turn-2",
    )

    assert first.native_session_id == "fake-kimi-session-1"
    assert second.native_session_id == first.native_session_id
    methods = [request["method"] for request in client_requests(env)]
    assert methods.count("session/new") == 1
    assert methods.count("session/resume") == 1
    assert manager.turn_count("scope/worker") == 2


def test_runner_stateless_calls_always_create_new_provider_sessions(tmp_path):
    env = fake_env(tmp_path, output="fresh")

    assert run_text("one", provider="kimi-code-cli", env=env) == "fresh"
    assert run_text("two", provider="kimi-code-cli", env=env) == "fresh"

    methods = [request["method"] for request in client_requests(env)]
    assert methods.count("session/new") == 2
    assert "session/resume" not in methods


def test_model_is_set_after_session_creation_and_before_prompt(tmp_path):
    env = fake_env(tmp_path)

    KimiCodeCliProvider(env=env).generate_text("prompt", model="custom-model-alias")

    requests = client_requests(env)
    assert [request["method"] for request in requests] == [
        "initialize",
        "authenticate",
        "session/new",
        "session/set_config_option",
        "session/prompt",
    ]
    assert requests[3]["params"] == {
        "sessionId": "fake-kimi-session-1",
        "configId": "model",
        "value": "custom-model-alias",
    }


def test_large_prompt_only_travels_in_acp_stdin_not_process_argv(tmp_path):
    env = fake_env(tmp_path)
    prompt = "private-large-prompt-" + "x" * 300_000

    assert KimiCodeCliProvider(env=env).generate_text(prompt) == "hello"

    boot = next(item for item in records(env) if item["kind"] == "boot")
    assert boot["argv"] == [str(FAKE_KIMI), "--max-retries-per-step", "1", "acp"]
    assert all(prompt not in arg for arg in boot["argv"])
    prompt_request = next(request for request in client_requests(env) if request["method"] == "session/prompt")
    assert prompt_request["params"]["prompt"] == [{"type": "text", "text": prompt}]


def test_child_env_forces_safety_switches_and_inherits_kimi_home(tmp_path):
    env = fake_env(tmp_path)
    env["KIMI_CODE_HOME"] = str(tmp_path / "existing-kimi-home")
    env["KIMI_CODE_NO_AUTO_UPDATE"] = "0"
    env["KIMI_DISABLE_TELEMETRY"] = "false"
    env["KIMI_DISABLE_CRON"] = "no"

    KimiCodeCliProvider(env=env).generate_text("prompt")

    boot = next(item for item in records(env) if item["kind"] == "boot")
    assert boot["cwd"] == os.getcwd()
    assert boot["env"] == {
        "KIMI_CODE_NO_AUTO_UPDATE": "1",
        "KIMI_DISABLE_TELEMETRY": "1",
        "KIMI_DISABLE_CRON": "1",
        "KIMI_CODE_HOME": str(tmp_path / "existing-kimi-home"),
    }


def test_stderr_is_drained_and_raw_artifacts_are_written(tmp_path):
    env = fake_env(tmp_path, scenario="stderr_flood")
    artifact_dir = tmp_path / "artifacts"

    response = KimiCodeCliProvider(env=env).generate_text_result("prompt", artifact_dir=artifact_dir)

    assert response.value == "hello"
    assert response.raw_events
    events = [json.loads(line) for line in (artifact_dir / "raw_events.jsonl").read_text().splitlines()]
    assert any(event.get("method") == "session/update" for event in events)
    assert '"method":"session/update"' in (artifact_dir / "raw_stdout.txt").read_text()
    stderr = (artifact_dir / "raw_stderr.txt").read_text()
    assert len(stderr) > 256_000
    assert stderr.endswith("FAKE_STDERR_END\n")


def test_json_direct_output_uses_canonical_schema_contract_and_null_usage(tmp_path):
    env = fake_env(tmp_path, output='{"ok":true}')
    schema = {
        "type": "object",
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
    }

    response = KimiCodeCliProvider(env=env).generate_json_result("return data", schema=schema)

    assert response.value == {"ok": True}
    assert response.structured_output is None
    assert len(response.candidate_material) == 1
    assert response.candidate_material[0].source == "kimi.session_prompt_message"
    assert response.candidate_material[0].text == '{"ok":true}'
    assert response.usage.input_tokens is None
    assert response.usage.cached_input_tokens is None
    assert response.usage.output_tokens is None
    assert response.usage.reasoning_output_tokens is None
    assert response.usage.cache_creation_input_tokens is None
    assert response.usage.cache_read_input_tokens is None
    prompt_request = next(request for request in client_requests(env) if request["method"] == "session/prompt")
    sent = prompt_request["params"]["prompt"][0]["text"]
    assert sent.startswith("return data")
    assert canonical_json(schema) in sent
    assert response.prompt_sent_sha256 == sha256_text(sent)


def test_json_strict_mode_accepts_relaxed_object_extraction(tmp_path):
    env = fake_env(tmp_path, output='prefix {"ok": true} suffix')

    response = KimiCodeCliProvider(env=env).generate_json_result(
        "prompt",
        schema={"type": "object"},
        output_recovery="strict",
    )

    assert response.value == {"ok": True}
    assert response.structured_output["severity"] == "minor"
    assert response.structured_output["recovery_strategy"] == "extract_json"


def test_json_warn_mode_recovers_plain_text(tmp_path):
    env = fake_env(tmp_path, output="plain natural language answer")

    response = KimiCodeCliProvider(env=env).generate_json_result(
        "prompt",
        schema={"type": "object"},
        output_recovery="warn",
    )

    assert response.value == {}
    assert response.structured_output["severity"] == "major"
    assert response.structured_output["recovery_strategy"] == "natural_language_fallback"


def test_json_strict_mode_rejects_plain_text(tmp_path):
    env = fake_env(tmp_path, output="plain natural language answer")

    with pytest.raises(LLMWorkerError, match="Kimi output was not JSON") as caught:
        KimiCodeCliProvider(env=env).generate_json_result(
            "prompt",
            schema={"type": "object"},
            output_recovery="strict",
        )

    assert caught.value.retryable is False


@pytest.mark.parametrize("output_recovery", ["strict", "warn"])
def test_empty_agent_response_is_non_retryable(output_recovery, tmp_path):
    env = fake_env(tmp_path, scenario="empty")

    with pytest.raises(LLMWorkerError, match="no agent message text") as caught:
        KimiCodeCliProvider(env=env).generate_json_result(
            "prompt",
            schema={"type": "object"},
            output_recovery=output_recovery,
        )

    assert caught.value.retryable is False
    assert caught.value.abort_batch is True


@pytest.mark.parametrize(
    ("scenario", "match"),
    [
        ("empty", "no agent message text"),
        ("rpc_rate_limit", "rate-limit"),
    ],
)
def test_runner_does_not_recreate_session_after_batch_aborting_response(
    scenario, match, tmp_path
):
    env = fake_env(tmp_path, scenario=scenario)

    with pytest.raises(RuntimeError, match=match) as caught:
        run_text("prompt", provider="kimi-code-cli", env=env)

    assert caught.value.__cause__.abort_batch is True
    methods = [request["method"] for request in client_requests(env)]
    assert methods.count("session/new") == 1
    assert methods.count("session/prompt") == 1


def test_reverse_permission_request_is_cancelled_without_claiming_sandbox(tmp_path):
    env = fake_env(tmp_path, scenario="reverse_permission")

    response = KimiCodeCliProvider(env=env).generate_text_result("prompt")
    assert response.value == "hello"

    reverse = next(
        item
        for item in records(env)
        if item.get("kind") == "reverse_response" and item.get("method") == "session/request_permission"
    )
    assert reverse["response"] == {
        "jsonrpc": "2.0",
        "id": "reverse-permission-1",
        "result": {"outcome": {"outcome": "cancelled"}},
    }
    assert [event for event in response.raw_events if event.get("direction") == "reverse_response"] == [{
        "direction": "reverse_response",
        "id": "reverse-permission-1",
        "method": "session/request_permission",
        "disposition": "permission_cancelled",
    }]
    assert "sandbox" not in EXPERIMENTAL_WARNING.lower()


def test_reverse_filesystem_requests_receive_json_rpc_errors(tmp_path):
    env = fake_env(tmp_path, scenario="reverse_fs")

    response = KimiCodeCliProvider(env=env).generate_text_result("prompt")
    assert response.value == "hello"

    reverse = {
        item["method"]: item["response"]
        for item in records(env)
        if item.get("kind") == "reverse_response"
    }
    assert reverse["fs/read_text_file"]["error"] == {
        "code": -32001,
        "message": "ARC denies ACP reverse filesystem access",
    }
    assert reverse["fs/write_text_file"]["error"] == {
        "code": -32001,
        "message": "ARC denies ACP reverse filesystem access",
    }
    assert [
        (event["method"], event["disposition"])
        for event in response.raw_events
        if event.get("direction") == "reverse_response"
    ] == [
        ("fs/read_text_file", "filesystem_denied"),
        ("fs/write_text_file", "filesystem_denied"),
    ]


@pytest.mark.parametrize(
    ("scenario", "match", "retryable", "abort_batch"),
    [
        ("auth_error", "run `kimi login`", False, True),
        ("invalid_session", "protocol error", False, False),
        ("invalid_json", "invalid JSON", False, False),
        ("old_version", "requires >=0.28.0", False, False),
        ("transport_eof", "exited before replying", False, False),
        ("transport_usage_limit", "usage limit", False, True),
        ("rpc_usage_limit", "usage limit", False, True),
        ("rpc_quota_exhausted", "quota-exhausted", False, True),
        ("rpc_rate_limit", "rate-limit", False, True),
    ],
)
def test_error_retryability_classification(
    scenario, match, retryable, abort_batch, tmp_path
):
    env = fake_env(tmp_path, scenario=scenario)
    session = None
    policy = "stateless"
    if scenario == "invalid_session":
        session = LLMSessionRef(
            key="scope/worker",
            provider="kimi-code-cli",
            model=None,
            runtime_fingerprint="fp",
            native_session_id="missing-session",
        )
        policy = "stateful"

    with pytest.raises(LLMWorkerError, match=match) as caught:
        KimiCodeCliProvider(env=env).generate_text_result(
            "prompt",
            session=session,
            session_policy=policy,
        )

    assert caught.value.retryable is retryable
    assert caught.value.abort_batch is abort_batch


@pytest.mark.parametrize(
    ("diagnostic", "retryable", "abort_batch"),
    [
        ("403 Forbidden", False, True),
        ("access forbidden", False, True),
        ("You've reached your usage limit", False, True),
        ("quota-exhausted", False, True),
        ("429 rate-limit exceeded", False, True),
        ("too many requests", False, True),
        ("temporarily unavailable", True, False),
    ],
)
def test_kimi_diagnostic_retryability_distinguishes_quota_from_throttling(
    diagnostic, retryable, abort_batch
):
    assert kimi_module._diagnostic_disposition(diagnostic, default=True) == (
        retryable,
        abort_batch,
    )


def test_stderr_rate_limit_aborts_hung_acp_immediately(tmp_path):
    env = fake_env(tmp_path, scenario="stderr_rate_limit_hang")
    started = time.monotonic()
    with pytest.raises(LLMWorkerError, match="rate_limit") as caught:
        KimiCodeCliProvider(env=env).generate_text_result(
            "prompt",
            idle_timeout_seconds=10,
        )
    assert time.monotonic() - started < 3
    assert caught.value.category == LLMFailureCategory.RATE_LIMIT
    assert caught.value.retryable is False


def test_missing_binary_is_non_retryable(tmp_path):
    env = fake_env(tmp_path)
    env["ARC_KIMI_BIN"] = str(tmp_path / "does-not-exist" / "kimi")

    with pytest.raises(LLMWorkerError, match="binary not found") as caught:
        KimiCodeCliProvider(env=env).generate_text("prompt")

    assert caught.value.retryable is False


def test_experimental_warning_is_emitted_once_before_execution(tmp_path):
    env = fake_env(tmp_path)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        provider = KimiCodeCliProvider(env=env)
        assert provider.generate_text("first") == "hello"
        assert provider.generate_text("second") == "hello"

    messages = [str(item.message) for item in seen if str(item.message) == EXPERIMENTAL_WARNING]
    assert messages == [EXPERIMENTAL_WARNING]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group assertion")
def test_idle_timeout_sends_cancel_and_kills_process_group(tmp_path):
    env = fake_env(tmp_path, scenario="timeout")
    env["ARC_KIMI_IDLE_TIMEOUT_SECONDS"] = "0.25"

    with pytest.raises(LLMWorkerTimeout, match="no meaningful output") as caught:
        KimiCodeCliProvider(env=env).generate_text("long prompt")

    assert caught.value.retryable is False
    seen = records(env)
    assert any(
        item.get("kind") == "client_message" and item.get("message", {}).get("method") == "session/cancel"
        for item in seen
    )
    child_pid = next(item["pid"] for item in seen if item.get("kind") == "child")
    deadline = time.monotonic() + 3
    while _process_is_live(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _process_is_live(child_pid)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group assertion")
def test_idle_timeout_attempt_timeline_records_acp_cancel_and_group_cleanup(tmp_path):
    env = fake_env(tmp_path, scenario="timeout")
    env["ARC_KIMI_IDLE_TIMEOUT_SECONDS"] = "0.2"
    diagnostics = AttemptDiagnostics(
        tmp_path / "artifacts",
        provider="kimi-code-cli",
        model=None,
        fallback_index=0,
        attempt=1,
        call_label="timeout",
        env=env,
    )

    with bind_attempt_diagnostics(diagnostics):
        with pytest.raises(LLMWorkerTimeout) as caught:
            KimiCodeCliProvider(env=env).generate_text("long prompt")
        reference = diagnostics.finalize(outcome="timeout", error=caught.value)

    record_path = (tmp_path / "artifacts") / reference.path
    record = json.loads(record_path.read_text())
    timeline = (record_path.parent / record["timeline"]["path"]).read_text()
    for event in (
        "cancellation_requested",
        "provider_cancel_sent",
        "stdin_close_attempted",
        "term_attempted",
        "process_group_outcome",
    ):
        assert f'"event": "{event}"' in timeline


def _process_is_live(pid: int) -> bool:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        fields = stat_path.read_text(encoding="utf-8").split()
    except OSError:
        return False
    return len(fields) < 3 or fields[2] != "Z"
