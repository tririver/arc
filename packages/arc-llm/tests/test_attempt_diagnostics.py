from __future__ import annotations

import gzip
import hashlib
import json
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate

from arc_llm import attempt_diagnostics as diagnostics_module
from arc_llm import runner as runner_module
from arc_llm.attempt_diagnostics import (
    AttemptDiagnostics,
    AttemptDiagnosticsError,
    DiagnosticRedactor,
    bind_attempt_diagnostics,
)
from arc_llm.call_record import ARC_LLM_CALL_RECORD_FIELD, ARC_LLM_CALL_RECORD_SCHEMA
from arc_llm.providers.activity import ActivityTracker
from arc_llm.host import HostDetection
from arc_llm.providers.base import (
    LLMSubmissionState,
    LLMWorkerError,
    LLMWorkerTimeout,
)
from arc_llm.providers.lifecycle import run_streaming_process_group
from arc_llm.runner import LLMConfig, LLMTaskError, run_json


def _read_stream(attempt_dir: Path, receipt: dict[str, object]) -> str:
    path = attempt_dir / str(receipt["path"])
    payload = path.read_bytes()
    if receipt["compression"] == "gzip":
        payload = gzip.decompress(payload)
    return payload.decode("utf-8", errors="replace")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_binding(tmp_path: Path) -> dict[str, object]:
    digest = "a" * 64
    return {
        "schema_version": "arc.llm.checkpoint_recomputation_binding.v1",
        "checkpoint_path": str(
            (tmp_path / "call-checkpoints" / "idempotency.json").resolve()
        ),
        "checkpoint_identity": "checkpoint-identity",
        "logical_identity": {
            "provider": "codex-cli",
            "model": "model",
            "session_key": "chapter:translation",
            "generation": 7,
            "idempotency_key": "segment-9",
        },
        "request_digest": digest,
        "request_recipe_sha256": "b" * 64,
        "idempotency_key": "segment-9",
        "session_key": "chapter:translation",
        "generation": 7,
        "prompt_sha256": "c" * 64,
        "schema_sha256": "d" * 64,
        "call_label_sha256": "e" * 64,
        "initial_native_authorization": {
            "control_address": str((tmp_path / "control-ledger.json").resolve()),
            "session_key": "chapter:translation",
            "logical_unit": "segment-9",
            "generation": 7,
            "idempotency_key": "segment-9",
        },
        "native_resume_authorization": {
            "control_address": str((tmp_path / "control-ledger.json").resolve()),
            "session_key": "chapter:translation",
            "logical_unit": "segment-9",
            "generation": 7,
            "idempotency_key": "segment-9",
        },
    }


def test_attempt_record_binds_exact_checkpoint_and_immutable_sources(
    tmp_path: Path,
) -> None:
    diagnostics = AttemptDiagnostics(
        tmp_path, provider="codex-cli", model="model", fallback_index=0,
        attempt=1, call_label="segment-9", env={},
    )
    binding = _checkpoint_binding(tmp_path)
    diagnostics.bind_checkpoint(binding)
    # A caller cannot mutate the already-bound association by retaining the
    # original nested mapping.
    binding["logical_identity"]["generation"] = 999  # type: ignore[index]
    diagnostics.capture_stdout("immutable provider output\n")
    diagnostics.record_candidate({"answer": 42}, source="provider")
    reference = diagnostics.finalize(outcome="success")
    record = json.loads((tmp_path / reference.path).read_text())

    assert record["checkpoint_path"] == "call-checkpoints/idempotency.json"
    assert record["checkpoint_identity"] == "checkpoint-identity"
    assert record["generation"] == 7
    assert record["prompt_sha256"] == "c" * 64
    assert record["schema_sha256"] == "d" * 64
    assert record["call_label_sha256"] == "e" * 64
    assert record["checkpoint_binding"]["logical_identity"]["generation"] == 7
    assert record["checkpoint_binding"]["native_resume_authorization"] == {
        "control_address": str((tmp_path / "control-ledger.json").resolve()),
        "session_key": "chapter:translation",
        "logical_unit": "segment-9",
        "generation": 7,
        "idempotency_key": "segment-9",
    }
    immutable = record["immutable_source"]
    encoded = json.dumps(
        immutable["manifest"], ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(encoded).hexdigest() == immutable["manifest_sha256"]
    assert immutable["manifest"]["streams"]["stdout"]["sha256"] == _sha256(
        (tmp_path / reference.path).parent / "stdout.txt"
    )


def test_attempt_checkpoint_binding_rejects_relabel_or_outside_path(
    tmp_path: Path,
) -> None:
    diagnostics = AttemptDiagnostics(
        tmp_path, provider="codex-cli", model=None, fallback_index=0,
        attempt=1, call_label="binding", env={},
    )
    binding = _checkpoint_binding(tmp_path)
    diagnostics.bind_checkpoint(binding)
    diagnostics.bind_checkpoint(binding)
    changed = deepcopy(binding)
    changed["prompt_sha256"] = "f" * 64
    with pytest.raises(AttemptDiagnosticsError, match="binding changed"):
        diagnostics.bind_checkpoint(changed)

    other = AttemptDiagnostics(
        tmp_path, provider="codex-cli", model=None, fallback_index=0,
        attempt=2, call_label="outside", env={},
    )
    outside = _checkpoint_binding(tmp_path)
    outside["checkpoint_path"] = str((tmp_path.parent / "foreign.json").resolve())
    with pytest.raises(AttemptDiagnosticsError, match="outside"):
        other.bind_checkpoint(outside)


def test_attempt_record_redacts_bounds_compresses_and_finalizes_once(tmp_path: Path) -> None:
    secret = "super-secret-environment-value"
    diagnostics = AttemptDiagnostics(
        tmp_path,
        provider="codex-cli",
        model="model",
        fallback_index=0,
        attempt=1,
        call_label="segment/one",
        env={"ARC_TEST_API_TOKEN": secret, "ARC_PROVIDER_CONFIG": '{"password":"hidden"}'},
        max_stream_bytes=32 * 1024,
    )
    diagnostics.mark_submitted()
    diagnostics.capture_stdout(
        json.dumps({"type": "thread.started", "thread_id": "native-1", "token": secret})
        + "\n"
    )
    diagnostics.capture_stdout("configuration leaf: hidden\n")
    diagnostics.capture_stderr((f"Authorization: Bearer {secret}\n" + "x" * 4000) * 40)
    diagnostics.record_candidate(
        {"answer": "usable", "api_key": secret}, source="provider_parsed_response"
    )
    error = LLMWorkerTimeout(
        f"provider timed out with password={secret}",
        submission_state=LLMSubmissionState.SUBMITTED,
    )
    reference = diagnostics.finalize(outcome="timeout", error=error)
    repeated = diagnostics.finalize(outcome="success")

    assert repeated == reference
    record_path = tmp_path / reference.path
    assert reference.sha256 == _sha256(record_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["outcome"] == "timeout"
    assert record["submission_state"] == "submitted"
    assert record["native_session_id"] == "native-1"
    assert secret not in json.dumps(record, ensure_ascii=False)
    assert record["error"]["category"] == "timeout"
    assert record["streams"]["stderr"]["truncated"] is True
    assert record["streams"]["stderr"]["compression"] == "gzip"
    assert record["streams"]["stderr"]["observed_bytes"] > 100_000
    stderr_text = _read_stream(record_path.parent, record["streams"]["stderr"])
    assert len(stderr_text.encode("utf-8")) <= 32 * 1024
    assert "[REDACTED" in stderr_text
    assert secret not in _read_stream(record_path.parent, record["streams"]["raw_events"])
    assert "hidden" not in _read_stream(record_path.parent, record["streams"]["stdout"])
    candidates = _read_stream(record_path.parent, record["streams"]["response_candidates"])
    assert '"answer": "usable"' in candidates
    assert secret not in candidates
    assert record_path.stat().st_mode & 0o777 == 0o400
    assert record_path.parent.stat().st_mode & 0o777 == 0o500


def test_attempt_directories_are_independent_and_never_overwritten(tmp_path: Path) -> None:
    first = AttemptDiagnostics(
        tmp_path,
        provider="claude-cli",
        model=None,
        fallback_index=0,
        attempt=1,
        call_label="same",
        env={},
    )
    first.capture_stdout("first\n")
    first_ref = first.finalize(outcome="error", error=RuntimeError("first failure"))
    first_bytes = (tmp_path / first_ref.path).read_bytes()

    second = AttemptDiagnostics(
        tmp_path,
        provider="claude-cli",
        model=None,
        fallback_index=0,
        attempt=2,
        call_label="same",
        env={},
    )
    second.capture_stdout("second\n")
    second_ref = second.finalize(outcome="success")

    assert first_ref.path != second_ref.path
    assert (tmp_path / first_ref.path).read_bytes() == first_bytes
    assert json.loads((tmp_path / first_ref.path).read_text())["attempt"] == 1
    assert json.loads((tmp_path / second_ref.path).read_text())["attempt"] == 2


def test_retry_and_fallback_attempts_each_keep_an_independent_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configs = [
        LLMConfig(
            provider=provider,
            model=None,
            host=HostDetection(host="test", confidence=1, signals=[]),
            signals=[],
        )
        for provider in ("codex-cli", "claude-cli")
    ]
    calls = 0

    def invoke(_selected, _config, _timeout):
        nonlocal calls
        calls += 1
        if calls < 4:
            raise RuntimeError(f"failure {calls}")
        return {"ok": True}

    monkeypatch.setattr(runner_module, "select_provider", lambda provider, **_kwargs: provider)
    monkeypatch.setattr(runner_module.time, "sleep", lambda _seconds: None)
    result = runner_module._run_with_retries(  # noqa: SLF001
        configs,
        provider_requested="auto",
        model_requested=None,
        model_tier_requested=None,
        attach_call_record=True,
        env={},
        process_chain=[],
        max_attempts=2,
        artifact_dir=tmp_path,
        diagnostic_call_label="retries",
        call=invoke,
    )

    attempts = result[ARC_LLM_CALL_RECORD_FIELD]["attempts"]
    assert [(item["fallback_index"], item["attempt"]) for item in attempts] == [
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
    ]
    paths = [item["diagnostic_path"] for item in attempts]
    assert len(set(paths)) == 4
    assert [json.loads((tmp_path / path).read_text())["outcome"] for path in paths] == [
        "error",
        "error",
        "error",
        "success",
    ]


def test_terminal_retry_error_carries_all_ordered_immutable_attempt_refs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    config = LLMConfig(
        provider="codex-cli", model=None,
        host=HostDetection(host="test", confidence=1, signals=[]), signals=[],
    )
    first = LLMWorkerError("transient", retryable=True)
    terminal = LLMWorkerError("terminal", retryable=False)
    errors = iter((first, terminal))
    monkeypatch.setattr(
        runner_module, "select_provider", lambda provider, **_kwargs: provider,
    )
    monkeypatch.setattr(runner_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(LLMTaskError) as caught:
        runner_module._run_with_retries(  # noqa: SLF001
            [config],
            provider_requested="codex-cli",
            model_requested=None,
            model_tier_requested=None,
            attach_call_record=False,
            env={},
            process_chain=[],
            max_attempts=2,
            artifact_dir=tmp_path,
            diagnostic_call_label="terminal-retry",
            call=lambda *_args: (_ for _ in ()).throw(next(errors)),
        )

    refs = caught.value.attempt_diagnostic_refs
    assert isinstance(refs, tuple)
    assert len(refs) == 2
    assert caught.value.diagnostic_ref == refs[-1]
    assert caught.value.__cause__ is terminal
    assert terminal.attempt_diagnostic_refs == refs
    assert terminal.diagnostic_ref == refs[-1]
    assert len(first.attempt_diagnostic_refs) == 1
    records = []
    for index, ref in enumerate(refs, 1):
        path = tmp_path / ref["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"]
        record = json.loads(path.read_text())
        records.append(record)
        assert record["attempt"] == index
        assert record["outcome"] == "error"
    assert [record["fallback_index"] for record in records] == [0, 0]


def test_identity_fields_and_oversized_timeline_details_are_sanitized_and_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = "identity-secret-value"
    monkeypatch.setattr(diagnostics_module.os, "fsync", lambda _fd: None)
    monkeypatch.setattr(diagnostics_module, "MAX_TIMELINE_BYTES", 800)
    diagnostics = AttemptDiagnostics(
        tmp_path,
        provider=secret,
        model=secret,
        fallback_index=0,
        attempt=1,
        call_label=secret,
        env={"ARC_PRIVATE_CONFIG": json.dumps({"secret": secret})},
    )
    diagnostics.record_native_session_id(secret)
    diagnostics.event(
        "e" * 100_000,
        sequence="must-not-overwrite",
        at="must-not-overwrite",
        payload="x" * 100_000,
    )
    for index in range(3):
        diagnostics.event("bounded", index=index, payload="y" * 80)
    reference = diagnostics.finalize(outcome="success")

    record_path = tmp_path / reference.path
    record_text = record_path.read_text()
    record = json.loads(record_text)
    timeline_path = record_path.parent / record["timeline"]["path"]
    assert secret not in reference.path
    assert secret not in record_text
    assert secret not in timeline_path.read_text()
    assert record["provider"] == "[REDACTED_ENV]"
    assert record["model"] == "[REDACTED_ENV]"
    assert record["call_label"] == "[REDACTED_ENV]"
    assert record["native_session_id"] == "[REDACTED_ENV]"
    assert timeline_path.stat().st_size <= diagnostics_module.MAX_TIMELINE_BYTES
    assert '"details_truncated": true' in timeline_path.read_text()
    assert record["timeline"]["dropped_events"] > 0
    timeline_events = [json.loads(line) for line in timeline_path.read_text().splitlines()]
    oversized = timeline_events[2]
    assert isinstance(oversized["sequence"], int)
    assert oversized["at"] != "must-not-overwrite"
    assert len(oversized["event"].encode("utf-8")) <= 128
    assert len(json.dumps(oversized, ensure_ascii=False).encode("utf-8")) < 16 * 1024


def test_mapping_keys_are_redacted_without_losing_colliding_entries() -> None:
    secret = "opaque-secret-as-a-key"
    sanitized = DiagnosticRedactor({"ARC_OPAQUE": secret}).value(
        {secret: "first", "[REDACTED_ENV]": "second"}
    )

    assert secret not in json.dumps(sanitized)
    assert sanitized == {"[REDACTED_ENV]": "[REDACTED]", "[REDACTED_ENV]#2": "second"}
    assert (
        DiagnosticRedactor({"ARC_SHORT_CONFIG": '{"credential":"xy"}'}).text(
            "bare config leaf xy"
        )
        == "bare config leaf [REDACTED_ENV]"
    )


def test_candidate_metadata_and_source_fields_have_hard_caps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(diagnostics_module.os, "fsync", lambda _fd: None)
    diagnostics = AttemptDiagnostics(
        tmp_path,
        provider="codex-cli",
        model=None,
        fallback_index=0,
        attempt=1,
        call_label="candidate-cap",
        env={},
    )
    monkeypatch.setattr(diagnostics, "event", lambda *_args, **_kwargs: None)
    with ThreadPoolExecutor(max_workers=24) as pool:
        list(
            pool.map(
                lambda index: diagnostics.record_candidate(
                    {"index": index},
                    source="s" * 100_000 if index == 0 else "candidate",
                ),
                range(300),
            )
        )
    reference = diagnostics.finalize(outcome="success")

    record = json.loads((tmp_path / reference.path).read_text())
    assert record["parsed_response_candidate_count"] == 300
    assert record["parsed_response_candidates_dropped"] == 44
    assert len(record["parsed_response_candidates"]) == 256
    assert [item["sequence"] for item in record["parsed_response_candidates"]] == list(
        range(1, 257)
    )
    assert all(
        len(item["source"].encode("utf-8")) <= 128
        for item in record["parsed_response_candidates"]
    )


def test_compression_expansion_falls_back_without_exceeding_disk_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(diagnostics_module.os, "fsync", lambda _fd: None)
    diagnostics = AttemptDiagnostics(
        tmp_path,
        provider="codex-cli",
        model=None,
        fallback_index=0,
        attempt=1,
        call_label="compression",
        env={},
        max_stream_bytes=32 * 1024,
    )
    diagnostics.capture_stderr("z" * 100_000)
    monkeypatch.setattr(
        diagnostics_module.gzip,
        "compress",
        lambda raw, **kwargs: b"expanded" * (len(raw) // 4),
    )
    reference = diagnostics.finalize(outcome="success")

    record_path = tmp_path / reference.path
    record = json.loads(record_path.read_text())
    receipt = record["streams"]["stderr"]
    assert receipt["compression"] == "none"
    assert not str(receipt["path"]).endswith(".gz")
    assert receipt["stored_bytes"] <= 32 * 1024
    assert (record_path.parent / receipt["path"]).stat().st_size <= 32 * 1024


def test_inherited_environment_is_redacted_when_runner_env_is_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = "opaque-inherited-environment-secret"
    monkeypatch.setenv("ARC_OPAQUE_PROVIDER_VALUE", secret)
    config = LLMConfig(
        provider="codex-cli",
        model=None,
        host=HostDetection(host="test", confidence=1, signals=[]),
        signals=[],
    )

    def invoke(_selected, _config, _timeout):
        active = diagnostics_module.current_attempt_diagnostics()
        assert active is not None
        active.capture_stderr(f"bare echo: {secret}\n")
        return {"ok": True}

    monkeypatch.setattr(runner_module, "select_provider", lambda provider, **_kwargs: provider)
    result = runner_module._run_with_retries(  # noqa: SLF001
        [config],
        provider_requested="codex-cli",
        model_requested=None,
        model_tier_requested=None,
        attach_call_record=True,
        env=None,
        process_chain=[],
        artifact_dir=tmp_path,
        diagnostic_call_label="inherited",
        call=invoke,
    )

    reference = result[ARC_LLM_CALL_RECORD_FIELD]["attempts"][0]
    record_path = tmp_path / reference["diagnostic_path"]
    record = json.loads(record_path.read_text())
    stderr = _read_stream(record_path.parent, record["streams"]["stderr"])
    assert secret not in stderr
    assert "[REDACTED_ENV]" in stderr


def test_finalize_reports_failure_when_immutability_cannot_be_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    diagnostics = AttemptDiagnostics(
        tmp_path,
        provider="codex-cli",
        model=None,
        fallback_index=0,
        attempt=1,
        call_label="chmod",
        env={},
    )
    monkeypatch.setattr(
        Path,
        "chmod",
        lambda self, mode: (_ for _ in ()).throw(OSError("chmod denied")),
    )

    with pytest.raises(AttemptDiagnosticsError, match="immutable"):
        diagnostics.finalize(outcome="success")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group timeline")
def test_streaming_timeout_captures_partial_output_stderr_and_kill_timeline(
    tmp_path: Path,
) -> None:
    secret = "diagnostic-secret-value"
    env = dict(os.environ)
    env["ARC_TEST_PASSWORD"] = secret
    script = (
        "import json,os,signal,sys,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "print(json.dumps({'type':'thread.started','thread_id':'partial-session','token':os.environ['ARC_TEST_PASSWORD']}),flush=True);"
        "sys.stderr.write(('noise-'+os.environ['ARC_TEST_PASSWORD']+'-'+'z'*4096+'\\n')*80);"
        "sys.stderr.flush();time.sleep(60)"
    )
    diagnostics = AttemptDiagnostics(
        tmp_path,
        provider="codex-cli",
        model=None,
        fallback_index=0,
        attempt=1,
        call_label="timeout",
        env=env,
        max_stream_bytes=32 * 1024,
    )
    with bind_attempt_diagnostics(diagnostics):
        with pytest.raises(LLMWorkerTimeout) as caught:
            run_streaming_process_group(
                [sys.executable, "-c", script],
                input_text="prompt",
                env=env,
                activity=ActivityTracker(provider="codex-cli", idle_timeout_seconds=0.15),
                stdout_line_callback=lambda _line: None,
                poll_interval_seconds=0.02,
                terminate_grace_seconds=0.1,
            )
        reference = diagnostics.finalize(outcome="timeout", error=caught.value)

    record_path = tmp_path / reference.path
    record = json.loads(record_path.read_text())
    timeline = (record_path.parent / record["timeline"]["path"]).read_text()
    assert record["native_session_id"] == "partial-session"
    assert record["streams"]["stderr"]["observed_bytes"] > 100_000
    assert record["streams"]["stderr"]["truncated"] is True
    assert secret not in _read_stream(record_path.parent, record["streams"]["stderr"])
    for event in (
        "submission_barrier_crossed",
        "stdin_close_attempted",
        "term_attempted",
        "kill_attempted",
        "process_group_outcome",
        "timeout_observed",
    ):
        assert f'"event": "{event}"' in timeline
    timeline_events = [json.loads(line) for line in timeline.splitlines()]
    assert next(
        event for event in timeline_events if event["event"] == "stdin_close_attempted"
    )["details"]["succeeded"] is True
    assert next(
        event for event in timeline_events if event["event"] == "term_attempted"
    )["details"]["delivered"] is True
    assert next(
        event for event in timeline_events if event["event"] == "kill_attempted"
    )["details"]["delivered"] is True


def test_call_record_references_attempt_record_without_inlining_streams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Provider:
        def generate_json_result(self, prompt, **kwargs):
            from arc_llm.usage import LLMProviderResponse

            return LLMProviderResponse(
                {"ok": True}, native_session_id="session-42", raw_output='{"ok":true}'
            )

    monkeypatch.setattr("arc_llm.runner.select_provider", lambda *_args, **_kwargs: Provider())
    result = run_json(
        "prompt",
        provider="codex-cli",
        env={},
        process_chain=[],
        artifact_dir=tmp_path,
        call_label="diagnosed",
    )

    call_record = result[ARC_LLM_CALL_RECORD_FIELD]
    validate(call_record, ARC_LLM_CALL_RECORD_SCHEMA)
    attempt = call_record["attempts"][0]
    record_path = tmp_path / attempt["diagnostic_path"]
    assert record_path.is_file()
    assert attempt["diagnostic_sha256"] == _sha256(record_path)
    assert "streams" not in attempt
    record = json.loads(record_path.read_text())
    assert record["native_session_id"] == "session-42"
    assert record["parsed_response_candidates"][0]["source"] == "provider_parsed_response"
    unsafe = deepcopy(call_record)
    unsafe["attempts"][0]["diagnostic_path"] = "../outside/record.json"
    with pytest.raises(ValidationError):
        validate(unsafe, ARC_LLM_CALL_RECORD_SCHEMA)


def test_diagnostic_finalize_failure_never_discards_a_paid_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Provider:
        def generate_json_result(self, prompt, **kwargs):
            from arc_llm.usage import LLMProviderResponse

            return LLMProviderResponse({"ok": True})

    monkeypatch.setattr(runner_module, "select_provider", lambda *_args, **_kwargs: Provider())
    monkeypatch.setattr(
        AttemptDiagnostics,
        "finalize",
        lambda self, **kwargs: (_ for _ in ()).throw(OSError("diagnostic disk failed")),
    )

    result = run_json(
        "prompt",
        provider="codex-cli",
        env={},
        process_chain=[],
        artifact_dir=tmp_path,
        call_label="paid",
    )

    assert result["ok"] is True
    record = result[ARC_LLM_CALL_RECORD_FIELD]
    assert "attempt_diagnostics.persistence_failed" in record["warnings"]
    assert record["attempts"][0]["diagnostic_path"] is None
    assert record["attempts"][0]["diagnostic_error_type"] == "OSError"


def test_diagnostic_creation_failure_is_nonretryable_and_precedes_provider_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact_file = tmp_path / "not-a-directory"
    artifact_file.write_text("occupied")
    calls = 0

    class Provider:
        def generate_json_result(self, prompt, **kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("provider must not be called")

    monkeypatch.setattr(runner_module, "select_provider", lambda *_args, **_kwargs: Provider())

    with pytest.raises(AttemptDiagnosticsError) as caught:
        runner_module._run_with_retries(  # noqa: SLF001
            [
                LLMConfig(
                    provider="codex-cli",
                    model=None,
                    host=HostDetection(host="test", confidence=1, signals=[]),
                    signals=[],
                )
            ],
            provider_requested="codex-cli",
            model_requested=None,
            model_tier_requested=None,
            attach_call_record=True,
            env={},
            process_chain=[],
            max_attempts=2,
            artifact_dir=artifact_file,
            diagnostic_call_label="creation",
            call=lambda selected, config, timeout: selected.generate_json_result("prompt"),
        )

    assert caught.value.retryable is False
    assert caught.value.submission_state == LLMSubmissionState.NOT_SUBMITTED
    assert calls == 0


def test_diagnostic_finalize_failure_never_replaces_original_provider_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider_error = LLMWorkerTimeout(
        "original provider timeout", submission_state=LLMSubmissionState.SUBMITTED
    )

    class Provider:
        def generate_json_result(self, prompt, **kwargs):
            raise provider_error

    monkeypatch.setattr(runner_module, "select_provider", lambda *_args, **_kwargs: Provider())
    monkeypatch.setattr(
        AttemptDiagnostics,
        "finalize",
        lambda self, **kwargs: (_ for _ in ()).throw(OSError("diagnostic disk failed")),
    )

    with pytest.raises(LLMWorkerTimeout) as caught:
        run_json(
            "prompt",
            provider="codex-cli",
            env={},
            process_chain=[],
            artifact_dir=tmp_path,
            call_label="failed",
        )

    assert caught.value is provider_error
    assert any("attempt diagnostics persistence failed" in note for note in provider_error.__notes__)


def test_parsed_candidate_is_retained_when_checkpoint_response_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Provider:
        def generate_json_result(self, prompt, **kwargs):
            from arc_llm.usage import LLMProviderResponse

            return LLMProviderResponse({"findings": ["retained"]})

    monkeypatch.setattr(runner_module, "select_provider", lambda *_args, **_kwargs: Provider())
    monkeypatch.setattr(
        runner_module,
        "record_response",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("checkpoint write failed")),
    )

    with pytest.raises(LLMTaskError, match="failed after 1 attempt"):
        run_json(
            "prompt",
            provider="codex-cli",
            env={},
            process_chain=[],
            artifact_dir=tmp_path,
            call_label="checkpoint-failure",
        )

    records = list((tmp_path / "attempts").glob("*/record.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    candidates = _read_stream(records[0].parent, record["streams"]["response_candidates"])
    assert '"retained"' in candidates
