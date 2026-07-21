from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import time

import pytest

from arc_llm.call_checkpoint import (
    LLMCallCheckpointError,
    LLMCallNeedsSupervision,
    LLMCallRetryDeferred,
    LLMCallRetryExhausted,
    checkpoint_path,
    prepare_call,
    record_failure,
    record_response,
    record_submitted,
    record_validated,
)
from arc_llm.usage import LLMProviderResponse, LLMUsage
from arc_llm.providers.base import (
    LLMSubmissionState,
    LLMWorkerCancelled,
    LLMWorkerError,
    LLMWorkerTimeout,
)
from arc_llm import runner
from arc_llm.runner import run_json


def _identity(tmp_path: Path) -> tuple[Path, str]:
    return checkpoint_path(
        tmp_path,
        prompt="prompt",
        schema={"type": "object"},
        provider="codex-cli",
        model="model",
        call_label="round/worker",
    )


def test_response_received_replays_without_provider_call(tmp_path: Path) -> None:
    path, identity = _identity(tmp_path)
    prepared = prepare_call(path, identity=identity, now=100)
    record_response(
        prepared,
        LLMProviderResponse(
            {"ok": True},
            usage=LLMUsage(input_tokens=3, output_tokens=2),
            raw_model_output='{"ok":true}',
        ),
    )

    replay = prepare_call(path, identity=identity, now=101)
    assert replay.replay_response is not None
    assert replay.replay_response.value == {"ok": True}
    assert replay.replay_response.usage.input_tokens == 3
    record_validated(replay)


def test_uncertain_call_never_retries_without_supervision(tmp_path: Path) -> None:
    path, identity = _identity(tmp_path)
    first = prepare_call(path, identity=identity, now=100)
    assert first.attempt == 1
    record_submitted(first)
    first.release_lock()  # simulate the owning process exiting without a response
    with pytest.raises(LLMCallRetryDeferred):
        prepare_call(path, identity=identity, now=3699)
    with pytest.raises(LLMCallRetryDeferred):
        prepare_call(path, identity=identity, now=7300)


def test_corrupt_checkpoint_fails_closed(tmp_path: Path) -> None:
    path, identity = _identity(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Could not read"):
        prepare_call(path, identity=identity, now=100)


def test_formatter_response_is_replayed_after_outer_checkpoint_crash(tmp_path: Path, monkeypatch) -> None:
    class Provider:
        calls = 0

        def generate_json_result(self, _prompt, *, schema=None, **_kwargs):
            self.calls += 1
            if isinstance(schema, dict) and "action" in schema.get("properties", {}):
                return LLMProviderResponse(
                    {
                        "action": "format",
                        "reason": "source contains the answer",
                        "formatted_output": {"ok": True},
                    }
                )
            return LLMProviderResponse(
                {"answer": "The source contains enough descriptive content to format safely."},
                raw_model_output="The source contains enough descriptive content to format safely.",
            )

    provider = Provider()
    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: provider)
    real_record_validated = runner.record_validated

    def crash_after_paid_recovery(prepared):
        if "schema_formatter" not in prepared.path.name:
            raise RuntimeError("simulated crash after formatter response")
        real_record_validated(prepared)

    monkeypatch.setattr(runner, "record_validated", crash_after_paid_recovery)
    kwargs = {
        "schema": {
            "type": "object",
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
            "additionalProperties": False,
        },
        "provider": "codex-cli",
        "env": {"ARC_HOME": str(tmp_path / "arc-home")},
        "process_chain": [],
        "output_recovery": "warn",
        "artifact_dir": tmp_path / "artifacts",
        "call_label": "loop/worker",
    }
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_json("prompt", **kwargs)
    assert provider.calls == 2

    monkeypatch.setattr(runner, "record_validated", real_record_validated)
    result = run_json("prompt", **kwargs)
    assert result["ok"] is True
    assert provider.calls == 2


def test_same_identity_is_single_flight_across_concurrent_callers(tmp_path: Path, monkeypatch) -> None:
    class Provider:
        calls = 0

        def generate_json_result(self, _prompt, **_kwargs):
            self.calls += 1
            time.sleep(0.15)
            return LLMProviderResponse({"ok": True})

    provider = Provider()
    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: provider)
    kwargs = {
        "schema": {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}},
        "provider": "codex-cli",
        "env": {"ARC_HOME": str(tmp_path / "arc-home")},
        "process_chain": [],
        "artifact_dir": tmp_path / "artifacts",
        "call_label": "same-call",
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run_json("prompt", **kwargs), range(2)))
    assert [result["ok"] for result in results] == [True, True]
    assert provider.calls == 1


def test_not_submitted_failure_does_not_consume_checkpoint_attempt(tmp_path: Path) -> None:
    path, identity = _identity(tmp_path)
    prepared = prepare_call(path, identity=identity, now=100)
    record_failure(
        prepared,
        LLMWorkerError("circuit open", submission_state=LLMSubmissionState.NOT_SUBMITTED),
    )
    retry = prepare_call(path, identity=identity, now=101)
    assert retry.attempt == 1
    retry.release_lock()


def test_unknown_submission_failure_requires_supervision(tmp_path: Path) -> None:
    path, identity = _identity(tmp_path)
    prepared = prepare_call(path, identity=identity, now=100)
    record_failure(
        prepared,
        LLMWorkerError("provider boundary uncertain", submission_state=LLMSubmissionState.UNKNOWN),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["state"] == "submitted"
    assert payload["submission_state"] == "unknown"
    with pytest.raises(LLMCallNeedsSupervision):
        prepare_call(path, identity=identity, now=101)


def test_untyped_failure_after_submission_requires_supervision(tmp_path: Path) -> None:
    path, identity = _identity(tmp_path)
    prepared = prepare_call(path, identity=identity, now=100)
    record_submitted(prepared)
    record_failure(prepared, RuntimeError("unexpected provider wrapper crash"))

    with pytest.raises(LLMCallNeedsSupervision):
        prepare_call(path, identity=identity, now=101)


def test_known_submitted_failure_is_terminal_and_never_replayed(tmp_path: Path) -> None:
    path, identity = _identity(tmp_path)
    prepared = prepare_call(path, identity=identity, now=100)
    record_failure(
        prepared,
        LLMWorkerError(
            "paid response was invalid",
            category="output_invalid",
            submission_state=LLMSubmissionState.SUBMITTED,
        ),
    )
    with pytest.raises(LLMCallRetryExhausted, match="known terminal failure") as caught:
        prepare_call(path, identity=identity, now=10_000)
    assert caught.value.checkpoint_path == path


def test_uncertain_checkpoint_exposes_supervision_path_and_disposition(tmp_path: Path) -> None:
    path, identity = _identity(tmp_path)
    prepared = prepare_call(path, identity=identity, now=100)
    record_submitted(prepared)
    prepared.release_lock()

    with pytest.raises(LLMCallNeedsSupervision) as caught:
        prepare_call(path, identity=identity, now=10_000)

    assert caught.value.checkpoint_path == path
    assert caught.value.submission_state == LLMSubmissionState.UNKNOWN


def test_prepared_not_submitted_checkpoint_is_safe_to_retry(tmp_path: Path) -> None:
    path, identity = _identity(tmp_path)
    first = prepare_call(path, identity=identity, now=100)
    first.release_lock()

    retry = prepare_call(path, identity=identity, now=101)

    assert retry.attempt == 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["state"] == "prepared"
    assert payload["submission_state"] == "not_submitted"
    retry.release_lock()


def test_v2_started_checkpoint_upgrades_without_replaying(tmp_path: Path) -> None:
    path, identity = _identity(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": "arc.llm.call_checkpoint.v2",
        "identity": identity,
        "state": "started",
        "submission_state": "unknown",
        "attempt": 1,
    }), encoding="utf-8")

    with pytest.raises(LLMCallNeedsSupervision):
        prepare_call(path, identity=identity, now=101)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "arc.llm.call_checkpoint.v3"
    assert payload["state"] == "submitted"


@pytest.mark.parametrize(
    "failure",
    [
        LLMWorkerTimeout("idle", submission_state=LLMSubmissionState.SUBMITTED),
        LLMWorkerCancelled("cancelled", submission_state=LLMSubmissionState.SUBMITTED),
    ],
)
def test_interrupted_submitted_call_records_recovery_metadata(
    tmp_path: Path, failure: LLMWorkerError
) -> None:
    artifact_dir = tmp_path / "artifacts"
    path, identity = _identity(artifact_dir)
    prepared = prepare_call(path, identity=identity, now=100)

    record_failure(prepared, failure)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["state"] == "failed"
    assert payload["resumable"] is True
    assert payload["progress_journal"] == str(artifact_dir / "progress.jsonl")


def test_resumable_submitted_checkpoint_requires_explicit_native_resume(tmp_path: Path) -> None:
    path, identity = _identity(tmp_path)
    prepared = prepare_call(path, identity=identity, now=100)
    record_failure(
        prepared,
        LLMWorkerTimeout("idle", submission_state=LLMSubmissionState.SUBMITTED),
    )

    with pytest.raises(LLMCallRetryExhausted):
        prepare_call(path, identity=identity, now=101)
    with pytest.raises(LLMCallCheckpointError, match="existing provider session id"):
        prepare_call(
            path,
            identity=identity,
            now=102,
            supervised_native_resume=True,
        )

    resumed = prepare_call(
        path,
        identity=identity,
        now=103,
        supervised_native_resume=True,
        native_session_available=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["state"] == "resuming"
    assert payload["resume_count"] == 1
    resumed.release_lock()

    with pytest.raises(LLMCallNeedsSupervision):
        prepare_call(path, identity=identity, now=104)
