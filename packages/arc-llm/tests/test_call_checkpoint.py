from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import os
import signal
import stat
import time

import pytest

from arc_llm.call_checkpoint import (
    LLMCallCheckpointError,
    LLMCallNeedsSupervision,
    LLMCallRetryDeferred,
    LLMCallRetryExhausted,
    checkpoint_recomputation_binding,
    checkpoint_path,
    prepare_call,
    promote_recovered_response,
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
from arc_llm import call_checkpoint as checkpoint_module
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


def _stateful_identity(
    tmp_path: Path, *, prompt: str = "prompt", model: str = "model",
) -> tuple[Path, str]:
    return checkpoint_path(
        tmp_path,
        prompt=prompt,
        schema={"type": "object"},
        provider="codex-cli",
        model=model,
        call_label="round/worker",
        session_policy="stateful",
        session_key="ch:translation",
        idempotency_key="logical-turn",
        generation=1,
        initial_native_authorization=_resume_authorization(tmp_path),
    )


def _resume_authorization(tmp_path: Path) -> tuple[str, str, str, int, str]:
    return (
        str((tmp_path / "control-ledger.json").resolve()),
        "ch:translation",
        "segment-1",
        1,
        "logical-turn",
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


def test_stateless_response_does_not_replay_changed_request_digest(tmp_path: Path) -> None:
    path, identity = _identity(tmp_path)
    prepared = prepare_call(path, identity=identity, now=100)
    record_response(prepared, LLMProviderResponse({"ok": True}))
    _other_path, changed = checkpoint_path(
        tmp_path, prompt="changed", schema={"type": "object"},
        provider="codex-cli", model="model", call_label="round/worker",
    )

    with pytest.raises(LLMCallCheckpointError, match="identity mismatch"):
        prepare_call(path, identity=changed, now=101)


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


def test_v4_response_checkpoint_upgrades_and_replays(tmp_path: Path) -> None:
    path, identity = _identity(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "arc.llm.call_checkpoint.v4",
                "identity": str(identity),
                "logical_identity": identity.logical_identity,
                "request_digest": identity.request_digest,
                "request_recipe": identity.request_recipe,
                "state": "response_received",
                "submission_state": "submitted",
                "attempt": 1,
                "response": {"value": {"ok": True}},
            }
        ),
        encoding="utf-8",
    )
    replay = prepare_call(path, identity=identity)
    assert replay.replay_response is not None
    assert replay.replay_response.value == {"ok": True}
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == (
        "arc.llm.call_checkpoint.v5"
    )


def test_v5_malformed_candidate_material_fails_as_checkpoint_error(tmp_path: Path) -> None:
    path, identity = _identity(tmp_path)
    prepared = prepare_call(path, identity=identity)
    record_response(prepared, LLMProviderResponse({"ok": True}))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["response"]["candidate_material"] = [{"protocol_position": "bad"}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LLMCallCheckpointError, match="invalid candidate material"):
        prepare_call(path, identity=identity)


def test_checkpoint_atomic_replace_fsyncs_parent_directory(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        calls.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(checkpoint_module.os, "fsync", record_fsync)
    path, identity = _identity(tmp_path)
    prepared = prepare_call(path, identity=identity)
    assert any(stat.S_ISDIR(mode) for mode in calls)
    prepared.release_lock()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_preexisting_checkpoint_lock_symlink_is_rejected_without_target_mutation(
    tmp_path: Path,
) -> None:
    path, identity = _identity(tmp_path)
    path.parent.mkdir(parents=True)
    outside = tmp_path / "outside-preexisting-lock"
    outside.write_bytes(b"outside preexisting lock sentinel\n")
    outside_before = outside.read_bytes()
    path.with_name(path.name + ".lock").symlink_to(outside)

    with pytest.raises(LLMCallCheckpointError, match="Could not lock"):
        prepare_call(path, identity=identity)

    assert outside.read_bytes() == outside_before


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_checkpoint_parent_swap_never_writes_external_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, identity = _identity(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    outside = external / path.name
    outside.write_bytes(b"outside checkpoint sentinel\n")
    outside_before = outside.read_bytes()
    moved = tmp_path / "saved-call-checkpoints"

    def swap_parent(_path: Path) -> None:
        path.parent.rename(moved)
        path.parent.symlink_to(external, target_is_directory=True)

    monkeypatch.setattr(checkpoint_module, "_before_checkpoint_replace", swap_parent)

    with pytest.raises(LLMCallCheckpointError):
        prepare_call(path, identity=identity)

    assert outside.read_bytes() == outside_before
    assert not (moved / path.name).exists()


def test_checkpoint_leaf_swap_never_overwrites_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, identity = _identity(tmp_path)
    prepared = prepare_call(path, identity=identity, now=100)
    prepared.release_lock()
    stale = b"stale checkpoint replacement\n"

    def swap_leaf(_path: Path) -> None:
        path.unlink()
        path.write_bytes(stale)

    monkeypatch.setattr(checkpoint_module, "_before_checkpoint_replace", swap_leaf)

    with pytest.raises(LLMCallCheckpointError, match="address changed"):
        prepare_call(path, identity=identity, now=101)

    assert path.read_bytes() == stale


def test_checkpoint_exchange_rolls_back_late_regular_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, identity = _identity(tmp_path)
    prepared = prepare_call(path, identity=identity)
    sentinel = b"late checkpoint replacement\n"

    def swap_at_exchange(_path: Path) -> None:
        path.unlink()
        path.write_bytes(sentinel)

    monkeypatch.setattr(
        checkpoint_module, "_before_checkpoint_exchange", swap_at_exchange,
    )
    try:
        with pytest.raises(LLMCallCheckpointError, match="final publication window"):
            record_submitted(prepared)
    finally:
        prepared.release_lock()
    assert path.read_bytes() == sentinel


def test_existing_checkpoint_update_fails_closed_without_atomic_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, identity = _identity(tmp_path)
    prepared = prepare_call(path, identity=identity)
    before = path.read_bytes()
    monkeypatch.setattr(
        checkpoint_module, "_checkpoint_rename_exchange", lambda *_args: False,
    )
    try:
        with pytest.raises(LLMCallCheckpointError, match="unsupported"):
            record_submitted(prepared)
    finally:
        prepared.release_lock()
    assert path.read_bytes() == before


def test_initial_checkpoint_hardlink_never_overwrites_late_appearance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, identity = _identity(tmp_path)
    sentinel = b"late initial checkpoint replacement\n"

    def appear_at_exchange(_path: Path) -> None:
        path.write_bytes(sentinel)

    monkeypatch.setattr(
        checkpoint_module, "_before_checkpoint_exchange", appear_at_exchange,
    )
    with pytest.raises(LLMCallCheckpointError, match="appeared"):
        prepare_call(path, identity=identity)
    assert path.read_bytes() == sentinel


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_checkpoint_lock_swap_never_writes_symlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, identity = _identity(tmp_path)
    outside = tmp_path / "outside-lock"
    outside.write_bytes(b"outside lock sentinel\n")
    outside_before = outside.read_bytes()
    lock_path = path.with_name(path.name + ".lock")

    def swap_lock(_path: Path) -> None:
        lock_path.unlink()
        lock_path.symlink_to(outside)

    monkeypatch.setattr(checkpoint_module, "_before_checkpoint_replace", swap_lock)

    with pytest.raises(LLMCallCheckpointError, match="lock address changed"):
        prepare_call(path, identity=identity)

    assert outside.read_bytes() == outside_before


def test_formatter_response_is_replayed_after_outer_checkpoint_crash(tmp_path: Path, monkeypatch) -> None:
    class Provider:
        calls = 0

        def generate_json_result(self, prompt, *, schema=None, **_kwargs):
            self.calls += 1
            if (
                isinstance(schema, dict) and "action" in schema.get("properties", {})
            ) or ('"action"' in prompt and '"formatted_output"' in prompt):
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


def test_prepared_not_submitted_request_change_is_rebuilt(tmp_path: Path) -> None:
    path, original = _identity(tmp_path)
    prepared = prepare_call(path, identity=original, now=100)
    prepared.release_lock()
    _unused, changed = checkpoint_path(
        tmp_path, prompt="changed", schema={"type": "object"},
        provider="codex-cli", model="model", call_label="round/worker",
    )

    rebuilt = prepare_call(path, identity=changed, now=101)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert rebuilt.identity == changed
    assert payload["logical_identity"] == changed.logical_identity
    assert payload["request_digest"] == changed.request_digest
    assert payload["submission_state"] == "not_submitted"
    rebuilt.release_lock()


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
    assert payload["schema_version"] == "arc.llm.call_checkpoint.v5"
    assert payload["state"] == "submitted"


def test_v3_changed_request_requires_validated_legacy_logical_identity(tmp_path: Path) -> None:
    path, original = checkpoint_path(
        tmp_path, prompt="old bootstrap", schema={"type": "object"},
        provider="codex-cli", model="m", call_label="turn",
        session_policy="stateful", session_key="ch:translation",
        idempotency_key="logical-turn", generation=1,
        initial_native_authorization=_resume_authorization(tmp_path),
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": "arc.llm.call_checkpoint.v3",
        "identity": str(original), "state": "failed",
        "submission_state": "submitted", "resumable": True, "attempt": 1,
    }), encoding="utf-8")
    _same_path, rebuilt = checkpoint_path(
        tmp_path, prompt="rebuilt delta", schema={"type": "object"},
        provider="codex-cli", model="m", call_label="turn",
        session_policy="stateful", session_key="ch:translation",
        idempotency_key="logical-turn", generation=1,
        initial_native_authorization=_resume_authorization(tmp_path),
    )

    with pytest.raises(LLMCallCheckpointError, match="initial authorization"):
        prepare_call(
            path, identity=rebuilt,
            supervised_native_resume=_resume_authorization(tmp_path),
            native_session_available=True,
        )
    resumed = prepare_call(
        path, identity=rebuilt,
        supervised_native_resume=_resume_authorization(tmp_path),
        native_session_available=True,
        validated_legacy_logical_identity=rebuilt.logical_identity,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "arc.llm.call_checkpoint.v5"
    assert payload["logical_identity"] == rebuilt.logical_identity
    assert payload["request_digest"] == str(original)
    assert payload["state"] == "resuming"
    resumed.release_lock()


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
    path, identity = _stateful_identity(tmp_path)
    authorization = _resume_authorization(tmp_path)
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
            supervised_native_resume=authorization,
        )

    resumed = prepare_call(
        path,
        identity=identity,
        now=103,
        supervised_native_resume=authorization,
        native_session_available=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["state"] == "resuming"
    assert payload["resume_count"] == 1
    resumed.release_lock()

    with pytest.raises(LLMCallNeedsSupervision):
        prepare_call(path, identity=identity, now=104)


def test_supervised_native_resume_preserves_submitted_identity_when_prompt_changes(
    tmp_path: Path,
) -> None:
    path, original_identity = _stateful_identity(tmp_path)
    authorization = _resume_authorization(tmp_path)
    prepared = prepare_call(path, identity=original_identity, now=100)
    record_failure(
        prepared,
        LLMWorkerTimeout("idle", submission_state=LLMSubmissionState.SUBMITTED),
    )

    _new_path, rebuilt_identity = _stateful_identity(
        tmp_path, prompt="rebuilt stateful stream"
    )
    with pytest.raises(LLMCallCheckpointError, match="identity mismatch"):
        prepare_call(path, identity=rebuilt_identity, now=101)
    with pytest.raises(LLMCallCheckpointError, match="identity mismatch"):
        prepare_call(
            path,
            identity=rebuilt_identity,
            now=102,
            supervised_native_resume=authorization,
            native_session_available=False,
        )

    resumed = prepare_call(
        path,
        identity=rebuilt_identity,
        now=103,
        supervised_native_resume=authorization,
        native_session_available=True,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert resumed.identity == original_identity
    assert payload["identity"] == original_identity
    assert payload["state"] == "resuming"
    assert payload["resume_count"] == 1
    resumed.release_lock()


def test_recomputation_binding_retains_exact_recipe_and_full_authorization(
    tmp_path: Path,
) -> None:
    path, identity = _stateful_identity(tmp_path)
    authorization = _resume_authorization(tmp_path)
    prepared = prepare_call(path, identity=identity, now=100)
    record_failure(
        prepared,
        LLMWorkerTimeout("idle", submission_state=LLMSubmissionState.SUBMITTED),
    )
    resumed = prepare_call(
        path,
        identity=identity,
        now=101,
        supervised_native_resume=authorization,
        native_session_available=True,
    )

    binding = resumed.recomputation_binding
    resumed.release_lock()
    assert checkpoint_recomputation_binding(path) == binding
    assert binding["checkpoint_path"] == str(path.resolve())
    assert binding["checkpoint_identity"] == str(identity)
    assert binding["logical_identity"]["control_address"] == authorization[0]
    assert binding["logical_identity"]["logical_unit"] == authorization[2]
    assert binding["idempotency_key"] == "logical-turn"
    assert binding["session_key"] == "ch:translation"
    assert binding["generation"] == 1
    assert binding["prompt_sha256"] == identity.request_recipe["prompt_sha256"]
    assert binding["schema_sha256"] == identity.request_recipe["schema_sha256"]
    assert binding["call_label_sha256"] == checkpoint_module.sha256_text(
        identity.request_recipe["call_label"]
    )
    assert binding["native_resume_authorization"] == {
        "control_address": authorization[0],
        "session_key": authorization[1],
        "logical_unit": authorization[2],
        "generation": authorization[3],
        "idempotency_key": authorization[4],
    }
    assert binding["initial_native_authorization"] == binding[
        "native_resume_authorization"
    ]


def test_recovered_promotion_rejects_changed_recomputation_binding_before_reading_receipt(
    tmp_path: Path,
) -> None:
    path, identity = _identity(tmp_path)
    prepared = prepare_call(path, identity=identity, now=100)
    record_submitted(prepared)
    binding = prepared.recomputation_binding
    prepared.release_lock()
    changed = dict(binding)
    changed["prompt_sha256"] = "f" * 64

    with pytest.raises(LLMCallCheckpointError, match="recomputation binding mismatch"):
        promote_recovered_response(
            path,
            LLMProviderResponse({"ok": True}),
            expected_logical_identity=identity.logical_identity,
            expected_schema_sha256=identity.request_recipe["schema_sha256"],
            selection_receipt_path=f"{path.stem}.candidate-selection.json",
            selection_receipt_sha256="0" * 64,
            expected_recomputation_binding=changed,
        )


def test_supervised_resume_never_rebinds_unsubmitted_identity(tmp_path: Path) -> None:
    path, original_identity = _stateful_identity(tmp_path)
    prepared = prepare_call(path, identity=original_identity, now=100)
    prepared.release_lock()

    with pytest.raises(LLMCallCheckpointError, match="structured call identity"):
        prepare_call(
            path,
            identity="rebuilt-prompt",
            now=101,
            supervised_native_resume=_resume_authorization(tmp_path),
            native_session_available=True,
        )

    with pytest.raises(LLMCallCheckpointError, match="unsubmitted checkpoint"):
        prepare_call(
            path,
            identity=original_identity,
            now=102,
            supervised_native_resume=_resume_authorization(tmp_path),
            native_session_available=True,
        )


@pytest.mark.parametrize("field_index", range(5))
def test_persisted_native_resume_authorization_rejects_each_field_mismatch(
    tmp_path: Path, field_index: int,
) -> None:
    path, identity = _stateful_identity(tmp_path)
    prepared = prepare_call(path, identity=identity, now=100)
    record_failure(
        prepared,
        LLMWorkerTimeout("idle", submission_state=LLMSubmissionState.SUBMITTED),
    )
    authorization = _resume_authorization(tmp_path)
    resumed = prepare_call(
        path,
        identity=identity,
        now=101,
        supervised_native_resume=authorization,
        native_session_available=True,
    )
    resumed.release_lock()
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["native_resume_authorization"] == {
        "control_address": authorization[0],
        "session_key": authorization[1],
        "logical_unit": authorization[2],
        "generation": authorization[3],
        "idempotency_key": authorization[4],
    }
    mismatched = list(authorization)
    replacements = (
        str((tmp_path / "other-ledger.json").resolve()),
        "other:translation",
        "segment-2",
        2,
        "other-turn",
    )
    mismatched[field_index] = replacements[field_index]

    with pytest.raises(LLMCallCheckpointError, match="authorization"):
        prepare_call(
            path,
            identity=identity,
            now=102,
            supervised_native_resume=tuple(mismatched),
            native_session_available=True,
        )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX process crash")
def test_real_process_crash_checkpoint_stage_is_cleaned_on_next_lock(
    tmp_path: Path,
) -> None:
    path, identity = _identity(tmp_path)
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child is intentionally SIGKILLed
        checkpoint_module._before_checkpoint_replace = (
            lambda _path: os.kill(os.getpid(), signal.SIGKILL)
        )
        prepare_call(path, identity=identity, now=100)
        os._exit(99)
    _finished, status = os.waitpid(pid, 0)
    assert os.WIFSIGNALED(status)
    assert os.WTERMSIG(status) == signal.SIGKILL
    assert list(path.parent.glob(f".{path.name}.arc-stage-*"))

    prepared = prepare_call(path, identity=identity, now=101)
    assert prepared.attempt == 1
    prepared.release_lock()
    assert path.is_file()
    assert not list(path.parent.glob(f".{path.name}.arc-stage-*"))
