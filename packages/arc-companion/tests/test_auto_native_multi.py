from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

import arc_companion.pipeline as pipeline
from arc_companion.ledger import (
    advance_block,
    initialize_lane_ledger,
    invalidate_suffix,
    mark_needs_supervision,
)
from arc_companion.resume_transaction import begin_transaction
from arc_companion.recovery_responses import (
    seal_submission_attempts,
    submission_receipt_reference,
    write_ledger_submission_receipt,
)
from arc_llm.attempt_diagnostics import AttemptDiagnostics
from arc_llm import run_json
from arc_llm.providers.base import LLMSubmissionState, LLMWorkerTimeout
from arc_llm.schema_cache import canonical_json
from arc_llm.sessions import LLMSessionManager


SESSION_KEY = "ch-0001:translation"
FAKE_KIMI = (
    Path(__file__).parents[2] / "arc-llm" / "tests" / "fixtures" / "fake_kimi_acp.py"
)


def _project(tmp_path: Path) -> tuple[Path, Path, Path, LLMSessionManager]:
    project = tmp_path / "run"
    checkpoint = project / "checkpoint"
    ledger_path = checkpoint / "chapters" / "ch-0001" / "translation-ledger.json"
    initialize_lane_ledger(
        ledger_path,
        chapter_id="ch-0001",
        lane="translation",
        segment_ids=["s1", "s2"],
    )
    manager = LLMSessionManager(checkpoint / "sessions")
    manager.get_or_create(
        key=SESSION_KEY,
        provider="codex-cli",
        model="test-model",
        runtime_fingerprint="runtime",
    )
    manager.update_native_session_id(SESSION_KEY, "native-1")
    project.mkdir(parents=True, exist_ok=True)
    (project / "state.json").write_text(json.dumps({
        "status": "needs_supervision",
        "checkpoint_dir": str(checkpoint),
        "recovery_options": {
            "paper_id": "local:auto-native-multi",
            "workers": 1,
            "recovery_policy": "auto",
        },
    }), encoding="utf-8")
    return project, checkpoint, ledger_path, manager


def _accept_remaining(path: Path) -> None:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    states = [
        "prepared", "submitted", "response_received", "schema_valid",
        "invariant_valid", "accepted",
    ]
    for block in ledger["blocks"]:
        segment_id = str(block["segment_id"])
        current = str(block["state"])
        for state in states[states.index(current) + 1:]:
            advance_block(
                path,
                segment_id=segment_id,
                state=state,
                input_sha256=f"input-{segment_id}",
                output_sha256=f"output-{segment_id}",
            )


def _typed_idle_context(
    checkpoint: Path,
    ledger_path: Path,
    key: str,
    *,
    response: dict | None = None,
) -> dict:
    artifact = checkpoint / "llm" / key.rsplit(":", 2)[-2]
    call_path = (
        artifact / "call-checkpoints"
        / f"idempotency-{hashlib.sha256(key.encode()).hexdigest()}.json"
    )
    call_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_identity = f"identity-{hashlib.sha256(key.encode()).hexdigest()}"
    schema = {
        "type": "object", "properties": {}, "additionalProperties": True,
    }
    prompt = "typed idle test prompt"
    authorization = {
        "control_address": str(ledger_path.resolve(strict=False)),
        "session_key": SESSION_KEY,
        "logical_unit": "s1",
        "generation": 1,
        "idempotency_key": key,
    }
    call_path.write_text(json.dumps({
        "identity": checkpoint_identity,
        "state": "response_received" if response is not None else "failed",
        "submission_state": "submitted",
        "resumable": response is None,
        "response": response,
        "logical_identity": {
            "provider": "codex-cli", "model": "test-model",
            "idempotency_key": key, "session_key": SESSION_KEY,
            "generation": 1,
            "control_address": authorization["control_address"],
            "logical_unit": "s1",
            "initial_native_authorization": authorization,
        },
        "initial_native_authorization": authorization,
        "request_recipe": {
            "runtime_fingerprint": "runtime",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "schema_sha256": hashlib.sha256(canonical_json(schema).encode()).hexdigest(),
            "call_label": "call-s1",
        },
    }), encoding="utf-8")
    latest = {
        "event": "idle_timeout", "idempotency_key": key,
        "session_key": SESSION_KEY, "generation": 1,
        "checkpoint_identity": checkpoint_identity,
        "native_session_id": "native-1",
    }
    (call_path.parent.parent / "progress.jsonl").write_text(
        json.dumps(latest) + "\n", encoding="utf-8",
    )
    context = {
        "idempotency_key": key,
        "checkpoint_path": str(call_path),
        "submission_state": "submitted",
        "resumable": response is None,
        "native_session_id": "native-1",
        "session_key": SESSION_KEY,
        "generation": 1,
        "provider": "codex-cli",
        "model": "test-model",
        "runtime_fingerprint": "runtime",
        "logical_unit": "s1",
        "latest_progress": latest,
    }
    if response is None:
        receipt_path = write_ledger_submission_receipt(
            checkpoint_dir=checkpoint,
            artifact_dir=artifact,
            ledger_path=ledger_path,
            session_key=SESSION_KEY,
            logical_unit="s1",
            generation=1,
            idempotency_key=key,
            schema=schema,
            prompt=prompt,
            recovery_unit="translation",
            input_sha256="input-s1",
            ordered_siblings=["s1", "s2"],
            suffix=["s1", "s2"],
            validator="translation-schema+invariants.v1",
            application="normal-translation-pipeline-replay.v1",
        )
        diagnostics = AttemptDiagnostics(
            artifact,
            provider="codex-cli",
            model="test-model",
            fallback_index=0,
            attempt=1,
            call_label="call-s1",
            env={},
        )
        diagnostics.bind_checkpoint_identity(checkpoint_identity)
        diagnostics.mark_submitted()
        attempt_ref = diagnostics.finalize(
            outcome="timeout",
            error=LLMWorkerTimeout(
                "idle", submission_state=LLMSubmissionState.SUBMITTED,
            ),
        )
        seal_submission_attempts(
            receipt_path,
            checkpoint_dir=checkpoint,
            attempt_references=[{
                "path": (artifact / attempt_ref.path).relative_to(checkpoint).as_posix(),
                "sha256": attempt_ref.sha256,
            }],
        )
        context["submission_receipt"] = submission_receipt_reference(
            receipt_path, checkpoint_dir=checkpoint,
        )
    return context


def test_same_lane_resumable_blockers_are_all_native_reconciled_before_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _checkpoint, ledger_path, manager = _project(tmp_path)
    keys = {
        segment_id: f"{SESSION_KEY}:call-{segment_id}:generation-1"
        for segment_id in ("s1", "s2")
    }
    for segment_id in ("s1", "s2"):
        mark_needs_supervision(
            ledger_path,
            segment_id=segment_id,
            reason=f"submitted {segment_id}",
            recovery_context={
                "idempotency_key": keys[segment_id],
                "submission_state": "submitted",
                "resumable": True,
                "native_session_id": "native-1",
                "session_key": SESSION_KEY,
                "generation": 1,
            },
        )

    validated_segments: list[str] = []

    def validate_context(**kwargs):
        ledger = kwargs["ledger"]
        supervision = kwargs.get("supervision") or ledger["needs_supervision"]
        segment_id = str(supervision["segment_id"])
        validated_segments.append(segment_id)
        return {
            "session_key": SESSION_KEY,
            "segment_id": segment_id,
            "ledger_path": str(kwargs["ledger_path"]),
            "idempotency_key": keys[segment_id],
            "provider": "codex-cli",
            "model": "test-model",
            "runtime_fingerprint": "runtime",
            "generation": 1,
            "native_session_id_to_restore": None,
        }

    captured_keys: list[str] = []

    def continuation(options):
        captured_keys.extend(options.supervised_native_resume_keys)
        _accept_remaining(ledger_path)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "_validate_native_resume_context", validate_context)
    monkeypatch.setattr(pipeline, "build_companion", continuation)

    result = pipeline.resume_companion(project, action="auto")

    assert result["ok"] is True, result
    assert validated_segments == ["s1", "s2"]
    assert set(captured_keys) == set(keys.values())
    assert manager.get_existing(SESSION_KEY).generation == 1


def test_rotated_generation_does_not_reuse_stale_native_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _checkpoint, ledger_path, manager = _project(tmp_path)
    stale_key = f"{SESSION_KEY}:call-s1:generation-1"
    manager.rotate(SESSION_KEY, reason="crash after automatic rotate")
    invalidate_suffix(ledger_path, from_segment_id="s1", generation=2)
    mark_needs_supervision(
        ledger_path,
        segment_id="s1",
        reason="generation two continuation pending",
        recovery_context={
            "idempotency_key": f"{SESSION_KEY}:call-s1:generation-2",
            "submission_state": "submitted",
            "resumable": False,
            "generation": 2,
        },
    )
    begin_transaction(
        project,
        action="auto",
        policy="auto",
        recovery_options={
            "paper_id": "local:auto-native-multi",
            "workers": 1,
            "recovery_policy": "auto",
        },
        entries=[{
            "ledger_path": str(ledger_path),
            "session_key": SESSION_KEY,
            "segment_id": "s1",
            "idempotency_key": stale_key,
            "initial_generation": 1,
            "target_generation": 2,
            "recovery_action": "generation_restart_required",
        }],
        native_resume_contexts=[{
            "ledger_path": str(ledger_path),
            "session_key": SESSION_KEY,
            "segment_id": "s1",
            "idempotency_key": stale_key,
            "provider": "codex-cli",
            "model": "test-model",
            "runtime_fingerprint": "runtime",
            "generation": 1,
            "native_session_id_to_restore": "native-1",
        }],
    )
    captured_keys: list[str] = []

    def continuation(options):
        captured_keys.extend(options.supervised_native_resume_keys)
        _accept_remaining(ledger_path)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", continuation)

    result = pipeline.resume_companion(project, action="auto")

    assert result["ok"] is True
    assert stale_key not in captured_keys
    assert manager.get_existing(SESSION_KEY).generation == 2
    assert json.loads(ledger_path.read_text(encoding="utf-8"))["generation"] == 2


def test_typed_idle_auto_skips_old_native_session_and_starts_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, checkpoint, ledger_path, manager = _project(tmp_path)
    key = f"{SESSION_KEY}:call-s1:generation-1"
    mark_needs_supervision(
        ledger_path,
        segment_id="s1",
        reason="provider became idle",
        recovery_context=_typed_idle_context(checkpoint, ledger_path, key),
    )
    validated = 0
    calls: list[tuple[int, tuple[str, ...]]] = []

    def forbidden_validate(**_kwargs):
        nonlocal validated
        validated += 1
        raise AssertionError("typed idle auto recovery must not validate old native context")

    def continuation(options):
        calls.append((manager.get_existing(SESSION_KEY).generation,
                      options.supervised_native_resume_keys))
        _accept_remaining(ledger_path)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "_validate_native_resume_context", forbidden_validate)
    monkeypatch.setattr(pipeline, "build_companion", continuation)

    result = pipeline.resume_companion(project, action="auto")

    assert result["ok"] is True, result
    assert validated == 0
    assert calls == [(2, ())]
    assert manager.get_existing(SESSION_KEY).generation == 2
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    entry = journal["entries"][0]
    assert entry["recovery_trigger"] == "idle_timeout"
    assert entry["automatic_native_resume_suppressed"] is True
    assert entry["fresh_generation_required"] is True
    assert journal["replacements"][0]["trigger_code"] == "idle_timeout"


def test_explicit_resume_native_keeps_typed_idle_old_session_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _checkpoint, ledger_path, manager = _project(tmp_path)
    key = f"{SESSION_KEY}:call-s1:generation-1"
    mark_needs_supervision(
        ledger_path,
        segment_id="s1",
        reason="provider became idle",
        recovery_context={
            "idempotency_key": key,
            "submission_state": "submitted",
            "resumable": True,
            "native_session_id": "native-1",
            "session_key": SESSION_KEY,
            "generation": 1,
            "logical_unit": "s1",
            "latest_progress": {
                "event": "idle_timeout", "idempotency_key": key,
                "session_key": SESSION_KEY, "generation": 1,
            },
        },
    )

    monkeypatch.setattr(
        pipeline,
        "_validate_native_resume_context",
        lambda **kwargs: {
            "session_key": SESSION_KEY,
            "segment_id": "s1",
            "ledger_path": str(kwargs["ledger_path"]),
            "idempotency_key": key,
            "provider": "codex-cli",
            "model": "test-model",
            "runtime_fingerprint": "runtime",
            "generation": 1,
            "native_session_id_to_restore": None,
        },
    )
    calls: list[tuple[int, tuple[str, ...]]] = []

    def continuation(options):
        calls.append((manager.get_existing(SESSION_KEY).generation,
                      options.supervised_native_resume_keys))
        _accept_remaining(ledger_path)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", continuation)

    result = pipeline.resume_companion(project, action="resume-native")

    assert result["ok"] is True
    assert calls == [(1, (key,))]
    assert manager.get_existing(SESSION_KEY).generation == 1


def test_typed_idle_complete_checkpoint_replays_without_fresh_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, checkpoint, ledger_path, manager = _project(tmp_path)
    key = f"{SESSION_KEY}:call-s1:generation-1"
    response = {"value": {"translations": [{"segment_id": "s1"}]}}
    mark_needs_supervision(
        ledger_path,
        segment_id="s1",
        reason="ledger acceptance crashed after provider became idle",
        recovery_context=_typed_idle_context(
            checkpoint, ledger_path, key, response=response,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_validate_native_resume_context",
        lambda **kwargs: {
            "session_key": SESSION_KEY,
            "segment_id": "s1",
            "ledger_path": str(kwargs["ledger_path"]),
            "idempotency_key": key,
            "provider": "codex-cli",
            "model": "test-model",
            "runtime_fingerprint": "runtime",
            "generation": 1,
            "native_session_id_to_restore": None,
        },
    )
    calls: list[tuple[int, tuple[str, ...]]] = []

    def continuation(options):
        calls.append((manager.get_existing(SESSION_KEY).generation,
                      options.supervised_native_resume_keys))
        _accept_remaining(ledger_path)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", continuation)

    result = pipeline.resume_companion(project, action="auto")

    assert result["ok"] is True
    assert calls == [(1, ())]
    assert manager.get_existing(SESSION_KEY).generation == 1
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert journal["entries"][0]["complete_response_scan"] == {
        "complete": True,
        "source": "call_checkpoint_response_received",
        "validation_status": "pending_normal_validation",
    }
    assert journal["entries"][0]["fresh_generation_required"] is False
    assert journal["replacements"] == []


def test_native_resume_authorization_matches_full_tuple_not_shared_key(
    tmp_path: Path,
) -> None:
    shared_key = "shared-provider-key"
    first_ledger = tmp_path / "checkpoint" / "chapters" / "a" / "translation-ledger.json"
    second_ledger = tmp_path / "checkpoint" / "chapters" / "b" / "companion-ledger.json"
    first = (
        str(first_ledger.resolve()), "a:translation", "a-1", 1, shared_key,
    )
    second = (
        str(second_ledger.resolve()), "b:companion", "b-1", 2, shared_key,
    )
    options = pipeline.BuildOptions(
        paper_id="local:full-tuple",
        project_dir=tmp_path,
        supervised_native_resume_identities=(second,),
    )

    assert pipeline._supervised_native_resume_authorized(
        options,
        ledger_path=first_ledger,
        session_key="a:translation",
        logical_unit="a-1",
        generation=1,
        idempotency_key=shared_key,
    ) is None
    assert pipeline._supervised_native_resume_authorized(
        options,
        ledger_path=second_ledger,
        session_key="b:companion",
        logical_unit="b-1",
        generation=2,
        idempotency_key=shared_key,
    ) == second
    # This property is audit compatibility only; equal keys never authorize
    # either stateful adapter without the exact five-field membership above.
    assert options.supervised_native_resume_keys == (shared_key,)


def test_callback_crash_reconstructs_receipt_and_starts_fresh_without_old_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, checkpoint, ledger_path, manager = _project(tmp_path)
    key = f"{SESSION_KEY}:call-s1:generation-1"
    # Persist the provider checkpoint, terminal idle event, T05 attempt, and
    # indexed sealed receipt, but deliberately omit the lane callback marker.
    _typed_idle_context(checkpoint, ledger_path, key)
    assert json.loads(ledger_path.read_text())["needs_supervision"] is None

    old_native_validations = 0
    calls: list[tuple[int, tuple[tuple[str, str, str, int, str], ...]]] = []

    def forbidden_validate(**_kwargs):
        nonlocal old_native_validations
        old_native_validations += 1
        raise AssertionError("callback-crash idle recovery must not resume old native")

    def continuation(options):
        calls.append((
            manager.get_existing(SESSION_KEY).generation,
            options.supervised_native_resume_identities,
        ))
        _accept_remaining(ledger_path)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "_validate_native_resume_context", forbidden_validate)
    monkeypatch.setattr(pipeline, "build_companion", continuation)

    result = pipeline.resume_companion(project, action="auto")

    assert result["ok"] is True
    assert old_native_validations == 0
    assert calls == [(2, ())]
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    entry = journal["entries"][0]
    reference = entry["recovery_context"]["submission_receipt"]
    assert set(reference) == {"path", "sha256", "identity_sha256"}
    assert entry["reconstructed_from_durable_state"] is True
    assert entry["fresh_generation_required"] is True


@pytest.mark.filterwarnings("ignore:kimi-code-cli is experimental.*:RuntimeWarning")
def test_callback_crash_runs_one_real_fresh_provider_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, checkpoint, ledger_path, manager = _project(tmp_path)
    key = f"{SESSION_KEY}:call-s1:generation-1"
    mark_needs_supervision(
        ledger_path,
        segment_id="s1",
        reason="controller callback crashed after an idle provider attempt",
        recovery_context=_typed_idle_context(checkpoint, ledger_path, key),
    )
    record_path = tmp_path / "fake-kimi-fresh.jsonl"
    for name, value in {
        "ARC_KIMI_BIN": str(FAKE_KIMI),
        "ARC_HOME": str(tmp_path / "arc-home"),
        "ARC_LLM_CACHE": str(tmp_path / "arc-home" / "cache" / "arc-llm"),
        "ARC_KIMI_WORK_DIR": str(tmp_path),
        "ARC_KIMI_IDLE_TIMEOUT_SECONDS": "5",
        "FAKE_KIMI_RECORD": str(record_path),
        "FAKE_KIMI_SCENARIO": "happy",
        "FAKE_KIMI_OUTPUT": '{"accepted":true}',
    }.items():
        monkeypatch.setenv(name, value)
    for name in (
        "ARC_LLM_TIMEOUT_SECONDS", "ARC_KIMI_TIMEOUT_SECONDS",
        "ARC_CODEX_TIMEOUT_SECONDS", "ARC_CLAUDE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    provider_sessions = LLMSessionManager(checkpoint / "fresh-provider-sessions")

    def forbidden_validate(**_kwargs):
        raise AssertionError("fresh recovery must not validate the abandoned native session")

    def continuation(options):
        assert manager.get_existing(SESSION_KEY).generation == 2
        assert options.supervised_native_resume_identities == ()
        value = run_json(
            "Complete the unresolved task in a fresh provider session.",
            schema={
                "type": "object", "additionalProperties": False,
                "required": ["accepted"],
                "properties": {"accepted": {"type": "boolean"}},
            },
            provider="kimi-code-cli",
            artifact_dir=checkpoint / "fresh-provider-call",
            call_label="callback-crash-fresh-provider",
            session_policy="stateful",
            session_key="callback-crash:fresh",
            session_manager=provider_sessions,
            idempotency_key="callback-crash:fresh:generation-2",
        )
        assert value["accepted"] is True
        _accept_remaining(ledger_path)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "_validate_native_resume_context", forbidden_validate)
    monkeypatch.setattr(pipeline, "build_companion", continuation)

    result = pipeline.resume_companion(project, action="auto")

    assert result["ok"] is True, result
    messages = [
        item["message"] for item in (
            json.loads(line) for line in record_path.read_text().splitlines()
        )
        if item.get("kind") == "client_message"
    ]
    methods = [str(item.get("method") or "") for item in messages]
    assert methods.count("session/prompt") == 1
    assert methods.count("session/new") == 1
    assert methods.count("session/resume") == 0


def test_stateless_callback_crash_reconstructs_from_indexed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, checkpoint, _chapter_ledger, manager = _project(tmp_path)
    artifact = checkpoint / "segmentation" / "window-0001"
    logical_unit = "window-0001"
    prompt = "segment this window"
    schema = {"type": "object", "properties": {}, "additionalProperties": True}
    descriptor = pipeline.submission_descriptor(
        unit="segmentation",
        logical_unit=logical_unit,
        checkpoint_dir=checkpoint,
        artifact_root=artifact,
        acceptance_checkpoint=checkpoint / "segmentation.json",
        input_sha256="a" * 64,
        ordered_siblings=[logical_unit],
        suffix=[logical_unit],
    )
    control = pipeline._prepare_pipeline_recovery_control(
        descriptor, artifact_dir=artifact,
    )
    key = str(control["idempotency_key"])
    receipt_path = write_ledger_submission_receipt(
        checkpoint_dir=checkpoint,
        artifact_dir=artifact,
        ledger_path=Path(control["ledger_path"]),
        session_key=str(control["session_key"]),
        logical_unit=logical_unit,
        generation=int(control["generation"]),
        idempotency_key=key,
        schema=schema,
        prompt=prompt,
        recovery_unit="segmentation",
        input_sha256="a" * 64,
        ordered_siblings=[logical_unit],
        suffix=[logical_unit],
        validator=str(control["validator"]),
        application=str(control["application"]),
        acceptance_checkpoint=Path(control["acceptance_checkpoint"]),
        stateful_checkpoint_identity=False,
    )
    checkpoint_identity = "stateless-checkpoint-identity"
    call_path = (
        artifact / "call-checkpoints"
        / f"idempotency-{hashlib.sha256(key.encode()).hexdigest()}.json"
    )
    call_path.parent.mkdir(parents=True)
    call_path.write_text(json.dumps({
        "identity": checkpoint_identity,
        "state": "failed",
        "submission_state": "submitted",
        "resumable": True,
        "failure_category": "timeout",
        "logical_identity": {
            "provider": "codex-cli", "model": "test-model",
            "idempotency_key": key, "session_key": None, "generation": None,
        },
        "request_recipe": {
            "runtime_fingerprint": "runtime",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "schema_sha256": hashlib.sha256(canonical_json(schema).encode()).hexdigest(),
            "call_label": "stateless-segmentation-window-0001",
        },
    }), encoding="utf-8")
    (artifact / "progress.jsonl").write_text(json.dumps({
        "event": "idle_timeout",
        "idempotency_key": key,
        "session_key": None,
        "generation": None,
        "checkpoint_identity": checkpoint_identity,
    }) + "\n", encoding="utf-8")
    diagnostics = AttemptDiagnostics(
        artifact,
        provider="codex-cli",
        model="test-model",
        fallback_index=0,
        attempt=1,
        call_label="stateless-segmentation-window-0001",
        env={},
    )
    diagnostics.bind_checkpoint_identity(checkpoint_identity)
    diagnostics.mark_submitted()
    attempt_ref = diagnostics.finalize(
        outcome="timeout",
        error=LLMWorkerTimeout(
            "idle", submission_state=LLMSubmissionState.SUBMITTED,
        ),
    )
    seal_submission_attempts(
        receipt_path,
        checkpoint_dir=checkpoint,
        attempt_references=[{
            "path": (artifact / attempt_ref.path).relative_to(checkpoint).as_posix(),
            "sha256": attempt_ref.sha256,
        }],
    )

    old_native_validations = 0

    def forbidden_validate(**_kwargs):
        nonlocal old_native_validations
        old_native_validations += 1
        raise AssertionError("stateless recovery must not validate an old native session")

    def continuation(options):
        assert options.supervised_native_resume_identities == ()
        # Stateless controls rotate their full5 ledger authority but never
        # manufacture or resume an LLMSessionRef.
        assert manager.get_existing(str(control["session_key"])) is None
        _accept_remaining(Path(control["ledger_path"]))
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "_validate_native_resume_context", forbidden_validate)
    monkeypatch.setattr(pipeline, "build_companion", continuation)

    result = pipeline.resume_companion(project, action="auto")

    assert result["ok"] is True
    assert old_native_validations == 0
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    context = journal["entries"][0]["recovery_context"]
    assert context["stateless_control"] is True
    assert set(context["submission_receipt"]) == {
        "path", "sha256", "identity_sha256",
    }
