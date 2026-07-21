from __future__ import annotations

from arc_llm.call_checkpoint import checkpoint_path, prepare_call, record_failure
from arc_llm.progress_journal import ProgressJournal
from arc_llm.providers.base import LLMSubmissionState, LLMWorkerTimeout
from arc_llm.recovery_context import read_recovery_context
from arc_llm.sessions import LLMSessionManager


def test_recovery_context_combines_checkpoint_progress_and_generation(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    manager = LLMSessionManager(tmp_path / "sessions")
    manager.get_or_create(
        key="chapter/lane",
        provider="codex-cli",
        model="m",
        runtime_fingerprint="fp",
    )
    path, identity = checkpoint_path(
        artifact_dir,
        prompt="prompt",
        schema=None,
        provider="codex-cli",
        model="m",
        call_label="turn",
        session_policy="stateful",
        session_key="chapter/lane",
        session_turn=0,
        runtime_fingerprint="fp",
        idempotency_key="turn-1",
        generation=1,
    )
    prepared = prepare_call(path, identity=identity)
    journal = ProgressJournal(
        artifact_dir=artifact_dir,
        call_label="turn",
        provider="codex-cli",
        callback=None,
        identity={
            "idempotency_key": "turn-1",
            "session_key": "chapter/lane",
            "generation": 1,
            "model": "m",
            "runtime_fingerprint": "fp",
        },
    )
    journal(
        {
            "event": "provider_progress",
            "native_session_id": "native-progress",
            "resumable": True,
        }
    )
    record_failure(
        prepared,
        LLMWorkerTimeout("idle", submission_state=LLMSubmissionState.SUBMITTED),
    )

    context = read_recovery_context(
        artifact_dir,
        idempotency_key="turn-1",
        session_manager=manager,
        session_key="chapter/lane",
    )

    assert context.checkpoint_state == "failed"
    assert context.submission_state == "submitted"
    assert context.native_session_id == "native-progress"
    assert context.resumable is True
    assert context.generation == 1
    assert context.provider == "codex-cli"
    assert context.model == "m"
    assert context.runtime_fingerprint == "fp"


def test_recovery_context_ignores_unrelated_latest_native_session(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    manager = LLMSessionManager(tmp_path / "sessions")
    manager.get_or_create(
        key="chapter/lane", provider="codex-cli", model="m", runtime_fingerprint="fp"
    )
    matching_identity = {
        "idempotency_key": "turn-1",
        "session_key": "chapter/lane",
        "generation": 1,
        "model": "m",
        "runtime_fingerprint": "fp",
    }
    ProgressJournal(
        artifact_dir=artifact_dir,
        call_label="turn",
        provider="codex-cli",
        callback=None,
        identity=matching_identity,
    )({"event": "provider_progress", "native_session_id": "native-matching", "resumable": True})
    ProgressJournal(
        artifact_dir=artifact_dir,
        call_label="other",
        provider="codex-cli",
        callback=None,
        identity={**matching_identity, "idempotency_key": "turn-2"},
    )({"event": "provider_progress", "native_session_id": "native-unrelated", "resumable": True})

    context = read_recovery_context(
        artifact_dir,
        idempotency_key="turn-1",
        session_manager=manager,
        session_key="chapter/lane",
    )

    assert context.native_session_id == "native-matching"
    assert context.latest_progress["idempotency_key"] == "turn-1"
