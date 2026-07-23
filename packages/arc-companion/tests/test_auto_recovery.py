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
from arc_companion.source import SourceError
from arc_llm.sessions import LLMSessionManager


BUILD_FINGERPRINT = "2" * 64


def test_resume_returns_structured_error_for_invalid_checkpoint_state(
    tmp_path: Path,
) -> None:
    project = tmp_path / "invalid-run"
    project.mkdir()
    (project / "state.json").write_text(json.dumps({
        "status": "failed",
        "fingerprint": BUILD_FINGERPRINT,
        "checkpoint_dir": str(tmp_path / "outside"),
    }), encoding="utf-8")

    result = pipeline._resume_companion_unlocked(
        project,
        action="auto",
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "companion_checkpoint_invalid"


def _project(
    tmp_path: Path,
    *,
    translation_segments: tuple[str, ...] = ("s1", "s2", "s3"),
    commentary_segments: tuple[str, ...] = ("s1", "s2", "s3"),
) -> tuple[Path, Path, Path, Path, LLMSessionManager]:
    project = tmp_path / "run"
    checkpoint = project / ".arc-companion" / "checkpoints" / BUILD_FINGERPRINT
    translation = checkpoint / "chapters" / "ch-0001" / "translation-ledger.json"
    commentary = checkpoint / "chapters" / "ch-0001" / "companion-ledger.json"
    initialize_lane_ledger(
        translation,
        chapter_id="ch-0001",
        lane="translation",
        segment_ids=list(translation_segments),
    )
    initialize_lane_ledger(
        commentary,
        chapter_id="ch-0001",
        lane="companion",
        segment_ids=list(commentary_segments),
    )
    manager = LLMSessionManager(checkpoint / "sessions")
    for session_key in ("ch-0001:translation", "ch-0001:companion"):
        manager.get_or_create(
            key=session_key,
            provider="codex-cli",
            model="test-model",
            runtime_fingerprint="runtime",
        )
    project.mkdir(parents=True, exist_ok=True)
    (project / "state.json").write_text(json.dumps({
        "status": "needs_supervision",
        "fingerprint": BUILD_FINGERPRINT,
        "checkpoint_dir": str(checkpoint),
        "recovery_options": {
            "paper_id": "local:auto-recovery",
            "workers": 1,
            "recovery_policy": "auto",
        },
    }), encoding="utf-8")
    return project, checkpoint, translation, commentary, manager


def _accept_generation_suffix(path: Path, generation: int) -> None:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    assert ledger["generation"] == generation
    for block in ledger["blocks"]:
        if block["state"] == "accepted":
            continue
        segment_id = str(block["segment_id"])
        for state in (
            "submitted", "response_received", "schema_valid", "invariant_valid",
            "accepted",
        ):
            advance_block(
                path,
                segment_id=segment_id,
                state=state,
                input_sha256=f"input-{generation}-{segment_id}",
                output_sha256=f"output-{generation}-{segment_id}",
            )


def _accept_next_block(path: Path, segment_id: str, generation: int = 1) -> None:
    for state in (
        "submitted", "response_received", "schema_valid", "invariant_valid", "accepted",
    ):
        advance_block(
            path,
            segment_id=segment_id,
            state=state,
            input_sha256=f"input-{generation}-{segment_id}",
            output_sha256=f"output-{generation}-{segment_id}",
        )


def _finish_generation_suffix(path: Path, generation: int) -> None:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    states = (
        "prepared", "submitted", "response_received", "schema_valid",
        "invariant_valid", "accepted",
    )
    for raw in ledger["blocks"]:
        if int(raw.get("generation") or 0) != generation:
            continue
        segment_id = str(raw["segment_id"])
        current = str(raw["state"])
        for state in states[states.index(current) + 1:]:
            advance_block(
                path,
                segment_id=segment_id,
                state=state,
                input_sha256=f"input-{generation}-{segment_id}",
                output_sha256=f"output-{generation}-{segment_id}",
            )


def _one_blocker_project(
    tmp_path: Path,
) -> tuple[Path, Path, LLMSessionManager, str]:
    project, _checkpoint, translation, _commentary, manager = _project(tmp_path)
    _accept_next_block(translation, "s1")
    logical_key = "ch-0001:translation:s2:generation-1"
    mark_needs_supervision(
        translation,
        segment_id="s2",
        reason="native session was lost",
        recovery_context={
            "idempotency_key": logical_key,
            "submission_state": "unknown",
            "resumable": False,
        },
    )
    return project, translation, manager, logical_key


def test_recovery_trigger_requires_typed_terminal_progress(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    key = "ch-0001:translation:s2:generation-1"
    call_path = (
        checkpoint / "llm" / "call-checkpoints"
        / f"idempotency-{hashlib.sha256(key.encode()).hexdigest()}.json"
    )
    progress = call_path.parent.parent / "progress.jsonl"
    call_path.parent.mkdir(parents=True)
    entry = {
        "session_key": "ch-0001:translation",
        "segment_id": "s2",
        "idempotency_key": key,
        "initial_generation": 1,
        "recovery_context": {
            "checkpoint_path": str(call_path),
            "failure_category": "timeout",
            "logical_unit": "s2",
        },
    }
    checkpoint_identity = "checkpoint-identity-1"
    call_path.write_text(json.dumps({
        "identity": checkpoint_identity,
        "state": "failed", "resumable": True,
        "progress_journal": str(progress),
        "logical_identity": {
            "idempotency_key": key,
            "session_key": "ch-0001:translation",
            "generation": 1,
        },
        "request_recipe": {},
    }))
    progress.write_text(json.dumps({
        "event": "provider_progress", "idempotency_key": key,
        "session_key": "ch-0001:translation", "generation": 1,
        "checkpoint_identity": checkpoint_identity,
    }) + "\n")

    assert pipeline._recovery_trigger(entry, checkpoint) is None

    with progress.open("a") as handle:
        handle.write(json.dumps({
            "event": "idle_timeout", "idempotency_key": key,
            "session_key": "ch-0001:translation", "generation": 1,
            "checkpoint_identity": checkpoint_identity,
        }) + "\n")

    assert pipeline._recovery_trigger(entry, checkpoint) == "idle_timeout"

    entry["recovery_context"]["native_session_id"] = "native-a"
    assert pipeline._recovery_trigger(entry, checkpoint) is None
    with progress.open("a") as handle:
        handle.write(json.dumps({
            "event": "idle_timeout", "idempotency_key": key,
            "session_key": "ch-0001:translation", "generation": 1,
            "checkpoint_identity": checkpoint_identity,
            "native_session_id": "native-b",
        }) + "\n")
    assert pipeline._recovery_trigger(entry, checkpoint) is None
    with progress.open("a") as handle:
        handle.write(json.dumps({
            "event": "idle_timeout", "idempotency_key": key,
            "session_key": "ch-0001:translation", "generation": 1,
            "checkpoint_identity": checkpoint_identity,
            "native_session_id": "native-a",
        }) + "\n")
    assert pipeline._recovery_trigger(entry, checkpoint) == "idle_timeout"

    entry["recovery_context"].pop("native_session_id")
    progress.write_text(json.dumps({
        "event": "idle_timeout", "idempotency_key": key,
        "session_key": "ch-0001:translation", "generation": 1,
        "checkpoint_identity": checkpoint_identity,
        "native_session_id": "native-b",
    }) + "\n")
    assert pipeline._recovery_trigger(entry, checkpoint) is None


@pytest.mark.parametrize(
    "reason,recovery_context",
    [
        (
            "native session was lost",
            {
                "idempotency_key": (
                    "ch-0001:translation:companion-translation-s2:generation-1"
                ),
                "submission_state": "unknown",
                "resumable": False,
            },
        ),
        (
            "persisted paid response failed local validation",
            {
                "idempotency_key": (
                    "ch-0001:translation:companion-translation-s2:generation-1"
                ),
                "submission_state": "submitted",
                "resumable": False,
                "recovery_action": "operator-supervision",
                "failure_category": "semantic_validation",
            },
        ),
        (
            "paid provider response failed output schema validation",
            {
                "idempotency_key": (
                    "ch-0001:translation:companion-translation-s2:generation-1"
                ),
                "submission_state": "submitted", "resumable": False,
                "failure_category": "schema",
            },
        ),
    ],
    ids=("native-session-missing", "invalid-paid-response", "invalid-paid-schema"),
)
def test_default_auto_starts_generation_two_for_replaceable_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    recovery_context: dict[str, object],
) -> None:
    project, _checkpoint, translation, commentary, manager = _project(tmp_path)
    _accept_next_block(translation, "s1")
    mark_needs_supervision(
        translation,
        segment_id="s2",
        reason=reason,
        recovery_context=recovery_context,
    )
    commentary_before = commentary.read_bytes()

    def continuation(_options):
        _accept_generation_suffix(translation, 2)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", continuation)

    result = pipeline.resume_companion(project)

    assert result["ok"] is True, result
    ledger = json.loads(translation.read_text(encoding="utf-8"))
    assert ledger["generation"] == 2
    assert [block["generation"] for block in ledger["blocks"]] == [1, 2, 2]
    assert all(block["state"] == "accepted" for block in ledger["blocks"][1:])
    assert manager.get_existing("ch-0001:translation").generation == 2
    assert commentary.read_bytes() == commentary_before


def test_auto_restart_records_legacy_owner_from_nonfirst_source_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, checkpoint, translation, _commentary, manager = _project(tmp_path)
    invalidate_suffix(translation, from_segment_id="s1", generation=2)
    invalidate_suffix(translation, from_segment_id="s1", generation=3)
    _accept_next_block(translation, "s1", generation=3)
    manager.rotate("ch-0001:translation", reason="prior rollover")
    manager.rotate("ch-0001:translation", reason="prior rollover")
    mark_needs_supervision(
        translation,
        segment_id="s2",
        reason="legacy generation lost its native session",
        recovery_context={
            "idempotency_key": "ch-0001:translation:s2:generation-3",
            "submission_state": "unknown", "resumable": False,
        },
    )

    def continuation(_options):
        assert pipeline._generation_segment_artifact_dir(
            checkpoint, "translations", "s2", 3,
        ) == checkpoint / "translations"
        assert pipeline._generation_segment_artifact_dir(
            checkpoint, "translations", "s2", 4,
        ) == checkpoint / "translations" / "generation-4"
        _accept_generation_suffix(translation, 4)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", continuation)
    result = pipeline.resume_companion(project)

    assert result["ok"] is True
    owners = json.loads(
        (checkpoint / "legacy-generation-owners.json").read_text(encoding="utf-8")
    )["owners"]
    assert owners["translations"]["s2"] == 3
    assert owners["translation-token-offset-attempts"]["s2"] == 3
    assert owners["llm/translations"]["s2"] == 3


def test_auto_groups_same_lane_blockers_at_earliest_suffix_and_rotates_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _checkpoint, translation, commentary, manager = _project(tmp_path)
    _accept_next_block(translation, "s1")
    # Model drained failures may be recorded out of source order; recovery must
    # nevertheless choose the earliest blocked segment in the lane.
    for segment_id in ("s3", "s2"):
        mark_needs_supervision(
            translation,
            segment_id=segment_id,
            reason=f"blocked {segment_id}",
            recovery_context={
                "idempotency_key": f"ch-0001:translation:{segment_id}:generation-1",
                "submission_state": "unknown",
                "resumable": False,
            },
        )
    commentary_before = commentary.read_bytes()
    generations_seen: list[int] = []

    def continuation(_options):
        generations_seen.append(
            manager.get_existing("ch-0001:translation").generation
        )
        _accept_generation_suffix(translation, 2)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", continuation)

    result = pipeline.resume_companion(project)

    assert result["ok"] is True
    assert generations_seen == [2]
    assert manager.get_existing("ch-0001:translation").generation == 2
    ledger = json.loads(translation.read_text(encoding="utf-8"))
    assert [block["generation"] for block in ledger["blocks"]] == [1, 2, 2]
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert len(journal["replacements"]) == 1
    assert {
        replacement["suffix_start_segment_id"]
        for replacement in journal["replacements"]
    } == {"s2"}
    assert all(
        replacement["suffix_segment_ids"] == ["s2", "s3"]
        for replacement in journal["replacements"]
    )
    assert commentary.read_bytes() == commentary_before


def test_typed_idle_restarts_at_first_nonaccepted_block_not_timeout_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _checkpoint, translation, _commentary, manager = _project(tmp_path)
    key = "ch-0001:translation:s2:generation-1"
    mark_needs_supervision(
        translation,
        segment_id="s2",
        reason="second block became idle",
        recovery_context={
            "idempotency_key": key,
            "submission_state": "submitted",
            "resumable": True,
            "native_session_id": "native-1",
            "generation": 1, "logical_unit": "s2",
            "latest_progress": {
                "event": "idle_timeout", "idempotency_key": key,
                "session_key": "ch-0001:translation", "generation": 1,
            },
        },
    )
    calls: list[tuple[int, tuple[str, ...]]] = []

    def continuation(options):
        calls.append((manager.get_existing("ch-0001:translation").generation,
                      options.supervised_native_resume_keys))
        _accept_generation_suffix(translation, 2)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", continuation)

    result = pipeline.resume_companion(project)

    assert result["ok"] is True
    assert calls == [(2, ())]
    ledger = json.loads(translation.read_text())
    assert [item["generation"] for item in ledger["blocks"]] == [2, 2, 2]
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    replacement = journal["replacements"][0]
    assert replacement["segment_id"] == "s2"
    assert replacement["abandoned_logical_key"] == key
    assert replacement["suffix_start_segment_id"] == "s1"
    assert replacement["suffix_segment_ids"] == ["s1", "s2", "s3"]
    assert journal["entries"][0]["fresh_task_start_segment_id"] == "s1"


def test_typed_idle_fresh_rotation_preserves_other_native_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _checkpoint, translation, commentary, manager = _project(
        tmp_path, translation_segments=("s1",), commentary_segments=("s1",),
    )
    translation_key = "ch-0001:translation:s1:generation-1"
    commentary_key = "ch-0001:companion:s1:generation-1"
    manager.update_native_session_id("ch-0001:companion", "native-commentary")
    mark_needs_supervision(
        translation, segment_id="s1", reason="typed idle",
        recovery_context={
            "idempotency_key": translation_key,
            "submission_state": "submitted", "resumable": True,
            "generation": 1, "logical_unit": "s1",
            "latest_progress": {
                "event": "idle_timeout", "idempotency_key": translation_key,
                "session_key": "ch-0001:translation", "generation": 1,
            },
        },
    )
    mark_needs_supervision(
        commentary, segment_id="s1", reason="ordinary resumable transport",
        recovery_context={
            "idempotency_key": commentary_key,
            "submission_state": "submitted", "resumable": True,
        },
    )

    def validate_context(**kwargs):
        ledger = kwargs["ledger"]
        assert ledger["lane"] == "companion"
        return {
            "session_key": "ch-0001:companion",
            "segment_id": "s1",
            "ledger_path": str(kwargs["ledger_path"]),
            "idempotency_key": commentary_key,
            "provider": "codex-cli", "model": "test-model",
            "runtime_fingerprint": "runtime", "generation": 1,
            "native_session_id_to_restore": None,
        }

    calls: list[tuple[int, int, tuple[str, ...]]] = []

    def continuation(options):
        calls.append((
            manager.get_existing("ch-0001:translation").generation,
            manager.get_existing("ch-0001:companion").generation,
            options.supervised_native_resume_keys,
        ))
        if options.supervised_native_resume_keys:
            _accept_generation_suffix(commentary, 1)
            return {"ok": False, "status": "needs_supervision"}
        _accept_generation_suffix(translation, 2)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "_validate_native_resume_context", validate_context)
    monkeypatch.setattr(pipeline, "build_companion", continuation)

    result = pipeline.resume_companion(project)

    assert result["ok"] is True
    assert calls == [
        (1, 1, (commentary_key,)),
        (2, 1, ()),
    ]
    assert json.loads(commentary.read_text())["generation"] == 1
    assert json.loads(translation.read_text())["generation"] == 2


def test_failed_replacement_uses_three_generations_before_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _checkpoint, translation, _commentary, manager = _project(tmp_path)
    _accept_next_block(translation, "s1")
    mark_needs_supervision(
        translation,
        segment_id="s2",
        reason="native session was lost",
        recovery_context={
            "idempotency_key": "ch-0001:translation:s2:generation-1",
            "submission_state": "unknown",
            "resumable": False,
        },
    )

    def failing_replacement(_options):
        mark_needs_supervision(
            translation,
            segment_id="s2",
            reason="replacement failed validation",
            recovery_context={
                "idempotency_key": "ch-0001:translation:s2:generation-2",
                "submission_state": "submitted",
                "resumable": False,
            },
        )
        return {"ok": False, "status": "needs_supervision"}

    monkeypatch.setattr(pipeline, "build_companion", failing_replacement)

    result = pipeline.resume_companion(project)

    assert result["error"]["code"] == "automatic_regeneration_exhausted"
    assert manager.get_existing("ch-0001:translation").generation == 4
    assert json.loads(translation.read_text())["generation"] == 4
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert journal["restart_budgets"][0]["attempts_used"] == 3
    assert [item["attempt"] for item in journal["replacements"]] == [1, 2, 3]
    assert all(item["status"] == "failed" for item in journal["replacements"])


@pytest.mark.parametrize(
    "category",
    (
        "cancelled", "authentication", "quota", "permission", "rate_limit",
        "local_io", "invalid_request",
    ),
)
def test_auto_does_not_rotate_excluded_failure_categories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, category: str,
) -> None:
    project, _checkpoint, translation, _commentary, manager = _project(tmp_path)
    mark_needs_supervision(
        translation,
        segment_id="s2",
        reason=f"provider failure: {category}",
        recovery_context={
            "idempotency_key": f"ch-0001:translation:s2:{category}:generation-1",
            "submission_state": "unknown",
            "resumable": False,
            "failure_category": category,
        },
    )
    continuation_calls = 0

    def continuation(_options):
        nonlocal continuation_calls
        continuation_calls += 1
        return {"ok": False, "status": "needs_supervision"}

    monkeypatch.setattr(pipeline, "build_companion", continuation)

    result = pipeline.resume_companion(project)

    assert result["status"] == "needs_supervision"
    assert manager.get_existing("ch-0001:translation").generation == 1
    assert json.loads(translation.read_text())["generation"] == 1
    assert continuation_calls == 0
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert journal["replacements"] == []
    assert journal["restart_budgets"] == []


def test_auto_reentry_after_rotate_before_invalidate_does_not_rotate_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, translation, manager, logical_key = _one_blocker_project(tmp_path)
    real_invalidate = pipeline.invalidate_suffix
    calls = 0

    def crash_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("crash after rotate")
        return real_invalidate(*args, **kwargs)

    monkeypatch.setattr(pipeline, "invalidate_suffix", crash_once)
    monkeypatch.setattr(
        pipeline,
        "build_companion",
        lambda _options: (
            _finish_generation_suffix(translation, 2)
            or {"ok": True, "status": "complete"}
        ),
    )

    with pytest.raises(RuntimeError, match="crash after rotate"):
        pipeline.resume_companion(project)
    assert manager.get_existing("ch-0001:translation").generation == 2
    assert json.loads(translation.read_text())["generation"] == 1

    result = pipeline.resume_companion(project)

    assert result["ok"] is True
    assert calls == 2
    assert manager.get_existing("ch-0001:translation").generation == 2
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert [entry["idempotency_key"] for entry in journal["entries"]] == [logical_key]
    assert journal["restart_budgets"][0]["attempts_used"] == 1
    assert journal["replacements"][0]["status"] == "accepted"


def test_auto_reentry_after_suffix_invalidation_reuses_generation_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, translation, manager, logical_key = _one_blocker_project(tmp_path)
    continuation_calls = 0

    def continuation(_options):
        nonlocal continuation_calls
        continuation_calls += 1
        if continuation_calls == 1:
            raise RuntimeError("crash after suffix invalidation")
        _finish_generation_suffix(translation, 2)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", continuation)

    with pytest.raises(RuntimeError, match="crash after suffix invalidation"):
        pipeline.resume_companion(project)
    invalidated = json.loads(translation.read_text())
    assert invalidated["generation"] == 2
    assert invalidated["blocks"][1]["state"] == "prepared"

    result = pipeline.resume_companion(project)

    assert result["ok"] is True
    assert continuation_calls == 2
    assert manager.get_existing("ch-0001:translation").generation == 2
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert [entry["idempotency_key"] for entry in journal["entries"]] == [logical_key]
    assert journal["restart_budgets"][0]["attempts_used"] == 1
    assert journal["replacements"][0]["status"] == "accepted"


def test_auto_reentry_after_response_persisted_finishes_existing_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, translation, manager, logical_key = _one_blocker_project(tmp_path)
    continuation_calls = 0

    def continuation(_options):
        nonlocal continuation_calls
        continuation_calls += 1
        if continuation_calls == 1:
            advance_block(translation, segment_id="s2", state="submitted")
            advance_block(
                translation, segment_id="s2", state="response_received",
                input_sha256="input-2-s2", output_sha256="output-2-s2",
            )
            return {"ok": False, "status": "needs_supervision"}
        if continuation_calls == 2:
            raise RuntimeError("crash after response persistence")
        _finish_generation_suffix(translation, 2)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", continuation)

    with pytest.raises(RuntimeError, match="crash after response persistence"):
        pipeline.resume_companion(project)
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert journal["replacements"][0]["status"] == "response_persisted"

    result = pipeline.resume_companion(project)

    assert result["ok"] is True
    assert continuation_calls == 3
    assert manager.get_existing("ch-0001:translation").generation == 2
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert [entry["idempotency_key"] for entry in journal["entries"]] == [logical_key]
    assert journal["restart_budgets"][0]["attempts_used"] == 1
    assert journal["replacements"][0]["status"] == "accepted"


def test_auto_reentry_after_block_acceptance_finalizes_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, translation, manager, logical_key = _one_blocker_project(tmp_path)
    real_finalize = pipeline._finalize_automatic_generation_restarts
    finalize_calls = 0

    def crash_once(*args, **kwargs):
        nonlocal finalize_calls
        block = json.loads(translation.read_text())["blocks"][1]
        if kwargs.get("replacements") and block["state"] == "accepted":
            finalize_calls += 1
        if finalize_calls == 1:
            raise RuntimeError("crash after block accepted")
        return real_finalize(*args, **kwargs)

    monkeypatch.setattr(
        pipeline, "_finalize_automatic_generation_restarts", crash_once,
    )
    monkeypatch.setattr(
        pipeline,
        "build_companion",
        lambda _options: (
            _finish_generation_suffix(translation, 2)
            or {"ok": True, "status": "complete"}
        ),
    )

    with pytest.raises(RuntimeError, match="crash after block accepted"):
        pipeline.resume_companion(project)
    assert json.loads(translation.read_text())["blocks"][1]["state"] == "accepted"

    result = pipeline.resume_companion(project)

    assert result["ok"] is True, result
    assert manager.get_existing("ch-0001:translation").generation == 2
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert [entry["idempotency_key"] for entry in journal["entries"]] == [logical_key]
    assert journal["restart_budgets"][0]["attempts_used"] == 1
    assert journal["entries"][0]["status"] == "resolved"
    assert journal["replacements"][0]["status"] == "accepted"


def test_confirmed_restart_can_continue_after_auto_budget_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _checkpoint, translation, _commentary, manager = _project(tmp_path)
    _accept_next_block(translation, "s1")
    for segment_id in ("s3", "s2"):
        mark_needs_supervision(
            translation,
            segment_id=segment_id,
            reason=f"generation one blocker {segment_id}",
            recovery_context={
                "idempotency_key": f"ch-0001:translation:{segment_id}:generation-1",
                "submission_state": "unknown",
                "resumable": False,
            },
        )

    def failed_auto_replacement(_options):
        mark_needs_supervision(
            translation,
            segment_id="s2",
            reason="generation two replacement failed",
            recovery_context={
                "idempotency_key": "ch-0001:translation:s2:generation-2",
                "submission_state": "submitted",
                "resumable": False,
            },
        )
        return {"ok": False, "status": "needs_supervision"}

    monkeypatch.setattr(pipeline, "build_companion", failed_auto_replacement)
    exhausted = pipeline.resume_companion(project)
    assert exhausted["error"]["code"] == "automatic_regeneration_exhausted"
    assert manager.get_existing("ch-0001:translation").generation == 4

    statuses_before_acceptance: list[list[str]] = []

    def accepted_manual_replacement(_options):
        journal = json.loads(
            (project / ".arc-companion" / "resume-transaction.json").read_text()
        )
        statuses_before_acceptance.append([
            str(entry["status"]) for entry in journal["entries"]
        ])
        _finish_generation_suffix(translation, 5)
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", accepted_manual_replacement)

    result = pipeline.resume_companion(
        project,
        action="restart-generation",
        confirm_possible_duplicate_charge=True,
    )

    assert result["ok"] is True, result
    assert statuses_before_acceptance
    assert all(
        status != "resolved" for status in statuses_before_acceptance[0]
    )
    assert manager.get_existing("ch-0001:translation").generation == 5
    ledger = json.loads(translation.read_text())
    assert [block["generation"] for block in ledger["blocks"]] == [1, 5, 5]
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert all(entry["status"] == "resolved" for entry in journal["entries"])


def test_source_preflight_failure_does_not_rotate_or_spend_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _checkpoint, translation, _commentary, manager = _project(tmp_path)
    _accept_next_block(translation, "s1")
    mark_needs_supervision(
        translation,
        segment_id="s2",
        reason="eligible submitted blocker",
        recovery_context={
            "idempotency_key": "ch-0001:translation:s2:generation-1",
            "submission_state": "unknown", "resumable": False,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_preflight_automatic_recovery_source",
        lambda _options: (_ for _ in ()).throw(SourceError("source snapshot missing")),
    )

    result = pipeline._resume_companion_unlocked(project, action="auto")

    assert result["error"]["code"] == "companion_source_unavailable"
    assert manager.get_existing("ch-0001:translation").generation == 1
    ledger = json.loads(translation.read_text())
    assert ledger["generation"] == 1
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert journal["restart_budgets"] == []
    assert journal["replacements"] == []


def test_inline_build_source_preflight_runs_before_generation_rotation(
    tmp_path: Path,
) -> None:
    project, _checkpoint, translation, _commentary, manager = _project(tmp_path)
    _accept_next_block(translation, "s1")
    mark_needs_supervision(
        translation,
        segment_id="s2",
        reason="eligible submitted blocker",
        recovery_context={
            "idempotency_key": "ch-0001:translation:s2:generation-1",
            "submission_state": "unknown",
            "resumable": False,
        },
    )
    preflight_calls: list[bool] = []

    def source_preflight() -> None:
        preflight_calls.append(True)
        raise SourceError("source snapshot missing")

    result = pipeline._resume_companion_unlocked(
        project,
        action="auto",
        continuation=lambda _options: {
            "ok": False,
            "status": "needs_supervision",
        },
        source_preflight=source_preflight,
    )

    assert result["error"]["code"] == "companion_source_unavailable"
    assert preflight_calls == [True]
    assert manager.get_existing("ch-0001:translation").generation == 1
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert journal["restart_budgets"] == []
    assert journal["replacements"] == []


def test_auto_repair_finalization_ignores_unindexed_marker_and_checkpoint(
    tmp_path: Path,
) -> None:
    _project_dir, checkpoint, _translation, _commentary, _manager = _project(
        tmp_path, translation_segments=("s1",), commentary_segments=("s1",),
    )
    pipeline.write_json(checkpoint / "document.json", {
        "paper_id": "local:unindexed-repair",
        "document": {
            "blocks": [{"block_id": "body-1", "type": "text", "text": "source"}],
            "equations": [], "figures": [], "tables": [], "bibliography": [],
        },
    })
    marker = pipeline._translation_token_attempt_path(checkpoint, "s1", 1)
    pipeline.write_json(marker, {
        "schema_version": "arc.companion.translation-token-attempt.v2",
        "status": "response_received",
        "segment_id": "s1",
        "generation": 1,
        "block_ids": ["body-1"],
        "raw_response": {"repairs": []},
    })
    rogue_checkpoint = (
        checkpoint / "llm" / "translations" / "generation-0001" / "s1"
        / marker.stem / "retry-offset-1" / "call-checkpoints" / "rogue.json"
    )
    pipeline.write_json(rogue_checkpoint, {
        "state": "validated",
        "submission_state": "submitted",
        "logical_identity": {"idempotency_key": "rogue"},
    })

    assert pipeline._finalize_paid_translation_repairs(checkpoint) == []
