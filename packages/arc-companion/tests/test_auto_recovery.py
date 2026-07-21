from __future__ import annotations

import json
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


def _project(
    tmp_path: Path,
    *,
    translation_segments: tuple[str, ...] = ("s1", "s2", "s3"),
    commentary_segments: tuple[str, ...] = ("s1", "s2", "s3"),
) -> tuple[Path, Path, Path, Path, LLMSessionManager]:
    project = tmp_path / "run"
    checkpoint = project / "checkpoint"
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
    assert len(journal["replacements"]) == 2
    assert {
        replacement["suffix_start_segment_id"]
        for replacement in journal["replacements"]
    } == {"s2"}
    assert all(
        replacement["suffix_segment_ids"] == ["s2", "s3"]
        for replacement in journal["replacements"]
    )
    assert commentary.read_bytes() == commentary_before


def test_failed_replacement_reports_exhaustion_without_generation_three(
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
    assert manager.get_existing("ch-0001:translation").generation == 2
    assert json.loads(translation.read_text())["generation"] == 2
    journal = json.loads(
        (project / ".arc-companion" / "resume-transaction.json").read_text()
    )
    assert journal["restart_budgets"][0]["attempts_used"] == 1
    assert journal["replacements"][0]["status"] == "failed"


@pytest.mark.parametrize("category", ("cancelled", "authentication", "rate_limit"))
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
    assert manager.get_existing("ch-0001:translation").generation == 2

    statuses_before_acceptance: list[list[str]] = []

    def accepted_manual_replacement(_options):
        journal = json.loads(
            (project / ".arc-companion" / "resume-transaction.json").read_text()
        )
        statuses_before_acceptance.append([
            str(entry["status"]) for entry in journal["entries"]
        ])
        _finish_generation_suffix(translation, 3)
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
    assert manager.get_existing("ch-0001:translation").generation == 3
    ledger = json.loads(translation.read_text())
    assert [block["generation"] for block in ledger["blocks"]] == [1, 3, 3]
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
