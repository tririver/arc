from __future__ import annotations

import json
from pathlib import Path

import pytest

import arc_companion.pipeline as pipeline_module
from arc_companion.pipeline import (
    BuildOptions,
    _coordinate_review_candidates,
    _pipeline_acceptance_checkpoint_valid,
    _review_arbitration_reference_valid,
)
from arc_companion.io import sha256_json
from arc_companion.review_arbitration import (
    ReviewArbitrationError,
    ReviewArbitrationNeedsSupervision,
    ReviewPatchSource,
)
from arc_llm.providers.base import LLMSubmissionState, LLMWorkerError


def _annotation(text: str = "old") -> dict:
    return {
        "commentary": text,
        "explanation": "",
        "commentary_sources": [],
        "prior_work": [],
        "later_work": [],
    }


def _patch(text: str) -> dict:
    return {
        "segment_id": "seg-1",
        "commentary": text,
        "explanation": f"explain {text}",
        "commentary_sources": None,
        "prior_work": None,
        "later_work": None,
        "reason": f"use {text}",
    }


def _source(text: str, order: int) -> ReviewPatchSource:
    return ReviewPatchSource.from_review(
        source_kind="commentary",
        stable_order=order,
        review={"patches": [_patch(text)], "issues": []},
        segment_set=["seg-1"],
    )


def _field_source(field: str, text: str, order: int) -> ReviewPatchSource:
    patch = _patch("unchanged")
    patch["commentary"] = None
    patch["explanation"] = None
    patch[field] = text
    return ReviewPatchSource.from_review(
        source_kind="commentary",
        stable_order=order,
        review={"patches": [patch], "issues": []},
        segment_set=["seg-1"],
    )


def _kwargs(tmp_path: Path, llm):
    return {
        "segments": [{"segment_id": "seg-1", "block_ids": []}],
        "translations": None,
        "annotations": {"seg-1": _annotation()},
        "blocks_by_id": {},
        "protected_names": [],
        "reader_evidence": {"seg-1": []},
        "options": BuildOptions(
            paper_id="arXiv:0000.0000",
            project_dir=tmp_path,
            workers=1,
            skip_translation=True,
        ),
        "llm": llm,
        "checkpoint_dir": tmp_path / "checkpoints",
        "freeze_binding": None,
        "cancel_check": None,
    }


def test_no_conflicts_write_zero_call_terminal_receipt(tmp_path: Path) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("arbitration provider must not run")

    _, annotations, review, reference = _coordinate_review_candidates(
        [_source("accepted", 0)],
        **_kwargs(tmp_path, forbidden),
    )
    assert annotations["seg-1"]["commentary"] == "accepted"
    assert review["patches"][0]["commentary"] == "accepted"
    receipt = json.loads(
        (tmp_path / "checkpoints" / reference["path"]).read_text()
    )
    assert receipt["status"] == "no_conflicts"
    assert receipt["provider_call_count"] == 0
    assert not receipt["unresolved_paths"]


def test_many_conflicts_use_one_low_stateless_tool_disabled_call_and_replay(
    tmp_path: Path,
) -> None:
    calls = []

    def llm(prompt: str, **kwargs):
        calls.append(kwargs)
        assert kwargs["call_label"].startswith(
            "companion-review-arbitration-"
        )
        assert kwargs["model_tier"] == "low"
        assert kwargs["session_policy"] == "stateless"
        env = kwargs["env"]
        assert env["ARC_CODEX_ALLOW_INTERNET"] == "false"
        assert env["ARC_CLAUDE_ALLOW_INTERNET"] == "false"
        assert env["ARC_LLM_INHERIT_HOST_TOOLS"] == "false"
        assert env["ARC_CODEX_ENABLE_MCP"] == "false"
        assert env["ARC_CLAUDE_ALLOW_MCP"] == "false"
        assert env["ARC_CLAUDE_TOOLS"] == ""
        assert env["ARC_CLAUDE_ALLOWED_TOOLS"] == ""
        assert env["ARC_PAPER_ACCESS"] == "none"
        payload = json.loads(prompt.split("REVIEW CONFLICTS:\n", 1)[1])
        return {"decisions": [
            {
                "path": conflict["path"],
                "action": "select_candidate",
                "selected_candidate_hashes": [
                    conflict["candidates"][0]["candidate_sha256"]
                ],
                "replacement_patch": conflict["candidates"][0][
                    "replacement_patch"
                ],
                "reason": "bounded supplied candidate",
            }
            for conflict in payload["conflicts"]
        ]}

    first = _coordinate_review_candidates(
        [_source("candidate-a", 0), _source("candidate-b", 1)],
        **_kwargs(tmp_path, llm),
    )
    assert len(calls) == 1
    assert first[3]["provider_call_count"] == 1
    assert first[3]["status"] == "resolved"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("terminal receipt replay must not call provider")

    second = _coordinate_review_candidates(
        [_source("candidate-a", 0), _source("candidate-b", 1)],
        **_kwargs(tmp_path, forbidden),
    )
    assert second[1] == first[1]
    assert second[3]["sha256"] == first[3]["sha256"]
    assert len(calls) == 1
    decision_path = next(
        (tmp_path / "checkpoints" / "review-arbitration").glob(
            "*/decision.json"
        )
    )
    decision_text = decision_path.read_text()
    assert "candidate-a" not in decision_text
    assert "candidate-b" not in decision_text
    assert "replacement_patch" not in decision_text
    assert '"output"' not in decision_text
    decision = json.loads(decision_text)
    validated_path = (
        tmp_path / "checkpoints" / decision["validated_output_path"]
    )
    assert validated_path.is_file()
    assert "candidate-" in validated_path.read_text()
    assert _pipeline_acceptance_checkpoint_valid(
        "review-arbitration",
        decision_path,
        {"input_sha256": decision["semantic_input_sha256"]},
        checkpoint_dir=tmp_path / "checkpoints",
    )
    original_validated = validated_path.read_text()
    validated_path.write_text('{"tampered":true}', encoding="utf-8")
    assert not _pipeline_acceptance_checkpoint_valid(
        "review-arbitration",
        decision_path,
        {"input_sha256": decision["semantic_input_sha256"]},
        checkpoint_dir=tmp_path / "checkpoints",
    )
    validated_path.write_text(original_validated, encoding="utf-8")
    rebound_path = tmp_path / "checkpoints" / "rebound-output.json"
    rebound_path.write_text(original_validated, encoding="utf-8")
    rebound_decision = {
        **decision,
        "validated_output_path": "rebound-output.json",
        "validated_output_sha256": pipeline_module.sha256_file(rebound_path),
    }
    decision_path.write_text(json.dumps(rebound_decision), encoding="utf-8")
    assert not _pipeline_acceptance_checkpoint_valid(
        "review-arbitration",
        decision_path,
        {"input_sha256": decision["semantic_input_sha256"]},
        checkpoint_dir=tmp_path / "checkpoints",
    )


def _select_first_arbitration(prompt: str) -> dict:
    payload = json.loads(prompt.split("REVIEW CONFLICTS:\n", 1)[1])
    return {"decisions": [
        {
            "path": conflict["path"],
            "action": "select_candidate",
            "selected_candidate_hashes": [
                conflict["candidates"][0]["candidate_sha256"]
            ],
            "replacement_patch": conflict["candidates"][0][
                "replacement_patch"
            ],
            "reason": "bounded decision",
        }
        for conflict in payload["conflicts"]
    ]}


def test_oversized_conflicts_supervise_without_call_and_write_body_free_ledger(
    tmp_path: Path,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("oversized arbitration must not call provider")

    large_a = "A" * 40_000
    large_b = "B" * 40_000
    with pytest.raises(ReviewArbitrationNeedsSupervision) as caught:
        _coordinate_review_candidates(
            [_source(large_a, 0), _source(large_b, 1)],
            **_kwargs(tmp_path, forbidden),
        )
    assert caught.value.paths
    receipt_path = (
        tmp_path / "checkpoints" / caught.value.receipt_path
    )
    receipt_text = receipt_path.read_text()
    receipt = json.loads(receipt_text)
    assert receipt["status"] == "needs_supervision"
    assert receipt["provider_call_count"] == 0
    assert large_a[:100] not in receipt_text
    assert str(tmp_path) not in receipt_text
    ledger_path = (
        tmp_path / "checkpoints" / "recovery-controls"
        / "review-arbitration" / "review-arbitration-ledger.json"
    )
    ledger_text = ledger_path.read_text()
    assert large_a[:100] not in ledger_text
    assert caught.value.paths[0] in ledger_text


def test_invalid_arbitration_output_becomes_terminal_exact_supervision(
    tmp_path: Path,
) -> None:
    calls = 0

    def llm(_prompt: str, **_kwargs):
        nonlocal calls
        calls += 1
        return {"decisions": []}

    with pytest.raises(ReviewArbitrationNeedsSupervision) as first:
        _coordinate_review_candidates(
            [_source("candidate-a", 0), _source("candidate-b", 1)],
            **_kwargs(tmp_path, llm),
        )
    assert calls == 1
    receipt = json.loads(
        (tmp_path / "checkpoints" / first.value.receipt_path).read_text()
    )
    assert receipt["status"] == "needs_supervision"
    assert receipt["provider_call_count"] == 1
    assert set(receipt["unresolved_paths"]) == set(first.value.paths)

    with pytest.raises(ReviewArbitrationNeedsSupervision):
        _coordinate_review_candidates(
            [_source("candidate-a", 0), _source("candidate-b", 1)],
            **_kwargs(tmp_path, lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("terminal supervision must replay without a call")
            )),
        )
    assert calls == 1


def test_receipt_tamper_fails_closed(tmp_path: Path) -> None:
    result = _coordinate_review_candidates(
        [_source("accepted", 0)],
        **_kwargs(tmp_path, lambda *_args, **_kwargs: None),
    )
    receipt_path = tmp_path / "checkpoints" / result[3]["path"]
    receipt = json.loads(receipt_path.read_text())
    receipt["provider"] = "tampered"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ReviewArbitrationError, match="tamper|mismatch"):
        _coordinate_review_candidates(
            [_source("accepted", 0)],
            **_kwargs(tmp_path, lambda *_args, **_kwargs: None),
        )


def test_first_chapter_freeze_rejects_candidate_before_call(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    baseline_annotations = {"seg-1": _annotation()}
    freeze_path = checkpoint_dir / "first-chapter-freeze.json"
    freeze_path.write_text(
        json.dumps({
            "schema_version": "arc.companion.first-chapter-freeze.v3",
            "translation_mode": "skipped",
            "pre_review_annotation_sha256": sha256_json(
                baseline_annotations
            ),
        }),
        encoding="utf-8",
    )
    kwargs = _kwargs(
        tmp_path,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frozen candidate must not call provider")
        ),
    )
    kwargs["freeze_binding"] = {
        "schema_version": "arc.companion.review-freeze-binding.v1",
        "freeze_sha256": pipeline_module.sha256_file(freeze_path),
        "segment_ids": ["seg-1"],
        "translation_mode": "skipped",
        "pre_review_translation_sha256": None,
        "pre_review_annotation_sha256": sha256_json(
            baseline_annotations
        ),
    }
    with pytest.raises(ReviewArbitrationNeedsSupervision) as caught:
        _coordinate_review_candidates(
            [_source("candidate", 0)],
            **kwargs,
        )
    assert caught.value.paths == (
        "/segments/seg-1/annotation/commentary",
        "/segments/seg-1/annotation/explanation",
    )


@pytest.mark.parametrize(
    "submission_state",
    [LLMSubmissionState.SUBMITTED, LLMSubmissionState.UNKNOWN],
)
def test_submitted_or_unknown_provider_failure_is_terminal_exact_supervision(
    submission_state: LLMSubmissionState,
    tmp_path: Path,
) -> None:
    def failing(_prompt: str, **_kwargs):
        raise LLMWorkerError(
            "provider boundary failed",
            submission_state=submission_state,
        )

    with pytest.raises(ReviewArbitrationNeedsSupervision) as caught:
        _coordinate_review_candidates(
            [_source("candidate-a", 0), _source("candidate-b", 1)],
            **_kwargs(tmp_path, failing),
        )
    assert caught.value.recovery_context["submission_state"] == (
        submission_state.value
    )
    receipt = json.loads(
        (tmp_path / "checkpoints" / caught.value.receipt_path).read_text()
    )
    assert receipt["status"] == "needs_supervision"
    assert receipt["submission_state"] == submission_state.value
    assert receipt["supervision_reason"] == (
        f"provider_{submission_state.value}_failure"
    )
    assert set(receipt["unresolved_paths"]) == set(caught.value.paths)
    assert str(tmp_path) not in json.dumps(receipt)


def test_not_submitted_provider_failure_keeps_retry_behavior(
    tmp_path: Path,
) -> None:
    error = LLMWorkerError(
        "local provider setup failed",
        submission_state=LLMSubmissionState.NOT_SUBMITTED,
    )
    with pytest.raises(LLMWorkerError) as caught:
        _coordinate_review_candidates(
            [_source("candidate-a", 0), _source("candidate-b", 1)],
            **_kwargs(
                tmp_path,
                lambda _prompt, **_kwargs: (_ for _ in ()).throw(error),
            ),
        )
    assert caught.value is error
    assert not list(
        (tmp_path / "checkpoints" / "review-arbitration").glob(
            "*/receipt.json"
        )
    )


def test_business_ledger_reconciles_conflict_sets_and_clears_on_resolution(
    tmp_path: Path,
) -> None:
    ledger_path = (
        tmp_path / "checkpoints" / "recovery-controls"
        / "review-arbitration" / "review-arbitration-ledger.json"
    )
    commentary_sources = [
        _field_source("commentary", "a", 0),
        _field_source("commentary", "b", 1),
    ]
    with pytest.raises(ReviewArbitrationNeedsSupervision):
        _coordinate_review_candidates(
            commentary_sources,
            **_kwargs(tmp_path, lambda _prompt, **_kwargs: {"decisions": []}),
        )
    first_ledger = json.loads(ledger_path.read_text())
    assert first_ledger["needs_supervision"]["recovery_context"][
        "review_arbitration_paths"
    ] == ["/segments/seg-1/annotation/commentary"]

    explanation_sources = [
        _field_source("explanation", "a", 0),
        _field_source("explanation", "b", 1),
    ]
    with pytest.raises(ReviewArbitrationNeedsSupervision):
        _coordinate_review_candidates(
            explanation_sources,
            **_kwargs(tmp_path, lambda _prompt, **_kwargs: {"decisions": []}),
        )
    second_ledger = json.loads(ledger_path.read_text())
    assert second_ledger["needs_supervision"]["recovery_context"][
        "review_arbitration_paths"
    ] == ["/segments/seg-1/annotation/explanation"]

    _coordinate_review_candidates(
        [
            _field_source("explanation", "resolved-a", 0),
            _field_source("explanation", "resolved-b", 1),
        ],
        **_kwargs(
            tmp_path,
            lambda prompt, **_kwargs: _select_first_arbitration(prompt),
        ),
    )
    closed = json.loads(ledger_path.read_text())
    assert closed["needs_supervision"] is None
    assert closed["supervision_entries"] == []


def test_runtime_prepared_oversize_prompt_makes_zero_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arc_llm import runner

    original = runner.prepare_runtime_prompt

    def oversized(*args, **kwargs):
        prompt, prefix, nested = original(*args, **kwargs)
        return prompt + ("R" * 80_000), prefix, nested

    monkeypatch.setattr(runner, "prepare_runtime_prompt", oversized)
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("runtime-sized oversize input must not submit")

    with pytest.raises(ReviewArbitrationNeedsSupervision) as caught:
        _coordinate_review_candidates(
            [_source("candidate-a", 0), _source("candidate-b", 1)],
            **_kwargs(tmp_path, forbidden),
        )
    assert calls == 0
    receipt = json.loads(
        (tmp_path / "checkpoints" / caught.value.receipt_path).read_text()
    )
    assert receipt["supervision_reason"] == "arbitration_input_too_large"


def test_receipt_reference_rejects_output_recipe_and_semantic_mismatch(
    tmp_path: Path,
) -> None:
    options = _kwargs(tmp_path, lambda *_args, **_kwargs: None)["options"]
    result = _coordinate_review_candidates(
        [_source("accepted", 0)],
        **_kwargs(tmp_path, lambda *_args, **_kwargs: None),
    )
    reference = result[3]
    assert _review_arbitration_reference_valid(
        {"review_arbitration_receipt": reference},
        tmp_path / "checkpoints",
        translations=None,
        annotations=result[1],
        options=options,
    )
    assert not _review_arbitration_reference_valid(
        {"review_arbitration_receipt": reference},
        tmp_path / "checkpoints",
        translations=None,
        annotations={"seg-1": _annotation("mismatched")},
        options=options,
    )
    tampered = {**reference, "semantic_input_sha256": "0" * 64}
    assert not _review_arbitration_reference_valid(
        {"review_arbitration_receipt": tampered},
        tmp_path / "checkpoints",
        translations=None,
        annotations=result[1],
        options=options,
    )


def test_stale_freeze_files_are_ignored_without_authoritative_binding(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    for name in ("first-chapter-freeze.json", "first-chapter-freeze.stale.json"):
        (checkpoint / name).write_text('{"stale":true}', encoding="utf-8")
    result = _coordinate_review_candidates(
        [_source("accepted", 0)],
        **_kwargs(tmp_path, lambda *_args, **_kwargs: None),
    )
    assert result[1]["seg-1"]["commentary"] == "accepted"


@pytest.mark.parametrize(
    "stage",
    [
        "before_partial",
        "after_partial_before_submit",
        "after_submit_before_decision",
        "after_decision_before_receipt",
        "after_receipt_before_reviewed_artifact",
    ],
)
def test_crash_cutpoints_replay_with_at_most_one_paid_call(
    stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injected = False
    calls = 0
    paid_responses = {}

    def cutpoint(current: str, _path: Path):
        nonlocal injected
        if current == stage and not injected:
            injected = True
            raise RuntimeError(f"crash:{stage}")

    def llm(prompt: str, **_kwargs):
        nonlocal calls
        key = prompt
        if key not in paid_responses:
            calls += 1
            paid_responses[key] = _select_first_arbitration(prompt)
        return paid_responses[key]

    monkeypatch.setattr(
        pipeline_module, "_review_arbitration_cutpoint", cutpoint,
    )
    with pytest.raises(RuntimeError, match="crash"):
        _coordinate_review_candidates(
            [_source("candidate-a", 0), _source("candidate-b", 1)],
            **_kwargs(tmp_path, llm),
        )
    result = _coordinate_review_candidates(
        [_source("candidate-a", 0), _source("candidate-b", 1)],
        **_kwargs(tmp_path, llm),
    )
    assert result[3]["status"] == "resolved"
    assert calls == 1
