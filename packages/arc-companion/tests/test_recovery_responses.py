from __future__ import annotations

import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Mapping

import pytest
import arc_companion.recovery_responses as recovery_module

from arc_companion.ledger import initialize_lane_ledger
from arc_companion.ledger_registry import (
    mutate_registered_lane_ledger,
    read_registered_lane_ledger,
)
from arc_companion.recovery_responses import (
    RecoveryResponseError,
    explicit_attempt_references,
    recover_complete_ledger_response,
    seal_submission_attempts,
    submission_receipt_reference,
    validate_ledger_submission_reference,
    write_ledger_submission_receipt,
    discover_submission_receipts,
)
from arc_llm.attempt_diagnostics import AttemptDiagnostics
from arc_llm.call_record import ARC_LLM_CALL_RECORD_FIELD
from arc_llm.call_checkpoint import (
    LLMCallCheckpointError,
    checkpoint_recomputation_binding,
    checkpoint_path,
    prepare_call,
    record_failure,
    record_submitted,
)
from arc_llm.providers.base import LLMSubmissionState, LLMWorkerTimeout
from arc_llm.response_candidates import LLMResponseCandidateConflict


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class _InjectedCrash(BaseException):
    pass


def _recover(reference: Mapping[str, str], **kwargs):
    return recover_complete_ledger_response(
        reference,
        idempotency_key="ch-0001:translation:call-s1:generation-1",
        expected_receipt_identity_sha256=reference["identity_sha256"],
        **kwargs,
    )


def _mutate_registered(checkpoint: Path, ledger_path: Path, mutate) -> None:
    _ledger, digest = read_registered_lane_ledger(checkpoint, ledger_path)
    mutate_registered_lane_ledger(
        checkpoint,
        ledger_path,
        expected_sha256=digest,
        mutate=mutate,
    )


def _receipt_kwargs(tmp_path: Path, *, logical_unit: str = "s1") -> dict:
    checkpoint = tmp_path / "checkpoint"
    ledger = checkpoint / "chapters" / "ch-0001" / "translation-ledger.json"
    if not ledger.exists():
        initialize_lane_ledger(
            ledger, chapter_id="ch-0001", lane="translation",
            segment_ids=["s1", "s2"],
        )
    return {
        "checkpoint_dir": checkpoint,
        "artifact_dir": checkpoint / "llm" / "translation" / logical_unit,
        "ledger_path": ledger,
        "session_key": "ch-0001:translation",
        "logical_unit": logical_unit,
        "generation": 1,
        "idempotency_key": f"ch-0001:translation:{logical_unit}:generation-1",
        "schema": SCHEMA,
        "prompt": f"translate {logical_unit}",
        "recovery_unit": "translation",
        "input_sha256": f"input-{logical_unit}",
        "ordered_siblings": ["s1", "s2"],
        "suffix": [logical_unit] if logical_unit == "s2" else ["s1", "s2"],
        "validator": "translation-schema+invariants.v1",
        "application": "normal-translation-pipeline-replay.v1",
    }


def _crash_on(
    monkeypatch: pytest.MonkeyPatch, point: str, *, occurrence: int = 1,
) -> None:
    seen = 0

    def inject(candidate: str) -> None:
        nonlocal seen
        if candidate != point:
            return
        seen += 1
        if seen == occurrence:
            raise _InjectedCrash(point)

    monkeypatch.setattr(recovery_module, "_recovery_write_fault", inject)


def _write_receipt_in_process(kwargs: dict) -> str:
    return str(write_ledger_submission_receipt(**kwargs))


def _submitted_fixture(
    tmp_path: Path,
    *,
    candidates: tuple[dict[str, str], ...] = ({"answer": "recovered"},),
    add_foreign_attempt: bool = False,
    raw_events: tuple[dict, ...] | None = None,
    parsed_candidates: tuple[dict[str, str], ...] | None = None,
    add_partial_attempt: bool = False,
) -> tuple[Path, Path, dict[str, str], Path]:
    checkpoint = tmp_path / "checkpoint"
    ledger = checkpoint / "chapters" / "ch-0001" / "translation-ledger.json"
    initialize_lane_ledger(
        ledger, chapter_id="ch-0001", lane="translation", segment_ids=["s1"],
    )
    artifact = checkpoint / "llm" / "translation" / "s1" / "generation-1"
    prompt = "translate s1"
    key = "ch-0001:translation:call-s1:generation-1"
    receipt_path = write_ledger_submission_receipt(
        checkpoint_dir=checkpoint,
        artifact_dir=artifact,
        ledger_path=ledger,
        session_key="ch-0001:translation",
        logical_unit="s1",
        generation=1,
        idempotency_key=key,
        schema=SCHEMA,
        prompt=prompt,
        recovery_unit="translation",
        input_sha256="input-s1",
        ordered_siblings=["s1"],
        suffix=["s1"],
        validator="translation-schema+invariants.v1",
        application="normal-translation-pipeline-replay.v1",
    )
    call_path, identity = checkpoint_path(
        artifact,
        prompt=prompt,
        schema=SCHEMA,
        provider="codex-cli",
        model="test-model",
        call_label="call-s1",
        session_policy="stateful",
        session_key="ch-0001:translation",
        runtime_fingerprint="runtime",
        idempotency_key=key,
        generation=1,
        progress_contract_scope="session",
        initial_native_authorization=(
            str(ledger.resolve(strict=False)),
            "ch-0001:translation",
            "s1",
            1,
            key,
        ),
    )
    prepared = prepare_call(call_path, identity=identity)
    record_submitted(prepared)
    timeout = LLMWorkerTimeout(
        "idle", submission_state=LLMSubmissionState.SUBMITTED,
    )
    record_failure(prepared, timeout)

    if add_foreign_attempt:
        foreign = AttemptDiagnostics(
            artifact,
            provider="codex-cli",
            model="test-model",
            fallback_index=0,
            attempt=1,
            call_label="foreign-call",
            env={},
        )
        foreign.bind_checkpoint_identity("foreign-checkpoint")
        foreign.mark_submitted()
        foreign.record_candidate(
            {"answer": "foreign"}, source="provider_parsed_response",
        )
        foreign.finalize(outcome="success")
    diagnostics = AttemptDiagnostics(
        artifact,
        provider="codex-cli",
        model="test-model",
        fallback_index=0,
        attempt=1,
        call_label="call-s1",
        env={},
    )
    diagnostics.bind_checkpoint_identity(identity)
    diagnostics.mark_submitted()
    events = raw_events
    if events is None:
        events = (
            {"type": "thread.started", "thread_id": "thread-recovery"},
            {"type": "turn.started"},
            *(
                {"type": "item.completed", "item": {
                    "type": "agent_message", "text": json.dumps(candidate),
                }} for candidate in candidates
            ),
            {"type": "turn.completed", "usage": {}},
        )
    for event in events:
        if event.get("direction") == "request":
            diagnostics.capture_raw_event(event)
        else:
            diagnostics.capture_stdout(
                json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
            )
    for candidate in parsed_candidates if parsed_candidates is not None else candidates:
        diagnostics.record_candidate(candidate, source="provider_parsed_response")
    attempt_ref = diagnostics.finalize(outcome="timeout", error=timeout)
    attempt_references = [{
        "path": (artifact / attempt_ref.path).relative_to(checkpoint).as_posix(),
        "sha256": attempt_ref.sha256,
    }]
    if add_partial_attempt:
        partial = AttemptDiagnostics(
            artifact,
            provider="codex-cli",
            model="test-model",
            fallback_index=0,
            attempt=2,
            call_label="call-s1",
            env={},
        )
        partial.bind_checkpoint_identity(identity)
        partial.mark_submitted()
        partial.capture_stdout(json.dumps({
            "type": "thread.started", "thread_id": "thread-recovery",
        }) + "\n")
        partial_ref = partial.finalize(outcome="timeout", error=timeout)
        attempt_references.append({
            "path": (artifact / partial_ref.path).relative_to(checkpoint).as_posix(),
            "sha256": partial_ref.sha256,
        })
    seal_submission_attempts(
        receipt_path,
        checkpoint_dir=checkpoint,
        attempt_references=attempt_references,
    )
    reference = submission_receipt_reference(
        receipt_path, checkpoint_dir=checkpoint,
    )
    return checkpoint, ledger, reference, call_path


def test_seal_binds_only_the_submitting_calls_explicit_attempt_refs(
    tmp_path: Path,
) -> None:
    checkpoint, _ledger, reference, _call = _submitted_fixture(
        tmp_path, add_foreign_attempt=True,
    )
    receipt = discover_submission_receipts(checkpoint)[0][1]

    assert len(list((checkpoint / receipt["artifact_dir"] / "attempts").iterdir())) == 2
    assert len(receipt["attempt_records"]) == 1
    assert "foreign-call" not in receipt["attempt_records"][0]["path"]


def _attempt_aggregate(
    artifact: Path,
    positions: list[tuple[int, int, str, str]],
) -> tuple[dict[str, str], ...]:
    refs: list[dict[str, str]] = []
    for fallback_index, attempt, outcome, label in positions:
        diagnostics = AttemptDiagnostics(
            artifact,
            provider=f"provider-{fallback_index}",
            model=f"model-{fallback_index}",
            fallback_index=fallback_index,
            attempt=attempt,
            call_label=label,
            env={},
        )
        error = RuntimeError("failed") if outcome not in {"success", "replayed"} else None
        ref = diagnostics.finalize(outcome=outcome, error=error)
        refs.append({"path": ref.path, "sha256": ref.sha256})
    return tuple(refs)


def test_explicit_attempt_references_consumes_complete_ordered_runner_aggregate(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    artifact = checkpoint / "llm" / "call"
    aggregate = _attempt_aggregate(artifact, [
        (0, 1, "error", "call"),
        (0, 2, "error", "call"),
        (1, 1, "timeout", "call"),
    ])
    error = RuntimeError("terminal")
    error.attempt_diagnostic_refs = aggregate

    resolved = explicit_attempt_references(
        error, checkpoint_dir=checkpoint, artifact_dir=artifact,
    )

    assert [item["sha256"] for item in resolved] == [
        item["sha256"] for item in aggregate
    ]


@pytest.mark.parametrize(
    "positions,duplicate,match",
    [
        ([(0, 2, "timeout", "call")], False, "start at one"),
        ([(0, 1, "error", "call"), (0, 3, "timeout", "call")], False, "contiguous"),
        ([(0, 1, "success", "call"), (0, 2, "error", "call")], False, "successful terminal"),
        ([(0, 1, "error", "call"), (0, 2, "timeout", "other")], False, "lifecycle is invalid"),
        ([(0, 1, "timeout", "call")], True, "duplicated"),
    ],
)
def test_explicit_attempt_references_rejects_incomplete_or_invalid_lifecycle(
    tmp_path: Path,
    positions: list[tuple[int, int, str, str]],
    duplicate: bool,
    match: str,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    artifact = checkpoint / "llm" / "call"
    aggregate = _attempt_aggregate(artifact, positions)
    if duplicate:
        aggregate = (*aggregate, aggregate[0])
    error = RuntimeError("terminal")
    error.attempt_diagnostic_refs = aggregate

    with pytest.raises(RecoveryResponseError, match=match):
        explicit_attempt_references(
            error, checkpoint_dir=checkpoint, artifact_dir=artifact,
        )


def test_call_record_attempt_aggregate_rejects_partial_diagnostic_reference(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    artifact = checkpoint / "llm" / "call"
    value = {ARC_LLM_CALL_RECORD_FIELD: {
        "attempts": [{"diagnostic_path": "attempts/x/record.json",
                      "diagnostic_sha256": None}],
    }}
    with pytest.raises(RecoveryResponseError, match="partial"):
        explicit_attempt_references(
            value, checkpoint_dir=checkpoint, artifact_dir=artifact,
        )


def _rewrite_sealed_attempt(
    checkpoint: Path, reference: dict[str, str],
    mutate: callable,
) -> None:
    receipt_path = checkpoint / reference["path"]
    sidecar_path = receipt_path.with_name(f"{receipt_path.stem}.sealed.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    record_ref = sidecar["attempt_records"][0]
    record_path = checkpoint / record_ref["path"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    mutate(record_path, record)
    record_path.chmod(0o600)
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    record_ref["sha256"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
    sidecar_path.chmod(0o600)
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    index_path = checkpoint / "recovery-submissions" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["sidecar_sha256"] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_complete_attempt_candidate_promotes_exact_checkpoint(tmp_path: Path) -> None:
    checkpoint, ledger, reference, call_path = _submitted_fixture(tmp_path)

    result = _recover(
        reference,
        checkpoint_dir=checkpoint,
        ledger_path=ledger,
        session_key="ch-0001:translation",
        logical_unit="s1",
        generation=1,
    )

    assert result["complete"] is True
    assert result["source"] == "promoted_response_pending_business"
    assert result["business_status"] == "pending_validation_and_application"
    persisted = json.loads(call_path.read_text(encoding="utf-8"))
    assert persisted["state"] == "response_received"
    assert persisted["response"]["value"] == {"answer": "recovered"}
    assert persisted["recovered_response"]["prior_failure"]["state"] == "failed"
    assert "failure_category" not in persisted
    assert "resumable" not in persisted


def test_receipt_binds_registered_semantics_but_allows_state_only_changes(
    tmp_path: Path,
) -> None:
    checkpoint, ledger_path, reference, _call_path = _submitted_fixture(tmp_path)
    receipt = discover_submission_receipts(checkpoint)[0][1]

    def mark_supervised(ledger):
        updated = dict(ledger)
        blocks = [dict(item) for item in updated["blocks"]]
        blocks[0].update(state="submitted", submission_state="submitted")
        updated["blocks"] = blocks
        updated["needs_supervision"] = {
            "segment_id": "s1", "reason": "timeout",
        }
        updated["updated_at"] = float(updated.get("updated_at") or 0) + 1
        return updated

    _mutate_registered(checkpoint, ledger_path, mark_supervised)
    _current, current_digest = read_registered_lane_ledger(checkpoint, ledger_path)
    validated = validate_ledger_submission_reference(
        reference,
        checkpoint_dir=checkpoint,
        expected_recovery_identity=(
            str(ledger_path),
            "ch-0001:translation",
            "s1",
            1,
            "ch-0001:translation:call-s1:generation-1",
        ),
        expected_receipt_identity_sha256=reference["identity_sha256"],
    )

    assert current_digest != receipt["registered_ledger_sha256_at_creation"]
    assert validated["validated_ledger_snapshot"] == receipt["ledger_snapshot"]
    assert validated["current_registered_ledger_sha256"] == current_digest
    assert _recover(
        reference,
        checkpoint_dir=checkpoint,
        ledger_path=ledger_path,
        session_key="ch-0001:translation",
        logical_unit="s1",
        generation=1,
    )["business_status"] == "pending_validation_and_application"


def test_recovery_rejects_changed_v5_native_authorization(
    tmp_path: Path,
) -> None:
    checkpoint, ledger_path, reference, call_path = _submitted_fixture(tmp_path)
    payload = json.loads(call_path.read_text(encoding="utf-8"))
    payload["logical_identity"]["initial_native_authorization"][
        "logical_unit"
    ] = "different-unit"
    call_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        RecoveryResponseError,
        match="call checkpoint does not match recovery submission",
    ):
        _recover(
            reference,
            checkpoint_dir=checkpoint,
            ledger_path=ledger_path,
            session_key="ch-0001:translation",
            logical_unit="s1",
            generation=1,
        )


def test_creation_digest_is_audit_only_and_does_not_readdress_same_receipt(
    tmp_path: Path,
) -> None:
    kwargs = _receipt_kwargs(tmp_path)
    first = write_ledger_submission_receipt(**kwargs)
    original = json.loads(first.read_text(encoding="utf-8"))

    def change_state_only(ledger):
        updated = dict(ledger)
        blocks = [dict(item) for item in updated["blocks"]]
        blocks[0].update(state="submitted", submission_state="submitted")
        updated["blocks"] = blocks
        updated["updated_at"] = float(updated.get("updated_at") or 0) + 1
        return updated

    _mutate_registered(kwargs["checkpoint_dir"], kwargs["ledger_path"], change_state_only)
    second = write_ledger_submission_receipt(**kwargs)

    assert second == first
    assert len(list(kwargs["artifact_dir"].glob("recovery-submission-*.json"))) == 1
    assert json.loads(second.read_text(encoding="utf-8"))[
        "registered_ledger_sha256_at_creation"
    ] == original["registered_ledger_sha256_at_creation"]


@pytest.mark.parametrize(
    "mutation",
    ["chapter", "lane", "ledger_generation", "block_generation", "input", "topology"],
)
def test_current_registered_ledger_semantic_changes_reject_recovery(
    tmp_path: Path, mutation: str,
) -> None:
    checkpoint, ledger_path, reference, _call_path = _submitted_fixture(tmp_path)

    def change(ledger):
        updated = dict(ledger)
        blocks = [dict(item) for item in updated["blocks"]]
        if mutation == "chapter":
            updated["chapter_id"] = "changed"
        elif mutation == "lane":
            updated["lane"] = "changed"
        elif mutation == "ledger_generation":
            updated["generation"] = 2
        elif mutation == "block_generation":
            blocks[0]["generation"] = 2
        elif mutation == "input":
            blocks[0]["input_sha256"] = "changed-input"
        elif mutation == "topology":
            blocks.append({
                "segment_id": "s2", "state": "prepared",
                "submission_state": "not_submitted", "generation": 1,
            })
        updated["blocks"] = blocks
        return updated

    _mutate_registered(checkpoint, ledger_path, change)
    with pytest.raises(RecoveryResponseError, match="registered ledger"):
        _recover(
            reference,
            checkpoint_dir=checkpoint,
            ledger_path=ledger_path,
            session_key="ch-0001:translation",
            logical_unit="s1",
            generation=1,
        )


def test_expected_full_recovery_identity_and_receipt_identity_are_required(
    tmp_path: Path,
) -> None:
    checkpoint, ledger_path, reference, _call_path = _submitted_fixture(tmp_path)
    with pytest.raises(RecoveryResponseError, match="does not match lane entry"):
        recover_complete_ledger_response(
            reference,
            checkpoint_dir=checkpoint,
            ledger_path=ledger_path,
            session_key="ch-0001:translation",
            logical_unit="s1",
            generation=1,
            idempotency_key="wrong-key",
            expected_receipt_identity_sha256=reference["identity_sha256"],
        )
    with pytest.raises(RecoveryResponseError, match="receipt identity changed"):
        recover_complete_ledger_response(
            reference,
            checkpoint_dir=checkpoint,
            ledger_path=ledger_path,
            session_key="ch-0001:translation",
            logical_unit="s1",
            generation=1,
            idempotency_key="ch-0001:translation:call-s1:generation-1",
            expected_receipt_identity_sha256="f" * 64,
        )


def test_conflicting_attempt_candidates_fail_closed(tmp_path: Path) -> None:
    checkpoint, ledger, reference, call_path = _submitted_fixture(
        tmp_path,
        candidates=({"answer": "one"}, {"answer": "two"}),
    )

    with pytest.raises(LLMResponseCandidateConflict):
        _recover(
            reference,
            checkpoint_dir=checkpoint,
            ledger_path=ledger,
            session_key="ch-0001:translation",
            logical_unit="s1",
            generation=1,
        )

    assert json.loads(call_path.read_text(encoding="utf-8"))["state"] == "failed"


def test_sealed_parsed_candidate_conflicts_with_raw_material(tmp_path: Path) -> None:
    checkpoint, ledger, reference, _call_path = _submitted_fixture(
        tmp_path,
        candidates=({"answer": "raw"},),
        parsed_candidates=({"answer": "parsed"},),
    )
    with pytest.raises(LLMResponseCandidateConflict):
        _recover(
            reference,
            checkpoint_dir=checkpoint,
            ledger_path=ledger,
            session_key="ch-0001:translation",
            logical_unit="s1",
            generation=1,
        )


def test_one_partial_attempt_fails_the_entire_sealed_group(tmp_path: Path) -> None:
    checkpoint, ledger, reference, _call_path = _submitted_fixture(
        tmp_path, add_partial_attempt=True,
    )
    with pytest.raises(
        RecoveryResponseError,
        match="incomplete or empty attempt conflicts with complete sealed material",
    ):
        _recover(
            reference,
            checkpoint_dir=checkpoint,
            ledger_path=ledger,
            session_key="ch-0001:translation",
            logical_unit="s1",
            generation=1,
        )


def test_empty_terminal_attempt_returns_no_candidate(tmp_path: Path) -> None:
    checkpoint, ledger, reference, _call_path = _submitted_fixture(
        tmp_path, candidates=(),
    )
    result = _recover(
        reference,
        checkpoint_dir=checkpoint,
        ledger_path=ledger,
        session_key="ch-0001:translation",
        logical_unit="s1",
        generation=1,
    )
    assert result == {"complete": False, "source": "no_complete_raw_candidate"}


def test_selection_receipt_commits_ordered_recomputable_attempt_evidence(
    tmp_path: Path,
) -> None:
    checkpoint, ledger, reference, call_path = _submitted_fixture(tmp_path)
    result = _recover(
        reference,
        checkpoint_dir=checkpoint,
        ledger_path=ledger,
        session_key="ch-0001:translation",
        logical_unit="s1",
        generation=1,
    )
    selection = json.loads(
        call_path.with_name(result["selection_receipt"]).read_text(encoding="utf-8")
    )
    evidence = selection["recovery_evidence"]
    assert [item["ordinal"] for item in evidence] == [1]
    assert [stream["name"] for stream in evidence[0]["streams"]] == [
        "raw_events", "stdout", "response_candidates", "stderr",
    ]
    assert selection["recovery_evidence_sha256"] == recovery_module.sha256_json(
        evidence
    )
    assert selection["material_sha256"]
    origins = {
        origin["source"]
        for candidate in selection["candidates"]
        for origin in candidate["origins"]
    }
    assert origins == {
        "codex.completed_message", "recovery.response_candidate_stream",
    }
    promoted = json.loads(call_path.read_text(encoding="utf-8"))
    assert promoted["response"]["candidate_selection"]["recovery_evidence"] == evidence


def test_recovered_promotion_rejects_changed_checkpoint_recomputation_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, ledger, reference, call_path = _submitted_fixture(tmp_path)
    actual_binding = checkpoint_recomputation_binding(call_path)
    changed_binding = dict(actual_binding)
    changed_binding["prompt_sha256"] = "f" * 64
    monkeypatch.setattr(
        recovery_module,
        "checkpoint_recomputation_binding",
        lambda path: changed_binding,
    )

    with pytest.raises(LLMCallCheckpointError, match="recomputation binding mismatch"):
        _recover(
            reference,
            checkpoint_dir=checkpoint,
            ledger_path=ledger,
            session_key="ch-0001:translation",
            logical_unit="s1",
            generation=1,
        )

    persisted = json.loads(call_path.read_text(encoding="utf-8"))
    assert persisted["state"] == "failed"
    assert persisted.get("response") is None
    assert checkpoint_recomputation_binding(call_path) == actual_binding


def test_tampered_attempt_stream_is_rejected(tmp_path: Path) -> None:
    checkpoint, ledger, reference, _call_path = _submitted_fixture(tmp_path)
    receipt_path = checkpoint / reference["path"]
    receipt = discover_submission_receipts(checkpoint)[0][1]
    record_path = checkpoint / receipt["attempt_records"][0]["path"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    stream = record_path.parent / record["streams"]["response_candidates"]["path"]
    stream.chmod(0o600)
    stream.write_text('{"sequence":1,"value":{"answer":"tampered"}}\n')

    with pytest.raises(RecoveryResponseError, match="stream hash changed"):
        _recover(
            reference,
            checkpoint_dir=checkpoint,
            ledger_path=ledger,
            session_key="ch-0001:translation",
            logical_unit="s1",
            generation=1,
        )


def test_receipt_reference_cannot_be_reused_for_other_generation(tmp_path: Path) -> None:
    checkpoint, ledger, reference, _call_path = _submitted_fixture(tmp_path)

    with pytest.raises(RecoveryResponseError, match="does not match lane entry"):
        _recover(
            reference,
            checkpoint_dir=checkpoint,
            ledger_path=ledger,
            session_key="ch-0001:translation",
            logical_unit="s1",
            generation=2,
        )


def test_crash_after_selection_receipt_reenters_without_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, ledger, reference, call_path = _submitted_fixture(tmp_path)
    real_promote = recovery_module.promote_recovered_response
    calls = 0

    def crash_before_promotion(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("crash after selection receipt")

    monkeypatch.setattr(
        recovery_module, "promote_recovered_response", crash_before_promotion,
    )
    with pytest.raises(RuntimeError, match="crash after selection receipt"):
        _recover(
            reference,
            checkpoint_dir=checkpoint,
            ledger_path=ledger,
            session_key="ch-0001:translation",
            logical_unit="s1",
            generation=1,
        )
    assert call_path.with_name(
        f"{call_path.stem}.candidate-selection.json"
    ).is_file()
    assert json.loads(call_path.read_text(encoding="utf-8"))["state"] == "failed"

    monkeypatch.setattr(recovery_module, "promote_recovered_response", real_promote)
    result = _recover(
        reference,
        checkpoint_dir=checkpoint,
        ledger_path=ledger,
        session_key="ch-0001:translation",
        logical_unit="s1",
        generation=1,
    )
    assert result["complete"] is True
    assert calls == 1


def test_attempt_from_other_checkpoint_identity_is_rejected(tmp_path: Path) -> None:
    checkpoint, ledger, reference, _ = _submitted_fixture(tmp_path)
    _rewrite_sealed_attempt(
        checkpoint, reference,
        lambda _path, record: record.__setitem__(
            "checkpoint_identity", "different-prepared-call",
        ),
    )

    with pytest.raises(
        RecoveryResponseError, match="evidence manifest|does not match submitted call",
    ):
        _recover(
            reference, checkpoint_dir=checkpoint, ledger_path=ledger,
            session_key="ch-0001:translation", logical_unit="s1", generation=1,
        )


def test_truncated_stream_is_rejected_even_with_rehashed_receipts(tmp_path: Path) -> None:
    checkpoint, ledger, reference, _ = _submitted_fixture(tmp_path)
    _rewrite_sealed_attempt(
        checkpoint, reference,
        lambda _path, record: record["streams"]["response_candidates"].__setitem__(
            "truncated", True,
        ),
    )

    with pytest.raises(RecoveryResponseError, match="evidence manifest|truncated"):
        _recover(
            reference, checkpoint_dir=checkpoint, ledger_path=ledger,
            session_key="ch-0001:translation", logical_unit="s1", generation=1,
        )


def test_existing_receipt_repairs_missing_discovery_index(tmp_path: Path) -> None:
    checkpoint, ledger, reference, _ = _submitted_fixture(tmp_path)
    (checkpoint / "recovery-submissions" / "index.json").unlink()
    assert discover_submission_receipts(checkpoint) == []

    receipt = json.loads((checkpoint / reference["path"]).read_text(encoding="utf-8"))
    repaired = write_ledger_submission_receipt(
        checkpoint_dir=checkpoint,
        artifact_dir=checkpoint / receipt["artifact_dir"],
        ledger_path=ledger,
        session_key=receipt["session_key"], logical_unit=receipt["logical_unit"],
        generation=receipt["generation"], idempotency_key=receipt["idempotency_key"],
        schema=receipt["schema"], prompt="translate s1", recovery_unit="translation",
        input_sha256="input-s1", ordered_siblings=["s1"], suffix=["s1"],
        validator="translation-schema+invariants.v1",
        application="normal-translation-pipeline-replay.v1",
    )

    assert repaired == checkpoint / reference["path"]
    assert discover_submission_receipts(checkpoint)[0][0] == repaired


def test_discovery_never_guesses_attempts_for_unsealed_crash_cut(tmp_path: Path) -> None:
    checkpoint, _ledger, reference, _ = _submitted_fixture(tmp_path)
    receipt_path = checkpoint / reference["path"]
    receipt_path.with_name(f"{receipt_path.stem}.sealed.json").unlink()
    index_path = checkpoint / "recovery-submissions" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0].update({
        "state": "prepared", "sidecar_path": None, "sidecar_sha256": None,
    })
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    discovered = discover_submission_receipts(checkpoint)

    assert len(discovered) == 1
    assert discovered[0][1]["sealed"] is False
    assert discovered[0][1]["attempt_records"] == []


@pytest.mark.parametrize("events, message", [
    (({"type": "mystery"}, {"type": "turn.completed"}), "unknown event"),
    (({"type": "thread.started", "sequence": 1}, {"type": "turn.completed", "sequence": 1}), "ordinals"),
])
def test_raw_replay_requires_closed_terminal_grammar(
    tmp_path: Path, events: tuple[dict, ...], message: str,
) -> None:
    checkpoint, ledger, reference, _ = _submitted_fixture(
        tmp_path, raw_events=events,
    )
    with pytest.raises(RecoveryResponseError, match=message):
        _recover(
            reference, checkpoint_dir=checkpoint, ledger_path=ledger,
            session_key="ch-0001:translation", logical_unit="s1", generation=1,
        )


def test_partial_raw_replay_fails_closed(tmp_path: Path) -> None:
    checkpoint, ledger, reference, _ = _submitted_fixture(
        tmp_path, raw_events=({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": '{"answer":"x"}'},
        },),
    )
    with pytest.raises(RecoveryResponseError, match="thread.started|final"):
        _recover(
            reference, checkpoint_dir=checkpoint, ledger_path=ledger,
            session_key="ch-0001:translation", logical_unit="s1", generation=1,
        )


def test_prepared_receipt_is_immutable_and_index_binds_seal(tmp_path: Path) -> None:
    checkpoint, _ledger, reference, _ = _submitted_fixture(tmp_path)
    receipt_path = checkpoint / reference["path"]
    prepared = receipt_path.read_bytes()
    sidecar = receipt_path.with_name(f"{receipt_path.stem}.sealed.json")
    index = json.loads(
        (checkpoint / "recovery-submissions" / "index.json").read_text()
    )["entries"][0]

    assert receipt_path.read_bytes() == prepared
    assert sidecar.is_file()
    assert index["state"] == "sealed"
    assert index["receipt_sha256"] == hashlib.sha256(prepared).hexdigest()
    assert index["sidecar_sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()


def test_submission_cap_is_checked_before_second_receipt_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, ledger, _reference, _ = _submitted_fixture(tmp_path)
    monkeypatch.setattr(recovery_module, "MAX_SUBMISSION_INDEX_ENTRIES", 1)
    second_artifact = checkpoint / "llm" / "translation" / "s2" / "generation-1"
    with pytest.raises(RecoveryResponseError, match="entry limit"):
        write_ledger_submission_receipt(
            checkpoint_dir=checkpoint, artifact_dir=second_artifact,
            ledger_path=ledger, session_key="ch-0001:translation",
            logical_unit="s1", generation=1, idempotency_key="second",
            schema=SCHEMA, prompt="second", recovery_unit="translation",
            input_sha256="input-s1", ordered_siblings=["s1"], suffix=["s1"],
            validator="translation-schema+invariants.v1",
            application="normal-translation-pipeline-replay.v1",
        )
    assert not list(second_artifact.glob("recovery-submission-*.json"))


def test_concurrent_same_identity_writers_converge(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    ledger = checkpoint / "chapters" / "ch-0001" / "translation-ledger.json"
    initialize_lane_ledger(
        ledger, chapter_id="ch-0001", lane="translation", segment_ids=["s1"],
    )
    kwargs = dict(
        checkpoint_dir=checkpoint, artifact_dir=checkpoint / "llm" / "same",
        ledger_path=ledger, session_key="ch-0001:translation", logical_unit="s1",
        generation=1, idempotency_key="same", schema=SCHEMA, prompt="same",
        recovery_unit="translation", input_sha256="input-s1",
        ordered_siblings=["s1"], suffix=["s1"],
        validator="translation-schema+invariants.v1",
        application="normal-translation-pipeline-replay.v1",
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda _item: write_ledger_submission_receipt(**kwargs), range(2)))
    assert paths[0] == paths[1]
    assert len(discover_submission_receipts(checkpoint)) == 1


@pytest.mark.parametrize(("point", "occurrence"), [
    ("reservation:durable", 1),
    ("prepared_receipt:after_file_write", 1),
    ("prepared_receipt:after_file_fsync", 1),
    ("prepared_receipt:after_publish", 1),
    ("prepared_receipt:after_directory_fsync", 1),
    ("index:after_file_write", 1),
    ("index:after_file_fsync", 1),
    ("index:after_replace", 1),
    ("index:after_directory_fsync", 1),
    ("index:after_file_write", 2),
    ("index:after_file_fsync", 2),
    ("index:after_replace", 2),
    ("index:after_directory_fsync", 2),
    ("prepared_index:durable", 1),
])
def test_prepared_receipt_crash_cuts_reconcile_exact_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    point: str,
    occurrence: int,
) -> None:
    kwargs = _receipt_kwargs(tmp_path)
    _crash_on(monkeypatch, point, occurrence=occurrence)

    with pytest.raises(_InjectedCrash):
        write_ledger_submission_receipt(**kwargs)

    monkeypatch.setattr(recovery_module, "_recovery_write_fault", lambda _point: None)
    receipt_path = write_ledger_submission_receipt(**kwargs)
    discovered = discover_submission_receipts(kwargs["checkpoint_dir"])
    assert [(path, value["logical_unit"]) for path, value in discovered] == [
        (receipt_path, "s1")
    ]
    index = json.loads(
        (kwargs["checkpoint_dir"] / "recovery-submissions" / "index.json").read_text()
    )
    assert index["entries"][0]["state"] == "prepared"


def test_post_prepared_pre_sidecar_orphan_is_exactly_discoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _receipt_kwargs(tmp_path)
    _crash_on(monkeypatch, "prepared_index:durable")
    with pytest.raises(_InjectedCrash):
        write_ledger_submission_receipt(**kwargs)

    discovered = discover_submission_receipts(kwargs["checkpoint_dir"])
    assert len(discovered) == 1
    receipt_path, receipt = discovered[0]
    assert receipt["sealed"] is False
    assert receipt["attempt_records"] == []
    assert receipt_path.parent == kwargs["artifact_dir"]

    monkeypatch.setattr(recovery_module, "_recovery_write_fault", lambda _point: None)
    assert write_ledger_submission_receipt(**kwargs) == receipt_path
    assert discover_submission_receipts(kwargs["checkpoint_dir"])[0][0] == receipt_path


@pytest.mark.parametrize(("point", "occurrence"), [
    ("sealed_sidecar:after_file_write", 1),
    ("sealed_sidecar:after_file_fsync", 1),
    ("sealed_sidecar:after_publish", 1),
    ("sealed_sidecar:after_directory_fsync", 1),
    ("index:after_file_write", 3),
    ("index:after_file_fsync", 3),
    ("index:after_replace", 3),
    ("index:after_directory_fsync", 3),
    ("sealed_index:durable", 1),
])
def test_sealed_sidecar_crash_cuts_reconcile_exact_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    point: str,
    occurrence: int,
) -> None:
    _crash_on(monkeypatch, point, occurrence=occurrence)
    with pytest.raises(_InjectedCrash):
        _submitted_fixture(tmp_path)

    checkpoint = tmp_path / "checkpoint"
    artifact = checkpoint / "llm" / "translation" / "s1" / "generation-1"
    receipt_path = next(
        path for path in artifact.glob("recovery-submission-*.json")
        if not path.name.endswith(".sealed.json")
    )
    record_path = next((artifact / "attempts").glob("*/record.json"))
    attempt_ref = {
        "path": record_path.relative_to(checkpoint).as_posix(),
        "sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(recovery_module, "_recovery_write_fault", lambda _point: None)
    sealed = seal_submission_attempts(
        receipt_path,
        checkpoint_dir=checkpoint,
        attempt_references=[attempt_ref],
    )

    assert sealed["sealed"] is True
    discovered = discover_submission_receipts(checkpoint)
    assert len(discovered) == 1
    assert discovered[0][1]["sealed"] is True


def test_concurrent_distinct_identity_writers_preserve_both_entries(
    tmp_path: Path,
) -> None:
    kwargs = [_receipt_kwargs(tmp_path, logical_unit=unit) for unit in ("s1", "s2")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda value: write_ledger_submission_receipt(**value), kwargs))

    assert len(set(paths)) == 2
    discovered = discover_submission_receipts(tmp_path / "checkpoint")
    assert {value["logical_unit"] for _path, value in discovered} == {"s1", "s2"}


def test_cross_process_same_identity_writers_converge(tmp_path: Path) -> None:
    kwargs = _receipt_kwargs(tmp_path)
    with ProcessPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(_write_receipt_in_process, [kwargs, kwargs]))

    assert paths[0] == paths[1]
    discovered = discover_submission_receipts(tmp_path / "checkpoint")
    assert len(discovered) == 1
    assert discovered[0][1]["logical_unit"] == "s1"


def test_missing_submission_index_parent_is_empty(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    assert discover_submission_receipts(checkpoint) == []


def test_duplicate_index_path_fails_closed_without_rewrite(tmp_path: Path) -> None:
    kwargs = _receipt_kwargs(tmp_path)
    write_ledger_submission_receipt(**kwargs)
    index_path = kwargs["checkpoint_dir"] / "recovery-submissions" / "index.json"
    index = json.loads(index_path.read_text())
    index["entries"].append(dict(index["entries"][0]))
    index_path.write_text(json.dumps(index))
    before = index_path.read_bytes()

    with pytest.raises(RecoveryResponseError, match="duplicate paths"):
        discover_submission_receipts(kwargs["checkpoint_dir"])
    with pytest.raises(RecoveryResponseError, match="duplicate paths"):
        write_ledger_submission_receipt(**kwargs)

    assert index_path.read_bytes() == before


def test_exact_ten_thousand_entry_cap_precedes_receipt_write(
    tmp_path: Path,
) -> None:
    kwargs = _receipt_kwargs(tmp_path)
    index_dir = tmp_path / "checkpoint" / "recovery-submissions"
    index_dir.mkdir(parents=True)
    entries = [{
        "path": f"llm/fake/recovery-submission-{number:05d}.json",
        "identity_sha256": f"{number:064x}",
        "receipt_sha256": f"{number + 1:064x}",
        "state": "reserved",
        "sidecar_path": None,
        "sidecar_sha256": None,
    } for number in range(10_000)]
    (index_dir / "index.json").write_text(json.dumps({
        "schema_version": recovery_module.SUBMISSION_INDEX_SCHEMA_VERSION,
        "entries": entries,
    }))

    with pytest.raises(RecoveryResponseError, match="entry limit"):
        write_ledger_submission_receipt(**kwargs)

    assert not list(kwargs["artifact_dir"].glob("recovery-submission-*.json"))
    persisted = json.loads((index_dir / "index.json").read_text())
    assert len(persisted["entries"]) == 10_000


def test_same_address_different_receipt_bytes_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recovery_module, "sha256_json", lambda _value: "a" * 64)
    first = _receipt_kwargs(tmp_path, logical_unit="s1")
    second = _receipt_kwargs(tmp_path, logical_unit="s2")
    second["artifact_dir"] = first["artifact_dir"]
    first_path = write_ledger_submission_receipt(**first)
    first_path.unlink()

    with pytest.raises(RecoveryResponseError, match="index collision"):
        write_ledger_submission_receipt(**second)

    with pytest.raises(RecoveryResponseError, match="missing"):
        discover_submission_receipts(tmp_path / "checkpoint")


def test_prepared_receipt_parent_component_swap_cannot_write_outside_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _receipt_kwargs(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    moved = tmp_path / "moved-artifact"

    def swap(point: str) -> None:
        if point == "prepared_receipt:after_file_write":
            kwargs["artifact_dir"].rename(moved)
            kwargs["artifact_dir"].symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(recovery_module, "_recovery_write_fault", swap)
    with pytest.raises(RecoveryResponseError, match="parent binding|unsafe"):
        write_ledger_submission_receipt(**kwargs)

    assert list(outside.iterdir()) == []


def test_prepared_receipt_leaf_swap_cannot_follow_attacker_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _receipt_kwargs(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.json"
    victim.write_text("untouched")

    def swap(point: str) -> None:
        if point == "prepared_receipt:after_file_write":
            leaf = next(kwargs["artifact_dir"].glob(".*.staged"))
            leaf.rename(leaf.with_suffix(".orphan"))
            leaf.symlink_to(victim)

    monkeypatch.setattr(recovery_module, "_recovery_write_fault", swap)
    with pytest.raises(RecoveryResponseError, match="leaf"):
        write_ledger_submission_receipt(**kwargs)

    assert victim.read_text() == "untouched"


def test_immutable_receipt_reader_rejects_named_leaf_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, _ledger, reference, _call = _submitted_fixture(tmp_path)
    receipt = checkpoint / reference["path"]
    victim = tmp_path / "victim.json"
    victim.write_text("untouched", encoding="utf-8")
    fired = False

    def swap(point: str) -> None:
        nonlocal fired
        if point == "immutable_read:after_open" and not fired:
            fired = True
            receipt.unlink()
            receipt.symlink_to(victim)

    monkeypatch.setattr(recovery_module, "_recovery_write_fault", swap)
    with pytest.raises(RecoveryResponseError, match="named identity"):
        recovery_module._try_read_immutable_path(checkpoint, receipt)
    assert victim.read_text(encoding="utf-8") == "untouched"


def test_index_parent_component_swap_cannot_publish_outside_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _receipt_kwargs(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    moved = tmp_path / "moved-index"

    def swap(point: str) -> None:
        if point == "index:after_replace":
            index_dir = tmp_path / "checkpoint" / "recovery-submissions"
            index_dir.rename(moved)
            index_dir.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(recovery_module, "_recovery_write_fault", swap)
    with pytest.raises(RecoveryResponseError, match="unsafe|binding"):
        write_ledger_submission_receipt(**kwargs)

    assert list(outside.iterdir()) == []


def test_index_cas_preserves_regular_replacement_of_existing_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _receipt_kwargs(tmp_path)
    index_path = kwargs["checkpoint_dir"] / "recovery-submissions" / "index.json"
    backup = index_path.with_name("prior-index.json")
    sentinel = b"regular sentinel must survive\n"
    seen = 0

    def replace_existing(point: str) -> None:
        nonlocal seen
        if point != "index:after_file_write":
            return
        seen += 1
        if seen == 2:
            index_path.rename(backup)
            index_path.write_bytes(sentinel)

    monkeypatch.setattr(recovery_module, "_recovery_write_fault", replace_existing)
    with pytest.raises(RecoveryResponseError, match="compare-and-swap"):
        write_ledger_submission_receipt(**kwargs)

    assert seen == 2
    assert index_path.read_bytes() == sentinel
    assert backup.is_file()


def test_index_cas_preserves_unexpected_regular_creation_when_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _receipt_kwargs(tmp_path)
    index_path = kwargs["checkpoint_dir"] / "recovery-submissions" / "index.json"
    sentinel = b"unexpected regular sentinel must survive\n"

    def create_unexpected(point: str) -> None:
        if point == "index:after_file_write":
            index_path.write_bytes(sentinel)

    monkeypatch.setattr(recovery_module, "_recovery_write_fault", create_unexpected)
    with pytest.raises(RecoveryResponseError, match="compare-and-swap"):
        write_ledger_submission_receipt(**kwargs)

    assert index_path.read_bytes() == sentinel
    assert not list(kwargs["artifact_dir"].glob("recovery-submission-*.json"))


def test_index_final_swap_uses_exchange_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _receipt_kwargs(tmp_path, logical_unit="s1")
    write_ledger_submission_receipt(**first)
    index_path = first["checkpoint_dir"] / "recovery-submissions" / "index.json"
    original = index_path.read_bytes()
    victim = tmp_path / "victim.json"
    victim.write_bytes(b"untouched")

    def swap(point: str) -> None:
        if point == "index:after_replace":
            index_path.unlink()
            index_path.symlink_to(victim)

    monkeypatch.setattr(recovery_module, "_recovery_write_fault", swap)
    second = _receipt_kwargs(tmp_path, logical_unit="s2")
    with pytest.raises(RecoveryResponseError):
        write_ledger_submission_receipt(**second)
    assert index_path.read_bytes() == original
    assert victim.read_bytes() == b"untouched"
    assert not list(index_path.parent.glob(".index.json.*"))


def test_checkpoint_root_rename_recreate_aborts_one_inode_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _receipt_kwargs(tmp_path)
    checkpoint = kwargs["checkpoint_dir"]
    moved = tmp_path / "moved-checkpoint"

    def swap(point: str) -> None:
        if point == "reservation:durable":
            checkpoint.rename(moved)
            checkpoint.mkdir()

    monkeypatch.setattr(recovery_module, "_recovery_write_fault", swap)
    with pytest.raises(RecoveryResponseError, match="root binding changed"):
        write_ledger_submission_receipt(**kwargs)
    assert not list(checkpoint.rglob("recovery-submission-*.json"))
    assert not list(moved.rglob("*.staged"))


def test_checkpoint_root_swap_between_sidecar_and_index_never_splits_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    moved = tmp_path / "moved-checkpoint"

    def swap(point: str) -> None:
        if point == "sealed_sidecar:after_directory_fsync":
            checkpoint.rename(moved)
            checkpoint.mkdir()

    monkeypatch.setattr(recovery_module, "_recovery_write_fault", swap)
    with pytest.raises(RecoveryResponseError, match="root binding changed"):
        _submitted_fixture(tmp_path)
    assert list(checkpoint.iterdir()) == []
    old_index = json.loads(
        (moved / "recovery-submissions" / "index.json").read_text(encoding="utf-8")
    )
    assert old_index["entries"][0]["state"] == "prepared"
    assert len(list(moved.rglob("*.sealed.json"))) == 1


def test_durable_sidecar_orphan_is_reconciled_without_resealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, ledger, reference, _call = _submitted_fixture(tmp_path)
    receipt_path = next(
        path for path in (checkpoint / "llm").rglob("recovery-submission-*.json")
        if not path.name.endswith(".sealed.json")
    )
    # Return the prepared state so the test models a crash after sidecar
    # durability but before the sealed index mutation.
    index_path = checkpoint / "recovery-submissions" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = index["entries"][0]
    entry.update(state="prepared", sidecar_path=None, sidecar_sha256=None)
    index_path.write_text(json.dumps(index), encoding="utf-8")

    discovered = discover_submission_receipts(checkpoint)
    assert discovered[0][0] == receipt_path
    assert discovered[0][1]["sealed"] is True
    repaired = json.loads(index_path.read_text(encoding="utf-8"))["entries"][0]
    assert repaired["state"] == "sealed"
    replayed = _recover(
        reference,
        checkpoint_dir=checkpoint,
        ledger_path=ledger,
        session_key="ch-0001:translation",
        logical_unit="s1",
        generation=1,
    )
    assert replayed["complete"] is True


def test_submission_lock_replacement_fails_before_index_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _receipt_kwargs(tmp_path)
    lock = kwargs["checkpoint_dir"] / "recovery-submissions" / ".index.lock"

    def replace_lock(point: str) -> None:
        if point == "index:after_file_fsync":
            lock.unlink()
            lock.write_bytes(b"replacement")

    monkeypatch.setattr(recovery_module, "_recovery_write_fault", replace_lock)
    with pytest.raises(RecoveryResponseError, match="lock changed"):
        write_ledger_submission_receipt(**kwargs)
    assert not list(kwargs["artifact_dir"].glob("recovery-submission-*.json"))
    monkeypatch.setattr(recovery_module, "_recovery_write_fault", lambda _point: None)
    write_ledger_submission_receipt(**kwargs)
    assert not list(lock.parent.glob(".index.json.*"))


def test_receipt_and_sidecar_limits_precede_immutable_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _receipt_kwargs(tmp_path)
    monkeypatch.setattr(recovery_module, "MAX_CONTROL_JSON_BYTES", 128)
    with pytest.raises(RecoveryResponseError, match="receipt exceeds"):
        write_ledger_submission_receipt(**kwargs)
    assert not (kwargs["checkpoint_dir"] / "recovery-submissions" / "index.json").exists()
    assert not list(kwargs["artifact_dir"].glob("recovery-submission-*.json"))


def test_attempt_reference_cap_precedes_sidecar_publication(tmp_path: Path) -> None:
    kwargs = _receipt_kwargs(tmp_path)
    receipt = write_ledger_submission_receipt(**kwargs)
    with pytest.raises(RecoveryResponseError, match="too many explicit attempt"):
        seal_submission_attempts(
            receipt,
            checkpoint_dir=kwargs["checkpoint_dir"],
            attempt_references=[{}] * (recovery_module.MAX_ATTEMPT_RECORDS + 1),
        )
    assert not receipt.with_name(f"{receipt.stem}.sealed.json").exists()


def test_duplicate_attempt_references_are_rejected_before_reseal(tmp_path: Path) -> None:
    checkpoint, _ledger, _reference, _call = _submitted_fixture(tmp_path)
    receipt_path, sealed = discover_submission_receipts(checkpoint)[0]
    reference = sealed["attempt_records"][0]
    with pytest.raises(RecoveryResponseError, match="duplicated"):
        seal_submission_attempts(
            receipt_path,
            checkpoint_dir=checkpoint,
            attempt_references=[reference] * recovery_module.MAX_ATTEMPT_RECORDS,
        )


def test_exact_receipt_byte_cap_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _receipt_kwargs(tmp_path / "baseline")
    baseline_path = write_ledger_submission_receipt(**baseline)
    exact_size = len(baseline_path.read_bytes())

    accepted = _receipt_kwargs(tmp_path / "accepted")
    monkeypatch.setattr(recovery_module, "MAX_CONTROL_JSON_BYTES", exact_size)
    assert write_ledger_submission_receipt(**accepted).is_file()

    rejected = _receipt_kwargs(tmp_path / "rejected")
    monkeypatch.setattr(recovery_module, "MAX_CONTROL_JSON_BYTES", exact_size - 1)
    with pytest.raises(RecoveryResponseError, match="receipt exceeds"):
        write_ledger_submission_receipt(**rejected)


def test_sidecar_byte_cap_precedes_immutable_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, _ledger, _reference, _call = _submitted_fixture(tmp_path)
    receipt_path, sealed = discover_submission_receipts(checkpoint)[0]
    attempts = list(sealed["attempt_records"])
    sidecar = receipt_path.with_name(f"{receipt_path.stem}.sealed.json")
    sidecar.unlink()
    index_path = checkpoint / "recovery-submissions" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0].update(
        state="prepared", sidecar_path=None, sidecar_sha256=None,
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")

    monkeypatch.setattr(recovery_module, "MAX_SUBMISSION_SEAL_BYTES", 128)
    with pytest.raises(RecoveryResponseError, match="sidecar exceeds"):
        seal_submission_attempts(
            receipt_path,
            checkpoint_dir=checkpoint,
            attempt_references=attempts,
        )
    assert not sidecar.exists()
