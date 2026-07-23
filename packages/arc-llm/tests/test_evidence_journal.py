from __future__ import annotations

import json
import multiprocessing
import queue
import stat
import threading
import time
from pathlib import Path

import pytest

from arc_llm import evidence_journal as journal_module
from arc_llm.evidence import EvidenceRequest, EvidenceResponse
from arc_llm.evidence_journal import (
    SCHEMA_VERSION,
    EvidenceJournal,
    EvidenceJournalContext,
    EvidenceJournalCorruptError,
    EvidenceJournalRecoveryError,
    EvidenceJournalStaleError,
    EvidenceOperationPolicy,
    canonical_hash,
)


_READ_POLICY = {"paper.read": EvidenceOperationPolicy(idempotent=True)}


def _process_resolve_worker(
    journal_root: str,
    start_event,
    controller_calls,
    results,
) -> None:
    context = EvidenceJournalContext(
        journal_root=Path(journal_root),
        run_id="run-1",
        lane_id="lane-1",
        worker_id="worker-a",
        logical_task_id="logical-round-1",
        source_generation=1,
        policy_hash="policy-v1",
        runtime_hash="runtime-v1",
    )
    request = _request()
    journal = EvidenceJournal(context.journal_root)

    def controller(requests, *, round_number):
        controller_calls.put((requests[0].request_id, round_number))
        time.sleep(0.1)
        return (EvidenceResponse("r1", True, {"round": round_number}),)

    start_event.wait(timeout=10)
    resolved = journal.resolve_round(
        context,
        (request,),
        controller,
        round_number=1,
        operation_policies=_READ_POLICY,
    )
    results.put(resolved[0].data)


def _context(
    tmp_path: Path,
    *,
    worker: str = "worker-a",
    policy_hash: str = "policy-v1",
    runtime_hash: str = "runtime-v1",
    generation: int = 1,
) -> EvidenceJournalContext:
    return EvidenceJournalContext(
        journal_root=tmp_path / "journal",
        run_id="run-1",
        lane_id="lane-1",
        worker_id=worker,
        logical_task_id="logical-round-1",
        source_generation=generation,
        policy_hash=policy_hash,
        runtime_hash=runtime_hash,
    )


def _request(
    request_id: str = "r1", *, arguments: dict | None = None, worker: str = "worker-a",
) -> EvidenceRequest:
    return EvidenceRequest(
        request_id=request_id,
        operation="paper.read",
        arguments=arguments or {"paper_id": "1234.5678"},
        reason="check a claim",
        worker_id=worker,
        role="proposer",
    )


def _controller(calls: list[tuple[str, ...]]):
    def resolve(requests, *, round_number):
        calls.append(tuple(request.request_id for request in requests))
        return tuple(
            EvidenceResponse(
                request.request_id,
                True,
                {"round": round_number, "request": request.request_id},
                provenance={"source": "fake"},
            )
            for request in requests
        )

    return resolve


def test_four_state_receipt_replays_and_redelivers_without_execution(tmp_path: Path) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request()
    calls: list[tuple[str, ...]] = []

    actions = journal.prepare_round(context, (request,), round_number=1)
    assert actions[0].action == "execute"
    prepared = journal.read_receipt(actions[0].address)
    assert prepared["schema_version"] == SCHEMA_VERSION
    assert prepared["state"] == "prepared"
    assert prepared["request"] == {
        "request_id": "r1",
        "operation": "paper.read",
        "arguments": {"paper_id": "1234.5678"},
        "reason": "check a claim",
    }
    assert "worker_id" not in prepared["request"]
    assert prepared["policy_hash"] != context.policy_hash
    assert prepared["runtime_hash"] != context.runtime_hash

    first = journal.resolve_round(
        context, (request,), _controller(calls), round_number=1,
        operation_policies=_READ_POLICY,
    )
    assert first[0].data["request"] == "r1"
    persisted = journal.read_receipt(actions[0].address)
    assert persisted["state"] == "response_persisted"
    assert set(persisted["timestamps"]) == {
        "prepared", "executed", "response_persisted",
    }

    journal.mark_delivered(
        context,
        (request,),
        round_number=1,
        target_generation=1,
        target_session="session-a",
        followup_id="followup-1",
    )
    journal.mark_delivered(
        context,
        (request,),
        round_number=1,
        target_generation=2,
        target_session="session-b",
        followup_id="followup-2",
    )
    replay = journal.resolve_round(
        context, (request,), _controller(calls), round_number=1,
    )

    assert replay == first
    assert calls == [("r1",)]
    delivered = journal.read_receipt(actions[0].address)
    assert delivered["state"] == "delivered"
    assert [item["target_generation"] for item in delivered["deliveries"]] == [1, 2]


def test_crash_after_executed_promotes_durable_response_without_reexecution(tmp_path: Path) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request()
    calls: list[tuple[str, ...]] = []

    def crash(state, _address, _receipt):
        if state == "executed":
            raise KeyboardInterrupt("simulated crash")

    with pytest.raises(KeyboardInterrupt, match="simulated crash"):
        journal.resolve_round(
            context, (request,), _controller(calls), round_number=1,
            transition_hook=crash,
        )
    address = context.address("r1", evidence_round=1)
    assert journal.read_receipt(address)["state"] == "executed"

    replay = journal.resolve_round(
        context, (request,), _controller(calls), round_number=1,
    )
    assert replay[0].ok is True
    assert calls == [("r1",)]
    assert journal.read_receipt(address)["state"] == "response_persisted"


def test_crash_after_prepared_recovers_idempotent_request(tmp_path: Path) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request()
    calls: list[tuple[str, ...]] = []

    def crash(state, _address, _receipt):
        if state == "prepared":
            raise RuntimeError("prepared crash")

    with pytest.raises(RuntimeError, match="prepared crash"):
        journal.resolve_round(
            context, (request,), _controller(calls), round_number=1,
            transition_hook=crash,
        )
    assert calls == []
    recovered = journal.resolve_round(
        context, (request,), _controller(calls), round_number=1,
        operation_policies=_READ_POLICY,
    )
    assert recovered[0].ok is True
    assert calls == [("r1",)]


def test_crash_after_response_persisted_replays_without_reexecution(tmp_path: Path) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request()
    calls: list[tuple[str, ...]] = []

    def crash(state, _address, _receipt):
        if state == "response_persisted":
            raise RuntimeError("append transcript crashed")

    with pytest.raises(RuntimeError, match="transcript"):
        journal.resolve_round(
            context, (request,), _controller(calls), round_number=1,
            transition_hook=crash,
        )
    replay = journal.resolve_round(
        context, (request,), _controller(calls), round_number=1,
    )
    assert replay[0].data["round"] == 1
    assert calls == [("r1",)]


def test_non_idempotent_recovery_requires_transaction_and_callback(tmp_path: Path) -> None:
    context = _context(tmp_path)
    request = _request()
    journal = EvidenceJournal(context.journal_root)
    policy = {"paper.read": EvidenceOperationPolicy(idempotent=False)}
    journal.prepare_round(
        context, (request,), round_number=1, operation_policies=policy,
    )

    with pytest.raises(EvidenceJournalRecoveryError, match="transaction receipt"):
        journal.resolve_round(
            context, (request,), _controller([]), round_number=1,
            operation_policies=policy,
        )


def test_non_idempotent_transaction_recovery_never_calls_controller(tmp_path: Path) -> None:
    context = _context(tmp_path)
    request = _request()
    journal = EvidenceJournal(context.journal_root)
    recovered: list[dict] = []

    def recover(item, receipt):
        recovered.append(dict(receipt))
        return EvidenceResponse(
            item.request_id, True, {"transaction": receipt["id"]},
            provenance={"source": "fake-transaction"},
        )

    policy = {
        "paper.read": EvidenceOperationPolicy(
            idempotent=False,
            transaction_receipt=lambda _request: {"id": "tx-1"},
            recover=recover,
        )
    }
    journal.prepare_round(
        context, (request,), round_number=1, operation_policies=policy,
    )
    calls: list[tuple[str, ...]] = []
    response = journal.resolve_round(
        context, (request,), _controller(calls), round_number=1,
        operation_policies=policy,
    )

    assert response[0].data == {"transaction": "tx-1"}
    assert recovered == [{"id": "tx-1"}]
    assert calls == []


@pytest.mark.parametrize("guard", ["arguments", "policy", "runtime"])
def test_same_address_rejects_stale_identity_guards(tmp_path: Path, guard: str) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request()
    journal.prepare_round(context, (request,), round_number=1)
    stale_context = context
    stale_request = request
    if guard == "arguments":
        stale_request = _request(arguments={"paper_id": "changed"})
    elif guard == "policy":
        stale_context = _context(tmp_path, policy_hash="policy-v2")
    else:
        stale_context = _context(tmp_path, runtime_hash="runtime-v2")

    with pytest.raises(EvidenceJournalStaleError, match=guard):
        journal.prepare_round(stale_context, (stale_request,), round_number=1)
    assert len(list(journal.receipts_root.glob("*.json"))) == 1


def test_reason_is_audit_data_not_identity(tmp_path: Path) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request()
    journal.prepare_round(context, (request,), round_number=1)
    changed_reason = EvidenceRequest(
        request.request_id,
        request.operation,
        request.arguments,
        reason="a clearer audit reason",
        worker_id=request.worker_id,
    )

    action = journal.prepare_round(
        context, (changed_reason,), round_number=1,
        operation_policies=_READ_POLICY,
    )
    assert action[0].action == "recover"


def test_round_cap_precedes_receipt_and_controller_side_effects(tmp_path: Path) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    calls: list[tuple[str, ...]] = []

    with pytest.raises(ValueError, match="exceeds max_rounds=3"):
        journal.resolve_round(
            context, (_request(),), _controller(calls), round_number=4,
        )

    assert calls == []
    assert list(journal.receipts_root.iterdir()) == []


def test_worker_isolation_allows_same_request_id(tmp_path: Path) -> None:
    first = _context(tmp_path, worker="worker-a")
    second = _context(tmp_path, worker="worker-b")
    journal = EvidenceJournal(first.journal_root)
    calls: list[tuple[str, ...]] = []

    a = journal.resolve_round(
        first, (_request(worker="worker-a"),), _controller(calls), round_number=1,
    )
    b = journal.resolve_round(
        second, (_request(worker="worker-b"),), _controller(calls), round_number=1,
    )

    assert a[0].request_id == b[0].request_id == "r1"
    assert calls == [("r1",), ("r1",)]
    assert len(list(journal.receipts_root.glob("*.json"))) == 2


def test_corrupt_receipt_fails_before_controller(tmp_path: Path) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request()
    action = journal.prepare_round(context, (request,), round_number=1)[0]
    journal.receipt_path(action.address).write_text("[]\n", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    with pytest.raises(EvidenceJournalCorruptError, match="root"):
        journal.resolve_round(
            context, (request,), _controller(calls), round_number=1,
        )
    assert calls == []


def test_tampered_persisted_operation_fails_its_guard_before_controller(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request()
    action = journal.prepare_round(context, (request,), round_number=1)[0]
    path = journal.receipt_path(action.address)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["request"]["operation"] = "paper.changed"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    with pytest.raises(EvidenceJournalCorruptError, match="operation/arguments guard"):
        journal.resolve_round(
            context, (request,), _controller(calls), round_number=1,
        )
    assert calls == []


def test_oversized_request_fails_before_receipt_or_controller(tmp_path: Path) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request(arguments={"blob": "x" * (1024 * 1024)})
    calls: list[tuple[str, ...]] = []

    with pytest.raises(ValueError, match="evidence request exceeds"):
        journal.resolve_round(
            context, (request,), _controller(calls), round_number=1,
        )

    assert calls == []
    assert list(journal.receipts_root.iterdir()) == []


def test_protocol_payloads_are_exact_and_first_response_equals_replay(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request(arguments={
        "paper_id": "1234.5678",
        "page_token": "page-2",
        "continuation_token": "continue-3",
        "session_token": "scientific-session-label",
        "note": "token: conserved tensor in the derivation",
        "author": "Ada Lovelace",
    })
    request = EvidenceRequest(
        request.request_id,
        request.operation,
        request.arguments,
        reason="check the token: operator notation exactly",
        worker_id=request.worker_id,
    )
    policy = {
        "paper.read": EvidenceOperationPolicy(
            idempotent=False,
            transaction_receipt=lambda _request: {
                "continuation_token": "transaction-page-4",
            },
        ),
    }
    controller_response = EvidenceResponse(
        request.request_id,
        True,
        {
            "page_token": "response-page-5",
            "continuation_token": "response-continuation-6",
            "session_token": "spectral-session-token",
            "message": "token: Ward identity insertion",
            "author": "Ada Lovelace",
        },
        provenance={
            "page_token": "provenance-page-7",
            "note": "token: exact scientific wording",
        },
    )
    responses = journal.resolve_round(
        context,
        (request,),
        lambda _requests, *, round_number: (controller_response,),
        round_number=1,
        operation_policies=policy,
    )
    assert responses == (controller_response,)
    path = journal.receipt_path(context.address("r1", evidence_round=1))
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["request"]["arguments"] == dict(request.arguments)
    assert receipt["request"]["reason"] == request.reason
    assert receipt["transaction_receipt"] == {
        "continuation_token": "transaction-page-4",
    }
    replay = journal.resolve_round(
        context,
        (request,),
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("controller must not run on replay")
        ),
        round_number=1,
        operation_policies=policy,
    )
    assert replay == responses == (controller_response,)


@pytest.mark.parametrize(
    "field", ["password", "api_key", "authorization", "private_key", "client_secret"],
)
def test_explicit_credential_fields_fail_closed_before_persistence(
    tmp_path: Path, field: str,
) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request(arguments={"paper_id": "1234.5678", field: "credential"})

    with pytest.raises(ValueError, match="prohibited credential field"):
        journal.resolve_round(
            context, (request,), _controller([]), round_number=1,
        )
    assert list(journal.receipts_root.iterdir()) == []


def test_response_and_transaction_credentials_fail_closed_without_leaking(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request()
    with pytest.raises(ValueError, match="transaction receipt.*credential field"):
        journal.resolve_round(
            context,
            (request,),
            _controller([]),
            round_number=1,
            operation_policies={
                "paper.read": EvidenceOperationPolicy(
                    transaction_receipt=lambda _request: {
                        "authorization": "credential",
                    },
                ),
            },
        )
    assert list(journal.receipts_root.iterdir()) == []

    with pytest.raises(ValueError, match="evidence response.*credential field"):
        journal.resolve_round(
            context,
            (request,),
            lambda _requests, *, round_number: (
                EvidenceResponse("r1", True, {"client_secret": "credential"}),
            ),
            round_number=1,
            operation_policies=_READ_POLICY,
        )
    receipt_path = journal.receipt_path(context.address("r1", evidence_round=1))
    assert "credential" not in receipt_path.read_text(encoding="utf-8")
    assert journal.read_receipt(context.address("r1", evidence_round=1))["state"] == "prepared"


def test_response_body_is_stored_once_and_bound_to_execution_and_address(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request()
    journal.resolve_round(
        context,
        (request,),
        lambda _requests, *, round_number: (
            EvidenceResponse("r1", True, {"unique_body": "only-once"}),
        ),
        round_number=1,
    )
    address = context.address("r1", evidence_round=1)
    path = journal.receipt_path(address)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["execution_receipt"] == {
        "request_id": "r1",
        "response_sha256": canonical_hash(receipt["response"]),
    }
    assert path.read_text(encoding="utf-8").count("only-once") == 1

    receipt["response"]["data"]["unique_body"] = "tampered"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(EvidenceJournalCorruptError, match="execution digest"):
        journal.read_receipt(address)


def test_response_request_id_is_bound_even_if_digest_is_recomputed(tmp_path: Path) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request()
    journal.resolve_round(
        context,
        (request,),
        lambda _requests, *, round_number: (EvidenceResponse("r1", True, {}),),
        round_number=1,
    )
    address = context.address("r1", evidence_round=1)
    path = journal.receipt_path(address)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["response"]["request_id"] = "different-request"
    receipt["execution_receipt"]["response_sha256"] = canonical_hash(
        receipt["response"]
    )
    path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(EvidenceJournalCorruptError, match="does not match its address"):
        journal.read_receipt(address)


def test_final_receipt_budget_is_preflighted_before_executed_state(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(journal_module, "_MAX_RESPONSE_BYTES", 4_000)
    monkeypatch.setattr(journal_module, "_MAX_RECEIPT_BYTES", 2_400)
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request()

    with pytest.raises(ValueError, match="evidence receipt exceeds"):
        journal.resolve_round(
            context,
            (request,),
            lambda _requests, *, round_number: (
                EvidenceResponse("r1", True, {"body": "x" * 2_000}),
            ),
            round_number=1,
            operation_policies=_READ_POLICY,
        )

    address = context.address("r1", evidence_round=1)
    assert journal.read_receipt(address)["state"] == "prepared"
    recovered = journal.resolve_round(
        context,
        (request,),
        lambda _requests, *, round_number: (
            EvidenceResponse("r1", True, {"body": "small"}),
        ),
        round_number=1,
        operation_policies=_READ_POLICY,
    )
    assert recovered[0].data == {"body": "small"}


def test_concurrent_recovery_executes_controller_once(tmp_path: Path) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request()
    calls: list[tuple[str, ...]] = []
    barrier = threading.Barrier(2)
    results: list[tuple[EvidenceResponse, ...]] = []

    def controller(requests, *, round_number):
        calls.append(tuple(item.request_id for item in requests))
        time.sleep(0.05)
        return (EvidenceResponse("r1", True, {"round": round_number}),)

    def worker() -> None:
        barrier.wait()
        results.append(journal.resolve_round(
            context, (request,), controller, round_number=1,
        ))

    threads = [threading.Thread(target=worker), threading.Thread(target=worker)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 2
    assert calls == [("r1",)]


def test_cross_process_recovery_executes_controller_once(tmp_path: Path) -> None:
    process_context = multiprocessing.get_context("spawn")
    start_event = process_context.Event()
    controller_calls = process_context.Queue()
    results = process_context.Queue()
    journal_root = str(tmp_path / "journal")
    processes = [
        process_context.Process(
            target=_process_resolve_worker,
            args=(journal_root, start_event, controller_calls, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start_event.set()
    try:
        for process in processes:
            process.join(timeout=15)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert [results.get(timeout=2), results.get(timeout=2)] == [
        {"round": 1}, {"round": 1},
    ]
    assert controller_calls.get(timeout=2) == ("r1", 1)
    with pytest.raises(queue.Empty):
        controller_calls.get(timeout=0.2)


def test_permissions_and_canonical_hash_are_deterministic(tmp_path: Path) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    action = journal.prepare_round(context, (_request(),), round_number=1)[0]

    assert stat.S_IMODE(context.journal_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(journal.locks_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(journal.receipt_path(action.address).stat().st_mode) == 0o600
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})


def test_controller_error_is_bounded_audit_without_fifth_state(tmp_path: Path) -> None:
    context = _context(tmp_path)
    journal = EvidenceJournal(context.journal_root)
    request = _request()

    for number in range(12):
        with pytest.raises(RuntimeError, match="offline failure"):
            journal.resolve_round(
                context,
                (request,),
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError(f"offline failure {number}")
                ),
                round_number=1,
                operation_policies=_READ_POLICY,
            )
    receipt = journal.read_receipt(context.address("r1", evidence_round=1))
    assert receipt["state"] == "prepared"
    assert len(receipt["errors"]) == 8
    assert all("message" not in item for item in receipt["errors"])
