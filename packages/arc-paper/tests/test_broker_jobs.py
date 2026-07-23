from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

from arc_jobs import JobManager, JobPaths
from arc_jobs.jobs import write_json
from arc_llm.budget import (
    BudgetExhausted,
    SharedBudget,
    current_shared_budget_binding,
    shared_budget_context,
)
from arc_llm import runner as llm_runner
from arc_llm.runner import run_json
from arc_llm.usage import LLMProviderResponse, LLMUsage
from arc_paper import broker_jobs
from arc_paper.execution import current_progress_callback
from arc_paper.broker_jobs import (
    BROKER_JOB_TYPE,
    BrokerJobError,
    BrokerJobManager,
    run_broker_job_worker,
)


def _budget(tmp_path: Path, *, calls: int = 8, tokens: int = 10_000) -> SharedBudget:
    return SharedBudget.create(
        tmp_path / "budget" / "budget.sqlite3",
        budget_id="parent",
        max_calls=calls,
        max_tokens=tokens,
    )


def _manager(tmp_path: Path, monkeypatch) -> BrokerJobManager:
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "jobs-cache"))
    jobs = JobManager(worker_mode="process")
    monkeypatch.setattr(jobs, "_launch_worker", lambda job_id: None)
    return BrokerJobManager(jobs)


def _submit(
    manager: BrokerJobManager,
    budget: SharedBudget,
    *,
    transaction_receipt_sha256: str = "3" * 64,
    source_sha256: str = "4" * 64,
    content_sha256: str = "5" * 64,
):
    return manager.submit(
        operation="get-llm-summary",
        arguments={
            "paper_ids": "2401.00001",
            "provider": "auto",
            "model": None,
            "model_tier": "medium",
            "refresh": False,
        },
        budget=budget,
        output_reserve_tokens=100,
        parent_run_id="run-a",
        policy_sha256="1" * 64,
        runtime_sha256="2" * 64,
        transaction_receipt_sha256=transaction_receipt_sha256,
        network_authorized=True,
        source_sha256=source_sha256,
        content_sha256=content_sha256,
    )


def test_missing_or_exhausted_budget_launches_no_job(tmp_path: Path, monkeypatch) -> None:
    manager = _manager(tmp_path, monkeypatch)
    with pytest.raises(Exception, match="finite parent budget"):
        _submit(manager, None)  # type: ignore[arg-type]
    assert not (tmp_path / "jobs-cache" / "jobs").exists()

    exhausted = _budget(tmp_path, calls=0)
    with pytest.raises(BudgetExhausted):
        _submit(manager, exhausted)
    jobs_root = tmp_path / "jobs-cache" / "jobs"
    assert not jobs_root.exists() or not any(
        path.name.startswith("paper-") for path in jobs_root.iterdir()
    )


def test_invalid_receipt_is_rejected_before_budget_admission(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path, calls=1)

    with pytest.raises(ValueError, match="transaction_receipt_sha256"):
        _submit(
            manager,
            budget,
            transaction_receipt_sha256="not-a-hash",
        )

    assert budget.snapshot().outstanding_calls == 0
    assert budget.snapshot().charged_calls == 0


def test_source_and_content_guards_change_deterministic_job_identity(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path, calls=3)

    baseline = _submit(manager, budget)
    changed_source = _submit(
        manager, budget,
        transaction_receipt_sha256="6" * 64,
        source_sha256="7" * 64,
    )
    changed_content = _submit(
        manager, budget,
        transaction_receipt_sha256="8" * 64,
        content_sha256="9" * 64,
    )

    assert len({
        baseline.job_id, changed_source.job_id, changed_content.job_id,
    }) == 3


def test_submit_is_deterministic_and_public_ticket_has_no_private_values(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path)
    first = _submit(manager, budget)
    second = _submit(manager, budget)

    assert first == second
    assert first.job_id.startswith("paper-")
    assert len(first.job_id) == len("paper-") + 20
    ticket = first.to_json()
    assert set(ticket) == {
        "job_id",
        "identity_sha256",
        "operation_version",
        "budget_identity_sha256",
        "transaction_receipt_sha256",
        "schema_version",
    }
    assert "2401.00001" not in json.dumps(ticket)
    stored = json.loads(
        JobPaths.for_job(first.job_id).job.read_text(encoding="utf-8")
    )
    assert stored["job_type"] == BROKER_JOB_TYPE
    assert stored["environment"]["ARC_PAPER_ACCESS"] == "none"
    assert stored["environment"]["ARC_LLM_INHERIT_HOST_TOOLS"] == "false"
    assert stored["environment"]["ARC_BROKER_JOB_DEPTH"] == "1"


def test_bounded_job_status_and_result_do_not_project_private_values(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    ticket = _submit(manager, _budget(tmp_path))
    paths = JobPaths.for_job(ticket.job_id)
    status = json.loads(paths.status.read_text(encoding="utf-8"))
    status.update({
        "progress": {
            "completed": 1,
            "total": 2,
            "input_path": "/private/input.pdf",
            "request": {"paper_ids": ["2401.00001"]},
        },
        "last_substantive_excerpt": "private prompt text",
        "meta": {"stdout_path": "/private/stdout.log"},
        "error": {
            "code": "bounded_error",
            "message": "failed for /private/input.pdf",
            "stderr_path": "/private/stderr.log",
        },
    })
    write_json(paths.status, status)

    public = manager.jobs.status(ticket.job_id)
    encoded = json.dumps(public)

    assert public["progress"] == {"completed": 1, "total": 2}
    assert public["error"] == {"code": "bounded_error"}
    for private in (
        "2401.00001", "/private", "private prompt text",
        "stdout_path", "stderr_path", "input_path",
    ):
        assert private not in encoded


def test_worker_uses_private_request_budget_and_persists_result_before_exit(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path)
    ticket = _submit(manager, budget)
    observed = {}

    def fake_dispatch(operation, arguments):
        binding = current_shared_budget_binding(required=True)
        observed.update(
            operation=operation,
            arguments=dict(arguments),
            reserve=binding.output_reserve_tokens,
        )
        reservation = binding.reserve(
            checkpoint_identity="provider-checkpoint-a",
            provider_attempt=1,
            prompt_bytes=12,
        )
        reservation.mark_submitted()
        progress = current_progress_callback()
        assert progress is not None
        progress({
            "event": "section_completed",
            "sections_completed": 1,
            "sections_total": 2,
            "paper_id": "private-paper-id",
            "title": "private section title",
        })
        reservation.settle_known(input_tokens=4, output_tokens=3)
        return {"ok": True, "data": {"summary": "bounded"}}

    monkeypatch.setattr(broker_jobs, "dispatch_operation", fake_dispatch)
    assert manager.wait(ticket, timeout=0.01) is None
    monkeypatch.setenv("ARC_JOB_ID", ticket.job_id)
    monkeypatch.setenv("ARC_JOB_TYPE", BROKER_JOB_TYPE)
    receipt = run_broker_job_worker()

    assert receipt["status"] == "done"
    assert observed["operation"] == "get-llm-summary"
    assert observed["reserve"] == 100
    terminal = manager.terminal(ticket)
    assert terminal is not None
    assert terminal.result == {"ok": True, "data": {"summary": "bounded"}}
    assert terminal.result_sha256
    assert budget.snapshot().charged_calls == 1
    assert budget.snapshot().charged_tokens == 7
    assert budget.snapshot().outstanding_calls == 0
    waiters = json.loads(
        (JobPaths.for_job(ticket.job_id).job_dir / "broker-waiters.json")
        .read_text(encoding="utf-8")
    )
    assert waiters["active_waiters"] == {}
    assert waiters["terminal_status"] == "done"
    public = manager.jobs.status(ticket.job_id)
    lifecycle = [item["event"] for item in public["events"]]
    assert "paper_broker_waiter_attached" in lifecycle
    assert "paper_broker_job_waiting" in lifecycle
    assert "paper_broker_job_submitted" in lifecycle
    assert "paper_broker_job_progress" in lifecycle
    assert "paper_broker_job_completed" in lifecycle
    progress_event = next(
        item for item in public["events"]
        if item["event"] == "paper_broker_job_progress"
    )
    assert progress_event["phase"] == "section_completed"
    assert progress_event["operation"] == "get-llm-summary"
    assert progress_event["identity_short"] == ticket.identity_sha256[:12]
    assert progress_event["completed"] == 1
    assert progress_event["total"] == 2
    assert "private-paper-id" not in json.dumps(public)
    assert "private section title" not in json.dumps(public)
    assert manager.cancel(ticket) == {
        "status": "done",
        "job_id": ticket.job_id,
    }
    second_waiter = _submit(
        manager,
        budget,
        transaction_receipt_sha256="6" * 64,
    )
    assert second_waiter.job_id == ticket.job_id
    assert second_waiter.transaction_receipt_sha256 != (
        ticket.transaction_receipt_sha256
    )
    waiters = json.loads(
        (JobPaths.for_job(ticket.job_id).job_dir / "broker-waiters.json")
        .read_text(encoding="utf-8")
    )
    assert waiters["active_waiters"] == {}
    assert waiters["terminal_status"] == "done"
    assert waiters["deduplicated"] is True
    assert budget.snapshot().charged_calls == 1
    assert budget.snapshot().charged_tokens == 7
    assert budget.snapshot().outstanding_calls == 0


def test_worker_cache_hit_releases_first_call_admission_uncharged(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path, calls=1)
    ticket = _submit(manager, budget)

    monkeypatch.setattr(
        broker_jobs,
        "dispatch_operation",
        lambda *_args, **_kwargs: {
            "ok": True,
            "data": {"summary": "cached"},
            "meta": {"cache": "hit"},
        },
    )
    monkeypatch.setenv("ARC_JOB_ID", ticket.job_id)
    monkeypatch.setenv("ARC_JOB_TYPE", BROKER_JOB_TYPE)
    receipt = run_broker_job_worker()

    assert receipt["status"] == "done"
    snapshot = budget.snapshot()
    assert snapshot.charged_calls == 0
    assert snapshot.charged_tokens == 0
    assert snapshot.outstanding_calls == 0


def test_worker_failure_projects_bounded_failed_event(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    ticket = _submit(manager, _budget(tmp_path, calls=1))

    def fail_dispatch(*_args, **_kwargs):
        raise RuntimeError("private failure details /private/provider.log")

    monkeypatch.setattr(broker_jobs, "dispatch_operation", fail_dispatch)
    monkeypatch.setenv("ARC_JOB_ID", ticket.job_id)
    monkeypatch.setenv("ARC_JOB_TYPE", BROKER_JOB_TYPE)
    receipt = run_broker_job_worker()

    assert receipt["status"] == "failed"
    public = manager.jobs.status(ticket.job_id)
    failed = next(
        item for item in public["events"]
        if item["event"] == "paper_broker_job_failed"
    )
    assert failed["status"] == "failed"
    assert failed["operation"] == "get-llm-summary"
    assert failed["identity_short"] == ticket.identity_sha256[:12]
    assert len(failed["error_sha256"]) == 64
    assert "/private/provider.log" not in json.dumps(public)


def test_worker_requires_complete_internal_context_and_rejects_nested_submit(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.delenv("ARC_JOB_ID", raising=False)
    monkeypatch.delenv("ARC_JOB_TYPE", raising=False)
    with pytest.raises(BrokerJobError) as authority:
        run_broker_job_worker()
    assert authority.value.code == "paper_broker_job_authority_invalid"

    manager = _manager(tmp_path, monkeypatch)
    monkeypatch.setenv("ARC_BROKER_JOB_DEPTH", "1")
    with pytest.raises(BrokerJobError) as nested:
        _submit(manager, _budget(tmp_path))
    assert nested.value.code == "paper_broker_nested_job_forbidden"


@pytest.mark.parametrize(
    "corrupt",
    ["arguments", "environment", "authorization", "request_identity"],
)
def test_worker_rejects_corrupt_private_job_record(
    tmp_path: Path, monkeypatch, corrupt: str,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    ticket = _submit(manager, _budget(tmp_path))
    paths = JobPaths.for_job(ticket.job_id)
    job = json.loads(paths.job.read_text(encoding="utf-8"))
    if corrupt == "arguments":
        job["payload"]["arguments"]["paper_ids"] = "9999.99999"
    elif corrupt == "environment":
        job["environment"]["ARC_PAPER_ACCESS"] = "read-write"
    elif corrupt == "authorization":
        job["payload"]["authorization"]["network_authorized"] = False
    else:
        job["request_identity_sha256"] = "0" * 64
    write_json(paths.job, job)
    monkeypatch.setenv("ARC_JOB_ID", ticket.job_id)
    monkeypatch.setenv("ARC_JOB_TYPE", BROKER_JOB_TYPE)

    with pytest.raises(BrokerJobError) as failure:
        run_broker_job_worker()

    assert failure.value.code in {
        "paper_broker_job_record_corrupt",
        "paper_broker_job_record_mismatch",
    }


def test_inline_and_admin_operations_are_not_routed(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path)
    kwargs = dict(
        budget=budget,
        output_reserve_tokens=10,
        parent_run_id="run-a",
        policy_sha256="1" * 64,
        runtime_sha256="2" * 64,
        transaction_receipt_sha256="3" * 64,
        network_authorized=True,
    )
    with pytest.raises(BrokerJobError) as inline:
        manager.submit(
            operation="get-title",
            arguments={"paper_ids": "2401.00001", "refresh": False},
            **kwargs,
        )
    assert inline.value.code == "paper_job_route_not_required"


def test_managed_batch_export_resolves_opaque_controller_handle(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path)
    root = tmp_path / "broker-root"
    (root / "handles").mkdir(parents=True)
    (root / "inputs").mkdir()
    output = root / "inputs" / "output-owned.bin"
    output.write_bytes(b"")
    handle_id = "w-" + "a" * 32
    handle_record = {
        "schema_version": "arc.companion.paper-broker-handle.v1",
        "handle_id": handle_id,
        "sha256": broker_jobs._canonical_hash({}),
        "size_bytes": 0,
        "media_type": "application/jsonl",
        "output_name": output.name,
        "run_id": "run-a",
        "access": "full",
        "policy_sha256": "1" * 64,
        "contexts": [["summary-batch.export", "output", "write"]],
    }
    write_json(root / "handles" / f"{handle_id}.json", handle_record)
    ticket = manager.submit(
        operation="summary-batch.export",
        arguments={
            "name": "batch-a",
            "output": {"handle_id": handle_id},
            "format": "jsonl",
        },
        budget=budget,
        output_reserve_tokens=10,
        parent_run_id="run-a",
        policy_sha256="1" * 64,
        runtime_sha256="2" * 64,
        transaction_receipt_sha256="3" * 64,
        network_authorized=True,
        artifact_root=root,
        artifact_authorizations={
            "output": {
                "handle_id": handle_id,
                "operation": "summary-batch.export",
                "parameter": "output",
                "access": "write",
                "handle_receipt_sha256": broker_jobs._canonical_hash(
                    handle_record
                ),
            },
        },
    )

    def fake_dispatch(operation, arguments, *, artifact_resolver):
        resolved = artifact_resolver(
            arguments["output"]["handle_id"],
            access="write",
            operation=operation,
            parameter="output",
        )
        assert resolved == output
        resolved.write_text('{"paper_id":"0911.3380"}\n', encoding="utf-8")
        return {"ok": True, "data": {"output": str(resolved)}}

    monkeypatch.setattr(broker_jobs, "dispatch_operation", fake_dispatch)
    monkeypatch.setenv("ARC_JOB_ID", ticket.job_id)
    monkeypatch.setenv("ARC_JOB_TYPE", BROKER_JOB_TYPE)
    receipt = run_broker_job_worker()

    assert receipt["status"] == "done"
    assert output.read_text(encoding="utf-8").startswith('{"paper_id"')
    public = manager.jobs.status(ticket.job_id)
    assert handle_id not in json.dumps(public)
    assert str(root) not in json.dumps(public)


def test_equivalent_requests_share_job_but_cancel_waiters_independently(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path)
    first = _submit(manager, budget, transaction_receipt_sha256="3" * 64)
    second = _submit(manager, budget, transaction_receipt_sha256="6" * 64)
    assert first.job_id == second.job_id
    assert first.transaction_receipt_sha256 != second.transaction_receipt_sha256

    first_cancel = manager.cancel(first)
    assert first_cancel["status"] == "waiter_cancelled"
    assert first_cancel["active_waiters"] == 1
    assert manager.jobs.status(first.job_id)["status"] != "cancel_requested"

    second_cancel = manager.cancel(second)
    assert second_cancel["status"] in {"cancel_requested", "cancelled"}


def test_last_waiter_cancel_and_new_attach_are_linearized(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path)
    first = _submit(manager, budget)
    barrier = threading.Barrier(2)

    def cancel_first():
        barrier.wait()
        return manager.cancel(first)

    def attach_second():
        barrier.wait()
        return _submit(
            manager,
            budget,
            transaction_receipt_sha256="6" * 64,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        cancelled = pool.submit(cancel_first)
        attached = pool.submit(attach_second)
        cancel_result = cancelled.result()
        second = attached.result()

    assert second.job_id == first.job_id
    paths = JobPaths.for_job(first.job_id)
    status = json.loads(paths.status.read_text(encoding="utf-8"))["status"]
    waiters = json.loads(
        (paths.job_dir / "broker-waiters.json").read_text(encoding="utf-8")
    )
    if status in {"cancel_requested", "cancelled"}:
        assert cancel_result["status"] in {"cancel_requested", "cancelled"}
        assert waiters["active_waiters"] == {}
    else:
        assert cancel_result["status"] == "waiter_cancelled"
        assert len(waiters["active_waiters"]) == 1


def test_last_waiter_cancel_charges_submitted_reservation(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path, calls=1)
    ticket = _submit(manager, budget)
    paths = JobPaths.for_job(ticket.job_id)
    job = json.loads(paths.job.read_text(encoding="utf-8"))
    admission_id = job["payload"]["admission_reservation_id"]
    with shared_budget_context(
        budget,
        output_reserve_tokens=100,
        admission_reservation_id=admission_id,
    ):
        reservation = current_shared_budget_binding(required=True).reserve(
            checkpoint_identity="submitted-before-cancel",
            provider_attempt=1,
            prompt_bytes=12,
        )
        reservation.mark_submitted()

    assert manager.cancel(ticket)["status"] in {
        "cancel_requested", "cancelled",
    }
    status = json.loads(paths.status.read_text(encoding="utf-8"))
    status.update({"status": "cancelled", "phase": "cancelled"})
    write_json(paths.status, status)
    terminal = manager.terminal(ticket)

    assert terminal is not None
    assert terminal.status == "cancelled"
    assert budget.snapshot().charged_calls == 1
    assert budget.snapshot().outstanding_calls == 0
    public = manager.jobs.status(ticket.job_id)
    assert any(
        item["event"] == "paper_broker_job_cancelled"
        and item["status"] == "cancelled"
        for item in public["events"]
    )


def test_two_workers_concurrently_share_one_job_with_distinct_waiters(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path, calls=1)
    receipts = ("3" * 64, "6" * 64)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tickets = list(pool.map(
            lambda receipt: _submit(
                manager,
                budget,
                transaction_receipt_sha256=receipt,
            ),
            receipts,
        ))

    assert tickets[0].job_id == tickets[1].job_id
    assert {
        ticket.transaction_receipt_sha256 for ticket in tickets
    } == set(receipts)
    waiters = json.loads(
        (
            JobPaths.for_job(tickets[0].job_id).job_dir
            / "broker-waiters.json"
        ).read_text(encoding="utf-8")
    )
    assert waiters["schema_version"] == "arc.paper.broker-job-waiters.v2"
    assert list(waiters["active_waiters"].values()).count("active") == 2
    assert waiters["deduplicated"] is True
    assert budget.snapshot().outstanding_calls == 1


def test_worker_loss_retries_only_proven_unsubmitted_then_supervises(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path, calls=1)
    ticket = _submit(manager, budget)
    paths = JobPaths.for_job(ticket.job_id)
    status = json.loads(paths.status.read_text(encoding="utf-8"))
    status.update({
        "status": "failed",
        "phase": "failed",
        "error": {"code": "job_worker_lost"},
    })
    write_json(paths.status, status)

    assert manager.terminal(ticket) is None
    retried = json.loads(paths.status.read_text(encoding="utf-8"))
    assert retried["status"] == "queued"
    assert retried["recovery_attempt"] == 1
    assert budget.snapshot().outstanding_calls == 1

    retried.update({
        "status": "failed",
        "phase": "failed",
        "error": {"code": "job_worker_lost"},
    })
    write_json(paths.status, retried)
    terminal = manager.terminal(ticket)

    assert terminal is not None
    assert terminal.status == "needs_supervision"
    assert terminal.error["submission_state"] == "not_submitted"
    assert budget.snapshot().outstanding_calls == 0


@pytest.mark.parametrize(
    "error_code",
    [
        "job_environment_invalid",
        "job_worker_launch_failed",
        "job_command_failed",
        "job_command_reported_failure",
        "job_failed",
    ],
)
def test_real_command_failure_codes_enter_verified_reconciliation(
    tmp_path: Path, monkeypatch, error_code: str,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path, calls=1)
    ticket = _submit(manager, budget)
    paths = JobPaths.for_job(ticket.job_id)
    status = json.loads(paths.status.read_text(encoding="utf-8"))
    status.update({
        "status": "failed",
        "phase": "failed",
        "error": {"code": error_code},
    })
    write_json(paths.status, status)

    assert manager.terminal(ticket) is None
    recovered = json.loads(paths.status.read_text(encoding="utf-8"))
    assert recovered["status"] == "queued"
    assert recovered["recovery_attempt"] == 1
    assert budget.snapshot().outstanding_calls == 1
    recovered.update({
        "status": "failed",
        "phase": "failed",
        "error": {"code": error_code},
    })
    write_json(paths.status, recovered)

    terminal = manager.terminal(ticket)

    assert terminal is not None
    assert terminal.status == "needs_supervision"
    assert budget.snapshot().outstanding_calls == 0


def test_invalid_persisted_environment_reconciles_only_retired_timeout_mismatch(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path, calls=1)
    ticket = _submit(manager, budget)
    paths = JobPaths.for_job(ticket.job_id)
    job = json.loads(paths.job.read_text(encoding="utf-8"))
    job["environment"]["ARC_LLM_TIMEOUT_SECONDS"] = "1"
    write_json(paths.job, job)

    JobManager._launch_worker(manager.jobs, ticket.job_id)

    failed = json.loads(paths.status.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "job_environment_invalid"
    assert manager.terminal(ticket) is None
    retried = json.loads(paths.status.read_text(encoding="utf-8"))
    assert retried["status"] == "queued"
    assert retried["recovery_attempt"] == 1
    assert budget.snapshot().outstanding_calls == 1

    JobManager._launch_worker(manager.jobs, ticket.job_id)
    terminal = manager.terminal(ticket)

    assert terminal is not None
    assert terminal.status == "needs_supervision"
    assert terminal.error["submission_state"] == "not_submitted"
    assert budget.snapshot().outstanding_calls == 0


def test_invalid_environment_reconciliation_rejects_other_record_mismatch(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path, calls=1)
    ticket = _submit(manager, budget)
    paths = JobPaths.for_job(ticket.job_id)
    job = json.loads(paths.job.read_text(encoding="utf-8"))
    job["environment"].update({
        "ARC_LLM_TIMEOUT_SECONDS": "1",
        "ARC_PAPER_ACCESS": "read-write",
    })
    write_json(paths.job, job)

    JobManager._launch_worker(manager.jobs, ticket.job_id)

    with pytest.raises(BrokerJobError) as failure:
        manager.terminal(ticket)
    assert failure.value.code == "paper_broker_job_record_corrupt"
    assert budget.snapshot().outstanding_calls == 1


def test_worker_loss_after_submission_charges_and_requires_supervision(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path, calls=1)
    ticket = _submit(manager, budget)
    job = json.loads(
        JobPaths.for_job(ticket.job_id).job.read_text(encoding="utf-8")
    )
    admission_id = job["payload"]["admission_reservation_id"]
    with shared_budget_context(
        budget,
        output_reserve_tokens=100,
        admission_reservation_id=admission_id,
    ):
        reservation = current_shared_budget_binding(required=True).reserve(
            checkpoint_identity="provider-checkpoint-a",
            provider_attempt=1,
            prompt_bytes=12,
        )
        reservation.mark_submitted()
    paths = JobPaths.for_job(ticket.job_id)
    status = json.loads(paths.status.read_text(encoding="utf-8"))
    status.update({
        "status": "failed",
        "phase": "failed",
        "error": {"code": "job_worker_lost"},
    })
    write_json(paths.status, status)

    terminal = manager.terminal(ticket)

    assert terminal is not None
    assert terminal.status == "needs_supervision"
    assert terminal.error["submission_state"] == "submitted"
    assert budget.snapshot().charged_calls == 1
    assert budget.snapshot().outstanding_calls == 0
    public = manager.jobs.status(ticket.job_id)
    event = next(
        item for item in public["events"]
        if item["event"] == "paper_broker_job_needs_supervision"
    )
    assert event["status"] == "needs_supervision"
    assert event["operation"] == "get-llm-summary"
    assert event["identity_short"] == ticket.identity_sha256[:12]
    assert event["budget_remaining_calls"] == 0
    assert event["budget_remaining_tokens"] == 9_897


def test_worker_loss_retries_when_completed_child_checkpoint_can_replay(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path, calls=1)
    ticket = _submit(manager, budget)
    paths = JobPaths.for_job(ticket.job_id)
    job = json.loads(paths.job.read_text(encoding="utf-8"))
    admission_id = job["payload"]["admission_reservation_id"]

    class Provider:
        name = "codex-cli"

        def generate_json_result(self, prompt, **kwargs):
            return LLMProviderResponse(
                {"ok": True},
                usage=LLMUsage(input_tokens=4, output_tokens=3),
            )

    monkeypatch.setattr(
        llm_runner, "select_provider", lambda *args, **kwargs: Provider(),
    )
    with shared_budget_context(
        budget,
        output_reserve_tokens=100,
        admission_reservation_id=admission_id,
    ):
        result = run_json(
            "prompt",
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            provider="codex-cli",
            env={},
            process_chain=[],
            artifact_dir=tmp_path / "child-checkpoint",
            call_label="summary",
            idempotency_key="summary-stable",
        )
    assert result["ok"] is True
    before = budget.snapshot()
    status = json.loads(paths.status.read_text(encoding="utf-8"))
    status.update({
        "status": "failed",
        "phase": "failed",
        "error": {"code": "job_worker_lost"},
    })
    write_json(paths.status, status)

    assert manager.terminal(ticket) is None
    retried = json.loads(paths.status.read_text(encoding="utf-8"))
    assert retried["status"] == "queued"
    assert retried["recovery_receipt"]["budget_disposition"] == (
        "completed_child_checkpoints_replayable"
    )
    assert budget.snapshot() == before


def test_worker_failure_reconciles_response_checkpoint_before_retry(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    budget = _budget(tmp_path, calls=1)
    ticket = _submit(manager, budget)
    paths = JobPaths.for_job(ticket.job_id)
    job = json.loads(paths.job.read_text(encoding="utf-8"))
    admission_id = job["payload"]["admission_reservation_id"]

    class Provider:
        name = "codex-cli"

        def generate_json_result(self, prompt, **kwargs):
            del prompt, kwargs
            return LLMProviderResponse(
                {"ok": True},
                usage=LLMUsage(input_tokens=4, output_tokens=3),
            )

    monkeypatch.setattr(
        llm_runner, "select_provider", lambda *args, **kwargs: Provider(),
    )
    real_settle = llm_runner._settle_descendant_success
    monkeypatch.setattr(
        llm_runner,
        "_settle_descendant_success",
        lambda *args, **kwargs: (_ for _ in ()).throw(SystemExit(74)),
    )
    with shared_budget_context(
        budget,
        output_reserve_tokens=100,
        admission_reservation_id=admission_id,
    ):
        with pytest.raises(SystemExit):
            run_json(
                "prompt",
                schema={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
                provider="codex-cli",
                env={},
                process_chain=[],
                artifact_dir=tmp_path / "child-response-checkpoint",
                call_label="summary",
                idempotency_key="summary-response-crash",
            )
    assert budget.snapshot().outstanding_calls == 1
    monkeypatch.setattr(
        llm_runner, "_settle_descendant_success", real_settle,
    )
    status = json.loads(paths.status.read_text(encoding="utf-8"))
    status.update({
        "status": "failed",
        "phase": "failed",
        "error": {"code": "job_command_failed"},
    })
    write_json(paths.status, status)

    assert manager.terminal(ticket) is None
    snapshot = budget.snapshot()
    assert snapshot.charged_calls == 1
    assert snapshot.charged_tokens == 7
    assert snapshot.outstanding_calls == 0
    assert json.loads(paths.status.read_text(encoding="utf-8"))["status"] == "queued"


def test_malformed_existing_broker_result_fails_stably(
    tmp_path: Path, monkeypatch,
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    ticket = _submit(manager, _budget(tmp_path))
    write_json(
        JobPaths.for_job(ticket.job_id).job_dir / "broker-result.json",
        ["not", "an", "object"],
    )

    with pytest.raises(BrokerJobError) as first:
        manager.terminal(ticket)
    with pytest.raises(BrokerJobError) as second:
        manager.terminal(ticket)

    assert first.value.code == "paper_broker_job_result_corrupt"
    assert second.value.code == first.value.code
