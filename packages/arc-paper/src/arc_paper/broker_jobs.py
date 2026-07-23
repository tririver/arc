"""Durable Controller jobs for catalog operations that use LLMs or jobs.

This module composes :mod:`arc_jobs`; job status, cancellation, process
ownership, and reconciliation remain owned by that package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from arc_jobs import (
    JobManager,
    JobPaths,
    append_event,
    is_cancel_requested,
    read_job,
    read_json,
    record_progress,
    submission_lock,
    write_json,
)
from arc_jobs.jobs import restored_environment
from arc_llm.budget import (
    BudgetExhausted,
    BudgetReference,
    BudgetRequired,
    SharedBudget,
    shared_budget_context,
)
from arc_llm.call_checkpoint import checkpoint_budget_state

from .capabilities import (
    CATALOG_SCHEMA_VERSION,
    catalog_document,
    dispatch_operation,
    get_operation_spec,
    validate_operation_parameters,
)
from .execution import managed_execution_scope


BROKER_JOB_SCHEMA_VERSION = "arc.paper.broker-job.v1"
BROKER_JOB_RESULT_SCHEMA_VERSION = "arc.paper.broker-job-result.v1"
BROKER_JOB_TICKET_SCHEMA_VERSION = "arc.paper.broker-job-ticket.v1"
BROKER_JOB_TYPE = "paper_broker_operation"
_DEPTH_ENV = "ARC_BROKER_JOB_DEPTH"
_CACHE_ENV_KEYS = frozenset({
    "ARC_PAPER_CACHE",
    "ARC_PAPER_WORKER_BASE_CACHE",
    "ARC_PAPER_WORKER_TOMBSTONE_DIR",
    "ARC_PAPER_WORKER_SESSION_ID",
    "ARC_PAPER_WORKER_SESSION_DIR",
})
_REMOVED_LLM_TIMEOUT_ENV_KEYS = frozenset({
    "ARC_LLM_TIMEOUT_SECONDS",
    "ARC_CODEX_TIMEOUT_SECONDS",
    "ARC_CLAUDE_TIMEOUT_SECONDS",
    "ARC_KIMI_TIMEOUT_SECONDS",
})


class BrokerJobError(RuntimeError):
    submission_state = "not_submitted"
    abort_batch = True

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class BrokerJobIdentity:
    operation_id: str
    operation_version: int
    catalog_schema_version: str
    catalog_sha256: str
    arguments_sha256: str
    policy_sha256: str
    runtime_sha256: str
    parent_run_sha256: str
    budget_identity_sha256: str
    source_sha256: str | None
    content_sha256: str | None
    artifact_authorizations_sha256: str
    artifact_root_sha256: str | None
    refresh: bool
    route: str = "controller_managed_job"
    schema_version: str = BROKER_JOB_SCHEMA_VERSION

    @property
    def sha256(self) -> str:
        return _canonical_hash(asdict(self))

    @property
    def job_id(self) -> str:
        return f"paper-{self.sha256[:20]}"


@dataclass(frozen=True)
class BrokerJobTicket:
    job_id: str
    identity_sha256: str
    operation_version: int
    budget_identity_sha256: str
    transaction_receipt_sha256: str
    schema_version: str = BROKER_JOB_TICKET_SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "BrokerJobTicket":
        expected = {
            "job_id",
            "identity_sha256",
            "operation_version",
            "budget_identity_sha256",
            "transaction_receipt_sha256",
            "schema_version",
        }
        if set(value) != expected:
            raise BrokerJobError(
                "paper_broker_job_ticket_corrupt",
                "Managed job ticket has an invalid shape.",
            )
        ticket = cls(
            job_id=str(value.get("job_id") or ""),
            identity_sha256=_required_hash(
                str(value.get("identity_sha256") or ""), "identity_sha256"
            ),
            operation_version=int(value.get("operation_version") or 0),
            budget_identity_sha256=_required_hash(
                str(value.get("budget_identity_sha256") or ""),
                "budget_identity_sha256",
            ),
            transaction_receipt_sha256=_required_hash(
                str(value.get("transaction_receipt_sha256") or ""),
                "transaction_receipt_sha256",
            ),
            schema_version=str(value.get("schema_version") or ""),
        )
        if (
            ticket.schema_version != BROKER_JOB_TICKET_SCHEMA_VERSION
            or ticket.job_id != f"paper-{ticket.identity_sha256[:20]}"
            or ticket.operation_version < 1
        ):
            raise BrokerJobError(
                "paper_broker_job_ticket_corrupt",
                "Managed job ticket identity is invalid.",
            )
        return ticket


@dataclass(frozen=True)
class BrokerJobTerminal:
    job_id: str
    status: str
    result: Mapping[str, Any] | None
    error: Mapping[str, Any] | None
    result_sha256: str | None
    receipt: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class BrokerJobExecutionContext:
    budget: BudgetReference
    output_reserve_tokens: int


class BrokerJobManager:
    """Submit, attach to, wait for, and cancel deterministic paper jobs."""

    def __init__(self, manager: JobManager | None = None) -> None:
        self.jobs = manager or JobManager(worker_mode="process")

    def submit(
        self,
        *,
        operation: str,
        arguments: Mapping[str, Any],
        budget: SharedBudget | None,
        output_reserve_tokens: int,
        parent_run_id: str,
        policy_sha256: str,
        runtime_sha256: str,
        transaction_receipt_sha256: str,
        network_authorized: bool,
        source_sha256: str | None = None,
        content_sha256: str | None = None,
        refresh: bool = False,
        cache_environment: Mapping[str, str] | None = None,
        artifact_root: Path | None = None,
        artifact_authorizations: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> BrokerJobTicket:
        if int(os.environ.get(_DEPTH_ENV, "0") or 0) >= 1:
            raise BrokerJobError(
                "paper_broker_nested_job_forbidden",
                "Managed paper jobs cannot recursively launch managed paper jobs.",
            )
        spec = get_operation_spec(operation)
        if spec is None:
            raise BrokerJobError("paper_operation_unknown", "Unknown ARC-paper operation.")
        if spec.admin or spec.destructive:
            raise BrokerJobError(
                "paper_operation_forbidden",
                "Administrative or destructive operations cannot use this route.",
            )
        if not (spec.uses_llm or spec.is_job or spec.recovery_class == "managed_job"):
            raise BrokerJobError(
                "paper_job_route_not_required",
                "Inline ARC-paper operations must not use the managed job route.",
            )
        errors = validate_operation_parameters(spec.name, arguments)
        if errors:
            raise BrokerJobError(
                "paper_operation_parameters_invalid",
                "Invalid ARC-paper operation parameters: " + "; ".join(errors),
            )
        if spec.network_access == "may" and not network_authorized:
            raise BrokerJobError(
                "paper_network_forbidden",
                "ARC-paper network access is not authorized for this job.",
            )
        if budget is None:
            raise BudgetRequired(
                "managed child ARC-paper operation requires a finite parent budget"
            )
        if type(output_reserve_tokens) is not int or output_reserve_tokens < 0:
            raise ValueError("output_reserve_tokens must be finite and non-negative")
        canonical_arguments = json.loads(_canonical_json(dict(arguments)))
        canonical_artifact_authorizations = json.loads(
            _canonical_json(dict(artifact_authorizations or {}))
        )
        canonical_artifact_root = (
            artifact_root.expanduser().resolve(strict=False)
            if artifact_root is not None else None
        )
        identity = BrokerJobIdentity(
            operation_id=spec.operation_id,
            operation_version=spec.version,
            catalog_schema_version=CATALOG_SCHEMA_VERSION,
            catalog_sha256=_canonical_hash(catalog_document()),
            arguments_sha256=_canonical_hash(canonical_arguments),
            policy_sha256=_required_hash(policy_sha256, "policy_sha256"),
            runtime_sha256=_required_hash(runtime_sha256, "runtime_sha256"),
            parent_run_sha256=_canonical_hash({"parent_run_id": parent_run_id}),
            budget_identity_sha256=budget.reference.identity_sha256,
            source_sha256=(
                _required_hash(source_sha256, "source_sha256")
                if source_sha256 is not None else None
            ),
            content_sha256=(
                _required_hash(content_sha256, "content_sha256")
                if content_sha256 is not None else None
            ),
            artifact_authorizations_sha256=_canonical_hash(
                canonical_artifact_authorizations
            ),
            artifact_root_sha256=(
                _canonical_hash({"artifact_root": str(canonical_artifact_root)})
                if canonical_artifact_root is not None else None
            ),
            refresh=bool(refresh),
        )
        full_identity = {**asdict(identity), "identity_sha256": identity.sha256}
        # Validate every caller-controlled value before finite budget admission.
        # After this point the only fallible external mutation is jobs.start(),
        # whose exception path releases the reservation below.
        transaction_receipt_sha256 = _required_hash(
            transaction_receipt_sha256, "transaction_receipt_sha256"
        )
        cache_values = dict(cache_environment or {})
        unknown_cache_keys = sorted(set(cache_values) - _CACHE_ENV_KEYS)
        if unknown_cache_keys:
            raise ValueError(
                "cache_environment contains unsupported keys: "
                + ", ".join(unknown_cache_keys)
            )
        admission = (
            budget.reserve(
                checkpoint_identity=f"broker-admission:{identity.sha256}",
                provider_attempt=1,
                prompt_bytes=0,
                output_reserve_tokens=output_reserve_tokens,
            )
            if spec.uses_llm else None
        )
        payload = {
            "broker_payload_schema_version": BROKER_JOB_SCHEMA_VERSION,
            "identity": full_identity,
            "operation": spec.name,
            "arguments": canonical_arguments,
            "arguments_sha256": identity.arguments_sha256,
            "budget": budget.reference.to_json(),
            "output_reserve_tokens": output_reserve_tokens,
            "admission_reservation_id": (
                admission.reservation_id if admission is not None else None
            ),
            "authorization": {
                "operation_id": spec.operation_id,
                "policy_sha256": identity.policy_sha256,
                "network_declared": spec.network_access,
                "network_authorized": bool(network_authorized),
                "cache_declared": spec.cache_access,
                "source_sha256": identity.source_sha256,
                "content_sha256": identity.content_sha256,
                "artifacts": canonical_artifact_authorizations,
            },
        }
        payload["authorization"]["receipt_sha256"] = _canonical_hash(
            payload["authorization"]
        )
        environment = {
            **cache_values,
            "ARC_PAPER_ACCESS": "none",
            "ARC_PAPER_CLI_ACCESS": "none",
            "ARC_INTERNAL_PAPER_ACCESS_LEGACY_MIRROR": "true",
            "ARC_LLM_INHERIT_HOST_TOOLS": "false",
            _DEPTH_ENV: "1",
        }
        if canonical_artifact_root is not None:
            environment["ARC_PAPER_BROKER_ROOT"] = str(canonical_artifact_root)
        existing_job = JobPaths.for_job(identity.job_id).job_dir.exists()
        try:
            self.jobs.start(
                job_type=BROKER_JOB_TYPE,
                payload=payload,
                argv=["arc-paper", "_broker-job-worker", "--json"],
                environment=environment,
                job_id=identity.job_id,
                full_identity=full_identity,
                public_payload={
                    "operation": spec.name,
                    "operation_version": spec.version,
                    "identity_sha256": identity.sha256,
                    "identity_short": identity.sha256[:12],
                    "budget_identity_sha256": (
                        budget.reference.identity_sha256
                    ),
                    "deduplicated": False,
                    "depth": 1,
                    "paper_access": "none",
                    "budget": {
                        "max_calls": budget.snapshot().max_calls,
                        "max_tokens": budget.snapshot().max_tokens,
                    },
                },
                public_projection="bounded",
            )
        except Exception:
            if admission is not None:
                row = budget.reservation_or_none(admission.reservation_id)
                if row is not None and row.get("state") == "reserved":
                    admission.release_not_submitted()
            raise
        waiter_terminal = _attach_waiter(
            identity.job_id,
            identity.sha256,
            transaction_receipt_sha256,
        )
        if not waiter_terminal:
            append_event(
                identity.job_id,
                {
                    "event": "paper_broker_waiter_attached",
                    "operation": spec.name,
                    "identity_short": identity.sha256[:12],
                    "status": "deduplicated" if existing_job else "created",
                },
            )
        return BrokerJobTicket(
            job_id=identity.job_id,
            identity_sha256=identity.sha256,
            operation_version=spec.version,
            budget_identity_sha256=budget.reference.identity_sha256,
            transaction_receipt_sha256=transaction_receipt_sha256,
        )

    def terminal(self, ticket: BrokerJobTicket) -> BrokerJobTerminal | None:
        persisted = _read_broker_result(ticket.job_id, ticket.identity_sha256)
        if persisted is not None:
            raw_status = read_json(JobPaths.for_job(ticket.job_id).status, {})
            return BrokerJobTerminal(
                ticket.job_id,
                str(persisted["status"]),
                persisted.get("result") if isinstance(persisted.get("result"), Mapping) else None,
                persisted.get("error") if isinstance(persisted.get("error"), Mapping) else None,
                str(persisted.get("result_sha256") or "") or None,
                _terminal_receipt(ticket, persisted, raw_status),
            )
        status = self.jobs.status(ticket.job_id)
        status_error = status.get("error")
        raw_status = read_json(JobPaths.for_job(ticket.job_id).status, {})
        if status.get("status") == "cancelled":
            recovery = _reconcile_lost_worker(ticket)
            if recovery["action"] == "retry":
                _release_lost_worker_reservation(recovery)
            append_event(
                ticket.job_id,
                {
                    "event": "paper_broker_job_cancelled",
                    "status": "cancelled",
                    "identity_short": ticket.identity_sha256[:12],
                },
            )
        if (
            status.get("status") == "failed"
            and isinstance(status_error, Mapping)
            and status_error.get("code") in {
                "job_worker_lost",
                "job_worker_unavailable",
                "job_environment_invalid",
                "job_worker_launch_failed",
                "job_command_failed",
                "job_command_reported_failure",
                "job_failed",
            }
        ):
            recovery = _reconcile_lost_worker(
                ticket,
                status_error_code=str(status_error.get("code") or ""),
                status=raw_status,
            )
            if (
                recovery["action"] == "retry"
                and int(status.get("recovery_attempt") or 0) < 1
            ):
                self.jobs.retry_verified_not_submitted(
                    ticket.job_id,
                    recovery_receipt={
                        "recovery_reason": "provider_not_submitted",
                        "identity_sha256": ticket.identity_sha256,
                        "budget_disposition": recovery["disposition"],
                    },
                )
                return None
            if recovery["action"] == "retry":
                _release_lost_worker_reservation(recovery)
            snapshot = SharedBudget.open(
                read_job(ticket.job_id)["payload"]["budget"],
            ).snapshot()
            append_event(
                ticket.job_id,
                {
                    "event": "paper_broker_job_needs_supervision",
                    "status": "needs_supervision",
                    "operation": read_job(ticket.job_id)["payload"]["operation"],
                    "identity_short": ticket.identity_sha256[:12],
                    "budget_remaining_calls": snapshot.remaining_calls,
                    "budget_remaining_tokens": snapshot.remaining_tokens,
                    "recovery_attempt": int(status.get("recovery_attempt") or 0),
                },
            )
            return BrokerJobTerminal(
                ticket.job_id,
                "needs_supervision",
                None,
                {
                    "code": "paper_broker_job_needs_supervision",
                    "category": "managed_job",
                    "submission_state": recovery["submission_state"],
                    "recovery_action": "operator-supervision",
                    "budget_disposition": recovery["disposition"],
                },
                None,
                _terminal_receipt(ticket, None, raw_status),
            )
        if status.get("status") not in {
            "done", "completed", "degraded", "failed", "cancelled",
            "needs_supervision",
        }:
            return None
        error = status.get("error")
        return BrokerJobTerminal(
            ticket.job_id,
            str(status.get("status")),
            None,
            dict(error) if isinstance(error, Mapping) else None,
            None,
            _terminal_receipt(ticket, None, status),
        )

    def wait(
        self, ticket: BrokerJobTicket, *, timeout: float,
    ) -> BrokerJobTerminal | None:
        append_event(
            ticket.job_id,
            {
                "event": "paper_broker_job_waiting",
                "status": "waiting",
                "identity_short": ticket.identity_sha256[:12],
            },
        )
        self.jobs.wait(ticket.job_id, timeout=timeout)
        return self.terminal(ticket)

    def cancel(self, ticket: BrokerJobTicket) -> Mapping[str, Any]:
        terminal = self.terminal(ticket)
        if terminal is not None:
            return {"status": terminal.status, "job_id": ticket.job_id}
        decision: dict[str, int | None] = {"active": None}

        def last_waiter() -> bool:
            decision["active"] = _cancel_waiter_locked(ticket)
            return decision["active"] == 0

        status = self.jobs.cancel(ticket.job_id, condition=last_waiter)
        active = decision["active"]
        if active is None:
            terminal = self.terminal(ticket)
            if terminal is not None:
                return {"status": terminal.status, "job_id": ticket.job_id}
            return status
        if active > 0:
            append_event(
                ticket.job_id,
                {
                    "event": "paper_broker_waiter_cancelled",
                    "status": "cancelled",
                    "identity_short": ticket.identity_sha256[:12],
                },
            )
            return {
                "status": "waiter_cancelled",
                "job_id": ticket.job_id,
                "active_waiters": active,
            }
        return status


def run_broker_job_worker() -> dict[str, Any]:
    """Run the private request selected solely by trusted arc-jobs env."""

    job_id = os.environ.get("ARC_JOB_ID", "")
    job_type = os.environ.get("ARC_JOB_TYPE", "")
    if not job_id or job_type != BROKER_JOB_TYPE:
        raise BrokerJobError(
            "paper_broker_job_authority_invalid",
            "Broker worker requires complete internal arc-jobs context.",
        )
    job = read_job(job_id)
    _verify_worker_job_record(job_id, job)
    payload = job.get("payload")
    if not isinstance(payload, Mapping):
        raise BrokerJobError(
            "paper_broker_job_corrupt", "Broker job payload is missing.",
        )
    identity = payload.get("identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("identity_sha256")
        != _canonical_hash({
            key: value for key, value in identity.items()
            if key != "identity_sha256"
        })
        or job_id != f"paper-{str(identity.get('identity_sha256'))[:20]}"
        or job.get("full_identity") != identity
    ):
        raise BrokerJobError(
            "paper_broker_job_identity_mismatch",
            "Broker job identity does not match its deterministic address.",
        )
    operation = str(payload.get("operation") or "")
    spec = get_operation_spec(operation)
    if (
        spec is None
        or spec.operation_id != identity.get("operation_id")
        or spec.version != identity.get("operation_version")
        or spec.admin
        or spec.destructive
        or not (spec.uses_llm or spec.is_job or spec.recovery_class == "managed_job")
    ):
        raise BrokerJobError(
            "paper_broker_job_authorization_mismatch",
            "Persisted Broker operation no longer matches the managed-job route.",
        )
    arguments = payload.get("arguments")
    if (
        not isinstance(arguments, Mapping)
        or _canonical_hash(arguments) != payload.get("arguments_sha256")
        or payload.get("arguments_sha256") != identity.get("arguments_sha256")
    ):
        raise BrokerJobError(
            "paper_broker_job_request_mismatch",
            "Persisted Broker arguments do not match their identity.",
        )
    budget = SharedBudget.open(payload.get("budget") or {})
    if budget.reference.identity_sha256 != identity.get(
        "budget_identity_sha256"
    ):
        raise BrokerJobError(
            "paper_broker_job_budget_mismatch",
            "Managed job budget identity changed.",
        )
    raw_admission_id = payload.get("admission_reservation_id")
    admission_id = (
        str(raw_admission_id) if isinstance(raw_admission_id, str) else None
    )
    result_path = _broker_result_path(job_id)
    existing = _read_broker_result(
        job_id, str(identity.get("identity_sha256") or ""),
    )
    if existing is not None:
        return _small_worker_receipt(existing)
    append_event(
        job_id,
        {
            "event": "paper_broker_job_submitted",
            "status": "submitted",
            "operation": operation,
            "identity_short": str(identity["identity_sha256"])[:12],
        },
    )

    def progress_callback(event: dict[str, Any]) -> None:
        snapshot = budget.snapshot()
        completed = event.get("sections_completed")
        total = event.get("sections_total")
        record_progress(
            job_id,
            {
                "event": "paper_broker_job_progress",
                "phase": str(event.get("event") or "provider"),
                "operation": operation,
                "identity_short": str(identity["identity_sha256"])[:12],
                **(
                    {"sections_completed": completed, "completed": completed}
                    if isinstance(completed, int) and not isinstance(completed, bool)
                    else {}
                ),
                **(
                    {"sections_total": total, "total": total}
                    if isinstance(total, int) and not isinstance(total, bool)
                    else {}
                ),
                "budget_remaining_calls": snapshot.remaining_calls,
                "budget_remaining_tokens": snapshot.remaining_tokens,
            },
        )

    try:
        with (
            managed_execution_scope(
                progress_callback=progress_callback,
                cancel_check=lambda: is_cancel_requested(job_id),
            ),
            shared_budget_context(
                budget,
                output_reserve_tokens=int(payload.get("output_reserve_tokens")),
                admission_reservation_id=admission_id,
            ),
        ):
            from .worker_session import WorkerCacheSession

            session = WorkerCacheSession.from_environment()
            job_status = read_json(JobPaths.for_job(job_id).status, {})
            recovery_attempt = (
                int(job_status.get("recovery_attempt") or 0)
                if isinstance(job_status, Mapping) else 0
            )
            call_id = (
                f"call-{str(identity['identity_sha256'])[:32]}"
                f"-attempt-{recovery_attempt + 1:02d}"
            )
            artifact_resolver = _worker_artifact_resolver(
                job, operation=operation,
            )
            if session is None:
                result = (
                    dispatch_operation(
                        operation,
                        arguments,
                        artifact_resolver=artifact_resolver,
                    )
                    if artifact_resolver is not None
                    else dispatch_operation(operation, arguments)
                )
            else:
                with session.in_process(call_id):
                    result = (
                        dispatch_operation(
                            operation,
                            arguments,
                            artifact_resolver=artifact_resolver,
                        )
                        if artifact_resolver is not None
                        else dispatch_operation(operation, arguments)
                    )
                session.record_call(
                    worker_id="broker-job-worker",
                    call_id=call_id,
                    operation=operation,
                    status="success" if result.get("ok") is True else "failed",
                    paper_ids=[],
                    parameters=arguments,
                    source={
                        "route": "controller_managed_job",
                        "operation_id": identity["operation_id"],
                    },
                )
                promotion = session.promote_call(call_id)
                if promotion.quarantined:
                    raise BrokerJobError(
                        "paper_cache_validation_failed",
                        "Managed ARC-paper cache artifacts failed validation.",
                    )
        if result.get("ok") is True:
            record = {
                "schema_version": BROKER_JOB_RESULT_SCHEMA_VERSION,
                "job_id": job_id,
                "identity_sha256": identity["identity_sha256"],
                "status": "done",
                "result": result,
                "error": None,
                "result_sha256": _canonical_hash(result),
                "error_sha256": None,
            }
        else:
            raw_error = (
                result.get("error")
                if isinstance(result.get("error"), Mapping)
                else {}
            )
            error = {
                "code": str(
                    raw_error.get("code") or "paper_broker_operation_failed"
                ),
                "message": str(
                    raw_error.get("message")
                    or "ARC-paper operation returned a failed result."
                ),
                "submission_state": str(
                    raw_error.get("submission_state") or "response_received"
                ),
            }
            record = {
                "schema_version": BROKER_JOB_RESULT_SCHEMA_VERSION,
                "job_id": job_id,
                "identity_sha256": identity["identity_sha256"],
                "status": "failed",
                "result": None,
                "error": error,
                "result_sha256": None,
                "error_sha256": _canonical_hash(error),
            }
    except Exception as exc:
        record = {
            "schema_version": BROKER_JOB_RESULT_SCHEMA_VERSION,
            "job_id": job_id,
            "identity_sha256": identity["identity_sha256"],
            "status": "failed",
            "result": None,
            "error": {
                "code": str(getattr(exc, "code", "paper_broker_job_failed")),
                "message": str(exc),
                "submission_state": str(
                    getattr(exc, "submission_state", "unknown")
                ),
            },
            "result_sha256": None,
            "error_sha256": None,
        }
        record["error_sha256"] = _canonical_hash(record["error"])
    finally:
        if admission_id is not None:
            admission = budget.reservation_or_none(admission_id)
            if admission is not None and admission.get("state") == "reserved":
                budget.release_not_submitted(admission_id)
    write_json(result_path, record)
    snapshot = budget.snapshot()
    append_event(
        job_id,
        {
            "event": (
                "paper_broker_job_completed"
                if record["status"] == "done"
                else "paper_broker_job_failed"
            ),
            "status": record["status"],
            "operation": operation,
            "identity_short": str(identity["identity_sha256"])[:12],
            "result_sha256": record.get("result_sha256"),
            "error_sha256": record.get("error_sha256"),
            "budget_remaining_calls": snapshot.remaining_calls,
            "budget_remaining_tokens": snapshot.remaining_tokens,
        },
    )
    return _small_worker_receipt(record)


def _verify_worker_job_record(
    job_id: str,
    job: Mapping[str, Any],
) -> None:
    payload = job.get("payload")
    identity = payload.get("identity") if isinstance(payload, Mapping) else None
    environment = job.get("environment")
    public_payload = job.get("public_payload")
    expected_payload_keys = {
        "broker_payload_schema_version", "identity", "operation",
        "arguments", "arguments_sha256", "budget",
        "output_reserve_tokens", "admission_reservation_id",
        "authorization",
    }
    if (
        job.get("schema_version") != "arc.job.v1"
        or job.get("job_id") != job_id
        or job.get("job_type") != BROKER_JOB_TYPE
        or job.get("execution_mode") != "process"
        or job.get("argv") != ["arc-paper", "_broker-job-worker", "--json"]
        or not isinstance(job.get("cwd"), str)
        or not isinstance(payload, Mapping)
        or set(payload) != expected_payload_keys
        or payload.get("broker_payload_schema_version")
        != BROKER_JOB_SCHEMA_VERSION
        or not isinstance(identity, Mapping)
        or job.get("full_identity") != identity
        or job.get("payload_sha256") != _canonical_hash(payload)
        or job.get("argv_sha256") != _canonical_hash(job.get("argv"))
        or not isinstance(environment, Mapping)
        or job.get("environment_identity_sha256")
        != _canonical_hash(environment)
        or not isinstance(public_payload, Mapping)
        or job.get("public_payload_sha256")
        != _canonical_hash(public_payload)
        or job.get("public_projection") != "bounded"
    ):
        raise BrokerJobError(
            "paper_broker_job_record_corrupt",
            "Managed job record is incomplete or inconsistent.",
        )
    request_identity = _canonical_hash({
        "job_type": BROKER_JOB_TYPE,
        "payload": dict(payload),
        "argv": list(job["argv"]),
        "cwd": job["cwd"],
        "environment": dict(environment),
        "execution_mode": "process",
        "full_identity": dict(identity),
        "public_payload": dict(public_payload),
        "public_projection": "bounded",
    })
    authorization = payload.get("authorization")
    spec = get_operation_spec(str(payload.get("operation") or ""))
    auth_material = (
        {
            key: value for key, value in authorization.items()
            if key != "receipt_sha256"
        }
        if isinstance(authorization, Mapping) else None
    )
    fixed_environment = {
        "ARC_PAPER_ACCESS": "none",
        "ARC_PAPER_CLI_ACCESS": "none",
        "ARC_INTERNAL_PAPER_ACCESS_LEGACY_MIRROR": "true",
        "ARC_LLM_INHERIT_HOST_TOOLS": "false",
        _DEPTH_ENV: "1",
    }
    if (
        job.get("request_identity_sha256") != request_identity
        or any(environment.get(key) != value for key, value in fixed_environment.items())
        or spec is None
        or identity.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION
        or identity.get("catalog_sha256") != _canonical_hash(catalog_document())
        or identity.get("operation_id") != spec.operation_id
        or identity.get("operation_version") != spec.version
        or identity.get("arguments_sha256") != payload.get("arguments_sha256")
        or identity.get("arguments_sha256") != _canonical_hash(payload.get("arguments"))
        or not isinstance(auth_material, Mapping)
        or authorization.get("receipt_sha256") != _canonical_hash(auth_material)
        or auth_material.get("operation_id") != spec.operation_id
        or auth_material.get("policy_sha256") != identity.get("policy_sha256")
        or auth_material.get("network_declared") != spec.network_access
        or (
            spec.network_access == "may"
            and auth_material.get("network_authorized") is not True
        )
        or auth_material.get("cache_declared") != spec.cache_access
        or auth_material.get("source_sha256") != identity.get("source_sha256")
        or auth_material.get("content_sha256") != identity.get("content_sha256")
        or not isinstance(auth_material.get("artifacts"), Mapping)
        or not _artifact_authorizations_match(
            spec,
            payload.get("arguments"),
            auth_material.get("artifacts"),
        )
        or _canonical_hash(auth_material.get("artifacts"))
        != identity.get("artifact_authorizations_sha256")
        or (
            _canonical_hash({
                "artifact_root": environment.get("ARC_PAPER_BROKER_ROOT")
            })
            if environment.get("ARC_PAPER_BROKER_ROOT") is not None
            else None
        ) != identity.get("artifact_root_sha256")
    ):
        raise BrokerJobError(
            "paper_broker_job_record_mismatch",
            "Managed job record no longer matches its catalog and policy receipt.",
        )


def _artifact_authorizations_match(
    spec: Any,
    arguments: Any,
    authorizations: Any,
) -> bool:
    if not isinstance(arguments, Mapping) or not isinstance(
        authorizations, Mapping
    ):
        return False
    expected_parameters = {
        parameter
        for parameter, _access in spec.artifact_parameters
        if arguments.get(parameter) is not None
    }
    if set(authorizations) != expected_parameters:
        return False
    access_by_parameter = dict(spec.artifact_parameters)
    for parameter in expected_parameters:
        handle = arguments.get(parameter)
        receipt = authorizations.get(parameter)
        if (
            not isinstance(handle, Mapping)
            or not isinstance(receipt, Mapping)
            or set(receipt) != {
                "handle_id", "operation", "parameter", "access",
                "handle_receipt_sha256",
            }
            or receipt.get("handle_id") != handle.get("handle_id")
            or receipt.get("operation") != spec.name
            or receipt.get("parameter") != parameter
            or receipt.get("access") != access_by_parameter[parameter]
            or not isinstance(receipt.get("handle_receipt_sha256"), str)
        ):
            return False
    return True


def _worker_artifact_resolver(
    job: Mapping[str, Any],
    *,
    operation: str,
):
    payload = job["payload"]
    authorizations = payload["authorization"]["artifacts"]
    root_value = job["environment"].get("ARC_PAPER_BROKER_ROOT")
    if not authorizations:
        return None
    if not isinstance(root_value, str):
        raise BrokerJobError(
            "paper_artifact_resolver_required",
            "Managed artifact resolver root is missing.",
        )
    root = Path(root_value).expanduser().resolve(strict=False)

    def resolve(
        handle_id: str,
        *,
        access: str,
        operation: str,
        parameter: str,
    ) -> Path:
        authorization = authorizations.get(parameter)
        if (
            not isinstance(authorization, Mapping)
            or authorization.get("handle_id") != handle_id
            or authorization.get("access") != access
            or authorization.get("operation") != operation
        ):
            raise BrokerJobError(
                "paper_artifact_handle_forbidden",
                "Managed artifact handle is not authorized.",
            )
        handle_path = root / "handles" / f"{handle_id}.json"
        record = read_json(handle_path)
        if (
            not isinstance(record, Mapping)
            or _canonical_hash(record)
            != authorization.get("handle_receipt_sha256")
            or [operation, parameter, access]
            not in record.get("contexts", [])
        ):
            raise BrokerJobError(
                "paper_artifact_handle_forbidden",
                "Managed artifact receipt changed.",
            )
        name_key = "output_name" if access == "write" else "object_name"
        name = record.get(name_key)
        if not isinstance(name, str) or Path(name).name != name:
            raise BrokerJobError(
                "paper_artifact_handle_forbidden",
                "Managed artifact ownership is invalid.",
            )
        directory = "inputs" if access == "write" else "objects"
        path = root / directory / name
        if path.is_symlink() or not path.is_file():
            raise BrokerJobError(
                "paper_artifact_handle_forbidden",
                "Managed artifact is unavailable.",
            )
        if access == "read":
            payload_bytes = path.read_bytes()
            if hashlib.sha256(payload_bytes).hexdigest() != record.get("sha256"):
                raise BrokerJobError(
                    "paper_artifact_handle_forbidden",
                    "Managed artifact integrity check failed.",
                )
        return path

    return resolve


def _broker_result_path(job_id: str) -> Path:
    return JobPaths.for_job(job_id).job_dir / "broker-result.json"


def _terminal_receipt(
    ticket: BrokerJobTicket,
    persisted: Mapping[str, Any] | None,
    status: Any,
) -> dict[str, Any]:
    job = read_job(ticket.job_id)
    status_error = status.get("error") if isinstance(status, Mapping) else None
    _verify_reconciliation_job_record(
        ticket.job_id,
        job,
        status_error_code=(
            str(status_error.get("code") or "")
            if isinstance(status_error, Mapping)
            else None
        ),
        status=status if isinstance(status, Mapping) else None,
    )
    payload = job["payload"]
    budget = SharedBudget.open(payload["budget"])
    snapshot = budget.snapshot()
    status_value = (
        str(persisted.get("status"))
        if isinstance(persisted, Mapping)
        else str(status.get("status") or "")
        if isinstance(status, Mapping)
        else ""
    )
    waiters = _seal_waiters_terminal(ticket, status_value)
    error = (
        persisted.get("error")
        if isinstance(persisted, Mapping)
        else status.get("error")
        if isinstance(status, Mapping)
        else None
    )
    return {
        "schema_version": "arc.paper.broker-job-terminal.v1",
        "job_schema_version": job.get("schema_version"),
        "job_id": ticket.job_id,
        "identity_sha256": ticket.identity_sha256,
        "request_identity_sha256": job.get("request_identity_sha256"),
        "payload_sha256": job.get("payload_sha256"),
        "status": status_value,
        "deduplicated": bool(waiters["deduplicated"]),
        "budget_identity_sha256": ticket.budget_identity_sha256,
        "budget": snapshot.to_json(),
        "depth": 1,
        "paper_access": "none",
        "result_sha256": (
            persisted.get("result_sha256")
            if isinstance(persisted, Mapping) else None
        ),
        "error_sha256": (
            persisted.get("error_sha256")
            if isinstance(persisted, Mapping)
            else _canonical_hash(error)
            if isinstance(error, Mapping)
            else None
        ),
        "recovery_attempt": (
            int(status.get("recovery_attempt") or 0)
            if isinstance(status, Mapping) else 0
        ),
    }


def _reconcile_lost_worker(
    ticket: BrokerJobTicket,
    *,
    status_error_code: str | None = None,
    status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    job = read_job(ticket.job_id)
    _verify_reconciliation_job_record(
        ticket.job_id,
        job,
        status_error_code=status_error_code,
        status=status,
    )
    payload = job["payload"]
    budget = SharedBudget.open(payload["budget"])
    admission_id = payload.get("admission_reservation_id")
    if not isinstance(admission_id, str):
        return {
            "action": "supervise",
            "submission_state": "unknown",
            "disposition": "no_provider_admission",
            "budget": budget,
            "reservation_id": None,
        }
    target_id = budget.adopted_target(admission_id)
    reservation_id = target_id or admission_id
    descendants = budget.descendant_reservations(admission_id)
    if descendants and _all_descendant_checkpoints_replayable(
        budget, descendants,
    ):
        return {
            "action": "retry",
            "submission_state": "response_received",
            "disposition": "completed_child_checkpoints_replayable",
            "budget": budget,
            "reservation_id": reservation_id,
        }
    row = budget.reservation(reservation_id)
    if (
        row["state"] == "reserved"
        and row["submission_state"] == "not_submitted"
    ) or (
        row["state"] == "released"
        and row["disposition"] == "proven_not_submitted"
    ):
        return {
            "action": "retry",
            "submission_state": "not_submitted",
            "disposition": str(row.get("disposition") or "reserved"),
            "budget": budget,
            "reservation_id": reservation_id,
        }
    if row["state"] == "reserved":
        settlement = budget.reconcile(
            reservation_id,
            checkpoint_submission_state=str(
                row.get("submission_state") or "unknown"
            ),
            owner_alive=False,
        )
        disposition = settlement.disposition
    else:
        disposition = str(row.get("disposition") or row["state"])
    return {
        "action": "supervise",
        "submission_state": str(row.get("submission_state") or "unknown"),
        "disposition": disposition,
        "budget": budget,
        "reservation_id": reservation_id,
    }


def _verify_reconciliation_job_record(
    job_id: str,
    job: Mapping[str, Any],
    *,
    status_error_code: str | None,
    status: Mapping[str, Any] | None,
) -> None:
    """Verify terminal reconciliation without weakening worker execution.

    A process launcher can reject a legacy total-timeout variable after the
    immutable job record was persisted.  If that variable was added without
    updating the record hashes, accept reconciliation only when removing the
    specific retired variables restores the exact original record identity.
    """

    verification_error: BrokerJobError | None = None
    try:
        _verify_worker_job_record(job_id, job)
        return
    except BrokerJobError as exc:
        if status_error_code != "job_environment_invalid":
            raise
        verification_error = exc

    assert verification_error is not None
    paths = JobPaths.for_job(job_id)
    persisted_error = read_json(paths.error, {})
    status_error = status.get("error") if isinstance(status, Mapping) else None
    if (
        not isinstance(status, Mapping)
        or status.get("status") != "failed"
        or status.get("phase") != "failed"
        or int(status.get("worker_launch_attempts") or 0) < 1
        or status.get("worker") is not None
        or status.get("process") is not None
        or paths.worker_process.exists()
        or not isinstance(status_error, Mapping)
        or status_error.get("schema_version") != "arc.job_error.v1"
        or status_error.get("code") != "job_environment_invalid"
        or persisted_error != status_error
    ):
        raise verification_error
    environment = job.get("environment")
    if not isinstance(environment, Mapping):
        raise verification_error
    removed = {
        key
        for key in _REMOVED_LLM_TIMEOUT_ENV_KEYS
        if str(environment.get(key) or "").strip()
    }
    if not removed:
        raise verification_error
    try:
        restored_environment(environment, base={})
    except ValueError as exc:
        if not str(exc).startswith(
            "LLM total-timeout environment variables were removed;"
        ):
            raise verification_error from exc
    else:
        raise verification_error

    sanitized = dict(job)
    sanitized["environment"] = {
        key: value
        for key, value in environment.items()
        if key not in removed
    }
    _verify_worker_job_record(job_id, sanitized)


def _all_descendant_checkpoints_replayable(
    budget: SharedBudget,
    descendants: list[Mapping[str, Any]],
) -> bool:
    for row in descendants:
        try:
            state = checkpoint_budget_state(
                Path(str(row.get("checkpoint_path") or "")),
                expected_identity=str(row.get("checkpoint_identity") or ""),
            )
            if state["state"] in {"response_received", "validated"}:
                usage = (
                    state["usage"]
                    if state.get("provider") != "kimi-code-cli"
                    and isinstance(state.get("usage"), Mapping)
                    else None
                )
                settlement = budget.reconcile(
                    str(row.get("reservation_id") or ""),
                    checkpoint_submission_state="response_received",
                    usage=usage,
                    owner_alive=False,
                )
                if settlement.charged_calls != 1:
                    return False
                continue
        except Exception:
            return False
        if (
            row.get("state") in {"reserved", "released"}
            and row.get("submission_state") == "not_submitted"
            and state["state"] == "prepared"
            and state["submission_state"] == "not_submitted"
        ):
            continue
        return False
    return True


def _release_lost_worker_reservation(recovery: Mapping[str, Any]) -> None:
    budget = recovery.get("budget")
    reservation_id = recovery.get("reservation_id")
    if isinstance(budget, SharedBudget) and isinstance(reservation_id, str):
        row = budget.reservation(reservation_id)
        if row["state"] == "reserved":
            budget.release_not_submitted(reservation_id)


def _waiters_path(job_id: str) -> Path:
    return JobPaths.for_job(job_id).job_dir / "broker-waiters.json"


def _waiter_id(job_id: str, transaction_receipt_sha256: str) -> str:
    return "waiter-" + _canonical_hash({
        "job_id": job_id,
        "transaction_receipt_sha256": transaction_receipt_sha256,
    })[:24]


def _attach_waiter(
    job_id: str,
    identity_sha256: str,
    transaction_receipt_sha256: str,
) -> bool:
    paths = JobPaths.for_job(job_id)
    with submission_lock(paths.job_dir.parent):
        path = _waiters_path(job_id)
        value = read_json(path, {})
        receipt = _normalized_waiter_receipt(value, identity_sha256)
        active = dict(receipt["active_waiters"])
        waiter_id = _waiter_id(job_id, transaction_receipt_sha256)
        known_ids = set(active)
        last_waiter_id = receipt.get("last_waiter_id")
        deduplicated = bool(receipt.get("deduplicated"))
        if (
            (known_ids and waiter_id not in known_ids)
            or (
                isinstance(last_waiter_id, str)
                and last_waiter_id
                and last_waiter_id != waiter_id
            )
        ):
            deduplicated = True
        terminal_status = _job_terminal_status_locked(job_id)
        if terminal_status is None:
            active[waiter_id] = "active"
        write_json(path, {
            "schema_version": "arc.paper.broker-job-waiters.v2",
            "identity_sha256": identity_sha256,
            "active_waiters": active,
            "deduplicated": deduplicated,
            "last_waiter_id": waiter_id,
            "terminal_status": terminal_status,
        })
        return terminal_status is not None


def _normalized_waiter_receipt(
    value: Any,
    identity_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BrokerJobError(
            "paper_broker_waiters_corrupt",
            "Managed job waiter receipt is invalid.",
        )
    if value and value.get("identity_sha256") != identity_sha256:
        raise BrokerJobError(
            "paper_broker_waiters_corrupt",
            "Managed job waiter identity changed.",
        )
    schema = value.get("schema_version")
    if not value:
        return {
            "active_waiters": {},
            "deduplicated": False,
            "last_waiter_id": None,
            "terminal_status": None,
        }
    if schema == "arc.paper.broker-job-waiters.v1":
        waiters = value.get("waiters")
        if not isinstance(waiters, Mapping):
            raise BrokerJobError(
                "paper_broker_waiters_corrupt",
                "Managed job waiter receipt is invalid.",
            )
        active = {
            str(key): "active"
            for key, state in waiters.items() if state == "active"
        }
        ids = list(waiters)
        return {
            "active_waiters": active,
            "deduplicated": len(ids) > 1,
            "last_waiter_id": str(ids[-1]) if ids else None,
            "terminal_status": None,
        }
    if schema != "arc.paper.broker-job-waiters.v2":
        raise BrokerJobError(
            "paper_broker_waiters_corrupt",
            "Managed job waiter receipt is invalid.",
        )
    active = value.get("active_waiters")
    if (
        not isinstance(active, Mapping)
        or any(state != "active" for state in active.values())
        or not isinstance(value.get("deduplicated"), bool)
        or (
            value.get("last_waiter_id") is not None
            and not isinstance(value.get("last_waiter_id"), str)
        )
        or (
            value.get("terminal_status") is not None
            and not isinstance(value.get("terminal_status"), str)
        )
    ):
        raise BrokerJobError(
            "paper_broker_waiters_corrupt",
            "Managed job waiter receipt is invalid.",
        )
    return {
        "active_waiters": dict(active),
        "deduplicated": bool(value["deduplicated"]),
        "last_waiter_id": value.get("last_waiter_id"),
        "terminal_status": value.get("terminal_status"),
    }


def _job_terminal_status_locked(job_id: str) -> str | None:
    result = read_json(_broker_result_path(job_id))
    if isinstance(result, Mapping) and result.get("status") in {"done", "failed"}:
        return str(result["status"])
    status = read_json(JobPaths.for_job(job_id).status, {})
    value = status.get("status") if isinstance(status, Mapping) else None
    if value in {
        "done", "completed", "degraded", "failed", "cancelled",
        "needs_supervision", "cancel_requested",
    }:
        return str(value)
    return None


def _cancel_waiter_locked(ticket: BrokerJobTicket) -> int | None:
    path = _waiters_path(ticket.job_id)
    receipt = _normalized_waiter_receipt(
        read_json(path), ticket.identity_sha256,
    )
    if receipt.get("terminal_status") is not None:
        return None
    active = dict(receipt["active_waiters"])
    waiter_id = _waiter_id(
        ticket.job_id, ticket.transaction_receipt_sha256,
    )
    if waiter_id not in active:
        if _job_terminal_status_locked(ticket.job_id) is not None:
            return None
        raise BrokerJobError(
            "paper_broker_waiter_unknown",
            "Managed job waiter is not attached.",
        )
    active.pop(waiter_id)
    write_json(path, {
        "schema_version": "arc.paper.broker-job-waiters.v2",
        "identity_sha256": ticket.identity_sha256,
        "active_waiters": active,
        "deduplicated": bool(receipt["deduplicated"]),
        "last_waiter_id": waiter_id,
        "terminal_status": None,
    })
    return len(active)


def _seal_waiters_terminal(
    ticket: BrokerJobTicket,
    terminal_status: str,
) -> dict[str, Any]:
    paths = JobPaths.for_job(ticket.job_id)
    with submission_lock(paths.job_dir.parent):
        path = _waiters_path(ticket.job_id)
        value = read_json(path, {})
        receipt = _normalized_waiter_receipt(
            value, ticket.identity_sha256,
        )
        sealed = {
            "schema_version": "arc.paper.broker-job-waiters.v2",
            "identity_sha256": ticket.identity_sha256,
            "active_waiters": {},
            "deduplicated": bool(receipt["deduplicated"]),
            "last_waiter_id": receipt.get("last_waiter_id"),
            "terminal_status": terminal_status,
        }
        if value != sealed:
            write_json(path, sealed)
        return sealed


def _read_broker_result(
    job_id: str, identity_sha256: str,
) -> Mapping[str, Any] | None:
    path = _broker_result_path(job_id)
    if not path.exists():
        return None
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise BrokerJobError(
            "paper_broker_job_result_corrupt",
            "Persisted Broker result is invalid.",
        )
    expected_keys = {
        "schema_version", "job_id", "identity_sha256", "status",
        "result", "error", "result_sha256", "error_sha256",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != BROKER_JOB_RESULT_SCHEMA_VERSION
        or value.get("job_id") != job_id
        or value.get("identity_sha256") != identity_sha256
        or value.get("status") not in {"done", "failed"}
        or (
            isinstance(value.get("result"), Mapping)
            and value.get("result_sha256")
            != _canonical_hash(value["result"])
        )
        or (
            isinstance(value.get("error"), Mapping)
            and value.get("error_sha256")
            != _canonical_hash(value["error"])
        )
        or (
            value.get("status") == "done"
            and (
                not isinstance(value.get("result"), Mapping)
                or value.get("error") is not None
                or not isinstance(value.get("result_sha256"), str)
                or value.get("error_sha256") is not None
            )
        )
        or (
            value.get("status") == "failed"
            and (
                not isinstance(value.get("error"), Mapping)
                or value.get("result") is not None
                or not isinstance(value.get("error_sha256"), str)
                or value.get("result_sha256") is not None
            )
        )
    ):
        raise BrokerJobError(
            "paper_broker_job_result_corrupt",
            "Persisted Broker result is invalid.",
        )
    return value


def _small_worker_receipt(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": record.get("status") == "done",
        "status": record.get("status"),
        "job_id": record.get("job_id"),
        "identity_sha256": record.get("identity_sha256"),
        "result_sha256": record.get("result_sha256"),
        "error_sha256": record.get("error_sha256"),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _required_hash(value: str, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    return normalized
