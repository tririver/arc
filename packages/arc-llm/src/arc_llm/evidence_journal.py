from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .evidence import (
    MAX_EVIDENCE_ROUNDS,
    EvidenceControllerCallback,
    EvidenceRequest,
    EvidenceResponse,
    resolve_evidence_round,
)


SCHEMA_VERSION = "arc.llm.evidence_exchange.v1"
STATES = ("prepared", "executed", "response_persisted", "delivered")
_ERROR_AUDIT_LIMIT = 8
_MAX_REQUEST_BYTES = 1024 * 1024
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_TRANSACTION_RECEIPT_BYTES = 1024 * 1024
_MAX_RECEIPT_BYTES = 32 * 1024 * 1024
_PROCESS_LOCKS: dict[Path, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


class EvidenceJournalError(RuntimeError):
    """Base class for durable evidence exchange failures."""


class EvidenceJournalCorruptError(EvidenceJournalError):
    """Raised when a receipt cannot be trusted or decoded."""


class EvidenceJournalStaleError(EvidenceJournalError):
    """Raised when an addressed receipt has different identity guards."""


class EvidenceJournalRecoveryError(EvidenceJournalError):
    """Raised when recovery would repeat unsafe work."""


@dataclass(frozen=True)
class EvidenceJournalAddress:
    run_id: str
    lane_id: str
    worker_id: str
    logical_task_id: str
    source_generation: int
    evidence_round: int
    request_id: str

    def __post_init__(self) -> None:
        for name in ("run_id", "lane_id", "worker_id", "logical_task_id", "request_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"evidence journal {name} is required")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.source_generation, bool)
            or not isinstance(self.source_generation, int)
            or self.source_generation < 1
        ):
            raise ValueError("evidence journal source_generation must be positive")
        if (
            isinstance(self.evidence_round, bool)
            or not isinstance(self.evidence_round, int)
            or self.evidence_round < 1
            or self.evidence_round > MAX_EVIDENCE_ROUNDS
        ):
            raise ValueError(
                f"evidence round must be between 1 and {MAX_EVIDENCE_ROUNDS}"
            )


@dataclass(frozen=True)
class EvidenceJournalContext:
    """Stable addressing and opaque guards supplied by an embedding workflow."""

    journal_root: Path
    run_id: str
    lane_id: str
    worker_id: str
    logical_task_id: str
    source_generation: int
    policy_hash: str
    runtime_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "journal_root", Path(self.journal_root))
        for name in (
            "run_id", "lane_id", "worker_id", "logical_task_id",
            "policy_hash", "runtime_hash",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"evidence journal {name} is required")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.source_generation, bool)
            or not isinstance(self.source_generation, int)
            or self.source_generation < 1
        ):
            raise ValueError("evidence journal source_generation must be positive")

    def address(self, request_id: str, *, evidence_round: int) -> EvidenceJournalAddress:
        return EvidenceJournalAddress(
            run_id=self.run_id,
            lane_id=self.lane_id,
            worker_id=self.worker_id,
            logical_task_id=self.logical_task_id,
            source_generation=self.source_generation,
            evidence_round=evidence_round,
            request_id=request_id,
        )


RecoveryCallback = Callable[[EvidenceRequest, Mapping[str, Any]], Any]
TransactionReceiptProvider = Callable[[EvidenceRequest], Mapping[str, Any] | None]


@dataclass(frozen=True)
class EvidenceOperationPolicy:
    """Operation semantics supplied by the controller-facing package.

    Arc-llm does not infer semantics from operation names.  Embeddings must
    explicitly opt known read/cache operations into idempotent recovery or
    provide transaction recovery for side-effecting work.
    """

    idempotent: bool = False
    recover: RecoveryCallback | None = None
    transaction_receipt: TransactionReceiptProvider | None = None


@dataclass(frozen=True)
class EvidenceExecution:
    response: EvidenceResponse
    transaction_receipt: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class EvidenceJournalAction:
    action: str
    address: EvidenceJournalAddress
    request: EvidenceRequest
    response: EvidenceResponse | None = None
    transaction_receipt: Mapping[str, Any] | None = None


TransitionHook = Callable[[str, EvidenceJournalAddress, Mapping[str, Any]], None]


def canonical_hash(value: Any) -> str:
    """Hash JSON-compatible policy/runtime material without persisting it."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class EvidenceJournal:
    """Durable four-state journal for controller-mediated evidence exchanges."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.receipts_root = self.root / "receipts"
        self.locks_root = self.root / "locks"
        for directory in (self.root, self.receipts_root, self.locks_root):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            _chmod(directory, 0o700)

    def prepare_round(
        self,
        context: EvidenceJournalContext,
        requests: Iterable[EvidenceRequest],
        *,
        round_number: int,
        max_rounds: int = MAX_EVIDENCE_ROUNDS,
        operation_policies: Mapping[str, EvidenceOperationPolicy] | None = None,
        transition_hook: TransitionHook | None = None,
    ) -> tuple[EvidenceJournalAction, ...]:
        """Persist new prepared receipts and classify execute/recover/replay."""

        material = _validate_round_input(
            context, requests, round_number=round_number, max_rounds=max_rounds
        )
        entries = self._entries(context, material, round_number, operation_policies)
        with self._locked_entries(entries):
            return self._prepare_locked(entries, transition_hook=transition_hook)

    def resolve_round(
        self,
        context: EvidenceJournalContext,
        requests: Iterable[EvidenceRequest],
        controller: EvidenceControllerCallback,
        *,
        round_number: int,
        max_rounds: int = MAX_EVIDENCE_ROUNDS,
        operation_policies: Mapping[str, EvidenceOperationPolicy] | None = None,
        transition_hook: TransitionHook | None = None,
    ) -> tuple[EvidenceResponse, ...]:
        """Resolve one worker's requests while holding all address locks.

        Persisted responses are replayed without invoking ``controller``.  The
        controller sees at most one group and request IDs therefore remain
        unambiguous despite being scoped per worker.
        """

        material = _validate_round_input(
            context, requests, round_number=round_number, max_rounds=max_rounds
        )
        if not material:
            return ()
        entries = self._entries(context, material, round_number, operation_policies)
        with self._locked_entries(entries):
            try:
                actions = self._prepare_locked(entries, transition_hook=transition_hook)
                responses: dict[str, EvidenceResponse] = {}
                controller_entries: list[_Entry] = []
                controller_requests: list[EvidenceRequest] = []
                for entry, action in zip(entries, actions, strict=True):
                    if action.action == "replay":
                        assert action.response is not None
                        responses[entry.request.request_id] = action.response
                    elif action.action == "recover" and not entry.policy.idempotent:
                        if entry.policy.recover is None or action.transaction_receipt is None:
                            raise EvidenceJournalRecoveryError(
                                "non-idempotent evidence recovery requires a transaction "
                                f"receipt and recovery callback: {entry.address.request_id}"
                            )
                        execution = _normalize_execution(
                            entry.policy.recover(entry.request, action.transaction_receipt),
                            expected_id=entry.request.request_id,
                            fallback_transaction=action.transaction_receipt,
                        )
                        responses[entry.request.request_id] = self._persist_execution(
                            entry, execution, transition_hook,
                        )
                    else:
                        controller_entries.append(entry)
                        controller_requests.append(entry.request)

                if controller_requests:
                    raw = resolve_evidence_round(
                        tuple(controller_requests),
                        controller,
                        round_number=round_number,
                        max_rounds=max_rounds,
                    )
                    for entry, response in zip(controller_entries, raw, strict=True):
                        execution = EvidenceExecution(response=response)
                        responses[entry.request.request_id] = self._persist_execution(
                            entry, execution, transition_hook,
                        )
                return tuple(responses[request.request_id] for request in material)
            except BaseException as exc:
                self._record_error(entries, exc)
                raise

    def mark_delivered(
        self,
        context: EvidenceJournalContext,
        requests: Iterable[EvidenceRequest],
        *,
        round_number: int,
        target_generation: int,
        target_session: str,
        followup_id: str,
        operation_policies: Mapping[str, EvidenceOperationPolicy] | None = None,
    ) -> None:
        """Append a delivery audit; delivery never consumes a response."""

        if (
            isinstance(target_generation, bool)
            or not isinstance(target_generation, int)
            or target_generation < 1
        ):
            raise ValueError("target_generation must be positive")
        target_session = str(target_session).strip()
        followup_id = str(followup_id).strip()
        if not target_session or not followup_id:
            raise ValueError("target_session and followup_id are required")
        material = _validate_round_input(
            context, requests, round_number=round_number,
            max_rounds=MAX_EVIDENCE_ROUNDS,
        )
        entries = self._entries(context, material, round_number, operation_policies)
        delivery = {
            "target_generation": target_generation,
            "target_session": target_session,
            "followup_id": followup_id,
        }
        with self._locked_entries(entries):
            for entry in entries:
                receipt = self._load_and_verify(entry)
                if receipt["state"] not in {"response_persisted", "delivered"}:
                    raise EvidenceJournalRecoveryError(
                        f"response is not durable before delivery: {entry.address.request_id}"
                    )
                deliveries = list(receipt["deliveries"])
                if any(
                    {key: item.get(key) for key in delivery} == delivery
                    for item in deliveries if isinstance(item, Mapping)
                ):
                    continue
                deliveries.append({**delivery, "delivered_at": _utc_now()})
                receipt["deliveries"] = deliveries
                receipt["state"] = "delivered"
                receipt["timestamps"].setdefault("delivered", deliveries[-1]["delivered_at"])
                self._write_receipt(entry.path, receipt)

    def receipt_path(self, address: EvidenceJournalAddress) -> Path:
        return self.receipts_root / f"{_address_hash(address)}.json"

    def read_receipt(self, address: EvidenceJournalAddress) -> dict[str, Any]:
        """Read and structurally validate a receipt for audit/testing."""

        return self._read_receipt(self.receipt_path(address))

    def _entries(
        self,
        context: EvidenceJournalContext,
        requests: Sequence[EvidenceRequest],
        round_number: int,
        operation_policies: Mapping[str, EvidenceOperationPolicy] | None,
    ) -> tuple["_Entry", ...]:
        if context.journal_root.resolve(strict=False) != self.root.resolve(strict=False):
            raise ValueError("evidence journal context root does not match journal root")
        policies = operation_policies or {}
        result = []
        for request in requests:
            if request.worker_id and request.worker_id != context.worker_id:
                raise ValueError(
                    f"request worker {request.worker_id} does not match journal worker "
                    f"{context.worker_id}"
                )
            address = context.address(request.request_id, evidence_round=round_number)
            operation_arguments_hash = canonical_hash({
                "operation": request.operation,
                "arguments": dict(request.arguments),
            })
            policy = policies.get(request.operation, EvidenceOperationPolicy())
            result.append(_Entry(
                address=address,
                request=request,
                operation_arguments_hash=operation_arguments_hash,
                policy_hash=_guard_hash(context.policy_hash),
                runtime_hash=_guard_hash(context.runtime_hash),
                policy=policy,
                path=self.receipt_path(address),
            ))
        return tuple(result)

    def _prepare_locked(
        self,
        entries: Sequence["_Entry"],
        *,
        transition_hook: TransitionHook | None = None,
    ) -> tuple[EvidenceJournalAction, ...]:
        actions = []
        for entry in entries:
            if not entry.path.exists():
                transaction = (
                    entry.policy.transaction_receipt(entry.request)
                    if entry.policy.transaction_receipt is not None else None
                )
                receipt = self._new_receipt(entry, transaction)
                self._write_receipt(entry.path, receipt)
                if transition_hook is not None:
                    transition_hook("prepared", entry.address, receipt)
                actions.append(EvidenceJournalAction(
                    "execute", entry.address, entry.request,
                    transaction_receipt=transaction,
                ))
                continue
            receipt = self._load_and_verify(entry)
            state = receipt["state"]
            if state in {"response_persisted", "delivered"}:
                response = _deserialize_response(
                    receipt.get("response"),
                    expected_request_id=entry.address.request_id,
                )
                actions.append(EvidenceJournalAction(
                    "replay", entry.address, entry.request, response=response,
                    transaction_receipt=receipt.get("transaction_receipt"),
                ))
                continue
            execution = receipt.get("execution_receipt")
            if state == "executed" and isinstance(execution, Mapping):
                stored_response = receipt.get("response")
                if stored_response is not None:
                    response = _deserialize_response(
                        stored_response,
                        expected_request_id=entry.address.request_id,
                    )
                    receipt["state"] = "response_persisted"
                    receipt["timestamps"]["response_persisted"] = _utc_now()
                    self._write_receipt(entry.path, receipt)
                    actions.append(EvidenceJournalAction(
                        "replay", entry.address, entry.request, response=response,
                        transaction_receipt=receipt.get("transaction_receipt"),
                    ))
                    continue
            transaction = receipt.get("transaction_receipt")
            if not entry.policy.idempotent and not isinstance(transaction, Mapping):
                raise EvidenceJournalRecoveryError(
                    "non-idempotent prepared/executed evidence has no transaction "
                    f"receipt: {entry.address.request_id}"
                )
            actions.append(EvidenceJournalAction(
                "recover", entry.address, entry.request,
                transaction_receipt=transaction if isinstance(transaction, Mapping) else None,
            ))
        return tuple(actions)

    def _persist_execution(
        self,
        entry: "_Entry",
        execution: EvidenceExecution,
        transition_hook: TransitionHook | None,
    ) -> EvidenceResponse:
        receipt = self._load_and_verify(entry)
        if receipt["state"] not in {"prepared", "executed"}:
            raise EvidenceJournalCorruptError(
                f"invalid execution transition from {receipt['state']}"
            )
        transaction = execution.transaction_receipt
        if transaction is not None:
            transaction = _bounded_json_object(
                transaction,
                "transaction receipt",
                max_bytes=_MAX_TRANSACTION_RECEIPT_BYTES,
            )
            existing = receipt.get("transaction_receipt")
            if existing is not None and canonical_hash(existing) != canonical_hash(transaction):
                raise EvidenceJournalStaleError("transaction receipt changed during recovery")
            receipt["transaction_receipt"] = transaction
        response = _serialize_response(execution.response)
        if response["request_id"] != entry.address.request_id:
            raise ValueError("evidence response request ID does not match journal address")
        response_digest = canonical_hash(response)
        executed_at = receipt["timestamps"].get("executed") or _utc_now()
        persisted_at = _utc_now()
        final_receipt = {
            **receipt,
            "execution_receipt": {
                "request_id": entry.address.request_id,
                "response_sha256": response_digest,
            },
            "response": response,
            "state": "response_persisted",
            "timestamps": {
                **receipt["timestamps"],
                "executed": executed_at,
                "response_persisted": persisted_at,
            },
        }
        # Preflight the final pretty-encoded receipt before publishing executed.
        # A large valid response therefore cannot strand recovery in executed.
        _encode_receipt(final_receipt)
        receipt["execution_receipt"] = {
            "request_id": entry.address.request_id,
            "response_sha256": response_digest,
        }
        receipt["response"] = response
        receipt["state"] = "executed"
        receipt["timestamps"].setdefault("executed", executed_at)
        self._write_receipt(entry.path, receipt)
        if transition_hook is not None:
            transition_hook("executed", entry.address, receipt)

        receipt["state"] = "response_persisted"
        receipt["timestamps"]["response_persisted"] = persisted_at
        self._write_receipt(entry.path, receipt)
        if transition_hook is not None:
            transition_hook("response_persisted", entry.address, receipt)
        return execution.response

    def _new_receipt(
        self, entry: "_Entry", transaction: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        now = _utc_now()
        address = asdict(entry.address)
        guards = {
            "operation_arguments_hash": entry.operation_arguments_hash,
            "policy_hash": entry.policy_hash,
            "runtime_hash": entry.runtime_hash,
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "address": address,
            "identity_hash": canonical_hash({"address": address, **guards}),
            **guards,
            "state": "prepared",
            "request": _serialize_request(entry.request),
            "response": None,
            "transaction_receipt": (
                _bounded_json_object(
                    transaction,
                    "transaction receipt",
                    max_bytes=_MAX_TRANSACTION_RECEIPT_BYTES,
                )
                if transaction is not None else None
            ),
            "execution_receipt": None,
            "timestamps": {"prepared": now},
            "deliveries": [],
            "errors": [],
        }

    def _load_and_verify(self, entry: "_Entry") -> dict[str, Any]:
        receipt = self._read_receipt(entry.path)
        if receipt["address"] != asdict(entry.address):
            raise EvidenceJournalCorruptError("receipt address does not match its filename")
        expected = {
            "operation_arguments_hash": entry.operation_arguments_hash,
            "policy_hash": entry.policy_hash,
            "runtime_hash": entry.runtime_hash,
        }
        stale = [name for name, value in expected.items() if receipt.get(name) != value]
        if stale:
            raise EvidenceJournalStaleError(
                "stale evidence receipt guard mismatch: " + ", ".join(stale)
            )
        identity = canonical_hash({"address": asdict(entry.address), **expected})
        if receipt.get("identity_hash") != identity:
            raise EvidenceJournalCorruptError("receipt identity hash is invalid")
        stored_request = receipt.get("request")
        if not isinstance(stored_request, Mapping):
            raise EvidenceJournalCorruptError("receipt request is not an object")
        if stored_request.get("request_id") != entry.request.request_id:
            raise EvidenceJournalCorruptError("receipt request ID is invalid")
        stored_operation_arguments_hash = canonical_hash({
            "operation": stored_request.get("operation"),
            "arguments": stored_request.get("arguments"),
        })
        if stored_operation_arguments_hash != receipt.get("operation_arguments_hash"):
            raise EvidenceJournalCorruptError(
                "persisted request does not match its operation/arguments guard"
            )
        return receipt

    def _read_receipt(self, path: Path) -> dict[str, Any]:
        try:
            if path.stat().st_size > _MAX_RECEIPT_BYTES:
                raise EvidenceJournalCorruptError("evidence receipt exceeds size limit")
            value = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
            )
        except FileNotFoundError:
            raise
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise EvidenceJournalCorruptError(f"cannot read evidence receipt {path.name}") from exc
        if not isinstance(value, dict):
            raise EvidenceJournalCorruptError("evidence receipt root must be an object")
        if value.get("schema_version") != SCHEMA_VERSION:
            raise EvidenceJournalCorruptError("unsupported evidence receipt schema")
        if value.get("state") not in STATES:
            raise EvidenceJournalCorruptError("unknown evidence receipt state")
        if not isinstance(value.get("address"), dict):
            raise EvidenceJournalCorruptError("evidence receipt address must be an object")
        try:
            EvidenceJournalAddress(**value["address"])
        except (TypeError, ValueError) as exc:
            raise EvidenceJournalCorruptError("evidence receipt address is invalid") from exc
        request = value.get("request")
        if (
            not isinstance(request, dict)
            or set(request) != {"request_id", "operation", "arguments", "reason"}
            or not isinstance(request.get("request_id"), str)
            or not request["request_id"].strip()
            or not isinstance(request.get("operation"), str)
            or not request["operation"].strip()
            or not isinstance(request.get("arguments"), dict)
            or not isinstance(request.get("reason"), str)
        ):
            raise EvidenceJournalCorruptError("evidence receipt request is malformed")
        try:
            _require_json_size(request, _MAX_REQUEST_BYTES, "evidence request")
        except (TypeError, ValueError) as exc:
            raise EvidenceJournalCorruptError(
                "persisted evidence request exceeds its size limit"
            ) from exc
        for name in (
            "identity_hash", "operation_arguments_hash", "policy_hash", "runtime_hash",
        ):
            if not isinstance(value.get(name), str) or not value[name].strip():
                raise EvidenceJournalCorruptError(f"evidence receipt {name} is invalid")
        if value.get("transaction_receipt") is not None and not isinstance(
            value["transaction_receipt"], dict
        ):
            raise EvidenceJournalCorruptError("transaction receipt must be an object or null")
        if value.get("transaction_receipt") is not None:
            try:
                _require_json_size(
                    value["transaction_receipt"],
                    _MAX_TRANSACTION_RECEIPT_BYTES,
                    "transaction receipt",
                )
            except (TypeError, ValueError) as exc:
                raise EvidenceJournalCorruptError(
                    "persisted transaction receipt exceeds its size limit"
                ) from exc
        if value.get("execution_receipt") is not None and not isinstance(
            value["execution_receipt"], dict
        ):
            raise EvidenceJournalCorruptError("execution receipt must be an object or null")
        if value.get("execution_receipt") is not None:
            execution_receipt = value["execution_receipt"]
            if (
                set(execution_receipt) != {"request_id", "response_sha256"}
                or execution_receipt.get("request_id")
                != value["address"].get("request_id")
                or not isinstance(execution_receipt.get("response_sha256"), str)
                or len(execution_receipt["response_sha256"]) != 64
            ):
                raise EvidenceJournalCorruptError(
                    "execution receipt envelope is malformed"
                )
        if not isinstance(value.get("timestamps"), dict):
            raise EvidenceJournalCorruptError("evidence receipt timestamps must be an object")
        if not isinstance(value.get("deliveries"), list):
            raise EvidenceJournalCorruptError("evidence receipt deliveries must be an array")
        if not isinstance(value.get("errors", []), list):
            raise EvidenceJournalCorruptError("evidence receipt errors must be an array")
        state_index = STATES.index(value["state"])
        allowed_timestamps = set(STATES[:state_index + 1])
        if set(value["timestamps"]) != allowed_timestamps:
            raise EvidenceJournalCorruptError(
                "receipt timestamps do not match its state transition"
            )
        if not all(
            isinstance(item, str) and item.strip()
            for item in value["timestamps"].values()
        ):
            raise EvidenceJournalCorruptError("receipt transition times are invalid")
        for required_state in STATES[:state_index + 1]:
            if required_state == "delivered" and not value["deliveries"]:
                raise EvidenceJournalCorruptError("delivered receipt has no delivery audit")
            if required_state not in value["timestamps"]:
                raise EvidenceJournalCorruptError(
                    f"receipt is missing {required_state} transition time"
                )
        for delivery in value["deliveries"]:
            if (
                not isinstance(delivery, dict)
                or set(delivery) != {
                    "target_generation", "target_session", "followup_id", "delivered_at",
                }
                or isinstance(delivery.get("target_generation"), bool)
                or not isinstance(delivery.get("target_generation"), int)
                or delivery["target_generation"] < 1
                or not isinstance(delivery.get("target_session"), str)
                or not delivery["target_session"].strip()
                or not isinstance(delivery.get("followup_id"), str)
                or not delivery["followup_id"].strip()
                or not isinstance(delivery.get("delivered_at"), str)
                or not delivery["delivered_at"].strip()
            ):
                raise EvidenceJournalCorruptError("evidence delivery audit is malformed")
        if value["state"] == "prepared" and value.get("execution_receipt") is not None:
            raise EvidenceJournalCorruptError("prepared receipt contains execution data")
        if value["state"] == "prepared" and value.get("response") is not None:
            raise EvidenceJournalCorruptError("prepared receipt contains a response")
        if value["state"] == "executed" and value.get("execution_receipt") is None:
            raise EvidenceJournalCorruptError("executed receipt has no execution data")
        if value["state"] in {"executed", "response_persisted", "delivered"}:
            if value.get("execution_receipt") is None:
                raise EvidenceJournalCorruptError("persisted response has no execution receipt")
            response = _deserialize_response(
                value.get("response"),
                expected_request_id=str(value["address"]["request_id"]),
            )
            try:
                serialized_response = _serialize_response(response)
            except (TypeError, ValueError) as exc:
                raise EvidenceJournalCorruptError(
                    "persisted evidence response exceeds its size limit"
                ) from exc
            if canonical_hash(serialized_response) != value["execution_receipt"].get(
                "response_sha256"
            ):
                raise EvidenceJournalCorruptError(
                    "persisted response does not match its execution digest"
                )
        return value

    def _record_error(self, entries: Sequence["_Entry"], exc: BaseException) -> None:
        audit = {
            "type": type(exc).__name__,
            "message_hash": canonical_hash(str(exc)),
            "at": _utc_now(),
        }
        for entry in entries:
            if not entry.path.exists():
                continue
            try:
                receipt = self._load_and_verify(entry)
                errors = list(receipt.get("errors") or [])[-(_ERROR_AUDIT_LIMIT - 1):]
                errors.append(audit)
                receipt["errors"] = errors
                self._write_receipt(entry.path, receipt)
            except (EvidenceJournalError, OSError, ValueError):
                continue

    @contextmanager
    def _locked_entries(self, entries: Sequence["_Entry"]) -> Iterator[None]:
        ordered = sorted({_address_hash(entry.address) for entry in entries})
        with ExitStack() as stack:
            for digest in ordered:
                stack.enter_context(_address_lock(self.locks_root / f"{digest}.lock"))
            yield

    def _write_receipt(self, path: Path, value: Mapping[str, Any]) -> None:
        _atomic_write_json(path, value, mode=0o600)


@dataclass(frozen=True)
class _Entry:
    address: EvidenceJournalAddress
    request: EvidenceRequest
    operation_arguments_hash: str
    policy_hash: str
    runtime_hash: str
    policy: EvidenceOperationPolicy
    path: Path


def _validate_round_input(
    context: EvidenceJournalContext,
    requests: Iterable[EvidenceRequest],
    *,
    round_number: int,
    max_rounds: int,
) -> tuple[EvidenceRequest, ...]:
    if (
        isinstance(max_rounds, bool)
        or not isinstance(max_rounds, int)
        or max_rounds < 1
        or max_rounds > MAX_EVIDENCE_ROUNDS
    ):
        raise ValueError(f"max_rounds must be between 1 and {MAX_EVIDENCE_ROUNDS}")
    if (
        isinstance(round_number, bool)
        or not isinstance(round_number, int)
        or round_number < 1
        or round_number > max_rounds
    ):
        raise ValueError(f"evidence round {round_number} exceeds max_rounds={max_rounds}")
    material = tuple(requests)
    if not all(isinstance(item, EvidenceRequest) for item in material):
        raise TypeError("evidence journal requests must be EvidenceRequest objects")
    ids = [item.request_id for item in material]
    if len(ids) != len(set(ids)):
        raise ValueError("evidence request IDs must be unique for an addressed worker round")
    return material


def _normalize_execution(
    value: Any,
    *,
    expected_id: str,
    fallback_transaction: Mapping[str, Any] | None,
) -> EvidenceExecution:
    if isinstance(value, EvidenceExecution):
        execution = value
    elif isinstance(value, EvidenceResponse):
        execution = EvidenceExecution(value, fallback_transaction)
    else:
        raise TypeError("evidence recovery must return EvidenceResponse or EvidenceExecution")
    if execution.response.request_id != expected_id:
        raise ValueError("evidence recovery response ID does not match request")
    return execution


def _serialize_request(request: EvidenceRequest) -> dict[str, Any]:
    raw = {
        "request_id": request.request_id,
        "operation": request.operation,
        "arguments": _json_object(request.arguments, "evidence arguments"),
        "reason": request.reason,
    }
    _require_json_size(raw, _MAX_REQUEST_BYTES, "evidence request")
    _reject_explicit_credentials(raw, label="evidence request")
    return raw


def _serialize_response(response: EvidenceResponse) -> dict[str, Any]:
    raw = {
        "request_id": response.request_id,
        "ok": response.ok,
        "data": _json_value(response.data, "evidence response data"),
        "error": response.error,
        "provenance": _json_object(response.provenance, "evidence provenance"),
    }
    _require_json_size(raw, _MAX_RESPONSE_BYTES, "evidence response")
    _reject_explicit_credentials(raw, label="evidence response")
    return raw


def _deserialize_response(
    value: Any, *, expected_request_id: str | None = None,
) -> EvidenceResponse:
    if not isinstance(value, Mapping):
        raise EvidenceJournalCorruptError("persisted evidence response is not an object")
    if set(value) != {"request_id", "ok", "data", "error", "provenance"}:
        raise EvidenceJournalCorruptError("persisted evidence response envelope is malformed")
    if not isinstance(value.get("ok"), bool) or not isinstance(value.get("provenance"), Mapping):
        raise EvidenceJournalCorruptError("persisted evidence response has invalid fields")
    if expected_request_id is not None and value.get("request_id") != expected_request_id:
        raise EvidenceJournalCorruptError(
            "persisted evidence response request ID does not match its address"
        )
    try:
        return EvidenceResponse(
            request_id=str(value.get("request_id") or ""),
            ok=value["ok"],
            data=value.get("data"),
            error=value.get("error"),
            provenance=dict(value["provenance"]),
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceJournalCorruptError("persisted evidence response is invalid") from exc


def _json_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    result = _json_value(dict(value), label)
    assert isinstance(result, dict)
    return result


def _bounded_json_object(
    value: Mapping[str, Any], label: str, *, max_bytes: int,
) -> dict[str, Any]:
    raw = _json_object(value, label)
    _require_json_size(raw, max_bytes, label)
    _reject_explicit_credentials(raw, label=label)
    return raw


_EXPLICIT_CREDENTIAL_FIELDS = {
    "password", "api_key", "authorization", "private_key", "client_secret",
}


def _reject_explicit_credentials(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for item_key, item_value in value.items():
            normalized = str(item_key).strip().casefold().replace("-", "_")
            if normalized in _EXPLICIT_CREDENTIAL_FIELDS:
                raise ValueError(
                    f"{label} contains prohibited credential field: {item_key}"
                )
            _reject_explicit_credentials(item_value, label=label)
        return
    if isinstance(value, list):
        for item in value:
            _reject_explicit_credentials(item, label=label)


def _require_json_size(value: Any, max_bytes: int, label: str) -> None:
    if len(_canonical_json(value).encode("utf-8")) > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} UTF-8 bytes")


def _json_value(value: Any, label: str) -> Any:
    try:
        encoded = _canonical_json(value)
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TypeError(f"{label} must be finite JSON data") from exc


def _canonical_json(value: Any) -> str:
    # json.dumps emits NaN by default; forbid it so identities transfer hosts.
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    json.loads(encoded, parse_constant=_reject_json_constant)
    return encoded


def _reject_json_constant(constant: str) -> None:
    raise ValueError(f"non-finite JSON number: {constant}")


def _address_hash(address: EvidenceJournalAddress) -> str:
    return canonical_hash(asdict(address))


def _guard_hash(value: str) -> str:
    normalized = str(value).strip().casefold()
    if len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    ):
        return normalized
    return canonical_hash(str(value))


@contextmanager
def _address_lock(path: Path) -> Iterator[None]:
    resolved = path.resolve(strict=False)
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.setdefault(resolved, threading.RLock())
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _chmod(path.parent, 0o700)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        _chmod(path, 0o600)
        fcntl = None
        msvcrt = None
        try:
            try:
                import fcntl
            except ImportError:  # pragma: no cover - Windows
                try:
                    import msvcrt
                except ImportError:  # pragma: no cover - unusual Python host
                    msvcrt = None  # type: ignore[assignment]
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            elif msvcrt is not None:  # pragma: no cover - Windows
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                    os.fsync(fd)
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:  # pragma: no cover
                raise EvidenceJournalError("host has no supported process file lock")
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            os.close(fd)


def _encode_receipt(value: Mapping[str, Any]) -> bytes:
    data = (json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n").encode("utf-8")
    if len(data) > _MAX_RECEIPT_BYTES:
        raise ValueError(f"evidence receipt exceeds {_MAX_RECEIPT_BYTES} UTF-8 bytes")
    return data


def _atomic_write_json(path: Path, value: Mapping[str, Any], *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _chmod(path.parent, 0o700)
    data = _encode_receipt(value)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(tmp, flags, mode)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _chmod(path, mode)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:  # pragma: no cover - filesystem dependent
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _chmod(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except OSError:  # pragma: no cover - permission model dependent
        pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
