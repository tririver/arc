"""Durable lifecycle accounting for parent-owned descendant LLM calls.

The parent creates one finite private ledger and passes only its verified
reference to child processes. Provider calls reserve after checkpoint replay is
ruled out, mark submission before later progress is accepted, and settle from
known usage or a conservative bound. Recovery combines the persisted provider
checkpoint with the reservation owner's PID/start identity: only a proven
unsubmitted call may be released or taken over; submitted, unknown, or
ambiguous work is charged and routed to supervision.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import threading
import time
from typing import Any, Iterator, Mapping


BUDGET_SCHEMA_VERSION = "arc.llm.shared-budget.v1"


@dataclass
class SharedBudgetBinding:
    budget: "SharedBudget"
    output_reserve_tokens: int
    admission_reservation_id: str | None = None
    _admission_consumed: bool = field(default=False, init=False, repr=False)
    _admission_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False,
    )

    def reserve(
        self,
        *,
        checkpoint_identity: str,
        provider_attempt: int,
        prompt_bytes: int,
        checkpoint_path: Path | None = None,
    ) -> "BudgetReservation":
        with self._admission_lock:
            if (
                self.admission_reservation_id is not None
                and not self._admission_consumed
            ):
                reservation = self.budget.adopt(
                    self.admission_reservation_id,
                    checkpoint_identity=checkpoint_identity,
                    provider_attempt=provider_attempt,
                    prompt_bytes=prompt_bytes,
                    output_reserve_tokens=self.output_reserve_tokens,
                )
                reservation = self.budget.recover_not_submitted(
                    reservation.reservation_id
                )
                if checkpoint_path is not None:
                    self.budget.bind_reservation_context(
                        reservation.reservation_id,
                        parent_admission_id=self.admission_reservation_id,
                        checkpoint_path=checkpoint_path,
                    )
                self._admission_consumed = True
                return reservation
        reservation = self.budget.reserve(
            checkpoint_identity=checkpoint_identity,
            provider_attempt=provider_attempt,
            prompt_bytes=prompt_bytes,
            output_reserve_tokens=self.output_reserve_tokens,
            recover_not_submitted=True,
        )
        if (
            self.admission_reservation_id is not None
            and checkpoint_path is not None
        ):
            self.budget.bind_reservation_context(
                reservation.reservation_id,
                parent_admission_id=self.admission_reservation_id,
                checkpoint_path=checkpoint_path,
            )
        return reservation

    def reconcile_replay(
        self,
        *,
        checkpoint_identity: str,
        provider_attempt: int,
        usage: Mapping[str, Any] | None,
    ) -> "BudgetSettlement | None":
        with self._admission_lock:
            existing = self.budget.lookup(
                checkpoint_identity=checkpoint_identity,
                provider_attempt=provider_attempt,
            )
            settlement = None
            if existing is not None:
                settlement = self.budget.reconcile(
                    existing.reservation_id,
                    checkpoint_submission_state="response_received",
                    usage=usage,
                )
            if (
                self.admission_reservation_id is not None
                and not self._admission_consumed
            ):
                admission = self.budget.reservation_or_none(
                    self.admission_reservation_id
                )
                if admission is not None and admission.get("state") == "reserved":
                    self.budget.release_not_submitted(
                        self.admission_reservation_id
                    )
                self._admission_consumed = True
            return settlement


_CURRENT_BUDGET: ContextVar["SharedBudgetBinding | None"] = ContextVar(
    "arc_llm_current_shared_budget_binding", default=None,
)


class BudgetError(RuntimeError):
    code = "child_budget_error"
    submission_state = "not_submitted"
    abort_batch = True


class BudgetRequired(BudgetError):
    code = "child_budget_required"


class BudgetExhausted(BudgetError):
    code = "child_budget_exhausted"


class BudgetCorrupt(BudgetError):
    code = "child_budget_corrupt"


@dataclass(frozen=True)
class BudgetReference:
    path: Path
    budget_id: str
    identity_sha256: str

    def to_json(self) -> dict[str, str]:
        return {
            "schema_version": BUDGET_SCHEMA_VERSION,
            "budget_id": self.budget_id,
            "identity_sha256": self.identity_sha256,
            "path": str(self.path),
        }


@dataclass(frozen=True)
class BudgetSnapshot:
    max_calls: int
    max_tokens: int
    charged_calls: int
    charged_tokens: int
    outstanding_calls: int
    outstanding_tokens: int

    @property
    def remaining_calls(self) -> int:
        return max(
            0, self.max_calls - self.charged_calls - self.outstanding_calls,
        )

    @property
    def remaining_tokens(self) -> int:
        return max(
            0, self.max_tokens - self.charged_tokens - self.outstanding_tokens,
        )

    def to_json(self) -> dict[str, int]:
        return {
            "max_calls": self.max_calls,
            "max_tokens": self.max_tokens,
            "charged_calls": self.charged_calls,
            "charged_tokens": self.charged_tokens,
            "outstanding_calls": self.outstanding_calls,
            "outstanding_tokens": self.outstanding_tokens,
            "remaining_calls": self.remaining_calls,
            "remaining_tokens": self.remaining_tokens,
        }


@dataclass(frozen=True)
class BudgetSettlement:
    reservation_id: str
    disposition: str
    charged_calls: int
    charged_tokens: int
    warning: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": "arc.llm.budget-settlement.v1",
            "reservation_id": self.reservation_id,
            "disposition": self.disposition,
            "charged_calls": self.charged_calls,
            "charged_tokens": self.charged_tokens,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class BudgetReservation:
    budget: "SharedBudget"
    reservation_id: str
    checkpoint_identity: str
    provider_attempt: int
    reserved_tokens: int

    def mark_submitted(self) -> None:
        self.budget.mark_submitted(self.reservation_id)

    def settle_known(self, *, input_tokens: int, output_tokens: int) -> BudgetSettlement:
        return self.budget.settle_known(
            self.reservation_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def settle_conservative(self, disposition: str) -> BudgetSettlement:
        return self.budget.settle_conservative(self.reservation_id, disposition)

    def settle_unknown_usage(self) -> BudgetSettlement:
        return self.budget._settle(
            self.reservation_id,
            disposition="unknown_usage",
            charged_calls=1,
            charged_tokens=self.reserved_tokens,
            warning="budget.usage_unknown_charged_reserved",
        )

    def release_not_submitted(self) -> BudgetSettlement:
        return self.budget.release_not_submitted(self.reservation_id)


class SharedBudget:
    """Finite parent-owned, cross-process budget for descendant provider calls."""

    def __init__(self, reference: BudgetReference) -> None:
        self.reference = reference
        self.path = reference.path
        self._verify_metadata()

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        budget_id: str,
        max_calls: int,
        max_tokens: int,
    ) -> "SharedBudget":
        if not budget_id.strip():
            raise ValueError("budget_id is required")
        if type(max_calls) is not int or max_calls < 0:
            raise ValueError("max_calls must be a finite non-negative integer")
        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("max_tokens must be a finite non-negative integer")
        canonical = path.expanduser().resolve(strict=False)
        canonical.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(canonical.parent, 0o700)
        identity = _hash({
            "schema_version": BUDGET_SCHEMA_VERSION,
            "budget_id": budget_id,
            "max_calls": max_calls,
            "max_tokens": max_tokens,
        })
        with _connect(canonical) as db:
            _initialize(db)
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT budget_id, identity_sha256, max_calls, max_tokens "
                "FROM budget_metadata WHERE singleton = 1"
            ).fetchone()
            expected = (budget_id, identity, max_calls, max_tokens)
            if existing is None:
                db.execute(
                    "INSERT INTO budget_metadata "
                    "(singleton, schema_version, budget_id, identity_sha256, "
                    "max_calls, max_tokens) VALUES (1, ?, ?, ?, ?, ?)",
                    (BUDGET_SCHEMA_VERSION, *expected),
                )
            elif tuple(existing) != expected:
                raise BudgetCorrupt("shared budget metadata changed")
            db.commit()
        _secure_sqlite(canonical)
        return cls(BudgetReference(canonical, budget_id, identity))

    @classmethod
    def open(cls, value: BudgetReference | Mapping[str, Any]) -> "SharedBudget":
        reference = (
            value
            if isinstance(value, BudgetReference)
            else BudgetReference(
                Path(str(value.get("path") or "")).expanduser().resolve(strict=False),
                str(value.get("budget_id") or ""),
                str(value.get("identity_sha256") or ""),
            )
        )
        return cls(reference)

    def reserve(
        self,
        *,
        checkpoint_identity: str,
        provider_attempt: int,
        prompt_bytes: int,
        output_reserve_tokens: int,
        recover_not_submitted: bool = False,
    ) -> BudgetReservation:
        if not checkpoint_identity:
            raise ValueError("checkpoint_identity is required")
        if type(provider_attempt) is not int or provider_attempt < 1:
            raise ValueError("provider_attempt must be positive")
        if type(prompt_bytes) is not int or prompt_bytes < 0:
            raise ValueError("prompt_bytes must be non-negative")
        if type(output_reserve_tokens) is not int or output_reserve_tokens < 0:
            raise ValueError("output_reserve_tokens must be finite and non-negative")
        reserved_tokens = math.ceil(prompt_bytes / 4) + output_reserve_tokens
        reservation_id = _reservation_id(
            self.reference.identity_sha256,
            checkpoint_identity,
            provider_attempt,
        )
        request_identity = _reservation_request_identity(
            checkpoint_identity, provider_attempt, reserved_tokens,
        )
        owner_pid = os.getpid()
        owner_started_at = _process_start_identity(owner_pid)
        with _connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["checkpoint_identity"] != checkpoint_identity
                    or int(existing["provider_attempt"]) != provider_attempt
                    or int(existing["reserved_tokens"]) != reserved_tokens
                    or existing["request_identity_sha256"] != request_identity
                ):
                    raise BudgetCorrupt("budget reservation identity changed")
                if recover_not_submitted:
                    self._recover_not_submitted_row(
                        db, existing, reservation_id,
                    )
                db.commit()
                return BudgetReservation(
                    self, reservation_id, checkpoint_identity,
                    provider_attempt, reserved_tokens,
                )
            snapshot = _snapshot(db)
            if (
                snapshot.charged_calls + snapshot.outstanding_calls + 1
                > snapshot.max_calls
                or snapshot.charged_tokens
                + snapshot.outstanding_tokens
                + reserved_tokens
                > snapshot.max_tokens
            ):
                raise BudgetExhausted("shared descendant LLM budget is exhausted")
            db.execute(
                "INSERT INTO reservations "
                "(reservation_id, checkpoint_identity, provider_attempt, "
                "request_identity_sha256, reserved_calls, reserved_tokens, state, "
                "submission_state, charged_calls, charged_tokens, owner_pid, "
                "owner_started_at, warning) "
                "VALUES (?, ?, ?, ?, 1, ?, 'reserved', 'not_submitted', "
                "0, 0, ?, ?, NULL)",
                (
                    reservation_id, checkpoint_identity, provider_attempt,
                    request_identity, reserved_tokens, owner_pid, owner_started_at,
                ),
            )
            db.commit()
        return BudgetReservation(
            self, reservation_id, checkpoint_identity,
            provider_attempt, reserved_tokens,
        )

    def recover_not_submitted(
        self, reservation_id: str,
    ) -> BudgetReservation:
        """Take over one durable, proven-unsubmitted provider reservation."""

        with _connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = _reservation(db, reservation_id)
            self._recover_not_submitted_row(db, row, reservation_id)
            refreshed = _reservation(db, reservation_id)
            db.commit()
        return BudgetReservation(
            self,
            reservation_id,
            str(refreshed["checkpoint_identity"]),
            int(refreshed["provider_attempt"]),
            int(refreshed["reserved_tokens"]),
        )

    def _recover_not_submitted_row(
        self,
        db: sqlite3.Connection,
        row: sqlite3.Row,
        reservation_id: str,
    ) -> None:
        if row["state"] == "reserved":
            if row["submission_state"] != "not_submitted":
                raise BudgetCorrupt(
                    "submitted reservation cannot be reused as unsubmitted"
                )
            alive = _reservation_owner_alive(row)
            if alive is True and int(row["owner_pid"]) != os.getpid():
                raise BudgetCorrupt(
                    "unsubmitted reservation still has a live owner"
                )
        elif (
            row["state"] == "released"
            and row["disposition"] == "proven_not_submitted"
        ):
            snapshot = _snapshot(db)
            if (
                snapshot.charged_calls + snapshot.outstanding_calls + 1
                > snapshot.max_calls
                or snapshot.charged_tokens
                + snapshot.outstanding_tokens
                + int(row["reserved_tokens"])
                > snapshot.max_tokens
            ):
                raise BudgetExhausted(
                    "shared descendant LLM budget is exhausted"
                )
        else:
            raise BudgetCorrupt(
                "provider checkpoint is unsubmitted but reservation is terminal"
            )
        owner_pid = os.getpid()
        db.execute(
            "UPDATE reservations SET state = 'reserved', "
            "submission_state = 'not_submitted', disposition = NULL, "
            "charged_calls = 0, charged_tokens = 0, warning = NULL, "
            "owner_pid = ?, owner_started_at = ? WHERE reservation_id = ?",
            (
                owner_pid,
                _process_start_identity(owner_pid),
                reservation_id,
            ),
        )

    def adopt(
        self,
        admission_reservation_id: str,
        *,
        checkpoint_identity: str,
        provider_attempt: int,
        prompt_bytes: int,
        output_reserve_tokens: int,
    ) -> BudgetReservation:
        """Atomically transfer a job admission to its first real provider call."""

        reserved_tokens = math.ceil(prompt_bytes / 4) + output_reserve_tokens
        reservation_id = _reservation_id(
            self.reference.identity_sha256,
            checkpoint_identity,
            provider_attempt,
        )
        request_identity = _reservation_request_identity(
            checkpoint_identity, provider_attempt, reserved_tokens,
        )
        with _connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            target = db.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            admission = db.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (admission_reservation_id,),
            ).fetchone()
            alias = db.execute(
                "SELECT target_reservation_id FROM reservation_aliases "
                "WHERE alias_reservation_id = ?",
                (admission_reservation_id,),
            ).fetchone()
            if target is not None:
                if (
                    target["checkpoint_identity"] != checkpoint_identity
                    or int(target["provider_attempt"]) != provider_attempt
                    or int(target["reserved_tokens"]) != reserved_tokens
                    or target["request_identity_sha256"] != request_identity
                ):
                    raise BudgetCorrupt("adopted budget reservation changed")
                if alias is not None and (
                    alias["target_reservation_id"] != reservation_id
                ):
                    raise BudgetCorrupt("first-call admission alias changed")
                if alias is None:
                    if admission is None:
                        raise BudgetCorrupt(
                            "first-call admission reservation is missing"
                        )
                    if admission["state"] == "reserved":
                        db.execute(
                            "UPDATE reservations SET state = 'released', "
                            "disposition = 'adopted', charged_calls = 0, "
                            "charged_tokens = 0, warning = NULL "
                            "WHERE reservation_id = ?",
                            (admission_reservation_id,),
                        )
                    elif (
                        admission["state"] != "released"
                        or admission["disposition"] != "adopted"
                    ):
                        raise BudgetCorrupt(
                            "first-call admission is not transferable"
                        )
                    db.execute(
                        "INSERT INTO reservation_aliases "
                        "(alias_reservation_id, target_reservation_id) "
                        "VALUES (?, ?)",
                        (admission_reservation_id, reservation_id),
                    )
                db.commit()
                return BudgetReservation(
                    self, reservation_id, checkpoint_identity,
                    provider_attempt, reserved_tokens,
                )
            if admission is None:
                raise BudgetCorrupt("first-call admission reservation is missing")
            if alias is not None:
                raise BudgetCorrupt("first-call admission alias target is missing")
            if (
                admission["state"] != "reserved"
                or admission["submission_state"] != "not_submitted"
            ):
                raise BudgetCorrupt(
                    "first-call admission is not transferable"
                )
            snapshot = _snapshot(db)
            prior_tokens = int(admission["reserved_tokens"])
            if (
                snapshot.charged_tokens
                + snapshot.outstanding_tokens
                - prior_tokens
                + reserved_tokens
                > snapshot.max_tokens
            ):
                raise BudgetExhausted(
                    "shared descendant LLM token budget cannot fit first call"
                )
            owner_pid = os.getpid()
            db.execute(
                "INSERT INTO reservations "
                "(reservation_id, checkpoint_identity, provider_attempt, "
                "request_identity_sha256, reserved_calls, reserved_tokens, state, "
                "submission_state, charged_calls, charged_tokens, owner_pid, "
                "owner_started_at, warning) "
                "VALUES (?, ?, ?, ?, 1, ?, 'reserved', 'not_submitted', "
                "0, 0, ?, ?, NULL)",
                (
                    reservation_id,
                    checkpoint_identity,
                    provider_attempt,
                    request_identity,
                    reserved_tokens,
                    owner_pid,
                    _process_start_identity(owner_pid),
                ),
            )
            db.execute(
                "UPDATE reservations SET state = 'released', "
                "disposition = 'adopted', charged_calls = 0, "
                "charged_tokens = 0, warning = NULL "
                "WHERE reservation_id = ?",
                (admission_reservation_id,),
            )
            db.execute(
                "INSERT INTO reservation_aliases "
                "(alias_reservation_id, target_reservation_id) VALUES (?, ?)",
                (admission_reservation_id, reservation_id),
            )
            db.commit()
        return BudgetReservation(
            self, reservation_id, checkpoint_identity,
            provider_attempt, reserved_tokens,
        )

    def lookup(
        self,
        *,
        checkpoint_identity: str,
        provider_attempt: int,
    ) -> BudgetReservation | None:
        reservation_id = _reservation_id(
            self.reference.identity_sha256,
            checkpoint_identity,
            provider_attempt,
        )
        row = self.reservation_or_none(reservation_id)
        if row is None:
            return None
        return BudgetReservation(
            self,
            reservation_id,
            checkpoint_identity,
            provider_attempt,
            int(row["reserved_tokens"]),
        )

    def mark_submitted(self, reservation_id: str) -> None:
        with _connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = _reservation(db, reservation_id)
            if row["state"] == "reserved":
                db.execute(
                    "UPDATE reservations SET submission_state = 'submitted' "
                    "WHERE reservation_id = ?",
                    (reservation_id,),
                )
            db.commit()

    def settle_known(
        self,
        reservation_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> BudgetSettlement:
        if min(input_tokens, output_tokens) < 0:
            raise ValueError("known usage tokens must be non-negative")
        return self._settle(
            reservation_id,
            disposition="known_usage",
            charged_calls=1,
            charged_tokens=input_tokens + output_tokens,
        )

    def settle_conservative(
        self, reservation_id: str, disposition: str,
    ) -> BudgetSettlement:
        with _connect(self.path) as db:
            row = _reservation(db, reservation_id)
        return self._settle(
            reservation_id,
            disposition=disposition,
            charged_calls=1,
            charged_tokens=int(row["reserved_tokens"]),
        )

    def release_not_submitted(self, reservation_id: str) -> BudgetSettlement:
        with _connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = _reservation(db, reservation_id)
            if row["state"] in {"settled", "released"}:
                db.commit()
                return BudgetSettlement(
                    reservation_id,
                    str(row["disposition"]),
                    int(row["charged_calls"]),
                    int(row["charged_tokens"]),
                    str(row["warning"]) if row["warning"] else None,
                )
            if row["submission_state"] == "not_submitted":
                disposition = "proven_not_submitted"
                charged_calls = 0
                charged_tokens = 0
                state = "released"
                warning = None
            else:
                disposition = (
                    f"release_denied_{row['submission_state']}"
                )
                charged_calls = 1
                charged_tokens = int(row["reserved_tokens"])
                state = "settled"
                warning = "budget.submitted_reservation_charged"
            db.execute(
                "UPDATE reservations SET state = ?, disposition = ?, "
                "charged_calls = ?, charged_tokens = ?, warning = ? "
                "WHERE reservation_id = ?",
                (
                    state, disposition, charged_calls, charged_tokens,
                    warning, reservation_id,
                ),
            )
            db.commit()
        return BudgetSettlement(
            reservation_id,
            disposition,
            charged_calls,
            charged_tokens,
            warning,
        )

    def reconcile(
        self,
        reservation_id: str,
        *,
        checkpoint_submission_state: str,
        usage: Mapping[str, Any] | None = None,
        owner_alive: bool | None = None,
    ) -> BudgetSettlement:
        persisted = self._persisted_reconciliation(
            reservation_id,
            checkpoint_submission_state=checkpoint_submission_state,
            usage=usage,
        )
        if persisted is not None:
            return persisted
        if owner_alive is None:
            with _connect(self.path) as db:
                owner_alive = _reservation_owner_alive(
                    _reservation(db, reservation_id)
                )
        if usage is not None:
            return self.settle_known(
                reservation_id,
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
            )
        if checkpoint_submission_state in {"submitted", "unknown", "response_received"}:
            return self.settle_conservative(
                reservation_id, f"reconciled_{checkpoint_submission_state}",
            )
        if checkpoint_submission_state == "not_submitted" and owner_alive is False:
            return self.release_not_submitted(reservation_id)
        return self._settle(
            reservation_id,
            disposition="reconciled_ambiguous",
            charged_calls=1,
            charged_tokens=self._reserved_tokens(reservation_id),
            warning="budget.reservation_ambiguous_charged",
        )

    def _persisted_reconciliation(
        self,
        reservation_id: str,
        *,
        checkpoint_submission_state: str,
        usage: Mapping[str, Any] | None,
    ) -> BudgetSettlement | None:
        with _connect(self.path) as db:
            row = _reservation(db, reservation_id)
        if row["state"] not in {"settled", "released"}:
            return None
        settlement = BudgetSettlement(
            reservation_id,
            str(row["disposition"]),
            int(row["charged_calls"]),
            int(row["charged_tokens"]),
            str(row["warning"]) if row["warning"] else None,
        )
        response_exists = checkpoint_submission_state in {
            "submitted", "unknown", "response_received",
        }
        if response_exists and settlement.charged_calls < 1:
            raise BudgetCorrupt(
                "persisted zero-charge settlement contradicts submitted checkpoint"
            )
        if usage is not None:
            observed_tokens = (
                int(usage.get("input_tokens") or 0)
                + int(usage.get("output_tokens") or 0)
            )
            if (
                observed_tokens < 0
                or settlement.charged_calls != 1
                or settlement.charged_tokens < observed_tokens
            ):
                raise BudgetCorrupt(
                    "persisted budget settlement contradicts replay usage"
                )
        return settlement

    def snapshot(self) -> BudgetSnapshot:
        with _connect(self.path) as db:
            return _snapshot(db)

    def reservation(self, reservation_id: str) -> dict[str, Any]:
        with _connect(self.path) as db:
            return dict(_reservation(db, reservation_id))

    def reservation_or_none(self, reservation_id: str) -> dict[str, Any] | None:
        with _connect(self.path) as db:
            row = db.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            return dict(row) if row is not None else None

    def adopted_target(self, admission_reservation_id: str) -> str | None:
        with _connect(self.path) as db:
            row = db.execute(
                "SELECT target_reservation_id FROM reservation_aliases "
                "WHERE alias_reservation_id = ?",
                (admission_reservation_id,),
            ).fetchone()
        return str(row["target_reservation_id"]) if row is not None else None

    def bind_reservation_context(
        self,
        reservation_id: str,
        *,
        parent_admission_id: str,
        checkpoint_path: Path,
    ) -> None:
        canonical_path = checkpoint_path.expanduser().resolve(strict=False)
        with _connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            _reservation(db, reservation_id)
            existing = db.execute(
                "SELECT parent_admission_id, checkpoint_path "
                "FROM reservation_context WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            expected = (parent_admission_id, str(canonical_path))
            if existing is None:
                db.execute(
                    "INSERT INTO reservation_context "
                    "(reservation_id, parent_admission_id, checkpoint_path) "
                    "VALUES (?, ?, ?)",
                    (reservation_id, *expected),
                )
            elif tuple(existing) != expected:
                raise BudgetCorrupt("budget reservation recovery context changed")
            db.commit()

    def descendant_reservations(
        self, parent_admission_id: str,
    ) -> list[dict[str, Any]]:
        with _connect(self.path) as db:
            rows = db.execute(
                "SELECT r.*, c.checkpoint_path "
                "FROM reservation_context c JOIN reservations r "
                "ON r.reservation_id = c.reservation_id "
                "WHERE c.parent_admission_id = ? "
                "ORDER BY r.checkpoint_identity, r.provider_attempt",
                (parent_admission_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _settle(
        self,
        reservation_id: str,
        *,
        disposition: str,
        charged_calls: int,
        charged_tokens: int,
        warning: str | None = None,
    ) -> BudgetSettlement:
        with _connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = _reservation(db, reservation_id)
            if row["state"] in {"settled", "released"}:
                existing = BudgetSettlement(
                    reservation_id,
                    str(row["disposition"]),
                    int(row["charged_calls"]),
                    int(row["charged_tokens"]),
                    str(row["warning"]) if row["warning"] else None,
                )
                if existing != BudgetSettlement(
                    reservation_id, disposition, charged_calls, charged_tokens, warning,
                ):
                    raise BudgetCorrupt("budget reservation settlement changed")
                db.commit()
                return existing
            state = "released" if charged_calls == 0 and charged_tokens == 0 else "settled"
            db.execute(
                "UPDATE reservations SET state = ?, disposition = ?, "
                "charged_calls = ?, charged_tokens = ?, warning = ? "
                "WHERE reservation_id = ?",
                (
                    state, disposition, charged_calls, charged_tokens,
                    warning, reservation_id,
                ),
            )
            db.commit()
        return BudgetSettlement(
            reservation_id, disposition, charged_calls, charged_tokens, warning,
        )

    def _reserved_tokens(self, reservation_id: str) -> int:
        with _connect(self.path) as db:
            return int(_reservation(db, reservation_id)["reserved_tokens"])

    def _verify_metadata(self) -> None:
        with _connect(self.path) as db:
            _initialize(db)
            row = db.execute(
                "SELECT schema_version, budget_id, identity_sha256, "
                "max_calls, max_tokens "
                "FROM budget_metadata WHERE singleton = 1"
            ).fetchone()
        metadata_identity = (
            _hash({
                "schema_version": str(row["schema_version"]),
                "budget_id": str(row["budget_id"]),
                "max_calls": int(row["max_calls"]),
                "max_tokens": int(row["max_tokens"]),
            })
            if row is not None else ""
        )
        if (
            row is None
            or row["schema_version"] != BUDGET_SCHEMA_VERSION
            or row["budget_id"] != self.reference.budget_id
            or row["identity_sha256"] != self.reference.identity_sha256
            or metadata_identity != self.reference.identity_sha256
        ):
            raise BudgetCorrupt("shared budget reference does not match metadata")
        _secure_sqlite(self.path)


def current_shared_budget_binding(
    *, required: bool = False,
) -> SharedBudgetBinding | None:
    binding = _CURRENT_BUDGET.get()
    if binding is None and required:
        raise BudgetRequired("managed child LLM call requires a finite parent budget")
    return binding


def current_shared_budget(*, required: bool = False) -> SharedBudget | None:
    binding = current_shared_budget_binding(required=required)
    return binding.budget if binding is not None else None


@contextmanager
def shared_budget_context(
    budget: SharedBudget,
    *,
    output_reserve_tokens: int,
    admission_reservation_id: str | None = None,
) -> Iterator[None]:
    if type(output_reserve_tokens) is not int or output_reserve_tokens < 0:
        raise ValueError("output_reserve_tokens must be finite and non-negative")
    token = _CURRENT_BUDGET.set(
        SharedBudgetBinding(
            budget,
            output_reserve_tokens,
            admission_reservation_id=admission_reservation_id,
        ),
    )
    try:
        yield
    finally:
        _CURRENT_BUDGET.reset(token)


def _connect(path: Path) -> sqlite3.Connection:
    if not path.parent.exists():
        raise BudgetCorrupt("shared budget parent directory is missing")
    db = sqlite3.connect(path, timeout=30.0)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=30000")
    # Two cooperating callers may create the same deterministic ledger at
    # once. SQLite's first WAL-mode transition can briefly reject the second
    # PRAGMA without honoring busy_timeout, so retry only that initialization
    # boundary and leave all transactional contention to SQLite.
    deadline = time.monotonic() + 30.0
    while True:
        try:
            db.execute("PRAGMA journal_mode=WAL")
            break
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).casefold() or time.monotonic() >= deadline:
                db.close()
                raise
            time.sleep(0.01)
    db.execute("PRAGMA synchronous=FULL")
    _secure_sqlite(path)
    return db


def _initialize(db: sqlite3.Connection) -> None:
    db.execute(
        "CREATE TABLE IF NOT EXISTS budget_metadata ("
        "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
        "schema_version TEXT NOT NULL, budget_id TEXT NOT NULL, "
        "identity_sha256 TEXT NOT NULL, max_calls INTEGER NOT NULL, "
        "max_tokens INTEGER NOT NULL)"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS reservations ("
        "reservation_id TEXT PRIMARY KEY, checkpoint_identity TEXT NOT NULL, "
        "provider_attempt INTEGER NOT NULL, request_identity_sha256 TEXT NOT NULL, "
        "reserved_calls INTEGER NOT NULL, reserved_tokens INTEGER NOT NULL, "
        "state TEXT NOT NULL, submission_state TEXT NOT NULL, "
        "disposition TEXT, charged_calls INTEGER NOT NULL, "
        "charged_tokens INTEGER NOT NULL, owner_pid INTEGER NOT NULL, "
        "owner_started_at TEXT NOT NULL, warning TEXT)"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS reservation_aliases ("
        "alias_reservation_id TEXT PRIMARY KEY, "
        "target_reservation_id TEXT NOT NULL UNIQUE, "
        "FOREIGN KEY(alias_reservation_id) REFERENCES reservations(reservation_id), "
        "FOREIGN KEY(target_reservation_id) REFERENCES reservations(reservation_id))"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS reservation_context ("
        "reservation_id TEXT PRIMARY KEY, "
        "parent_admission_id TEXT NOT NULL, checkpoint_path TEXT NOT NULL, "
        "FOREIGN KEY(reservation_id) REFERENCES reservations(reservation_id))"
    )


def _snapshot(db: sqlite3.Connection) -> BudgetSnapshot:
    metadata = db.execute(
        "SELECT max_calls, max_tokens FROM budget_metadata WHERE singleton = 1"
    ).fetchone()
    if metadata is None:
        raise BudgetCorrupt("shared budget metadata is missing")
    totals = db.execute(
        "SELECT "
        "COALESCE(SUM(CASE WHEN state = 'settled' THEN charged_calls ELSE 0 END), 0), "
        "COALESCE(SUM(CASE WHEN state = 'settled' THEN charged_tokens ELSE 0 END), 0), "
        "COALESCE(SUM(CASE WHEN state = 'reserved' THEN reserved_calls ELSE 0 END), 0), "
        "COALESCE(SUM(CASE WHEN state = 'reserved' THEN reserved_tokens ELSE 0 END), 0) "
        "FROM reservations"
    ).fetchone()
    return BudgetSnapshot(
        int(metadata["max_calls"]),
        int(metadata["max_tokens"]),
        *(int(value) for value in totals),
    )


def _reservation(db: sqlite3.Connection, reservation_id: str) -> sqlite3.Row:
    row = db.execute(
        "SELECT * FROM reservations WHERE reservation_id = ?",
        (reservation_id,),
    ).fetchone()
    if row is None:
        raise BudgetCorrupt("shared budget reservation is missing")
    return row


def _hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reservation_id(
    budget_identity_sha256: str,
    checkpoint_identity: str,
    provider_attempt: int,
) -> str:
    return _hash({
        "budget_identity_sha256": budget_identity_sha256,
        "checkpoint_identity": checkpoint_identity,
        "provider_attempt": provider_attempt,
    })


def _reservation_request_identity(
    checkpoint_identity: str,
    provider_attempt: int,
    reserved_tokens: int,
) -> str:
    return _hash({
        "checkpoint_identity": checkpoint_identity,
        "provider_attempt": provider_attempt,
        "reserved_tokens": reserved_tokens,
    })


def _process_start_identity(pid: int) -> str:
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return stat_text.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return "unknown"


def _reservation_owner_alive(row: Mapping[str, Any]) -> bool | None:
    pid = int(row["owner_pid"])
    expected = str(row["owner_started_at"])
    current = _process_start_identity(pid)
    if current == "unknown":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return None
        return None if expected != "unknown" else True
    return expected == current


def _secure_sqlite(path: Path) -> None:
    if path.parent.exists():
        os.chmod(path.parent, 0o700)
    for candidate in path.parent.glob(f"{path.name}*"):
        try:
            mode = stat.S_IMODE(candidate.stat().st_mode)
            if mode != 0o600:
                os.chmod(candidate, 0o600)
        except OSError:
            continue
