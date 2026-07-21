from __future__ import annotations

import contextlib
import hashlib
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping
from urllib.parse import urlsplit, urlunsplit

from .paths import llm_cache_root
from .providers.base import (
    LLMAbortScope,
    LLMFailureCategory,
    LLMSubmissionState,
    LLMWorkerCancelled,
    LLMWorkerError,
    failure_disposition,
)


GLOBAL_MAX_CONCURRENCY = 24
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 15 * 60.0
DEFAULT_SLOT_LEASE_SECONDS = 300.0
DEFAULT_HEARTBEAT_SECONDS = 5.0

_PERSISTENT_CIRCUIT_CATEGORIES = {
    LLMFailureCategory.QUOTA,
    LLMFailureCategory.AUTHENTICATION,
    LLMFailureCategory.PERMISSION,
}


class LLMSafetyConfigurationError(LLMWorkerError):
    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            category=LLMFailureCategory.INVALID_REQUEST,
            abort_scope=LLMAbortScope.BATCH,
            submission_state=LLMSubmissionState.NOT_SUBMITTED,
        )


class LLMCircuitOpen(LLMWorkerError):
    def __init__(
        self,
        message: str,
        *,
        category: LLMFailureCategory,
        retry_after_seconds: float | None,
    ) -> None:
        super().__init__(
            message,
            category=category,
            abort_scope=LLMAbortScope.PROVIDER,
            submission_state=LLMSubmissionState.NOT_SUBMITTED,
            retry_after_seconds=retry_after_seconds,
        )


@dataclass(frozen=True)
class CircuitPermit:
    provider_key: str
    probe_token: str | None = None


class GlobalLLMSlot:
    def __init__(self, controller: "LLMSafetyController", token: str) -> None:
        self.controller = controller
        self.token = token
        self._stop = threading.Event()
        self._released = False
        self._thread = threading.Thread(target=self._heartbeat, name="arc-llm-slot-heartbeat", daemon=True)
        self._thread.start()

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.controller.heartbeat_seconds):
            self.controller._heartbeat_slot(self.token)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.controller.heartbeat_seconds * 2))
        # A provider may already have returned a paid response. Do not turn a
        # post-response SQLite cleanup failure into a retry that duplicates it;
        # the still-owned slot fails closed until this process exits.
        with contextlib.suppress(sqlite3.Error):
            self.controller._release_slot(self.token)

    def __enter__(self) -> "GlobalLLMSlot":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


@dataclass
class LLMCallPermit:
    controller: "LLMSafetyController"
    provider: str
    endpoint: str | None
    slot: GlobalLLMSlot
    circuit: CircuitPermit

    def report_success(self) -> None:
        self.controller.report_success(self.provider, endpoint=self.endpoint, permit=self.circuit)

    def report_failure(self, exc: BaseException) -> None:
        self.controller.report_failure(self.provider, exc, endpoint=self.endpoint, permit=self.circuit)

    def release(self) -> None:
        self.slot.release()

    def __enter__(self) -> "LLMCallPermit":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if exc is not None:
                self.report_failure(exc)
        finally:
            self.release()


class LLMSafetyController:
    """ARC_HOME-wide concurrency and provider-circuit coordination.

    Provider adapters should acquire this immediately before spawning the real
    provider process. The permit must remain held until that process tree is
    fully reaped.
    """

    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        db_path: str | Path | None = None,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        slot_lease_seconds: float = DEFAULT_SLOT_LEASE_SECONDS,
    ) -> None:
        self.env = dict(os.environ if env is None else env)
        self.db_path = Path(db_path) if db_path is not None else llm_cache_root(self.env) / "safety.sqlite3"
        self.now = now
        self.sleep = sleep
        self.heartbeat_seconds = heartbeat_seconds
        self.slot_lease_seconds = slot_lease_seconds
        self._initialize()

    def effective_max_concurrency(self, provider: str | None = None) -> int:
        global_limit = self._limit_from_env("ARC_LLM_MAX_CONCURRENCY") or GLOBAL_MAX_CONCURRENCY
        provider_limit = self._limit_from_env(_provider_limit_key(provider))
        return min(global_limit, provider_limit) if provider_limit is not None else global_limit

    def _limit_from_env(self, key: str | None) -> int | None:
        if not key or not self.env.get(key):
            return None
        try:
            value = int(str(self.env[key]).strip())
        except ValueError as exc:
            raise LLMSafetyConfigurationError(f"{key} must be an integer from 1 to 24") from exc
        if not 1 <= value <= GLOBAL_MAX_CONCURRENCY:
            raise LLMSafetyConfigurationError(f"{key} must be an integer from 1 to 24")
        return value

    def acquire_slot(
        self,
        provider: str,
        *,
        timeout_seconds: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
        call_label: str | None = None,
    ) -> GlobalLLMSlot:
        global_limit = self.effective_max_concurrency()
        provider_limit = self.effective_max_concurrency(provider)
        started = time.monotonic()
        token = uuid.uuid4().hex
        pid = os.getpid()
        pid_start = _process_start_identity(pid)
        now = self.now()
        with self._transaction() as connection:
            self._purge_stale_owners(connection, now)
            connection.execute(
                "INSERT INTO waiters(token,pid,pid_start,provider,queued_at,heartbeat_at) VALUES(?,?,?,?,?,?)",
                (token, pid, pid_start, provider, now, now),
            )
        acquired = False
        try:
            while True:
                if cancel_check is not None and cancel_check():
                    raise LLMWorkerCancelled("LLM call cancelled while waiting for global capacity")
                now = self.now()
                with self._transaction() as connection:
                    self._purge_stale_owners(connection, now)
                    connection.execute("UPDATE waiters SET heartbeat_at=? WHERE token=?", (now, token))
                    active = int(connection.execute("SELECT COUNT(*) FROM slots").fetchone()[0])
                    active_provider = int(
                        connection.execute("SELECT COUNT(*) FROM slots WHERE provider=?", (provider,)).fetchone()[0]
                    )
                    if active < global_limit and active_provider < provider_limit:
                        connection.execute(
                            "INSERT INTO slots(token,pid,pid_start,provider,call_label,acquired_at,heartbeat_at) "
                            "VALUES(?,?,?,?,?,?,?)",
                            # Labels can originate in user material. They are not needed for
                            # coordination, so never persist them in the shared database.
                            (token, pid, pid_start, provider, None, now, now),
                        )
                        connection.execute("DELETE FROM waiters WHERE token=?", (token,))
                        acquired = True
                        return GlobalLLMSlot(self, token)
                if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
                    raise LLMWorkerError(
                        "timed out waiting for ARC-wide LLM capacity",
                        category=LLMFailureCategory.TIMEOUT,
                        submission_state=LLMSubmissionState.NOT_SUBMITTED,
                    )
                self.sleep(0.05)
        finally:
            if not acquired:
                with contextlib.suppress(sqlite3.Error):
                    with self._transaction() as connection:
                        connection.execute("DELETE FROM waiters WHERE token=?", (token,))

    def acquire_call(
        self,
        provider: str,
        *,
        endpoint: str | None = None,
        timeout_seconds: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
        call_label: str | None = None,
    ) -> LLMCallPermit:
        # Fast rejection avoids waiting behind active work for a known-open provider.
        self.check_circuit(provider, endpoint=endpoint, reserve_probe=False)
        slot = self.acquire_slot(
            provider,
            timeout_seconds=timeout_seconds,
            cancel_check=cancel_check,
            call_label=call_label,
        )
        try:
            circuit = self.check_circuit(provider, endpoint=endpoint, reserve_probe=True)
        except BaseException:
            slot.release()
            raise
        return LLMCallPermit(self, provider, endpoint, slot, circuit)

    def check_circuit(
        self,
        provider: str,
        *,
        endpoint: str | None = None,
        reserve_probe: bool = True,
    ) -> CircuitPermit:
        key, normalized = _provider_key(provider, endpoint)
        now = self.now()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT category,reason,cooldown_until,probe_token,probe_started_at,probe_pid,probe_pid_start "
                "FROM circuits WHERE provider_key=?",
                (key,),
            ).fetchone()
            if row is None:
                return CircuitPermit(key)
            category = LLMFailureCategory(row[0])
            reason = str(row[1] or category.value)
            cooldown_until = float(row[2]) if row[2] is not None else None
            if category in _PERSISTENT_CIRCUIT_CATEGORIES or cooldown_until is None:
                raise LLMCircuitOpen(reason, category=category, retry_after_seconds=None)
            remaining = cooldown_until - now
            if remaining > 0:
                raise LLMCircuitOpen(reason, category=category, retry_after_seconds=remaining)
            probe_token = row[3]
            probe_started = float(row[4]) if row[4] is not None else None
            probe_pid = int(row[5]) if row[5] is not None else None
            probe_pid_start = row[6]
            if probe_token and probe_pid is not None and _owner_alive(probe_pid, probe_pid_start):
                raise LLMCircuitOpen(
                    f"{reason}; half-open probe already active",
                    category=category,
                    retry_after_seconds=None,
                )
            if not reserve_probe:
                return CircuitPermit(key)
            token = uuid.uuid4().hex
            pid = os.getpid()
            connection.execute(
                "UPDATE circuits SET provider=?,endpoint=?,probe_token=?,probe_started_at=?,probe_pid=?,probe_pid_start=? "
                "WHERE provider_key=?",
                (provider, normalized, token, now, pid, _process_start_identity(pid), key),
            )
            return CircuitPermit(key, token)

    def report_success(
        self,
        provider: str,
        *,
        endpoint: str | None = None,
        permit: CircuitPermit | None = None,
    ) -> None:
        key, _ = _provider_key(provider, endpoint)
        # Success is reported after the paid provider response exists. Failing
        # closed (leaving a circuit open) is safer than throwing and causing the
        # caller to submit the same request again.
        with contextlib.suppress(sqlite3.Error):
            with self._transaction() as connection:
                if permit is not None and permit.probe_token:
                    connection.execute(
                        "DELETE FROM circuits WHERE provider_key=? AND probe_token=?",
                        (key, permit.probe_token),
                    )
                # A normal call may have been admitted before a concurrent call
                # opened the circuit. Its later success must not erase that newer
                # 429 signal. Only the uniquely reserved half-open probe can close
                # a transient circuit.

    def report_failure(
        self,
        provider: str,
        exc: BaseException,
        *,
        endpoint: str | None = None,
        permit: CircuitPermit | None = None,
    ) -> None:
        disposition = failure_disposition(exc)
        if disposition is None:
            return
        if disposition.category not in _PERSISTENT_CIRCUIT_CATEGORIES | {LLMFailureCategory.RATE_LIMIT}:
            return
        key, normalized = _provider_key(provider, endpoint)
        now = self.now()
        cooldown_until = None
        if disposition.category == LLMFailureCategory.RATE_LIMIT:
            cooldown = max(DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS, disposition.retry_after_seconds or 0.0)
            cooldown_until = now + cooldown
        # Provider exceptions can echo commands, credentials, or response bodies.
        # Persist only a stable diagnostic category in ARC_HOME.
        reason = f"{provider} provider circuit opened: {disposition.category.value}"
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO circuits(provider_key,provider,endpoint,category,reason,opened_at,cooldown_until,probe_token,probe_started_at,probe_pid,probe_pid_start) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(provider_key) DO UPDATE SET "
                "category=excluded.category,reason=excluded.reason,opened_at=excluded.opened_at,"
                "cooldown_until=excluded.cooldown_until,probe_token=NULL,probe_started_at=NULL,probe_pid=NULL,probe_pid_start=NULL",
                (key, provider, normalized, disposition.category.value, reason, now, cooldown_until, None, None, None, None),
            )

    def reset_circuit(self, provider: str | None = None, *, endpoint: str | None = None) -> int:
        with self._transaction() as connection:
            if provider is None:
                cursor = connection.execute("DELETE FROM circuits")
            elif endpoint is None:
                cursor = connection.execute("DELETE FROM circuits WHERE provider=?", (provider,))
            else:
                key, _ = _provider_key(provider, endpoint)
                cursor = connection.execute("DELETE FROM circuits WHERE provider_key=?", (key,))
            return max(0, cursor.rowcount)

    def status(self) -> dict[str, object]:
        now = self.now()
        with self._transaction() as connection:
            self._purge_stale_owners(connection, now)
            slots = connection.execute(
                "SELECT provider,COUNT(*) FROM slots GROUP BY provider ORDER BY provider"
            ).fetchall()
            circuits = connection.execute(
                "SELECT provider,endpoint,category,reason,opened_at,cooldown_until,probe_token "
                "FROM circuits ORDER BY provider,endpoint"
            ).fetchall()
            waiters = connection.execute(
                "SELECT provider,COUNT(*) FROM waiters GROUP BY provider ORDER BY provider"
            ).fetchall()
        return {
            "schema_version": "arc.llm.safety_status.v1",
            "max_concurrency": self.effective_max_concurrency(),
            "active_slots": sum(int(row[1]) for row in slots),
            "active_by_provider": {str(row[0]): int(row[1]) for row in slots},
            "queued_calls": sum(int(row[1]) for row in waiters),
            "queued_by_provider": {str(row[0]): int(row[1]) for row in waiters},
            "circuits": [
                {
                    "provider": row[0],
                    "endpoint": row[1] or None,
                    "category": row[2],
                    "reason": row[3],
                    "opened_at": row[4],
                    "cooldown_until": row[5],
                    "half_open_probe_active": bool(row[6]),
                }
                for row in circuits
            ],
        }

    def _initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS slots(
                    token TEXT PRIMARY KEY,
                    pid INTEGER NOT NULL,
                    pid_start TEXT,
                    provider TEXT NOT NULL,
                    call_label TEXT,
                    acquired_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS circuits(
                    provider_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    category TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    opened_at REAL NOT NULL,
                    cooldown_until REAL,
                    probe_token TEXT,
                    probe_started_at REAL,
                    probe_pid INTEGER,
                    probe_pid_start TEXT
                );
                CREATE TABLE IF NOT EXISTS waiters(
                    token TEXT PRIMARY KEY,
                    pid INTEGER NOT NULL,
                    pid_start TEXT,
                    provider TEXT NOT NULL,
                    queued_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL
                );
                """
            )
            circuit_columns = {row[1] for row in connection.execute("PRAGMA table_info(circuits)")}
            if "probe_pid" not in circuit_columns:
                connection.execute("ALTER TABLE circuits ADD COLUMN probe_pid INTEGER")
            if "probe_pid_start" not in circuit_columns:
                connection.execute("ALTER TABLE circuits ADD COLUMN probe_pid_start TEXT")
        with contextlib.suppress(OSError):
            self.db_path.chmod(0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _purge_stale_owners(self, connection: sqlite3.Connection, now: float) -> None:
        del now
        for table in ("slots", "waiters"):
            rows = connection.execute(f"SELECT token,pid,pid_start FROM {table}").fetchall()
            for token, pid, pid_start in rows:
                if not _owner_alive(int(pid), pid_start):
                    connection.execute(f"DELETE FROM {table} WHERE token=?", (token,))

    def _heartbeat_slot(self, token: str) -> None:
        with contextlib.suppress(sqlite3.Error):
            with self._transaction() as connection:
                connection.execute("UPDATE slots SET heartbeat_at=? WHERE token=?", (self.now(), token))

    def _release_slot(self, token: str) -> None:
        with self._transaction() as connection:
            connection.execute("DELETE FROM slots WHERE token=?", (token,))


def _provider_limit_key(provider: str | None) -> str | None:
    if not provider:
        return None
    normalized = provider.upper().replace("-CLI", "").replace("-CODE", "").replace("-", "_")
    return f"ARC_{normalized}_MAX_CONCURRENCY"


def _provider_key(provider: str, endpoint: str | None) -> tuple[str, str]:
    normalized_endpoint = _normalize_endpoint(endpoint)
    material = f"{provider.strip().lower()}\0{normalized_endpoint}".encode()
    return hashlib.sha256(material).hexdigest(), normalized_endpoint


def _normalize_endpoint(endpoint: str | None) -> str:
    if not endpoint:
        return ""
    value = endpoint.strip()
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        # Opaque endpoint identifiers might themselves contain credentials.
        return f"opaque:{hashlib.sha256(value.encode()).hexdigest()[:16]}"
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((parsed.scheme.lower(), f"{hostname.lower()}{port}", "", "", ""))


def _process_start_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return fields[21]
    except (OSError, IndexError):
        return None


def _owner_alive(pid: int, expected_start: str | None) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if expected_start is None:
        return True
    actual = _process_start_identity(pid)
    return actual is None or actual == expected_start
