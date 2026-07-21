from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping
from urllib.parse import urlsplit, urlunsplit

if os.name == "nt":  # pragma: no cover - exercised through the portable helper tests
    import msvcrt as _msvcrt

    _fcntl = None
else:  # pragma: no branch - one implementation is selected at import time
    import fcntl as _fcntl

    _msvcrt = None

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
DEFAULT_PROVIDER_FAILURE_COOLDOWN_SECONDS = 15 * 60.0
DEFAULT_SLOT_LEASE_SECONDS = 300.0
DEFAULT_HEARTBEAT_SECONDS = 5.0

_PROVIDER_CIRCUIT_CATEGORIES = {
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
    probe_handle: object | None = None


class GlobalLLMSlot:
    def __init__(
        self,
        controller: "LLMSafetyController",
        token: str,
        handles: tuple[object, object],
    ) -> None:
        self.controller = controller
        self.token = token
        self._handles = handles
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        # A provider may already have returned a paid response. Do not turn a
        # post-response SQLite cleanup failure into a retry that duplicates it;
        # the still-owned slot fails closed until this process exits.
        with contextlib.suppress(sqlite3.Error):
            self.controller._release_slot(self.token)
        for handle in reversed(self._handles):
            _advisory_unlock(handle)
            with contextlib.suppress(OSError):
                handle.close()

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
        now = self.now()
        waiter_handle = self._open_locked(self._waiter_lock_path(token))
        with self._transaction() as connection:
            self._reconcile_lock_rows(connection)
            connection.execute(
                "INSERT INTO waiters(token,pid,pid_start,provider,queued_at,heartbeat_at,lock_path) "
                "VALUES(?,?,?,?,?,?,?)",
                (token, 0, None, provider, now, now, str(self._waiter_lock_path(token))),
            )
        acquired = False
        try:
            while True:
                if cancel_check is not None and cancel_check():
                    raise LLMWorkerCancelled("LLM call cancelled while waiting for global capacity")
                now = self.now()
                lock_pair = self._try_slot_locks(provider, global_limit, provider_limit)
                if lock_pair is not None:
                    global_index, global_handle, provider_index, provider_handle = lock_pair
                    try:
                        with self._transaction() as connection:
                            self._reconcile_lock_rows(connection)
                            connection.execute("DELETE FROM slots WHERE global_lock_index=?", (global_index,))
                            connection.execute(
                                "DELETE FROM slots WHERE provider=? AND provider_lock_index=?",
                                (provider, provider_index),
                            )
                            connection.execute(
                                "INSERT INTO slots(token,pid,pid_start,provider,call_label,acquired_at,heartbeat_at,"
                                "global_lock_index,provider_lock_index) VALUES(?,?,?,?,?,?,?,?,?)",
                                (token, 0, None, provider, None, now, now, global_index, provider_index),
                            )
                            connection.execute("DELETE FROM waiters WHERE token=?", (token,))
                    except BaseException:
                        self._unlock_close(provider_handle)
                        self._unlock_close(global_handle)
                        raise
                    self._unlock_close(waiter_handle)
                    acquired = True
                    return GlobalLLMSlot(self, token, (global_handle, provider_handle))
                if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
                    raise LLMWorkerError(
                        "timed out waiting for ARC-wide LLM capacity",
                        category=LLMFailureCategory.TIMEOUT,
                        submission_state=LLMSubmissionState.NOT_SUBMITTED,
                    )
                self.sleep(0.05)
        finally:
            if not acquired:
                self._unlock_close(waiter_handle)
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
            if cooldown_until is None:
                # Migrate old indefinitely-open provider circuits to a bounded
                # window measured from their recorded open time.
                opened_at = connection.execute(
                    "SELECT opened_at FROM circuits WHERE provider_key=?", (key,)
                ).fetchone()[0]
                cooldown_until = float(opened_at) + DEFAULT_PROVIDER_FAILURE_COOLDOWN_SECONDS
                connection.execute(
                    "UPDATE circuits SET cooldown_until=? WHERE provider_key=?",
                    (cooldown_until, key),
                )
            remaining = cooldown_until - now
            if remaining > 0:
                raise LLMCircuitOpen(reason, category=category, retry_after_seconds=remaining)
            probe_handle = self._try_lock(self._probe_lock_path(key))
            if probe_handle is None:
                raise LLMCircuitOpen(
                    f"{reason}; half-open probe already active",
                    category=category,
                    retry_after_seconds=None,
                )
            if not reserve_probe:
                self._unlock_close(probe_handle)
                return CircuitPermit(key)
            token = uuid.uuid4().hex
            connection.execute(
                "UPDATE circuits SET provider=?,endpoint=?,probe_token=?,probe_started_at=?,probe_pid=?,probe_pid_start=? "
                "WHERE provider_key=?",
                (provider, normalized, token, now, None, None, key),
            )
            return CircuitPermit(key, token, probe_handle)

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
        if permit is not None and permit.probe_handle is not None:
            self._unlock_close(permit.probe_handle)

    def report_failure(
        self,
        provider: str,
        exc: BaseException,
        *,
        endpoint: str | None = None,
        permit: CircuitPermit | None = None,
    ) -> None:
        try:
            disposition = failure_disposition(exc)
            if disposition is None:
                return
            if disposition.category not in _PROVIDER_CIRCUIT_CATEGORIES | {LLMFailureCategory.RATE_LIMIT}:
                return
            key, normalized = _provider_key(provider, endpoint)
            now = self.now()
            cooldown = DEFAULT_PROVIDER_FAILURE_COOLDOWN_SECONDS
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
        finally:
            if permit is not None and permit.probe_handle is not None:
                self._unlock_close(permit.probe_handle)

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
        with self._transaction() as connection:
            self._reconcile_lock_rows(connection)
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
            slot_columns = {row[1] for row in connection.execute("PRAGMA table_info(slots)")}
            if "global_lock_index" not in slot_columns:
                connection.execute("ALTER TABLE slots ADD COLUMN global_lock_index INTEGER")
            if "provider_lock_index" not in slot_columns:
                connection.execute("ALTER TABLE slots ADD COLUMN provider_lock_index INTEGER")
            waiter_columns = {row[1] for row in connection.execute("PRAGMA table_info(waiters)")}
            if "lock_path" not in waiter_columns:
                connection.execute("ALTER TABLE waiters ADD COLUMN lock_path TEXT")
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

    def _reconcile_lock_rows(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT token,global_lock_index FROM slots"
        ).fetchall()
        for token, index in rows:
            if index is None:
                connection.execute("DELETE FROM slots WHERE token=?", (token,))
                continue
            handle = self._try_lock(self._global_lock_path(int(index)))
            if handle is not None:
                self._unlock_close(handle)
                connection.execute("DELETE FROM slots WHERE token=?", (token,))
        waiters = connection.execute("SELECT token,lock_path FROM waiters").fetchall()
        for token, lock_path in waiters:
            if not lock_path:
                connection.execute("DELETE FROM waiters WHERE token=?", (token,))
                continue
            handle = self._try_lock(Path(str(lock_path)))
            if handle is not None:
                self._unlock_close(handle)
                connection.execute("DELETE FROM waiters WHERE token=?", (token,))

    def _try_slot_locks(
        self,
        provider: str,
        global_limit: int,
        provider_limit: int,
    ) -> tuple[int, object, int, object] | None:
        gate = self._try_lock(self._lock_root / "acquire.lock")
        if gate is None:
            return None
        global_candidate: tuple[int, object] | None = None
        provider_candidate: tuple[int, object] | None = None
        try:
            global_active = 0
            for index in range(GLOBAL_MAX_CONCURRENCY):
                handle = self._try_lock(self._global_lock_path(index))
                if handle is None:
                    global_active += 1
                elif global_candidate is None:
                    global_candidate = (index, handle)
                else:
                    self._unlock_close(handle)
            provider_active = 0
            for index in range(GLOBAL_MAX_CONCURRENCY):
                handle = self._try_lock(self._provider_lock_path(provider, index))
                if handle is None:
                    provider_active += 1
                elif provider_candidate is None:
                    provider_candidate = (index, handle)
                else:
                    self._unlock_close(handle)
            if (
                global_active >= global_limit
                or provider_active >= provider_limit
                or global_candidate is None
                or provider_candidate is None
            ):
                if global_candidate is not None:
                    self._unlock_close(global_candidate[1])
                if provider_candidate is not None:
                    self._unlock_close(provider_candidate[1])
                return None
            return (
                global_candidate[0],
                global_candidate[1],
                provider_candidate[0],
                provider_candidate[1],
            )
        finally:
            self._unlock_close(gate)

    @property
    def _lock_root(self) -> Path:
        return self.db_path.with_suffix(self.db_path.suffix + ".locks")

    def _global_lock_path(self, index: int) -> Path:
        return self._lock_root / "global" / f"{index}.lock"

    def _provider_lock_path(self, provider: str, index: int) -> Path:
        digest = hashlib.sha256(provider.strip().lower().encode()).hexdigest()
        return self._lock_root / "providers" / digest / f"{index}.lock"

    def _probe_lock_path(self, provider_key: str) -> Path:
        return self._lock_root / "probes" / f"{provider_key}.lock"

    def _waiter_lock_path(self, token: str) -> Path:
        return self._lock_root / "waiters" / f"{token}.lock"

    @staticmethod
    def _try_lock(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
        if not _advisory_try_lock(handle):
            handle.close()
            return None
        return handle

    @classmethod
    def _open_locked(cls, path: Path):
        handle = cls._try_lock(path)
        if handle is None:
            raise RuntimeError(f"failed to acquire unique advisory lock {path}")
        return handle

    @staticmethod
    def _unlock_close(handle: object) -> None:
        _advisory_unlock(handle)
        with contextlib.suppress(OSError):
            handle.close()

    def _release_slot(self, token: str) -> None:
        with self._transaction() as connection:
            connection.execute("DELETE FROM slots WHERE token=?", (token,))


def _provider_limit_key(provider: str | None) -> str | None:
    if not provider:
        return None
    normalized = provider.upper().replace("-CLI", "").replace("-CODE", "").replace("-", "_")
    return f"ARC_{normalized}_MAX_CONCURRENCY"


def _advisory_try_lock(handle: object) -> bool:
    try:
        if _fcntl is not None:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        else:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            _msvcrt.locking(handle.fileno(), _msvcrt.LK_NBLCK, 1)
        return True
    except OSError as exc:
        if isinstance(exc, BlockingIOError) or exc.errno in {errno.EACCES, errno.EAGAIN}:
            return False
        raise


def _advisory_unlock(handle: object) -> None:
    with contextlib.suppress(OSError):
        if _fcntl is not None:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
        else:
            handle.seek(0)
            _msvcrt.locking(handle.fileno(), _msvcrt.LK_UNLCK, 1)


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
