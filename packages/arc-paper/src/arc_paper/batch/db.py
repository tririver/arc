from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..cache import cache_root, now_iso
from ..ids import normalize_paper_id


@dataclass(frozen=True)
class BatchItem:
    batch_name: str
    paper_id: str
    status: str
    attempts: int = 0
    provider: str | None = None
    model: str | None = None
    source_hash: str | None = None
    summary_path: str | None = None
    last_error: str | None = None
    worker_id: str | None = None
    lease_token: str | None = None
    lease_until: str | None = None
    owner_pid: int | None = None
    owner_started_at: str | None = None
    heartbeat_at: str | None = None
    updated_at: str = ""


class BatchDB:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    @classmethod
    def default(cls) -> "BatchDB":
        return cls(cache_root() / "index.sqlite")

    def create_batch(self, name: str, paper_ids: list[str], prompt_version: str) -> None:
        now = now_iso()
        unique = list(dict.fromkeys(normalize_paper_id(item) for item in paper_ids if item.strip()))
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO batches(name, created_at, prompt_version) VALUES (?, ?, ?)",
                (name, now, prompt_version),
            )
            conn.execute("DELETE FROM batch_items WHERE batch_name = ?", (name,))
            for paper_id in unique:
                conn.execute(
                    """
                    INSERT INTO batch_items(
                      batch_name, paper_id, status, attempts, updated_at
                    ) VALUES (?, ?, 'queued', 0, ?)
                    """,
                    (name, paper_id, now),
                )

    def next_items(self, name: str, *, status: str, limit: int) -> list[BatchItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM batch_items
                WHERE batch_name = ? AND status = ?
                ORDER BY rowid
                LIMIT ?
                """,
                (name, status, limit),
            ).fetchall()
        return [_item_from_row(row) for row in rows]

    def claim_ready_items(
        self,
        name: str,
        *,
        limit: int,
        worker_id: str,
        lease_seconds: int = 300,
    ) -> list[BatchItem]:
        if limit <= 0:
            return []
        now = now_iso()
        lease_until = (datetime.now(timezone.utc) + timedelta(seconds=max(1, lease_seconds))).isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT * FROM batch_items
                WHERE batch_name = ?
                  AND (
                    status = 'ready'
                    OR (status = 'running' AND lease_until IS NOT NULL AND lease_until < ?)
                  )
                ORDER BY rowid
                """,
                (name, now),
            ).fetchall()
            claimed: list[sqlite3.Row] = []
            owner_pid = os.getpid()
            owner_started_at = _process_start_identity(owner_pid)
            for row in rows:
                if len(claimed) >= limit:
                    break
                if row["status"] == "running" and _owner_is_alive(row):
                    continue
                paper_id = str(row["paper_id"])
                lease_token = uuid.uuid4().hex
                updated = conn.execute(
                    """
                    UPDATE batch_items
                    SET status = 'running',
                        attempts = attempts + 1,
                        worker_id = ?,
                        lease_token = ?,
                        lease_until = ?,
                        owner_pid = ?,
                        owner_started_at = ?,
                        heartbeat_at = ?,
                        updated_at = ?
                    WHERE batch_name = ?
                      AND paper_id = ?
                      AND (
                        status = 'ready'
                        OR (status = 'running' AND lease_until IS NOT NULL AND lease_until < ?)
                      )
                    """,
                    (
                        worker_id,
                        lease_token,
                        lease_until,
                        owner_pid,
                        owner_started_at,
                        now,
                        now,
                        name,
                        paper_id,
                        now,
                    ),
                )
                if updated.rowcount:
                    claimed.append(
                        conn.execute(
                            "SELECT * FROM batch_items WHERE batch_name = ? AND paper_id = ? AND lease_token = ?",
                            (name, paper_id, lease_token),
                        ).fetchone()
                    )
        return [_item_from_row(row) for row in claimed]

    def heartbeat(
        self,
        name: str,
        paper_id: str,
        *,
        lease_token: str,
        lease_seconds: int = 300,
    ) -> bool:
        now = now_iso()
        lease_until = (datetime.now(timezone.utc) + timedelta(seconds=max(1, lease_seconds))).isoformat()
        with self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE batch_items
                SET heartbeat_at = ?, lease_until = ?, updated_at = ?
                WHERE batch_name = ? AND paper_id = ? AND status = 'running' AND lease_token = ?
                """,
                (now, lease_until, now, name, normalize_paper_id(paper_id), lease_token),
            )
        return bool(updated.rowcount)

    def mark_status(
        self,
        name: str,
        paper_id: str,
        status: str,
        *,
        lease_token: str | None = None,
        **fields: Any,
    ) -> bool:
        allowed = {
            "attempts",
            "provider",
            "model",
            "source_hash",
            "summary_path",
            "last_error",
            "worker_id",
            "lease_until",
            "owner_pid",
            "owner_started_at",
            "heartbeat_at",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["status"] = status
        if status not in {"running", "prefetching"}:
            updates.setdefault("worker_id", None)
            updates.setdefault("lease_token", None)
            updates.setdefault("lease_until", None)
            updates.setdefault("owner_pid", None)
            updates.setdefault("owner_started_at", None)
            updates.setdefault("heartbeat_at", None)
        updates["updated_at"] = now_iso()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values())
        values.extend([name, normalize_paper_id(paper_id)])
        ownership_clause = ""
        if lease_token is not None:
            ownership_clause = " AND status = 'running' AND lease_token = ?"
            values.append(lease_token)
        with self._connect() as conn:
            updated = conn.execute(
                f"UPDATE batch_items SET {assignments} WHERE batch_name = ? AND paper_id = ?{ownership_clause}",
                values,
            )
        return bool(updated.rowcount)

    def status_counts(self, name: str) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM batch_items WHERE batch_name = ? GROUP BY status",
                (name,),
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def retry_failed(self, name: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE batch_items
                SET status = 'queued', last_error = NULL, updated_at = ?
                WHERE batch_name = ? AND status = 'failed'
                """,
                (now_iso(), name),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS batches (
                  name TEXT PRIMARY KEY,
                  created_at TEXT NOT NULL,
                  prompt_version TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS batch_items (
                  batch_name TEXT NOT NULL,
                  paper_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  attempts INTEGER NOT NULL DEFAULT 0,
                  provider TEXT,
                  model TEXT,
                  source_hash TEXT,
                  summary_path TEXT,
                  last_error TEXT,
                  worker_id TEXT,
                  lease_token TEXT,
                  lease_until TEXT,
                  owner_pid INTEGER,
                  owner_started_at TEXT,
                  heartbeat_at TEXT,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY (batch_name, paper_id)
                );
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(batch_items)").fetchall()}
            if "worker_id" not in columns:
                conn.execute("ALTER TABLE batch_items ADD COLUMN worker_id TEXT")
            if "lease_until" not in columns:
                conn.execute("ALTER TABLE batch_items ADD COLUMN lease_until TEXT")
            if "lease_token" not in columns:
                conn.execute("ALTER TABLE batch_items ADD COLUMN lease_token TEXT")
            if "owner_pid" not in columns:
                conn.execute("ALTER TABLE batch_items ADD COLUMN owner_pid INTEGER")
            if "owner_started_at" not in columns:
                conn.execute("ALTER TABLE batch_items ADD COLUMN owner_started_at TEXT")
            if "heartbeat_at" not in columns:
                conn.execute("ALTER TABLE batch_items ADD COLUMN heartbeat_at TEXT")


def _item_from_row(row: sqlite3.Row) -> BatchItem:
    return BatchItem(**{key: row[key] for key in row.keys()})


def _process_start_identity(pid: int) -> str | None:
    try:
        # Linux /proc stat field 22 is stable for a process lifetime and protects
        # against PID reuse. Split after the parenthesized command because the
        # command itself may contain spaces. The suffix starts at field 3.
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return stat[stat.rfind(")") + 2 :].split()[19]
    except (OSError, IndexError):
        return None


def _owner_is_alive(row: sqlite3.Row) -> bool:
    pid = row["owner_pid"] if "owner_pid" in row.keys() else None
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    expected = row["owner_started_at"] if "owner_started_at" in row.keys() else None
    actual = _process_start_identity(pid)
    return not expected or not actual or str(expected) == actual
