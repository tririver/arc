from __future__ import annotations

import sqlite3
from dataclasses import dataclass
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

    def mark_status(self, name: str, paper_id: str, status: str, **fields: Any) -> None:
        allowed = {"attempts", "provider", "model", "source_hash", "summary_path", "last_error"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["status"] = status
        updates["updated_at"] = now_iso()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values())
        values.extend([name, normalize_paper_id(paper_id)])
        with self._connect() as conn:
            conn.execute(
                f"UPDATE batch_items SET {assignments} WHERE batch_name = ? AND paper_id = ?",
                values,
            )

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
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY (batch_name, paper_id)
                );
                """
            )


def _item_from_row(row: sqlite3.Row) -> BatchItem:
    return BatchItem(**{key: row[key] for key in row.keys()})
