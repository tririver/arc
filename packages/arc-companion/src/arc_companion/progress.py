from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import RLock
import time
from typing import Any, Callable, Mapping


PROGRESS_VERSION = "arc.companion.progress.v1"
DEFAULT_REVIEW_INTERVAL_SECONDS = 30 * 60


class CompanionProgress:
    """Forward durable companion events and emit review checkpoints at safe boundaries."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        review_interval_seconds: float = DEFAULT_REVIEW_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        configured = path or os.environ.get("ARC_JOB_PROGRESS_FILE")
        self.path = Path(configured) if configured else None
        self.review_interval_seconds = review_interval_seconds
        self.clock = clock
        self.started_at = clock()
        self.last_review_at = self.started_at
        self.review_sequence = 0
        self._lock = RLock()

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            item = {
                "schema_version": PROGRESS_VERSION,
                "event": event,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            return item

    def safe_boundary(self, event: str, **payload: Any) -> list[dict[str, Any]]:
        with self._lock:
            events = [self.emit(event, **payload)]
            now = self.clock()
            if now - self.last_review_at >= self.review_interval_seconds:
                self.review_sequence += 1
                self.last_review_at = now
                events.append(self.emit(
                    "review_due",
                    review_sequence=self.review_sequence,
                    elapsed_seconds=now - self.started_at,
                    safe_boundary=event,
                ))
            return events

    def provider_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        return self.emit("provider_progress", provider_event=dict(event))
