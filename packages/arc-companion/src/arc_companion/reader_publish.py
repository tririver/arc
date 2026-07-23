from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import weakref

from .io import sha256_json


READER_PUBLISH_STATE_VERSION = "arc.companion.reader-publish-state.v1"
READER_SEMANTIC_VERSION = "arc.companion.reader-semantic.v1"
READER_PUBLISH_INTERVAL_SECONDS = 60.0

_NONSEMANTIC_SNAPSHOT_KEYS = {
    "created_at",
    "updated_at",
    "published_at",
    "committed_at",
    "revision",
    "revision_time",
    "manifest_created_at",
    "checkpoint_dir",
    "state_path",
    "lock_path",
    "run_id",
}


@dataclass(frozen=True)
class ReaderObject:
    relative_path: str
    data: bytes
    kind: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def semantic_record(self) -> dict[str, object]:
        return {
            "path": self.relative_path,
            "sha256": self.sha256,
            "bytes": len(self.data),
            "kind": self.kind,
        }


@dataclass(frozen=True)
class PreparedReaderCandidate:
    snapshot: Mapping[str, Any]
    web_render_version: str
    builtin_objects: tuple[ReaderObject, ...] = ()
    source_objects: tuple[ReaderObject, ...] = ()
    payload: Any = None

    @property
    def semantic_sha256(self) -> str:
        return reader_semantic_sha256(
            snapshot=self.snapshot,
            web_render_version=self.web_render_version,
            builtin_objects=self.builtin_objects,
            source_objects=self.source_objects,
        )


@dataclass(frozen=True)
class ReaderPublishResult:
    status: str
    semantic_sha256: str
    state: Mapping[str, Any]
    published: bool = False
    dirty: bool = False


def reader_semantic_sha256(
    *,
    snapshot: Mapping[str, Any],
    web_render_version: str,
    builtin_objects: Sequence[ReaderObject],
    source_objects: Sequence[ReaderObject],
) -> str:
    """Hash only exact Reader-visible semantics and exact asset bytes."""

    builtin_records = sorted(
        (item.semantic_record() for item in builtin_objects),
        key=lambda item: (
            str(item["path"]),
            str(item["kind"]),
            str(item["sha256"]),
        ),
    )
    source_records = sorted(
        (item.semantic_record() for item in source_objects),
        key=lambda item: (
            str(item["path"]),
            str(item["kind"]),
            str(item["sha256"]),
        ),
    )
    return sha256_json({
        "semantic_version": READER_SEMANTIC_VERSION,
        "snapshot": _semantic_snapshot(snapshot),
        "web_render_version": web_render_version,
        "builtin_assets_sha256": sha256_json(builtin_records),
        "source_assets": source_records,
    })


def _semantic_snapshot(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        return {
            str(item_key): _semantic_snapshot(item, key=str(item_key))
            for item_key, item in sorted(
                value.items(), key=lambda pair: str(pair[0]),
            )
            if not _nonsemantic_snapshot_key(str(item_key))
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_snapshot(item, key=key) for item in value]
    return value


def _nonsemantic_snapshot_key(key: str) -> bool:
    return (
        key in _NONSEMANTIC_SNAPSHOT_KEYS
        or key.startswith("output_")
        or key.endswith("_path")
        or key.endswith("_lock")
        or key.endswith("_deadline")
    )


def parse_reader_commit_utc(value: object) -> datetime | None:
    if (
        not isinstance(value, str)
        or not value
        or not value.endswith("+00:00")
    ):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
    ):
        return None
    return parsed.astimezone(timezone.utc)


class ReaderPublishCoordinator:
    """Serialize semantic dedupe and exact monotonic Reader throttling."""

    def __init__(
        self,
        *,
        state_loader: Callable[[], Mapping[str, Any]],
        state_merger: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        preparer: Callable[[Any], PreparedReaderCandidate],
        publisher: Callable[
            [PreparedReaderCandidate], Mapping[str, Any]
        ],
        utc_now: Callable[[], datetime] = (
            lambda: datetime.now(timezone.utc)
        ),
        monotonic: Callable[[], float] = time.monotonic,
        interval_seconds: float = READER_PUBLISH_INTERVAL_SECONDS,
        lock: threading.RLock | None = None,
    ) -> None:
        if interval_seconds < 0:
            raise ValueError("reader publish interval must be nonnegative")
        self._state_loader = state_loader
        self._state_merger = state_merger
        self._preparer = preparer
        self._publisher = publisher
        self._utc_now = utc_now
        self._monotonic = monotonic
        self._interval = float(interval_seconds)
        self._owned_lock = threading.RLock() if lock is None else None
        self._lock_ref = weakref.ref(lock) if lock is not None else None
        state = dict(state_loader())
        monotonic_now = float(monotonic())
        self._deadline = monotonic_now + self._remaining_interval(
            state, utc_now(),
        )

    def mark_dirty(self) -> Mapping[str, Any]:
        with self._active_lock():
            state = dict(self._state_loader())
            if (
                state.get("reader_publish_state_version")
                == READER_PUBLISH_STATE_VERSION
                and state.get("reader_dirty") is True
            ):
                return state
            return self._state_merger({
                "reader_publish_state_version": (
                    READER_PUBLISH_STATE_VERSION
                ),
                "reader_dirty": True,
            })

    def request(
        self,
        latest_supplier: Callable[[], Any],
        *,
        final: bool = False,
        strict: bool = False,
    ) -> ReaderPublishResult:
        with self._active_lock():
            state = dict(self._state_loader())
            candidate = self._preparer(latest_supplier())
            semantic = candidate.semantic_sha256
            current = str(
                state.get("reader_committed_semantic_sha256") or ""
            )
            if semantic == current:
                repaired = self._merge_if_changed(
                    state,
                    {
                        "reader_publish_state_version": (
                            READER_PUBLISH_STATE_VERSION
                        ),
                        "reader_dirty": False,
                    },
                )
                return ReaderPublishResult(
                    "deduplicated",
                    semantic,
                    repaired,
                    dirty=False,
                )

            monotonic_now = float(self._monotonic())
            if not final and monotonic_now < self._deadline:
                deferred = self._merge_if_changed(
                    state,
                    {
                        "reader_publish_state_version": (
                            READER_PUBLISH_STATE_VERSION
                        ),
                        "reader_dirty": True,
                    },
                )
                return ReaderPublishResult(
                    "deferred",
                    semantic,
                    deferred,
                    dirty=True,
                )

            try:
                outputs = dict(self._publisher(candidate))
            except BaseException as exc:
                self._merge_if_changed(
                    state,
                    {
                        "reader_publish_state_version": (
                            READER_PUBLISH_STATE_VERSION
                        ),
                        "reader_dirty": True,
                    },
                )
                if (
                    not isinstance(exc, Exception)
                    or strict
                    or final
                ):
                    raise
                return ReaderPublishResult(
                    "failed",
                    semantic,
                    dict(self._state_loader()),
                    dirty=True,
                )

            committed_utc = self._aware_utc(self._utc_now())
            latest_after = self._preparer(latest_supplier())
            dirty_after = (
                latest_after.semantic_sha256 != semantic
            )
            merged = self._state_merger({
                **outputs,
                "reader_publish_state_version": (
                    READER_PUBLISH_STATE_VERSION
                ),
                "reader_dirty": dirty_after,
                "reader_committed_semantic_sha256": semantic,
                "reader_committed_at": committed_utc.isoformat(),
            })
            self._deadline = (
                float(self._monotonic()) + self._interval
            )
            return ReaderPublishResult(
                "published",
                semantic,
                merged,
                published=True,
                dirty=dirty_after,
            )

    def _remaining_interval(
        self,
        state: Mapping[str, Any],
        utc_now: datetime,
    ) -> float:
        committed = parse_reader_commit_utc(
            state.get("reader_committed_at")
        )
        if committed is None:
            raw = state.get("reader_committed_at")
            return 0.0 if raw in {None, ""} else self._interval
        now = self._aware_utc(utc_now)
        elapsed = (now - committed).total_seconds()
        if elapsed < 0:
            return self._interval
        return max(0.0, self._interval - elapsed)

    def _active_lock(self) -> threading.RLock:
        if self._owned_lock is not None:
            return self._owned_lock
        assert self._lock_ref is not None
        lock = self._lock_ref()
        if lock is None:
            raise RuntimeError("Reader publish lock expired")
        return lock

    @staticmethod
    def _aware_utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reader publish UTC clock must be aware")
        return value.astimezone(timezone.utc)

    def _merge_if_changed(
        self,
        state: Mapping[str, Any],
        values: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if all(state.get(key) == value for key, value in values.items()):
            return state
        return self._state_merger(values)
