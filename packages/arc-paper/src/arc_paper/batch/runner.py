from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextvars import copy_context
from pathlib import Path
from threading import Event, Thread, get_ident
from typing import Any

from arc_llm.runner import resolve_llm_config

from .. import service
from ..execution import check_cancelled
from ..summary.checkpoint import schema_canary_scope
from ..summary.model import DEFAULT_SUMMARY_MODEL_TIER
from .db import BatchDB, BatchItem


def prefetch_batch(name: str, *, workers: int = 4, db: BatchDB | None = None) -> dict[str, Any]:
    db = db or BatchDB.default()
    items = db.next_items(name, status="queued", limit=1_000_000)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_prefetch_one, item, db): item for item in items}
        for future in as_completed(futures):
            future.result()
    return {"batch": name, "counts": db.status_counts(name)}


def run_batch(
    name: str,
    *,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = DEFAULT_SUMMARY_MODEL_TIER,
    concurrency: int = 1,
    max_items: int | None = None,
    db: BatchDB | None = None,
) -> dict[str, Any]:
    resolve_llm_config(provider=provider, model=model, model_tier=model_tier)
    db = db or BatchDB.default()
    if max_items is not None and max_items < 0:
        raise ValueError("max_items must be non-negative")
    limit = 1_000_000 if max_items is None else max_items
    if limit == 0:
        return {"batch": name, "counts": db.status_counts(name)}
    workers = max(1, concurrency)
    submitted = 0
    worker_id = _worker_id()
    schema_canary_root = _schema_canary_root(db, name)
    abort_error: BaseException | None = None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        while abort_error is None:
            check_cancelled()
            while len(futures) < workers and submitted < limit:
                check_cancelled()
                claim_limit = min(workers - len(futures), limit - submitted)
                items = db.claim_ready_items(name, limit=claim_limit, worker_id=worker_id)
                if not items:
                    break
                for item in items:
                    future = executor.submit(
                        copy_context().run,
                        _run_one,
                        item,
                        db,
                        provider=provider,
                        model=model,
                        model_tier=model_tier,
                        schema_canary_root=schema_canary_root,
                    )
                    futures[future] = item
                submitted += len(items)
                if len(items) < claim_limit:
                    break
            if not futures:
                break
            done, _pending = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                try:
                    future.result()
                except BaseException as exc:
                    if _abort_batch_exception(exc):
                        abort_error = exc
                        break
                    raise
        if abort_error is not None:
            for future, item in list(futures.items()):
                if future.cancel() and item.lease_token:
                    db.mark_status(item.batch_name, item.paper_id, "ready", lease_token=item.lease_token)
    if abort_error is not None:
        raise abort_error
    return {"batch": name, "counts": db.status_counts(name)}


def export_batch(name: str, *, output: Path, db: BatchDB | None = None) -> dict[str, Any]:
    db = db or BatchDB.default()
    items = db.next_items(name, status="done", limit=1_000_000)
    output.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    with output.open("w", encoding="utf-8") as handle:
        for item in items:
            if not item.summary_path:
                continue
            path = Path(item.summary_path)
            if not path.exists():
                continue
            handle.write(path.read_text(encoding="utf-8").strip() + "\n")
            exported += 1
    return {"batch": name, "output": str(output), "exported": exported}


def _prefetch_one(item: BatchItem, db: BatchDB) -> None:
    db.mark_status(item.batch_name, item.paper_id, "prefetching")
    results = [
        service.get_title(item.paper_id),
        service.get_abstract(item.paper_id),
        service.get_toc(item.paper_id),
    ]
    failed = [result for result in results if not result.get("ok")]
    if failed:
        db.mark_status(
            item.batch_name,
            item.paper_id,
            "failed",
            attempts=item.attempts + 1,
            last_error=json.dumps(failed[0].get("error", failed[0]), ensure_ascii=False),
        )
    else:
        db.mark_status(item.batch_name, item.paper_id, "ready")


def _run_one(
    item: BatchItem,
    db: BatchDB,
    *,
    provider: str,
    model: str | None,
    model_tier: str | None,
    schema_canary_root: Path,
) -> None:
    check_cancelled()
    if not item.lease_token:
        raise RuntimeError(f"Claimed batch item has no lease token: {item.paper_id}")
    heartbeat = _LeaseHeartbeat(db, item)
    with heartbeat:
        try:
            with schema_canary_scope(schema_canary_root):
                result = service.generate_llm_summary(
                    item.paper_id,
                    provider=provider,
                    model=model,
                    model_tier=model_tier,
                )
        except BaseException as exc:
            if _budget_exception(exc):
                db.mark_status(
                    item.batch_name,
                    item.paper_id,
                    "ready",
                    lease_token=item.lease_token,
                    provider=provider,
                    model=model,
                    last_error=str(exc),
                )
            else:
                db.mark_status(
                    item.batch_name,
                    item.paper_id,
                    "failed",
                    lease_token=item.lease_token,
                    provider=provider,
                    model=model,
                    last_error=str(exc),
                )
            raise
    if heartbeat.failed:
        db.mark_status(
            item.batch_name,
            item.paper_id,
            "failed",
            lease_token=item.lease_token,
            last_error="Batch lease heartbeat failed",
        )
        raise RuntimeError(f"Lost batch lease while generating summary for {item.paper_id}")
    if result.get("ok"):
        summary_path = result.get("meta", {}).get("summary_path") or result.get("summary_path")
        committed = db.mark_status(
            item.batch_name,
            item.paper_id,
            "done",
            lease_token=item.lease_token,
            provider=provider,
            model=model,
            summary_path=summary_path,
        )
    else:
        committed = db.mark_status(
            item.batch_name,
            item.paper_id,
            "failed",
            lease_token=item.lease_token,
            provider=provider,
            model=model,
            last_error=json.dumps(result.get("error", result), ensure_ascii=False),
        )
    if not committed:
        raise RuntimeError(f"Lost batch lease before committing summary status for {item.paper_id}")


def _worker_id() -> str:
    return f"pid:{os.getpid()}:thread:{get_ident()}"


def _schema_canary_root(db: BatchDB, name: str) -> Path:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return db.path.parent / "summary-batch-artifacts" / digest


class _LeaseHeartbeat:
    def __init__(self, db: BatchDB, item: BatchItem, *, interval_seconds: float = 60.0):
        self.db = db
        self.item = item
        self.interval_seconds = interval_seconds
        self.failed = False
        self._stop = Event()
        self._thread: Thread | None = None

    def __enter__(self):
        self._thread = Thread(target=self._run, name="arc-paper-lease-heartbeat", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                alive = self.db.heartbeat(
                    self.item.batch_name,
                    self.item.paper_id,
                    lease_token=str(self.item.lease_token),
                )
            except Exception:
                alive = False
            if not alive:
                self.failed = True
                return


def _abort_batch_exception(exc: BaseException) -> bool:
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        if bool(getattr(current, "abort_batch", False)):
            return True
        scope = getattr(current, "abort_scope", None)
        if str(getattr(scope, "value", scope) or "").lower() in {"batch", "provider"}:
            return True
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None and current.__context__ is not current.__cause__:
            pending.append(current.__context__)
    return False


def _budget_exception(exc: BaseException) -> bool:
    return any(
        getattr(current, "code", "") in {
            "child_budget_required", "child_budget_exhausted",
        }
        for current in _exception_chain(exc)
    )


def _exception_chain(exc: BaseException):
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None and current.__context__ is not current.__cause__:
            pending.append(current.__context__)
