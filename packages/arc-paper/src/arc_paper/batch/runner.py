from __future__ import annotations

import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from threading import get_ident
from typing import Any

from arc_llm.runner import resolve_llm_config

from .. import service
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
    model_tier: str | None = None,
    concurrency: int = 1,
    max_items: int | None = None,
    db: BatchDB | None = None,
) -> dict[str, Any]:
    resolve_llm_config(provider=provider, model=model, model_tier=model_tier)
    db = db or BatchDB.default()
    limit = max_items or 1_000_000
    workers = max(1, concurrency)
    submitted = 0
    worker_id = _worker_id()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        while True:
            while len(futures) < workers and submitted < limit:
                claim_limit = min(workers - len(futures), limit - submitted)
                items = db.claim_ready_items(name, limit=claim_limit, worker_id=worker_id)
                if not items:
                    break
                for item in items:
                    future = executor.submit(_run_one, item, db, provider=provider, model=model, model_tier=model_tier)
                    futures[future] = item
                submitted += len(items)
                if len(items) < claim_limit:
                    break
            if not futures:
                break
            done, _pending = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                future.result()
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


def _run_one(item: BatchItem, db: BatchDB, *, provider: str, model: str | None, model_tier: str | None) -> None:
    result = service.generate_llm_summary(item.paper_id, provider=provider, model=model, model_tier=model_tier)
    if result.get("ok"):
        summary_path = result.get("meta", {}).get("summary_path") or result.get("summary_path")
        db.mark_status(
            item.batch_name,
            item.paper_id,
            "done",
            provider=provider,
            model=model,
            summary_path=summary_path,
        )
    else:
        db.mark_status(
            item.batch_name,
            item.paper_id,
            "failed",
            provider=provider,
            model=model,
            last_error=json.dumps(result.get("error", result), ensure_ascii=False),
        )


def _worker_id() -> str:
    return f"pid:{os.getpid()}:thread:{get_ident()}"
