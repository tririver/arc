from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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
    concurrency: int = 1,
    max_items: int | None = None,
    db: BatchDB | None = None,
) -> dict[str, Any]:
    db = db or BatchDB.default()
    limit = max_items or 1_000_000
    items = db.next_items(name, status="ready", limit=limit)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {
            executor.submit(_run_one, item, db, provider=provider, model=model): item
            for item in items
        }
        for future in as_completed(futures):
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
        service.get_references(item.paper_id),
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


def _run_one(item: BatchItem, db: BatchDB, *, provider: str, model: str | None) -> None:
    db.mark_status(item.batch_name, item.paper_id, "running", attempts=item.attempts + 1)
    result = service.generate_llm_summary(item.paper_id, provider=provider, model=model)
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
