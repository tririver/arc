from __future__ import annotations

import argparse
from typing import Any

from arc_domain import service as domain_service
from arc_paper import service as paper_service
from arc_paper.batch.runner import run_batch
from arc_paper.ids import normalize_paper_id

from .jobs import (
    MCPJobCancelled,
    acquire_worker_lock,
    finish_job,
    is_cancel_requested,
    read_job,
    record_progress,
    set_error,
    start_running,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a persisted ARC MCP background job")
    parser.add_argument("job_id")
    args = parser.parse_args(argv)
    return run_job(args.job_id)


def run_job(job_id: str) -> int:
    if not acquire_worker_lock(job_id):
        return 0
    start_running(job_id)
    try:
        job = read_job(job_id)
        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        result = _dispatch(str(job.get("job_type") or ""), payload, job_id=job_id)
        if is_cancel_requested(job_id):
            raise MCPJobCancelled("MCP job cancellation was requested.")
        finish_job(job_id, result, _result_status(result))
        return 0
    except MCPJobCancelled as exc:
        set_error(job_id, "job_cancelled", str(exc), cancelled=True)
        return 0
    except Exception as exc:
        set_error(job_id, "job_failed", str(exc))
        return 1


def _dispatch(job_type: str, payload: dict[str, Any], *, job_id: str) -> Any:
    if job_type == "paper_summary":
        return _paper_summary(payload, job_id=job_id)
    if job_type == "domain_build":
        return _domain_build(payload, job_id=job_id)
    if job_type == "summary_batch_run":
        return _summary_batch_run(payload, job_id=job_id)
    raise ValueError(f"Unsupported ARC MCP job type: {job_type}")


def _paper_summary(payload: dict[str, Any], *, job_id: str) -> dict[str, Any]:
    _check_cancel(job_id)
    record_progress(job_id, {"event": "job_started"})
    return paper_service.generate_llm_summary(
        payload.get("paper_ids"),
        provider=str(payload.get("provider") or "auto"),
        model=payload.get("model"),
        refresh=bool(payload.get("refresh", False)),
        progress_callback=lambda event: record_progress(job_id, event),
    )


def _domain_build(payload: dict[str, Any], *, job_id: str) -> dict[str, Any]:
    _check_cancel(job_id)
    seed_paper = str(payload["seed_paper"])
    record_progress(
        job_id,
        {
            "event": "domain_started",
            "seed_paper": normalize_paper_id(seed_paper),
            "intent": str(payload.get("intent", "")),
        },
    )
    result = domain_service.build_domain(
        seed_paper,
        intent=str(payload.get("intent", "")),
        domain_id=payload.get("domain_id"),
        provider=str(payload.get("provider") or "auto"),
        model=payload.get("model"),
        refresh=bool(payload.get("refresh", False)),
        workers=int(payload.get("workers", 8)),
    )
    record_progress(job_id, {"event": "domain_completed" if _result_ok(result) else "domain_failed"})
    return result


def _summary_batch_run(payload: dict[str, Any], *, job_id: str) -> dict[str, Any]:
    _check_cancel(job_id)
    name = str(payload["name"])
    record_progress(job_id, {"event": "summary_batch_started", "name": name})
    result = run_batch(
        name,
        provider=str(payload.get("provider") or "auto"),
        model=payload.get("model"),
        concurrency=int(payload.get("concurrency", 1)),
        max_items=payload.get("max_items"),
    )
    record_progress(job_id, {"event": "summary_batch_completed", "name": name})
    return {"ok": True, "data": result, "errors": [], "meta": {}}


def _check_cancel(job_id: str) -> None:
    if is_cancel_requested(job_id):
        raise MCPJobCancelled("MCP job cancellation was requested.")


def _result_status(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("status") == "needs_llm":
            return "needs_llm"
        if result.get("ok") is False:
            return "failed"
        if result.get("ok") is True:
            return "done"
    return "done"


def _result_ok(result: Any) -> bool:
    if isinstance(result, dict) and "ok" in result:
        return result.get("ok") is True
    if isinstance(result, dict):
        return bool(result) and all(isinstance(item, dict) and item.get("ok") is True for item in result.values())
    return False


if __name__ == "__main__":
    raise SystemExit(main())
