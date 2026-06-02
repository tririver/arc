from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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


def _bool_arg(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default
    return bool(value)


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
    if job_type == "main_reference_inference":
        return _main_reference_inference(payload, job_id=job_id)
    if job_type == "domain_build":
        return _domain_build(payload, job_id=job_id)
    if job_type == "summary_batch_run":
        return _summary_batch_run(payload, job_id=job_id)
    if job_type == "md2pdf":
        return _md2pdf(payload, job_id=job_id)
    if job_type == "translate":
        return _translate(payload, job_id=job_id)
    if job_type == "batch_translate":
        return _batch_translate(payload, job_id=job_id)
    raise ValueError(f"Unsupported ARC MCP job type: {job_type}")


def _paper_summary(payload: dict[str, Any], *, job_id: str) -> dict[str, Any]:
    from arc_paper import service as paper_service

    _check_cancel(job_id)
    record_progress(job_id, {"event": "job_started"})
    return paper_service.generate_llm_summary(
        payload.get("paper_ids"),
        provider=str(payload.get("provider") or "auto"),
        model=payload.get("model"),
        model_tier=payload.get("model_tier"),
        refresh=_bool_arg(payload.get("refresh"), False),
        progress_callback=lambda event: record_progress(job_id, event),
    )


def _main_reference_inference(payload: dict[str, Any], *, job_id: str) -> dict[str, Any]:
    from arc_paper import service as paper_service

    _check_cancel(job_id)
    record_progress(job_id, {"event": "reference_inference_started"})
    result = paper_service.llm_infer_main_references(
        str(payload.get("text") or ""),
        provider=str(payload.get("provider") or "auto"),
        model=payload.get("model"),
        model_tier=payload.get("model_tier"),
        refresh=_bool_arg(payload.get("refresh"), False),
    )
    event = "reference_inference_completed" if _result_ok(result) else "reference_inference_failed"
    record_progress(job_id, {"event": event})
    return result


def _domain_build(payload: dict[str, Any], *, job_id: str) -> dict[str, Any]:
    from arc_domain import service as domain_service
    from arc_paper.ids import normalize_paper_id

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
    build_kwargs: dict[str, Any] = {
        "intent": str(payload.get("intent", "")),
        "domain_id": payload.get("domain_id"),
        "provider": str(payload.get("provider") or "auto"),
        "model": payload.get("model"),
        "refresh": _bool_arg(payload.get("refresh"), False),
        "workers": int(payload.get("workers", 8)),
    }
    if payload.get("model_tier") is not None:
        build_kwargs["model_tier"] = payload.get("model_tier")
    result = domain_service.build_domain(seed_paper, **build_kwargs)
    record_progress(job_id, {"event": "domain_completed" if _result_ok(result) else "domain_failed"})
    return result


def _summary_batch_run(payload: dict[str, Any], *, job_id: str) -> dict[str, Any]:
    from arc_paper.batch.runner import run_batch

    _check_cancel(job_id)
    name = str(payload["name"])
    record_progress(job_id, {"event": "summary_batch_started", "name": name})
    result = run_batch(
        name,
        provider=str(payload.get("provider") or "auto"),
        model=payload.get("model"),
        model_tier=payload.get("model_tier"),
        concurrency=int(payload.get("concurrency", 1)),
        max_items=payload.get("max_items"),
    )
    record_progress(job_id, {"event": "summary_batch_completed", "name": name})
    return {"ok": True, "data": result, "errors": [], "meta": {}}


def _md2pdf(payload: dict[str, Any], *, job_id: str) -> dict[str, Any]:
    from arc_typeset import md2pdf as typeset_md2pdf

    _check_cancel(job_id)
    input_path = Path(str(payload["input"]))
    output_path = Path(str(payload["output"])) if payload.get("output") else None
    texlive_bin_raw = payload.get("texlive_bin", str(typeset_md2pdf.DEFAULT_TEXLIVE_BIN))
    texlive_bin = Path(str(texlive_bin_raw)) if texlive_bin_raw else None
    record_progress(
        job_id,
        {
            "event": "md2pdf_started",
            "input": str(input_path),
            "output": str(output_path) if output_path else None,
        },
    )
    result = typeset_md2pdf.convert_markdown_to_pdf(
        input_path=input_path,
        output_path=output_path,
        texlive_bin=texlive_bin,
        margin=str(payload.get("margin", typeset_md2pdf.DEFAULT_MARGIN)),
        mainfont=str(payload.get("mainfont", typeset_md2pdf.DEFAULT_MAINFONT)),
        cjk_mainfont=str(payload.get("cjk_mainfont", typeset_md2pdf.DEFAULT_CJK_MAINFONT)),
        resource_paths=[Path(str(path)) for path in payload.get("resource_path") or []] or None,
    )
    record_progress(job_id, {"event": "md2pdf_completed" if _result_ok(result) else "md2pdf_failed"})
    return result


def _translate(payload: dict[str, Any], *, job_id: str) -> dict[str, Any]:
    from arc_typeset import translate as typeset_translate

    _check_cancel(job_id)
    input_path = Path(str(payload["input"]))
    output_path = Path(str(payload["output"])) if payload.get("output") else None
    target_locale = str(payload.get("target_locale", typeset_translate.DEFAULT_TARGET_LOCALE))
    record_progress(job_id, {"event": "translate_started", "input": str(input_path), "target_locale": target_locale})
    result = typeset_translate.translate_markdown(
        input_path=input_path,
        output_path=output_path,
        target_language=str(payload.get("target_language", typeset_translate.DEFAULT_TARGET_LANGUAGE)),
        target_locale=target_locale,
        provider=str(payload.get("provider") or "auto"),
        model=payload.get("model"),
        model_tier=str(payload.get("model_tier", typeset_translate.DEFAULT_MODEL_TIER)),
        quality=_bool_arg(payload.get("quality"), False),
        convert_pdf=True,
        overwrite=_bool_arg(payload.get("overwrite"), False),
    )
    record_progress(job_id, {"event": "translate_completed" if _result_ok(result) else "translate_failed"})
    return result


def _batch_translate(payload: dict[str, Any], *, job_id: str) -> dict[str, Any]:
    from arc_typeset import translate as typeset_translate

    _check_cancel(job_id)
    project_dir = Path(str(payload["project_dir"]))
    target_locale = str(payload.get("target_locale", typeset_translate.DEFAULT_TARGET_LOCALE))
    record_progress(
        job_id,
        {"event": "batch_translate_started", "project_dir": str(project_dir), "target_locale": target_locale},
    )
    result = typeset_translate.batch_translate_project(
        project_dir=project_dir,
        target_language=str(payload.get("target_language", typeset_translate.DEFAULT_TARGET_LANGUAGE)),
        target_locale=target_locale,
        provider=str(payload.get("provider") or "auto"),
        model=payload.get("model"),
        model_tier=str(payload.get("model_tier", typeset_translate.DEFAULT_MODEL_TIER)),
        quality=_bool_arg(payload.get("quality"), False),
        overwrite=_bool_arg(payload.get("overwrite"), False),
    )
    record_progress(job_id, {"event": "batch_translate_completed" if _result_ok(result) else "batch_translate_failed"})
    return result


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
