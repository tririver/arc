from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from arc_llm.call_record import ARC_LLM_CALL_RECORD_FIELD, ARC_LLM_CALL_RECORD_SCHEMA, strip_arc_llm_call_records
from jsonschema import Draft202012Validator

from ..checkpoint import run_json_checkpointed
from ..store import read_section_summary, store_section_summary

RunJson = Callable[[str, dict[str, Any], str | None], dict[str, Any]]
ProgressCallback = Callable[[dict[str, Any]], None]

SECTION_SUMMARY_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "arc.section-summary-v1",
    "type": "object",
    "additionalProperties": False,
    "required": ["section_id", "title", "summary", "warnings"],
    "properties": {
        "section_id": {"type": "string"},
        "title": {"type": "string"},
        "summary": {"type": "string", "minLength": 1},
        "warnings": {"type": "array", "items": {"type": "string"}},
        ARC_LLM_CALL_RECORD_FIELD: ARC_LLM_CALL_RECORD_SCHEMA,
    },
}

PAPER_SYNTHESIS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "arc.paper-summary-synthesis-v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "paper_id",
        "title",
        "authors_short",
        "high_value_summary",
        "reading_guide",
        "warnings",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": "arc.paper_summary_synthesis.v1"},
        "paper_id": {"type": "string", "minLength": 1},
        "title": {"type": "string", "minLength": 1},
        "authors_short": {"type": "string", "minLength": 1},
        "high_value_summary": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "reading_guide": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["purpose", "sections", "reason"],
                "properties": {
                    "purpose": {"type": "string"},
                    "sections": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        ARC_LLM_CALL_RECORD_FIELD: ARC_LLM_CALL_RECORD_SCHEMA,
    },
}


def generate_summary_with_section_pipeline(
    task: dict[str, Any],
    *,
    model: str | None,
    run_json: RunJson,
    provider: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    if task.get("pipeline") != "section_then_paper":
        return run_json(_task_prompt(task), task.get("output_schema") or {}, model)

    input_pack = task.get("input_pack") or {}
    sections = list(input_pack.get("sections") or [])
    section_summaries = summarize_sections(
        input_pack,
        sections,
        prompt_version=str(task.get("prompt_version") or "paper-summary-v1"),
        provider=provider,
        model=model,
        run_json=run_json,
        use_cache=not _bool_value(task.get("refresh"), False),
        progress_callback=progress_callback,
    )
    _emit(
        progress_callback,
        {
            "event": "final_started",
            "paper_id": str(input_pack.get("paper_id") or ""),
            "sections_total": len(section_summaries),
            "sections_completed": len(section_summaries),
        },
    )
    final_task = _final_task(task, section_summaries)
    final_prompt = _task_prompt(final_task)
    synthesis = run_json_checkpointed(
        paper_id=str(input_pack.get("paper_id") or ""),
        call_kind="paper-synthesis",
        identity={
            "prompt_version": str(task.get("prompt_version") or "paper-summary-v1"),
            "source_hash": str(input_pack.get("source_hash") or ""),
            "provider": provider,
        },
        prompt=final_prompt,
        schema=PAPER_SYNTHESIS_SCHEMA,
        model=model,
        run_json=run_json,
        validate=Draft202012Validator(PAPER_SYNTHESIS_SCHEMA).validate,
        use_cache=not _bool_value(task.get("refresh"), False),
    )
    summary = _assemble_summary(task, section_summaries, synthesis)
    _emit(
        progress_callback,
        {
            "event": "final_completed",
            "paper_id": str(input_pack.get("paper_id") or ""),
            "sections_total": len(section_summaries),
            "sections_completed": len(section_summaries),
        },
    )
    return summary


def apply_provider_provenance(
    summary: dict[str, Any],
    task: dict[str, Any],
    *,
    method: str,
    model: str | None,
) -> dict[str, Any]:
    provenance = summary.setdefault("provenance", {})
    provenance.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    provenance["method"] = method
    if model:
        provenance["model"] = model
    provenance["prompt_version"] = task.get("prompt_version") or "paper-summary-v1"
    source_hash = (task.get("input_pack") or {}).get("source_hash")
    if source_hash:
        provenance["source_hash"] = source_hash
    return summary


def summarize_sections(
    input_pack: dict[str, Any],
    sections: list[dict[str, Any]],
    *,
    prompt_version: str,
    model: str | None,
    run_json: RunJson,
    provider: str | None = None,
    use_cache: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    paper_id = str(input_pack.get("paper_id") or "")
    source_hash = str(input_pack.get("source_hash") or "")
    total = len(sections)
    section_summaries: list[dict[str, Any]] = []
    _emit(
        progress_callback,
        {
            "event": "sections_started",
            "paper_id": paper_id,
            "sections_total": total,
            "sections_completed": 0,
        },
    )

    for index, section in enumerate(sections, start=1):
        section_id = str(section.get("section_id") or "")
        title = str(section.get("title") or "")
        event_base = {
            "paper_id": paper_id,
            "section_index": index,
            "sections_total": total,
            "section_id": section_id,
            "title": title,
        }

        if use_cache and paper_id and source_hash:
            cached = _read_cached_section_summary(
                paper_id,
                prompt_version=prompt_version,
                source_hash=source_hash,
                provider=provider,
                model=model,
                section_index=index,
                section_id=section_id,
            )
            if cached:
                section_summaries.append(cached)
                _emit(
                    progress_callback,
                    {
                        "event": "section_cached",
                        **event_base,
                        "sections_completed": len(section_summaries),
                    },
                )
                continue

        _emit(
            progress_callback,
            {
                "event": "section_started",
                **event_base,
                "sections_completed": len(section_summaries),
            },
        )
        summary = _summarize_section(
            input_pack,
            section,
            prompt_version=prompt_version,
            provider=provider,
            model=model,
            run_json=run_json,
            use_cache=use_cache,
        )
        if paper_id and source_hash:
            # A paid response that cannot be persisted must stop the workflow;
            # continuing would make a resume silently pay for it again.
            store_section_summary(
                paper_id,
                prompt_version=prompt_version,
                source_hash=source_hash,
                provider=provider,
                model=model,
                section_index=index,
                section_id=section_id,
                summary=summary,
            )
        section_summaries.append(summary)
        _emit(
            progress_callback,
            {
                "event": "section_completed",
                **event_base,
                "sections_completed": len(section_summaries),
            },
        )

    return section_summaries


def _summarize_section(
    input_pack: dict[str, Any],
    section: dict[str, Any],
    *,
    prompt_version: str,
    provider: str | None,
    model: str | None,
    run_json: RunJson,
    use_cache: bool,
) -> dict[str, Any]:
    prompt = "\n\n".join(
        [
            "You are summarizing one section of a physics paper for a research agent.",
            "Use only the supplied title, abstract, and section text.",
            "Return JSON only, conforming exactly to the supplied schema.",
            "The summary must be one short paragraph of at most 3 sentences.",
            "Paper metadata:",
            json.dumps(_metadata_for_section(input_pack), ensure_ascii=False, indent=2),
            "Section:",
            json.dumps(section, ensure_ascii=False, indent=2),
        ]
    )
    summary = run_json_checkpointed(
        paper_id=str(input_pack.get("paper_id") or ""),
        call_kind="section-summary",
        identity={
            "prompt_version": prompt_version,
            "source_hash": str(input_pack.get("source_hash") or ""),
            "provider": provider,
            "section_id": str(section.get("section_id") or ""),
        },
        prompt=prompt,
        schema=SECTION_SUMMARY_SCHEMA,
        model=model,
        run_json=run_json,
        validate=Draft202012Validator(SECTION_SUMMARY_SCHEMA).validate,
        use_cache=use_cache,
    )
    result = {
        "section_id": str(summary.get("section_id") or section.get("section_id") or ""),
        "title": str(summary.get("title") or section.get("title") or ""),
        "summary": str(summary.get("summary") or "").strip(),
        "warnings": list(summary.get("warnings") or []),
    }
    if isinstance(summary.get(ARC_LLM_CALL_RECORD_FIELD), dict):
        result[ARC_LLM_CALL_RECORD_FIELD] = summary[ARC_LLM_CALL_RECORD_FIELD]
    return result


def _metadata_for_section(input_pack: dict[str, Any]) -> dict[str, Any]:
    metadata = input_pack.get("metadata") or {}
    return {
        "paper_id": input_pack.get("paper_id", ""),
        "title": metadata.get("title", ""),
        "abstract": metadata.get("abstract", ""),
        "authors": metadata.get("authors", []),
    }


def _final_task(task: dict[str, Any], section_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    input_pack = task.get("input_pack") or {}
    compact_pack = {
        "paper_id": input_pack.get("paper_id", ""),
        "metadata": input_pack.get("metadata") or {},
        "toc": input_pack.get("toc") or [],
        "section_summaries": strip_arc_llm_call_records(section_summaries),
        "source_hash": input_pack.get("source_hash", ""),
    }
    return {
        **task,
        "user_prompt": (
            "Generate only the paper-level synthesis JSON for the supplied compact input pack. "
            "Use the title, abstract, table of contents, and section summaries as evidence. "
            "Do not output table of contents, section_summaries, or provenance; ARC will attach those "
            "fields exactly and deterministically. Do not rewrite the supplied section summaries. "
            "Do not infer from references; references are intentionally omitted."
        ),
        "input_pack": compact_pack,
        "output_schema": PAPER_SYNTHESIS_SCHEMA,
    }


def _assemble_summary(
    task: dict[str, Any],
    section_summaries: list[dict[str, Any]],
    synthesis: dict[str, Any],
) -> dict[str, Any]:
    input_pack = task.get("input_pack") or {}
    metadata = input_pack.get("metadata") or {}
    result = {
        "schema_version": "arc.paper_llm_summary.v1",
        "paper_id": str(input_pack.get("paper_id") or synthesis.get("paper_id") or ""),
        "title": str(synthesis.get("title") or metadata.get("title") or ""),
        "authors_short": str(synthesis.get("authors_short") or ""),
        "high_value_summary": list(synthesis.get("high_value_summary") or []),
        "toc": _canonical_toc(input_pack.get("toc") or []),
        "section_summaries": section_summaries,
        "reading_guide": list(synthesis.get("reading_guide") or []),
        "warnings": list(synthesis.get("warnings") or []),
        "provenance": {},
    }
    if isinstance(synthesis.get(ARC_LLM_CALL_RECORD_FIELD), dict):
        result[ARC_LLM_CALL_RECORD_FIELD] = synthesis[ARC_LLM_CALL_RECORD_FIELD]
    return result


def _canonical_toc(toc: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical = []
    for item in toc:
        section_id = str(item.get("section_id") or item.get("id") or "")
        entry = {
            "section_id": section_id,
            "title": str(item.get("title") or ""),
        }
        if item.get("level") is not None:
            try:
                entry["level"] = int(item["level"])
            except (TypeError, ValueError):
                pass
        canonical.append(entry)
    return canonical


def _task_prompt(task: dict[str, Any]) -> str:
    return "\n\n".join(
        part
        for part in [
            task.get("system_prompt", ""),
            task.get("user_prompt", ""),
            "Input pack:",
            json.dumps(task.get("input_pack", {}), ensure_ascii=False, indent=2),
            "Return JSON only.",
        ]
        if part
    )


def _bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return default


def _read_cached_section_summary(
    paper_id: str,
    *,
    prompt_version: str,
    source_hash: str,
    provider: str | None,
    model: str | None,
    section_index: int,
    section_id: str,
) -> dict[str, Any] | None:
    cached = read_section_summary(
        paper_id,
        prompt_version=prompt_version,
        source_hash=source_hash,
        provider=provider,
        model=model,
        section_index=section_index,
        section_id=section_id,
    )
    if not isinstance(cached, dict):
        return None
    try:
        Draft202012Validator(SECTION_SUMMARY_SCHEMA).validate(cached)
    except Exception:
        return None
    return cached


def _emit(progress_callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if progress_callback:
        progress_callback(dict(event))
