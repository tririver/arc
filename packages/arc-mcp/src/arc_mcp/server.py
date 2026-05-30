from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Callable

from arc_domain import service as domain_service
from arc_llm.host import detect_host, select_llm_provider
from arc_llm.runner import resolve_llm_config
from arc_paper import service
from arc_paper.batch.db import BatchDB
from arc_paper.batch.runner import export_batch, prefetch_batch, run_batch
from arc_paper.ids import normalize_paper_id
from arc_typeset import md2pdf as typeset_md2pdf
from arc_typeset import translate as typeset_translate
from pydantic import Field

from .jobs import MCPJobCancelled, MCPJobManager, resolve_inline_wait_seconds


ToolHandler = Callable[[dict[str, Any]], Any]
MCP_JOBS = MCPJobManager(max_workers=1)

SERVER_INSTRUCTIONS = (
    "Use ARC when a user asks about theoretical-physics papers or arXiv papers: "
    "titles, abstracts, authors, references, citing papers, citation counts, "
    "ar5iv table of contents, sections, full-text search, equation context, cached LLM paper summaries, "
    "cached research-domain artifacts from a seed paper, converting Markdown reports to PDF, "
    "or translating Markdown reports. "
    "Tools that may call an LLM provider are prefixed with llm_. "
    "Paper IDs may be passed with or without the arXiv: prefix, as INSPIRE recids, "
    "or as DOI identifiers, for example 0911.3380, arXiv:0911.3380, "
    "hep-th/0601001, inspire:837197, or doi:10.1088/1475-7516/2010/04/027. "
    "For one paper use paper_id; for multiple papers use paper_ids."
)

PAPER_ID_DESCRIPTION = (
    "Single paper identifier. arXiv IDs may be written as 0911.3380, "
    "arXiv:0911.3380, or hep-th/0601001. DOI and INSPIRE IDs may be written "
    "as doi:10.1088/1475-7516/2010/04/027 or inspire:837197."
)
PAPER_IDS_DESCRIPTION = "Multiple paper identifiers. Use this instead of paper_id for batch queries."
REFRESH_DESCRIPTION = "Bypass cached data and refetch source metadata or full text when possible."
ENRICH_REFERENCES_DESCRIPTION = (
    "When true, fetch and cache each referenced paper's INSPIRE metadata through arc-paper, "
    "including title, abstract, authors, and identifiers when available."
)
SECTION_DESCRIPTION = "Section heading, section number, or section id to retrieve from the ar5iv full text."
QUERY_DESCRIPTION = "Equation label, symbol, or phrase to find nearby equation context in the paper."
FULL_TEXT_QUERY_DESCRIPTION = "Word or phrase to search for in cached parsed ar5iv text."
SEARCH_LIMIT_DESCRIPTION = "Maximum number of full-text search hits to return, clamped to 1..200."
SEARCH_CONTEXT_DESCRIPTION = "Number of nearby parsed section lines to include in each hit snippet, clamped to 0..5."
CASE_SENSITIVE_DESCRIPTION = "When true, full-text search is case-sensitive."
TEXT_DESCRIPTION = "Natural-language text that may contain arXiv, INSPIRE, or DOI paper identifiers."
BATCH_NAME_DESCRIPTION = "Name of a summary batch stored in ARC's local SQLite batch database."
DOMAIN_INTENT_DESCRIPTION = "Optional description of the user's scientific interest or desired subfield scope."
DOMAIN_ID_DESCRIPTION = "Optional ARC domain id returned by llm_domain_build or arc-domain init."
CITER_LIMIT_DESCRIPTION = "Maximum number of citing papers to return from INSPIRE, clamped to 1..1000."
CITER_SORT_DESCRIPTION = "INSPIRE citer sort order: mostrecent or mostcited."
LLM_PROVIDER_DESCRIPTION = "LLM provider: auto or a built-in provider (codex-cli, claude-cli, manual)."
LLM_MODEL_DESCRIPTION = "Optional model name passed to the selected LLM provider."
LLM_MODEL_TIER_DESCRIPTION = "Optional LLM model tier: low, medium, or high."
BACKGROUND_DESCRIPTION = (
    "When true, start the job and return a background job id immediately instead of waiting inline."
)
JOB_CANCEL_DESCRIPTION = (
    "Cancel an MCP job. Do not use this unless the user explicitly asks; cancelling may waste work "
    "and leave a requested cached artifact unfinished."
)
MD2PDF_INPUT_DESCRIPTION = "Markdown file to convert to PDF."
MD2PDF_OUTPUT_DESCRIPTION = "Optional output PDF path. Defaults to the input path with a .pdf suffix."
MD2PDF_TEXLIVE_BIN_DESCRIPTION = (
    "Optional TeX Live bin directory to prepend to PATH. Pass an empty string to avoid modifying PATH."
)
MD2PDF_RESOURCE_PATH_DESCRIPTION = (
    "Optional Pandoc resource path entries for resolving relative images and assets."
)
MD2PDF_TIMEOUT_DESCRIPTION = "Pandoc/XeLaTeX timeout in seconds. Defaults to 600."
TRANSLATE_INPUT_DESCRIPTION = "Markdown file to translate."
TRANSLATE_OUTPUT_DESCRIPTION = "Optional translated Markdown output path. Defaults to input.<target_locale>.md."
TRANSLATE_TARGET_LANGUAGE_DESCRIPTION = "Target natural language for translation. Defaults to Chinese."
TRANSLATE_TARGET_LOCALE_DESCRIPTION = "Target locale suffix for output files. Defaults to zh_CN."
TRANSLATE_PROJECT_DIR_DESCRIPTION = "Project directory to scan for same-folder Markdown/PDF report pairs."
TRANSLATE_QUALITY_DESCRIPTION = "When true, run an additional LLM QA/revision pass after fast translation."
TRANSLATE_OVERWRITE_DESCRIPTION = "When true, overwrite existing translated Markdown/PDF outputs."
MODEL_TIER_DESCRIPTION = "LLM model tier for translation work. Defaults to low for speed and cost."
PARSE_SOURCE_PATH_DESCRIPTION = "Optional local source path. Extension may be .html, .tex, or .pdf."
PARSE_SOURCE_DESCRIPTION = "Source adapter: auto, ar5iv, html, tex, pdf, or tex-pdf."
PARSE_ID_DESCRIPTION = "Parsed source id to store in paper_id and use for sources cache filename."
PARSE_HTML_PATH_DESCRIPTION = "Optional local HTML source path."
PARSE_TEX_PATH_DESCRIPTION = "Optional local TeX source path."
PARSE_PDF_PATH_DESCRIPTION = "Optional local PDF source path."
PARSED_SOURCE_ID_DESCRIPTION = "Parsed source id whose cached equation should be annotated."
PARSED_EQUATION_ID_DESCRIPTION = "Parsed equation id to annotate, for example eq_00042."
PARSED_EQUATION_STATUS_DESCRIPTION = "Equation annotation status: problematic, needs_recache, or resolved."
PARSED_EQUATION_REASON_DESCRIPTION = "Short reason explaining why this parsed equation annotation was added."

PaperId = Annotated[str | None, Field(description=PAPER_ID_DESCRIPTION)]
PaperIds = Annotated[list[str] | None, Field(description=PAPER_IDS_DESCRIPTION)]
Refresh = Annotated[bool, Field(description=REFRESH_DESCRIPTION)]
EnrichReferences = Annotated[bool, Field(description=ENRICH_REFERENCES_DESCRIPTION)]
Section = Annotated[str, Field(description=SECTION_DESCRIPTION)]
EquationQuery = Annotated[str, Field(description=QUERY_DESCRIPTION)]
FullTextQuery = Annotated[str, Field(description=FULL_TEXT_QUERY_DESCRIPTION)]
SearchLimit = Annotated[int, Field(description=SEARCH_LIMIT_DESCRIPTION)]
SearchContext = Annotated[int, Field(description=SEARCH_CONTEXT_DESCRIPTION)]
CaseSensitive = Annotated[bool, Field(description=CASE_SENSITIVE_DESCRIPTION)]
NaturalText = Annotated[str, Field(description=TEXT_DESCRIPTION)]
BatchName = Annotated[str, Field(description=BATCH_NAME_DESCRIPTION)]
DomainIntent = Annotated[str, Field(description=DOMAIN_INTENT_DESCRIPTION)]
DomainId = Annotated[str | None, Field(description=DOMAIN_ID_DESCRIPTION)]
CiterLimit = Annotated[int, Field(description=CITER_LIMIT_DESCRIPTION)]
CiterSort = Annotated[str, Field(description=CITER_SORT_DESCRIPTION)]
LLMProvider = Annotated[str, Field(description=LLM_PROVIDER_DESCRIPTION)]
LLMModel = Annotated[str | None, Field(description=LLM_MODEL_DESCRIPTION)]
LLMModelTier = Annotated[str | None, Field(description=LLM_MODEL_TIER_DESCRIPTION)]
Background = Annotated[bool, Field(description=BACKGROUND_DESCRIPTION)]
Md2PdfInput = Annotated[str, Field(description=MD2PDF_INPUT_DESCRIPTION)]
Md2PdfOutput = Annotated[str | None, Field(description=MD2PDF_OUTPUT_DESCRIPTION)]
Md2PdfTexliveBin = Annotated[str | None, Field(description=MD2PDF_TEXLIVE_BIN_DESCRIPTION)]
Md2PdfResourcePath = Annotated[list[str] | None, Field(description=MD2PDF_RESOURCE_PATH_DESCRIPTION)]
Md2PdfTimeout = Annotated[float, Field(description=MD2PDF_TIMEOUT_DESCRIPTION)]
TranslateInput = Annotated[str, Field(description=TRANSLATE_INPUT_DESCRIPTION)]
TranslateOutput = Annotated[str | None, Field(description=TRANSLATE_OUTPUT_DESCRIPTION)]
TranslateTargetLanguage = Annotated[str, Field(description=TRANSLATE_TARGET_LANGUAGE_DESCRIPTION)]
TranslateTargetLocale = Annotated[str, Field(description=TRANSLATE_TARGET_LOCALE_DESCRIPTION)]
TranslateProjectDir = Annotated[str, Field(description=TRANSLATE_PROJECT_DIR_DESCRIPTION)]
TranslateQuality = Annotated[bool, Field(description=TRANSLATE_QUALITY_DESCRIPTION)]
TranslateOverwrite = Annotated[bool, Field(description=TRANSLATE_OVERWRITE_DESCRIPTION)]
ModelTier = Annotated[str, Field(description=MODEL_TIER_DESCRIPTION)]
ParseSourcePath = Annotated[str | None, Field(description=PARSE_SOURCE_PATH_DESCRIPTION)]
ParseSource = Annotated[str, Field(description=PARSE_SOURCE_DESCRIPTION)]
ParseId = Annotated[str | None, Field(description=PARSE_ID_DESCRIPTION)]
ParseHtmlPath = Annotated[str | None, Field(description=PARSE_HTML_PATH_DESCRIPTION)]
ParseTexPath = Annotated[str | None, Field(description=PARSE_TEX_PATH_DESCRIPTION)]
ParsePdfPath = Annotated[str | None, Field(description=PARSE_PDF_PATH_DESCRIPTION)]
ParsedSourceId = Annotated[str, Field(description=PARSED_SOURCE_ID_DESCRIPTION)]
ParsedEquationId = Annotated[str, Field(description=PARSED_EQUATION_ID_DESCRIPTION)]
ParsedEquationStatus = Annotated[str, Field(description=PARSED_EQUATION_STATUS_DESCRIPTION)]
ParsedEquationReason = Annotated[str, Field(description=PARSED_EQUATION_REASON_DESCRIPTION)]


class ToolInputError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    try:
        handler = TOOL_HANDLERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown ARC MCP tool: {name}") from exc
    try:
        return handler(arguments)
    except ToolInputError as exc:
        return _error(exc.code, str(exc))


def _paper_ids(args: dict[str, Any]):
    return _select_paper_ids(args.get("paper_id"), args.get("paper_ids"), required=True)


def _optional_paper_ids(args: dict[str, Any]):
    return _select_paper_ids(args.get("paper_id"), args.get("paper_ids"), required=False)


def _select_paper_ids(paper_id: Any = None, paper_ids: Any = None, *, required: bool):
    has_one = paper_id is not None and str(paper_id).strip() != ""
    if isinstance(paper_ids, list):
        has_many = any(str(item).strip() for item in paper_ids)
    else:
        has_many = paper_ids is not None
    if has_one and has_many:
        raise ToolInputError("paper_ids_ambiguous", "Exactly one of paper_id or paper_ids must be provided.")
    if not has_one and not has_many:
        if not required:
            return None
        raise ToolInputError("paper_ids_required", "Exactly one of paper_id or paper_ids must be provided.")
    return paper_ids if has_many else paper_id


TOOL_HANDLERS: dict[str, ToolHandler] = {
    "md2pdf": lambda args: _start_md2pdf_job_response(args),
    "translate": lambda args: _start_translate_job_response(args),
    "batch_translate": lambda args: _start_batch_translate_job_response(args),
    "extract_paper_ids": lambda args: service.extract_paper_ids(str(args.get("text", ""))),
    "paper_ids_safe_dir_name": lambda args: service.paper_ids_safe_dir_name(_paper_ids(args)),
    "get_title": lambda args: service.get_title(_paper_ids(args), refresh=bool(args.get("refresh", False))),
    "get_abstract": lambda args: service.get_abstract(_paper_ids(args), refresh=bool(args.get("refresh", False))),
    "get_authors": lambda args: service.get_authors(_paper_ids(args), refresh=bool(args.get("refresh", False))),
    "get_metadata": lambda args: service.get_metadata(_paper_ids(args), refresh=bool(args.get("refresh", False))),
    "get_citers": lambda args: service.get_citers(
        _paper_ids(args),
        refresh=bool(args.get("refresh", False)),
        limit=int(args.get("limit", 1000)),
        sort=str(args.get("sort", "mostrecent")),
    ),
    "get_citer_count": lambda args: service.get_citer_count(_paper_ids(args), refresh=bool(args.get("refresh", False))),
    "get_references": lambda args: service.get_references(
        _paper_ids(args),
        refresh=bool(args.get("refresh", False)),
        enrich=bool(args.get("enrich", False)),
    ),
    "get_toc": lambda args: service.get_toc(_paper_ids(args), refresh=bool(args.get("refresh", False))),
    "get_section": lambda args: service.get_section(
        _paper_ids(args),
        str(args["section"]),
        refresh=bool(args.get("refresh", False)),
    ),
    "search_full_text": lambda args: service.search_full_text(
        _optional_paper_ids(args),
        query=str(args.get("query", "")),
        refresh=bool(args.get("refresh", False)),
        limit=int(args.get("limit", 20)),
        context=int(args.get("context", 1)),
        case_sensitive=bool(args.get("case_sensitive", False)),
    ),
    "get_equation_context": lambda args: service.get_equation_context(
        _paper_ids(args),
        str(args["query"]),
        refresh=bool(args.get("refresh", False)),
    ),
    "parse": lambda args: service.parse_source(
        args.get("source_path"),
        source=str(args.get("source", "auto")),
        source_id=args.get("source_id") or args.get("id"),
        paper_id=args.get("paper_id"),
        html_path=args.get("html_path"),
        tex_path=args.get("tex_path"),
        pdf_path=args.get("pdf_path"),
        refresh=bool(args.get("refresh", False)),
    ),
    "mark_parsed_equation": lambda args: service.mark_parsed_equation(
        str(args["source_id"]),
        str(args["equation_id"]),
        status=str(args.get("status", "problematic")),
        reason=str(args.get("reason", "")),
    ),
    "llm_get_summary": lambda args: _cached_or_start_summary_job(args),
    "llm_generate_summary": lambda args: _start_summary_job_response(
        _paper_ids(args),
        provider=str(args.get("provider", "auto")),
        model=args.get("model"),
        model_tier=args.get("model_tier"),
        refresh=bool(args.get("refresh", False)),
        background=bool(args.get("background", False)),
    ),
    "llm_infer_main_references": lambda args: _start_reference_inference_job_response(args),
    "job_status": lambda args: job_status(str(args["job_id"])),
    "job_result": lambda args: job_result(str(args["job_id"])),
    "cancel_job": lambda args: cancel_job(str(args["job_id"])),
    "list_jobs": lambda args: list_jobs(),
    "store_llm_summary": lambda args: service.store_llm_summary(str(args["paper_id"]), args["summary"]),
    "doctor_host": lambda args: {
        "ok": True,
        "data": detect_host().__dict__,
        "errors": [],
        "meta": {},
    },
    "doctor_provider": lambda args: {
        "ok": True,
        "data": {
            "provider": select_llm_provider().provider,
            "host": select_llm_provider().host.host,
            "signals": select_llm_provider().signals,
        },
        "errors": [],
        "meta": {},
    },
    "doctor_cache": lambda args: service.doctor_cache(args.get("paper_id")),
    "llm_domain_build": lambda args: _start_domain_job_response(args),
    "domain_status": lambda args: _domain_status_response(args),
    "domain_get_summary": lambda args: _domain_artifact(args, artifact="summary"),
    "domain_get_graph": lambda args: _domain_artifact(args, artifact="graph"),
    "llm_domain_get_summary": lambda args: _domain_artifact_or_start(args, artifact="summary"),
    "llm_domain_get_graph": lambda args: _domain_artifact_or_start(args, artifact="graph"),
    "summary_batch_create": lambda args: _summary_batch_create_response(args),
    "summary_batch_prefetch": lambda args: _summary_batch_prefetch_response(args),
    "llm_summary_batch_run": lambda args: _run_summary_batch_inline(args),
    "summary_batch_status": lambda args: _summary_batch_status_response(args),
    "summary_batch_export": lambda args: _summary_batch_export_response(args),
    "summary_batch_retry_failed": lambda args: _summary_batch_retry_failed_response(args),
}


def _start_md2pdf_job_response(args: dict[str, Any]) -> dict[str, Any]:
    input_path = Path(str(args["input"]))
    output_path = Path(str(args["output"])) if args.get("output") else None
    texlive_bin_raw = args.get("texlive_bin", str(typeset_md2pdf.DEFAULT_TEXLIVE_BIN))
    texlive_bin = Path(str(texlive_bin_raw)) if texlive_bin_raw else None
    margin = str(args.get("margin", typeset_md2pdf.DEFAULT_MARGIN))
    mainfont = str(args.get("mainfont", typeset_md2pdf.DEFAULT_MAINFONT))
    cjk_mainfont = str(args.get("cjk_mainfont", typeset_md2pdf.DEFAULT_CJK_MAINFONT))
    resource_paths = [Path(str(path)) for path in args.get("resource_path") or []] or None
    try:
        timeout_seconds = _optional_float(args.get("timeout_seconds", typeset_md2pdf.DEFAULT_TIMEOUT_SECONDS))
    except ValueError as exc:
        return _error("invalid_timeout", str(exc))

    payload = {
        "input": str(input_path),
        "output": str(output_path) if output_path else None,
        "texlive_bin": str(texlive_bin) if texlive_bin else "",
        "margin": margin,
        "mainfont": mainfont,
        "cjk_mainfont": cjk_mainfont,
        "resource_path": [str(path) for path in resource_paths or []],
        "timeout_seconds": timeout_seconds,
        "background": True,
    }
    job_id = MCP_JOBS.start(
        job_type="md2pdf",
        payload=payload,
        runner=lambda progress, cancel: _run_md2pdf_job(
            input_path=input_path,
            output_path=output_path,
            texlive_bin=texlive_bin,
            margin=margin,
            mainfont=mainfont,
            cjk_mainfont=cjk_mainfont,
            resource_paths=resource_paths,
            timeout_seconds=timeout_seconds,
            progress=progress,
            cancel=cancel,
        ),
        status_resolver=_arc_result_status,
    )
    return _wait_or_background(
        job_id,
        message="Markdown to PDF conversion is running in the background.",
        poll_after_seconds=1,
        background=True,
    )


def _run_md2pdf_job(
    *,
    input_path: Path,
    output_path: Path | None,
    texlive_bin: Path | None,
    margin: str,
    mainfont: str,
    cjk_mainfont: str,
    resource_paths: list[Path] | None,
    timeout_seconds: float | None,
    progress: Callable[[dict[str, Any]], None],
    cancel: Callable[[], bool],
) -> dict[str, Any]:
    if cancel():
        raise MCPJobCancelled("MCP job cancellation was requested.")
    progress(
        {
            "event": "md2pdf_started",
            "input": str(input_path),
            "output": str(output_path) if output_path else None,
        }
    )
    result = typeset_md2pdf.convert_markdown_to_pdf(
        input_path=input_path,
        output_path=output_path,
        texlive_bin=texlive_bin,
        margin=margin,
        mainfont=mainfont,
        cjk_mainfont=cjk_mainfont,
        resource_paths=resource_paths,
        timeout_seconds=timeout_seconds,
    )
    progress({"event": "md2pdf_completed" if _all_ok(result) else "md2pdf_failed"})
    return result


def _start_translate_job_response(args: dict[str, Any]) -> dict[str, Any]:
    input_path = Path(str(args["input"]))
    output_path = Path(str(args["output"])) if args.get("output") else None
    target_language = str(args.get("target_language", typeset_translate.DEFAULT_TARGET_LANGUAGE))
    target_locale = str(args.get("target_locale", typeset_translate.DEFAULT_TARGET_LOCALE))
    provider = str(args.get("provider", "auto"))
    model = args.get("model")
    model_tier = str(args.get("model_tier", typeset_translate.DEFAULT_MODEL_TIER))
    quality = bool(args.get("quality", False))
    overwrite = bool(args.get("overwrite", False))
    payload = {
        "input": str(input_path),
        "output": str(output_path) if output_path else None,
        "target_language": target_language,
        "target_locale": target_locale,
        "provider": provider,
        "model": model,
        "model_tier": model_tier,
        "quality": quality,
        "overwrite": overwrite,
        "background": True,
    }
    job_id = MCP_JOBS.start(
        job_type="translate",
        payload=payload,
        runner=lambda progress, cancel: _run_translate_job(
            input_path=input_path,
            output_path=output_path,
            target_language=target_language,
            target_locale=target_locale,
            provider=provider,
            model=model,
            model_tier=model_tier,
            quality=quality,
            overwrite=overwrite,
            progress=progress,
            cancel=cancel,
        ),
        status_resolver=_arc_result_status,
    )
    return _wait_or_background(
        job_id,
        message="Markdown translation is running in the background.",
        poll_after_seconds=5,
        background=True,
    )


def _run_translate_job(
    *,
    input_path: Path,
    output_path: Path | None,
    target_language: str,
    target_locale: str,
    provider: str,
    model: str | None,
    model_tier: str,
    quality: bool,
    overwrite: bool,
    progress: Callable[[dict[str, Any]], None],
    cancel: Callable[[], bool],
) -> dict[str, Any]:
    if cancel():
        raise MCPJobCancelled("MCP job cancellation was requested.")
    progress({"event": "translate_started", "input": str(input_path), "target_locale": target_locale})
    result = typeset_translate.translate_markdown(
        input_path=input_path,
        output_path=output_path,
        target_language=target_language,
        target_locale=target_locale,
        provider=provider,
        model=model,
        model_tier=model_tier,
        quality=quality,
        convert_pdf=True,
        overwrite=overwrite,
    )
    progress({"event": "translate_completed" if _all_ok(result) else "translate_failed"})
    return result


def _start_batch_translate_job_response(args: dict[str, Any]) -> dict[str, Any]:
    project_dir = Path(str(args["project_dir"]))
    target_language = str(args.get("target_language", typeset_translate.DEFAULT_TARGET_LANGUAGE))
    target_locale = str(args.get("target_locale", typeset_translate.DEFAULT_TARGET_LOCALE))
    provider = str(args.get("provider", "auto"))
    model = args.get("model")
    model_tier = str(args.get("model_tier", typeset_translate.DEFAULT_MODEL_TIER))
    quality = bool(args.get("quality", False))
    overwrite = bool(args.get("overwrite", False))
    payload = {
        "project_dir": str(project_dir),
        "target_language": target_language,
        "target_locale": target_locale,
        "provider": provider,
        "model": model,
        "model_tier": model_tier,
        "quality": quality,
        "overwrite": overwrite,
        "background": True,
    }
    job_id = MCP_JOBS.start(
        job_type="batch_translate",
        payload=payload,
        runner=lambda progress, cancel: _run_batch_translate_job(
            project_dir=project_dir,
            target_language=target_language,
            target_locale=target_locale,
            provider=provider,
            model=model,
            model_tier=model_tier,
            quality=quality,
            overwrite=overwrite,
            progress=progress,
            cancel=cancel,
        ),
        status_resolver=_arc_result_status,
    )
    return _wait_or_background(
        job_id,
        message="Batch Markdown translation is running in the background.",
        poll_after_seconds=10,
        background=True,
    )


def _run_batch_translate_job(
    *,
    project_dir: Path,
    target_language: str,
    target_locale: str,
    provider: str,
    model: str | None,
    model_tier: str,
    quality: bool,
    overwrite: bool,
    progress: Callable[[dict[str, Any]], None],
    cancel: Callable[[], bool],
) -> dict[str, Any]:
    if cancel():
        raise MCPJobCancelled("MCP job cancellation was requested.")
    progress({"event": "batch_translate_started", "project_dir": str(project_dir), "target_locale": target_locale})
    result = typeset_translate.batch_translate_project(
        project_dir=project_dir,
        target_language=target_language,
        target_locale=target_locale,
        provider=provider,
        model=model,
        model_tier=model_tier,
        quality=quality,
        overwrite=overwrite,
    )
    progress({"event": "batch_translate_completed" if _all_ok(result) else "batch_translate_failed"})
    return result


def _cached_or_start_summary_job(args: dict[str, Any]) -> dict[str, Any]:
    try:
        paper_ids = _paper_ids(args)
    except ToolInputError as exc:
        return _error(exc.code, str(exc))
    if not bool(args.get("refresh", False)):
        cached = service.get_cached_llm_summary(paper_ids)
        if _all_ok(cached):
            return cached
    return _start_summary_job_response(
        paper_ids,
        provider=str(args.get("provider", "auto")),
        model=args.get("model"),
        model_tier=args.get("model_tier"),
        refresh=bool(args.get("refresh", False)),
        background=bool(args.get("background", False)),
    )


def _error(code: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message},
        "errors": [{"code": code, "message": message}],
        "meta": {},
    }


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _start_summary_job_response(
    paper_ids: Any,
    *,
    provider: str,
    model: str | None,
    model_tier: str | None,
    refresh: bool,
    background: bool,
) -> dict[str, Any]:
    try:
        resolve_llm_config(provider=provider, model=model, model_tier=model_tier)
    except Exception as exc:
        return _error("invalid_llm_config", str(exc))
    normalized = _normalize_ids(paper_ids)
    job_id = MCP_JOBS.start(
        job_type="paper_summary",
        payload={
            "paper_ids": normalized,
            "provider": provider,
            "model": model,
            "model_tier": model_tier,
            "refresh": refresh,
            "sections_total": None,
            "sections_completed": 0,
            "current_section": None,
            "background": background,
        },
        runner=lambda progress, cancel: _run_summary_job(
            normalized, provider, model, model_tier, refresh, progress, cancel
        ),
        status_resolver=_arc_result_status,
    )
    return _wait_or_background(
        job_id,
        message="LLM summary is still running in the background.",
        poll_after_seconds=5,
        background=background,
    )


def _start_reference_inference_job_response(args: dict[str, Any]) -> dict[str, Any]:
    text = str(args.get("text") or "")
    provider = str(args.get("provider", "auto"))
    model = args.get("model")
    refresh = bool(args.get("refresh", False))
    extracted = service.extract_paper_ids(text)
    if extracted.get("ok") and extracted.get("data"):
        return service.llm_infer_main_references(text, provider=provider, model=model, refresh=refresh)

    background = bool(args.get("background", False))
    job_id = MCP_JOBS.start(
        job_type="main_reference_inference",
        payload={
            "text": text,
            "provider": provider,
            "model": model,
            "refresh": refresh,
            "background": background,
        },
        runner=lambda progress, cancel: _run_reference_inference_job(
            text,
            provider,
            model,
            refresh,
            progress,
            cancel,
        ),
        status_resolver=_arc_result_status,
    )
    return _wait_or_background(
        job_id,
        message="Main reference inference is still running in the background.",
        poll_after_seconds=5,
        background=background,
    )


def get_summary_job_status(job_id: str) -> dict[str, Any]:
    return job_status(job_id)


def _start_domain_job_response(args: dict[str, Any]) -> dict[str, Any]:
    seed_paper = str(args["seed_paper"])
    intent = str(args.get("intent", ""))
    domain_id = args.get("domain_id")
    provider = str(args.get("provider", "auto"))
    model = args.get("model")
    model_tier = args.get("model_tier")
    refresh = bool(args.get("refresh", False))
    workers = int(args.get("workers", 8))
    background = bool(args.get("background", False))
    job_id = MCP_JOBS.start(
        job_type="domain_build",
        payload={
            "seed_paper": normalize_paper_id(seed_paper),
            "intent": intent,
            "domain_id": domain_id,
            "provider": provider,
            "model": model,
            "model_tier": model_tier,
            "refresh": refresh,
            "workers": workers,
            "background": background,
        },
        runner=lambda progress, cancel: _run_domain_job(
            seed_paper,
            intent,
            domain_id,
            provider,
            model,
            model_tier,
            refresh,
            workers,
            progress,
            cancel,
        ),
        status_resolver=_arc_result_status,
    )
    return _wait_or_background(
        job_id,
        message="Domain build is still running in the background.",
        poll_after_seconds=10,
        background=background,
    )


def _domain_status_response(args: dict[str, Any]) -> dict[str, Any]:
    if args.get("job_id"):
        return get_domain_job_status(str(args["job_id"]))
    return domain_service.status(
        args.get("seed_paper"),
        intent=str(args.get("intent", "")),
        domain_id=args.get("domain_id"),
    )


def _domain_artifact(args: dict[str, Any], *, artifact: str) -> dict[str, Any]:
    if artifact == "summary":
        return domain_service.get_domain_summary(
            args.get("seed_paper"),
            intent=str(args.get("intent", "")),
            domain_id=args.get("domain_id"),
        )
    elif artifact == "graph":
        return domain_service.get_domain_graph(
            args.get("seed_paper"),
            intent=str(args.get("intent", "")),
            domain_id=args.get("domain_id"),
        )
    raise ValueError(f"Unsupported domain artifact: {artifact}")


def _domain_artifact_or_start(args: dict[str, Any], *, artifact: str) -> dict[str, Any]:
    result = _domain_artifact(args, artifact=artifact)
    if result.get("ok") or not args.get("seed_paper"):
        return result
    error_code = (result.get("error") or {}).get("code")
    if error_code not in {"domain_summary_not_available", "domain_summary_invalid", "domain_graph_not_available"}:
        return result
    return _start_domain_job_response(args)


def get_domain_job_status(job_id: str) -> dict[str, Any]:
    snapshot = job_status(job_id)
    if snapshot.get("status") == "job_unknown":
        return snapshot
    if snapshot.get("job_type") != "domain_build":
        return snapshot
    try:
        snapshot["domain_status"] = domain_service.status(
            snapshot.get("seed_paper"),
            intent=str(snapshot.get("intent", "")),
            domain_id=snapshot.get("domain_id"),
        )
    except Exception as exc:
        snapshot["domain_status_error"] = str(exc)
    return snapshot


def _run_domain_job(
    seed_paper: str,
    intent: str,
    domain_id: str | None,
    provider: str,
    model: str | None,
    model_tier: str | None,
    refresh: bool,
    workers: int,
    progress: Callable[[dict[str, Any]], None],
    cancel: Callable[[], bool],
) -> dict[str, Any]:
    if cancel():
        raise MCPJobCancelled("MCP job cancellation was requested.")
    progress({"event": "domain_started", "seed_paper": normalize_paper_id(seed_paper), "intent": intent})
    build_kwargs: dict[str, Any] = {
        "intent": intent,
        "domain_id": domain_id,
        "provider": provider,
        "model": model,
        "refresh": refresh,
        "workers": workers,
    }
    if model_tier is not None:
        build_kwargs["model_tier"] = model_tier
    result = domain_service.build_domain(seed_paper, **build_kwargs)
    progress({"event": "domain_completed" if _all_ok(result) else "domain_failed"})
    return result


def _run_summary_job(
    paper_ids: Any,
    provider: str,
    model: str | None,
    model_tier: str | None,
    refresh: bool,
    progress: Callable[[dict[str, Any]], None],
    cancel: Callable[[], bool],
) -> dict[str, Any]:
    if cancel():
        raise MCPJobCancelled("MCP job cancellation was requested.")
    progress({"event": "job_started"})
    return service.generate_llm_summary(
        paper_ids,
        provider=provider,
        model=model,
        model_tier=model_tier,
        refresh=refresh,
        progress_callback=progress,
    )


def _run_reference_inference_job(
    text: str,
    provider: str,
    model: str | None,
    refresh: bool,
    progress: Callable[[dict[str, Any]], None],
    cancel: Callable[[], bool],
) -> dict[str, Any]:
    if cancel():
        raise MCPJobCancelled("MCP job cancellation was requested.")
    progress({"event": "reference_inference_started"})
    result = service.llm_infer_main_references(text, provider=provider, model=model, refresh=refresh)
    event = "reference_inference_completed" if _all_ok(result) else "reference_inference_failed"
    progress({"event": event})
    return result


def _run_summary_batch_inline(args: dict[str, Any]) -> dict[str, Any]:
    name = str(args["name"])
    provider = str(args.get("provider", "auto"))
    model = args.get("model")
    model_tier = args.get("model_tier")
    try:
        resolve_llm_config(provider=provider, model=model, model_tier=model_tier)
    except Exception as exc:
        return _error("invalid_llm_config", str(exc))
    concurrency = int(args.get("concurrency", 1))
    max_items = args.get("max_items")
    max_items_int = int(max_items) if max_items is not None else None
    background = bool(args.get("background", False))
    job_id = MCP_JOBS.start(
        job_type="summary_batch_run",
        payload={
            "name": name,
            "provider": provider,
            "model": model,
            "model_tier": model_tier,
            "concurrency": concurrency,
            "max_items": max_items_int,
            "background": background,
        },
        runner=lambda progress, cancel: _run_summary_batch_job(
            name, provider, model, model_tier, concurrency, max_items_int, progress, cancel
        ),
        status_resolver=_arc_result_status,
    )
    return _wait_or_background(
        job_id,
        message="LLM summary batch is still running in the background.",
        poll_after_seconds=10,
        background=background,
    )


def _run_summary_batch_job(
    name: str,
    provider: str,
    model: str | None,
    model_tier: str | None,
    concurrency: int,
    max_items: int | None,
    progress: Callable[[dict[str, Any]], None],
    cancel: Callable[[], bool],
) -> dict[str, Any]:
    if cancel():
        raise MCPJobCancelled("MCP job cancellation was requested.")
    progress({"event": "summary_batch_started", "name": name})
    result = run_batch(
        name,
        provider=provider,
        model=model,
        model_tier=model_tier,
        concurrency=concurrency,
        max_items=max_items,
    )
    progress({"event": "summary_batch_completed", "name": name})
    return {"ok": True, "data": result, "errors": [], "meta": {}}


def _summary_batch_create_response(args: dict[str, Any]) -> dict[str, Any]:
    name = str(args["name"])
    prompt_version = str(args.get("prompt_version", "paper-summary-v1"))
    db = BatchDB.default()
    with open(str(args["papers_file"]), encoding="utf-8") as handle:
        paper_ids = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
    db.create_batch(name, paper_ids, prompt_version)
    return {"ok": True, "data": {"batch": name, "counts": db.status_counts(name)}, "errors": [], "meta": {}}


def _summary_batch_prefetch_response(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "data": prefetch_batch(str(args["name"]), workers=int(args.get("workers", 4))),
        "errors": [],
        "meta": {},
    }


def _summary_batch_status_response(args: dict[str, Any]) -> dict[str, Any]:
    name = str(args["name"])
    db = BatchDB.default()
    return {"ok": True, "data": {"batch": name, "counts": db.status_counts(name)}, "errors": [], "meta": {}}


def _summary_batch_export_response(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "data": export_batch(str(args["name"]), output=Path(str(args["output"]))),
        "errors": [],
        "meta": {},
    }


def _summary_batch_retry_failed_response(args: dict[str, Any]) -> dict[str, Any]:
    name = str(args["name"])
    db = BatchDB.default()
    db.retry_failed(name)
    return {"ok": True, "data": {"batch": name, "counts": db.status_counts(name)}, "errors": [], "meta": {}}


def _wait_or_background(
    job_id: str,
    *,
    message: str,
    poll_after_seconds: int,
    background: bool,
) -> dict[str, Any]:
    inline_wait = 0.0 if background else resolve_inline_wait_seconds(server_name="arc")
    if not background and MCP_JOBS.wait(job_id, timeout=inline_wait):
        status = job_status(job_id)
        wrapped = MCP_JOBS.result(job_id)
        result = wrapped.get("result") if isinstance(wrapped, dict) else None
        if isinstance(result, dict):
            return _attach_job_meta(result, status)
        return wrapped
    status = job_status(job_id)
    return {
        "ok": False,
        "status": "job_running",
        "job_id": job_id,
        "job_type": status.get("job_type"),
        "message": message,
        "inline_wait_seconds": inline_wait,
        "background_requested": background,
        "next": {
            "cli_command": f"arc-mcp watch {job_id} --json",
            "tool": "job_status",
            "arguments": {"job_id": job_id},
            "poll_after_seconds": poll_after_seconds,
        },
        "job": status,
        "errors": [],
        "meta": {},
    }


def job_status(job_id: str) -> dict[str, Any]:
    status = MCP_JOBS.status(job_id)
    if status.get("job_type") == "paper_summary":
        _normalize_summary_status(status)
    return status


def job_result(job_id: str) -> dict[str, Any]:
    return MCP_JOBS.result(job_id)


def cancel_job(job_id: str) -> dict[str, Any]:
    return MCP_JOBS.cancel(job_id)


def list_jobs() -> dict[str, Any]:
    return MCP_JOBS.list_jobs()


def _normalize_summary_status(status: dict[str, Any]) -> None:
    event_name = str(status.get("phase") or "")
    if event_name in {"section_started", "section_cached", "section_completed"}:
        status["current_section"] = {
            "paper_id": status.get("paper_id"),
            "section_index": status.get("section_index"),
            "sections_total": status.get("sections_total"),
            "section_id": status.get("section_id"),
            "title": status.get("title"),
        }
    elif event_name.startswith("final_") or event_name in {"done", "failed", "needs_llm", "cancelled"}:
        status["current_section"] = None


def _attach_job_meta(result: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    output = dict(result)
    meta = dict(output.get("meta") or {})
    meta["job"] = {
        "job_id": status.get("job_id"),
        "job_type": status.get("job_type"),
        "status": status.get("status"),
        "phase": status.get("phase"),
        "started_at": status.get("started_at"),
        "updated_at": status.get("updated_at"),
        "finished_at": status.get("finished_at"),
    }
    output["meta"] = meta
    return output


def _arc_result_status(result: Any) -> str:
    if _all_ok(result):
        return "done"
    if isinstance(result, dict) and result.get("status") == "needs_llm":
        return "needs_llm"
    return "failed"


def _all_ok(result: Any) -> bool:
    if isinstance(result, dict) and "ok" in result:
        return result.get("ok") is True
    if isinstance(result, dict):
        return bool(result) and all(isinstance(item, dict) and item.get("ok") is True for item in result.values())
    return False


def _normalize_ids(paper_ids: Any) -> Any:
    if isinstance(paper_ids, str):
        return normalize_paper_id(paper_ids)
    return [normalize_paper_id(str(item)) for item in paper_ids or []]


def run_mcp_server() -> int:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit("The 'mcp' package is required to run arc-mcp.") from exc

    app = FastMCP("arc", instructions=SERVER_INSTRUCTIONS)
    _register_tools(app)

    app.run()
    return 0


def main() -> None:
    raise SystemExit(run_mcp_server())


def _one_or_many(paper_id: str | None = None, paper_ids: list[str] | None = None):
    return _select_paper_ids(paper_id, paper_ids, required=True)


def _optional_one_or_many(paper_id: str | None = None, paper_ids: list[str] | None = None):
    return _select_paper_ids(paper_id, paper_ids, required=False)


def _tool_input_result(call: Callable[[], Any]) -> Any:
    try:
        return call()
    except ToolInputError as exc:
        return _error(exc.code, str(exc))


def _register_tools(app: Any) -> None:
    @app.tool(
        description=(
            "Convert a Markdown file to PDF using Pandoc and XeLaTeX. "
            "This starts a background job immediately, does not wait inline, "
            "and is intended for math-heavy ARC reports with CJK font support."
        )
    )
    def md2pdf(
        input: Md2PdfInput,
        output: Md2PdfOutput = None,
        texlive_bin: Md2PdfTexliveBin = str(typeset_md2pdf.DEFAULT_TEXLIVE_BIN),
        margin: Annotated[str, Field(description="LaTeX geometry margin value.")] = typeset_md2pdf.DEFAULT_MARGIN,
        mainfont: Annotated[str, Field(description="Main font passed to Pandoc's LaTeX template.")] = (
            typeset_md2pdf.DEFAULT_MAINFONT
        ),
        cjk_mainfont: Annotated[str, Field(description="CJK main font passed to Pandoc's LaTeX template.")] = (
            typeset_md2pdf.DEFAULT_CJK_MAINFONT
        ),
        resource_path: Md2PdfResourcePath = None,
        timeout_seconds: Md2PdfTimeout = typeset_md2pdf.DEFAULT_TIMEOUT_SECONDS,
    ) -> Any:
        return _start_md2pdf_job_response(
            {
                "input": input,
                "output": output,
                "texlive_bin": texlive_bin,
                "margin": margin,
                "mainfont": mainfont,
                "cjk_mainfont": cjk_mainfont,
                "resource_path": resource_path,
                "timeout_seconds": timeout_seconds,
            }
        )

    @app.tool(
        description=(
            "Translate a Markdown report with arc-llm, then convert the translated Markdown to PDF. "
            "This starts a background job immediately and defaults to Chinese with low-tier LLMs."
        )
    )
    def translate(
        input: TranslateInput,
        output: TranslateOutput = None,
        target_language: TranslateTargetLanguage = typeset_translate.DEFAULT_TARGET_LANGUAGE,
        target_locale: TranslateTargetLocale = typeset_translate.DEFAULT_TARGET_LOCALE,
        provider: LLMProvider = "auto",
        model: LLMModel = None,
        model_tier: ModelTier = typeset_translate.DEFAULT_MODEL_TIER,
        quality: TranslateQuality = False,
        overwrite: TranslateOverwrite = False,
    ) -> Any:
        return _start_translate_job_response(
            {
                "input": input,
                "output": output,
                "target_language": target_language,
                "target_locale": target_locale,
                "provider": provider,
                "model": model,
                "model_tier": model_tier,
                "quality": quality,
                "overwrite": overwrite,
            }
        )

    @app.tool(
        description=(
            "Find same-folder Markdown/PDF report pairs in a project and translate missing locale outputs. "
            "This starts a background job immediately and defaults to Chinese with low-tier LLMs."
        )
    )
    def batch_translate(
        project_dir: TranslateProjectDir,
        target_language: TranslateTargetLanguage = typeset_translate.DEFAULT_TARGET_LANGUAGE,
        target_locale: TranslateTargetLocale = typeset_translate.DEFAULT_TARGET_LOCALE,
        provider: LLMProvider = "auto",
        model: LLMModel = None,
        model_tier: ModelTier = typeset_translate.DEFAULT_MODEL_TIER,
        quality: TranslateQuality = False,
        overwrite: TranslateOverwrite = False,
    ) -> Any:
        return _start_batch_translate_job_response(
            {
                "project_dir": project_dir,
                "target_language": target_language,
                "target_locale": target_locale,
                "provider": provider,
                "model": model,
                "model_tier": model_tier,
                "quality": quality,
                "overwrite": overwrite,
            }
        )

    @app.tool(
        description=(
            "Extract normalized paper identifiers from natural-language text. "
            "Finds explicit arXiv, INSPIRE, DOI, and bare arXiv-like identifiers."
        )
    )
    def extract_paper_ids(text: NaturalText) -> Any:
        return service.extract_paper_ids(text)

    @app.tool(
        description=(
            "Create a filesystem-safe directory-name stem from one or more paper identifiers. "
            "Examples: arXiv:0911.3380 -> 0911.3380; multiple ids are joined with _x_."
        )
    )
    def paper_ids_safe_dir_name(paper_id: PaperId = None, paper_ids: PaperIds = None) -> Any:
        return _tool_input_result(lambda: service.paper_ids_safe_dir_name(_one_or_many(paper_id, paper_ids)))

    @app.tool(
        description=(
            "Get the title of one or more arXiv papers from INSPIRE. "
            "Use this when the user asks for a paper title or to identify a paper."
        )
    )
    def get_title(paper_id: PaperId = None, paper_ids: PaperIds = None, refresh: Refresh = False) -> Any:
        return _tool_input_result(lambda: service.get_title(_one_or_many(paper_id, paper_ids), refresh=refresh))

    @app.tool(
        description=(
            "Get the abstract of one or more arXiv papers from INSPIRE. "
            "Use this for paper overview, motivation, or abstract lookup."
        )
    )
    def get_abstract(paper_id: PaperId = None, paper_ids: PaperIds = None, refresh: Refresh = False) -> Any:
        return _tool_input_result(lambda: service.get_abstract(_one_or_many(paper_id, paper_ids), refresh=refresh))

    @app.tool(
        description=(
            "Get authors for one or more arXiv papers from INSPIRE. "
            "Use this when the user asks who wrote a paper."
        )
    )
    def get_authors(paper_id: PaperId = None, paper_ids: PaperIds = None, refresh: Refresh = False) -> Any:
        return _tool_input_result(lambda: service.get_authors(_one_or_many(paper_id, paper_ids), refresh=refresh))

    @app.tool(
        description=(
            "Get normalized INSPIRE metadata for one or more arXiv papers, including title, abstract, "
            "authors, identifiers, year, DOI, INSPIRE record id, and citation count."
        )
    )
    def get_metadata(paper_id: PaperId = None, paper_ids: PaperIds = None, refresh: Refresh = False) -> Any:
        return _tool_input_result(lambda: service.get_metadata(_one_or_many(paper_id, paper_ids), refresh=refresh))

    @app.tool(
        description=(
            "Get papers that cite one or more arXiv papers using INSPIRE. "
            "Citer data is cached with a time limit because it changes over time."
        )
    )
    def get_citers(
        paper_id: PaperId = None,
        paper_ids: PaperIds = None,
        refresh: Refresh = False,
        limit: CiterLimit = 1000,
        sort: CiterSort = "mostrecent",
    ) -> Any:
        return _tool_input_result(
            lambda: service.get_citers(_one_or_many(paper_id, paper_ids), refresh=refresh, limit=limit, sort=sort)
        )

    @app.tool(
        description=(
            "Get the INSPIRE citation count for one or more arXiv papers. "
            "Use this when only the number of citing papers is needed."
        )
    )
    def get_citer_count(paper_id: PaperId = None, paper_ids: PaperIds = None, refresh: Refresh = False) -> Any:
        return _tool_input_result(lambda: service.get_citer_count(_one_or_many(paper_id, paper_ids), refresh=refresh))

    @app.tool(
        description=(
            "Get the bibliography or reference list for one or more arXiv papers from INSPIRE. "
            "Use this when the user asks what a paper cites. Set enrich=true when titles, abstracts, "
            "authors, and identifiers are needed for referenced papers."
        )
    )
    def get_references(
        paper_id: PaperId = None,
        paper_ids: PaperIds = None,
        refresh: Refresh = False,
        enrich: EnrichReferences = False,
    ) -> Any:
        return _tool_input_result(
            lambda: service.get_references(_one_or_many(paper_id, paper_ids), refresh=refresh, enrich=enrich)
        )

    @app.tool(
        description=(
            "Get the table of contents from ar5iv full text for one or more arXiv papers. "
            "Use this before selecting sections to read."
        )
    )
    def get_toc(paper_id: PaperId = None, paper_ids: PaperIds = None, refresh: Refresh = False) -> Any:
        return _tool_input_result(lambda: service.get_toc(_one_or_many(paper_id, paper_ids), refresh=refresh))

    @app.tool(
        description=(
            "Get a specific section from ar5iv full text for one or more arXiv papers. "
            "If the section is not found, the response includes the table of contents."
        )
    )
    def get_section(
        section: Section,
        paper_id: PaperId = None,
        paper_ids: PaperIds = None,
        refresh: Refresh = False,
    ) -> Any:
        return _tool_input_result(
            lambda: service.get_section(_one_or_many(paper_id, paper_ids), section, refresh=refresh)
        )

    @app.tool(
        description=(
            "Search cached parsed ar5iv text for one or more papers, or all cached papers when no paper_id "
            "is supplied. Use returned MCP or CLI commands to fetch the whole section or paper metadata."
        )
    )
    def search_full_text(
        query: FullTextQuery,
        paper_id: PaperId = None,
        paper_ids: PaperIds = None,
        refresh: Refresh = False,
        limit: SearchLimit = 20,
        context: SearchContext = 1,
        case_sensitive: CaseSensitive = False,
    ) -> Any:
        return _tool_input_result(
            lambda: service.search_full_text(
                _optional_one_or_many(paper_id, paper_ids),
                query=query,
                refresh=refresh,
                limit=limit,
                context=context,
                case_sensitive=case_sensitive,
            )
        )

    @app.tool(
        description=(
            "Find equation context in ar5iv full text for one or more arXiv papers. "
            "Use this for equation labels, symbols, or nearby explanatory text."
        )
    )
    def get_equation_context(
        query: EquationQuery,
        paper_id: PaperId = None,
        paper_ids: PaperIds = None,
        refresh: Refresh = False,
    ) -> Any:
        return _tool_input_result(
            lambda: service.get_equation_context(_one_or_many(paper_id, paper_ids), query, refresh=refresh)
        )

    @app.tool(
        name="parse",
        description=(
            "Parse ar5iv, local HTML, TeX, PDF, or TeX+PDF into ARC's canonical parsed JSON cache."
        )
    )
    def parse_tool(
        source_path: ParseSourcePath = None,
        source: ParseSource = "auto",
        source_id: ParseId = None,
        paper_id: PaperId = None,
        html_path: ParseHtmlPath = None,
        tex_path: ParseTexPath = None,
        pdf_path: ParsePdfPath = None,
        refresh: Refresh = False,
    ) -> Any:
        return service.parse_source(
            source_path,
            source=source,
            source_id=source_id,
            paper_id=paper_id,
            html_path=html_path,
            tex_path=tex_path,
            pdf_path=pdf_path,
            refresh=refresh,
        )

    @app.tool(
        description=(
            "Mark a cached parsed equation as problematic, needing re-cache, or resolved. "
            "The marker is stored in a sidecar annotation cache and does not modify canonical parsed JSON."
        )
    )
    def mark_parsed_equation(
        source_id: ParsedSourceId,
        equation_id: ParsedEquationId,
        reason: ParsedEquationReason,
        status: ParsedEquationStatus = "problematic",
    ) -> Any:
        return service.mark_parsed_equation(source_id, equation_id, status=status, reason=reason)

    @app.tool(
        name="llm_get_summary",
        description=(
            "Get a cached high-quality LLM summary for one or more arXiv papers. "
            "If no cached summary is available, this may call the host LLM provider. "
            "The tool waits only until the MCP deadline margin, then returns a background job id."
        ),
    )
    def llm_get_summary(
        paper_id: PaperId = None,
        paper_ids: PaperIds = None,
        provider: LLMProvider = "auto",
        model: LLMModel = None,
        model_tier: LLMModelTier = None,
        refresh: Refresh = False,
        background: Background = False,
    ) -> Any:
        return _cached_or_start_summary_job(
            {
                "paper_id": paper_id,
                "paper_ids": paper_ids,
                "provider": provider,
                "model": model,
                "model_tier": model_tier,
                "refresh": refresh,
                "background": background,
            }
        )

    @app.tool(
        name="llm_generate_summary",
        description=(
            "Generate and cache a high-quality LLM summary for one or more arXiv papers. "
            "This calls the host LLM provider. The tool waits only until the MCP deadline margin, "
            "then returns a background job id."
        ),
    )
    def llm_generate_summary(
        paper_id: PaperId = None,
        paper_ids: PaperIds = None,
        provider: LLMProvider = "auto",
        model: LLMModel = None,
        model_tier: LLMModelTier = None,
        refresh: Refresh = False,
        background: Background = False,
    ) -> Any:
        return _tool_input_result(
            lambda: _start_summary_job_response(
                _one_or_many(paper_id, paper_ids),
                provider=provider,
                model=model,
                model_tier=model_tier,
                refresh=refresh,
                background=background,
            )
        )

    @app.tool(
        name="llm_infer_main_references",
        description=(
            "Infer the main reference paper IDs from natural-language text. "
            "If explicit arXiv, INSPIRE, DOI, or bare arXiv-like IDs are present, "
            "this returns them immediately without calling an LLM. Otherwise it calls "
            "the host LLM provider with web search enabled, verifies candidates through INSPIRE, "
            "and may return a background job id."
        ),
    )
    def llm_infer_main_references(
        text: NaturalText,
        provider: LLMProvider = "auto",
        model: LLMModel = None,
        refresh: Refresh = False,
        background: Background = False,
    ) -> Any:
        return _start_reference_inference_job_response(
            {
                "text": text,
                "provider": provider,
                "model": model,
                "refresh": refresh,
                "background": background,
            }
        )

    @app.tool(
        name="job_status",
        description="Check an ARC MCP background job. Poll this tool; ARC does not push completion notifications.",
    )
    def job_status_tool(job_id: Annotated[str, Field(description="MCP job id returned by a long-running tool.")]) -> Any:
        return job_status(job_id)

    @app.tool(
        name="job_result",
        description="Read the result of an ARC MCP background job when complete, or get a not-ready status.",
    )
    def job_result_tool(job_id: Annotated[str, Field(description="MCP job id returned by a long-running tool.")]) -> Any:
        return job_result(job_id)

    @app.tool(
        name="cancel_job",
        description=JOB_CANCEL_DESCRIPTION,
    )
    def cancel_job_tool(job_id: Annotated[str, Field(description="MCP job id to cancel.")]) -> Any:
        return cancel_job(job_id)

    @app.tool(name="list_jobs", description="List persisted ARC MCP jobs known to the local job store.")
    def list_jobs_tool() -> Any:
        return list_jobs()

    @app.tool(
        name="store_llm_summary",
        description=(
            "Validate and cache a schema-valid LLM paper summary for an arXiv paper. "
            "Use after an agent generated summary JSON from a needs_llm task."
        ),
    )
    def store_llm_summary(
        paper_id: Annotated[str, Field(description=PAPER_ID_DESCRIPTION)],
        summary: Annotated[dict[str, Any], Field(description="Schema-valid paper-summary-v1 JSON object.")],
    ) -> Any:
        return service.store_llm_summary(paper_id, summary)

    @app.tool(
        description=(
            "Create a resumable batch for many paper summaries from a text file of paper IDs. "
            "Use this for large jobs such as tens or hundreds of arXiv papers."
        )
    )
    def summary_batch_create(
        papers_file: Annotated[str, Field(description="Path to a text file containing one paper ID per line.")],
        name: BatchName,
        prompt_version: Annotated[str, Field(description="Summary prompt/schema version to use.")] = "paper-summary-v1",
    ) -> Any:
        return _summary_batch_create_response(
            {"name": name, "papers_file": papers_file, "prompt_version": prompt_version}
        )

    @app.tool(
        description=(
            "Prefetch deterministic paper data for a summary batch using local cache, ar5iv, and INSPIRE. "
            "Run this before LLM generation for large batches."
        )
    )
    def summary_batch_prefetch(
        name: BatchName,
        workers: Annotated[int, Field(description="Number of parallel prefetch worker threads.")] = 4,
    ) -> Any:
        return _summary_batch_prefetch_response({"name": name, "workers": workers})

    @app.tool(
        name="llm_summary_batch_run",
        description=(
            "Run LLM summary generation for queued or ready items in a summary batch. "
            "This calls the host LLM provider and may return a background job id."
        ),
    )
    def llm_summary_batch_run(
        name: BatchName,
        provider: LLMProvider = "auto",
        model: LLMModel = None,
        model_tier: LLMModelTier = None,
        concurrency: Annotated[int, Field(description="Number of concurrent LLM summary generation workers.")] = 1,
        max_items: Annotated[int | None, Field(description="Optional cap on items to process in this run.")] = None,
        background: Background = False,
    ) -> Any:
        return _run_summary_batch_inline(
            {
                "name": name,
                "provider": provider,
                "model": model,
                "model_tier": model_tier,
                "concurrency": concurrency,
                "max_items": max_items,
                "background": background,
            }
        )

    @app.tool(description="Get status counts for a resumable paper-summary batch.")
    def summary_batch_status(name: BatchName) -> Any:
        return _summary_batch_status_response({"name": name})

    @app.tool(description="Export completed paper summaries from a batch to a JSONL file.")
    def summary_batch_export(
        name: BatchName,
        output: Annotated[str, Field(description="Output path for exported JSONL summaries.")],
    ) -> Any:
        return _summary_batch_export_response({"name": name, "output": output})

    @app.tool(description="Move failed items in a paper-summary batch back to queued status for retry.")
    def summary_batch_retry_failed(name: BatchName) -> Any:
        return _summary_batch_retry_failed_response({"name": name})

    @app.tool(description="Diagnose which coding-agent host ARC detected, such as Codex or Claude Code.")
    def doctor_host() -> Any:
        detected = detect_host()
        return {"ok": True, "data": detected.__dict__, "errors": [], "meta": {}}

    @app.tool(description="Diagnose which LLM provider ARC will use for summary generation.")
    def doctor_provider() -> Any:
        selected = select_llm_provider()
        return {
            "ok": True,
            "data": {
                "provider": selected.provider,
                "host": selected.host.host,
                "signals": selected.signals,
            },
            "errors": [],
            "meta": {},
        }

    @app.tool(description="Diagnose ARC's local cache directory and whether a paper summary is cached.")
    def doctor_cache(paper_id: PaperId = None) -> Any:
        return service.doctor_cache(paper_id)

    @app.tool(
        name="llm_domain_build",
        description=(
            "Build a cached ARC research-domain package from a seed arXiv paper and optional user intent. "
            "This calls the host LLM provider during foundation selection and domain summarization. "
            "The tool waits only until the MCP deadline margin, then returns a background job id."
        ),
    )
    def llm_domain_build(
        seed_paper: Annotated[str, Field(description=PAPER_ID_DESCRIPTION)],
        intent: DomainIntent = "",
        domain_id: DomainId = None,
        provider: LLMProvider = "auto",
        model: LLMModel = None,
        model_tier: LLMModelTier = None,
        refresh: Refresh = False,
        workers: Annotated[int, Field(description="Number of parallel arc-paper workers.")] = 8,
        background: Background = False,
    ) -> Any:
        return _start_domain_job_response(
            {
                "seed_paper": seed_paper,
                "intent": intent,
                "domain_id": domain_id,
                "provider": provider,
                "model": model,
                "model_tier": model_tier,
                "refresh": refresh,
                "workers": workers,
                "background": background,
            }
        )

    @app.tool(
        description=(
            "Check a domain build background job by job_id, or inspect cached domain artifacts by seed_paper/domain_id."
        )
    )
    def domain_status(
        job_id: Annotated[str | None, Field(description="Background job id returned by llm_domain_build.")] = None,
        seed_paper: PaperId = None,
        intent: DomainIntent = "",
        domain_id: DomainId = None,
    ) -> Any:
        return _domain_status_response(
            {
                "job_id": job_id,
                "seed_paper": seed_paper,
                "intent": intent,
                "domain_id": domain_id,
            }
        )

    @app.tool(
        description=(
            "Get a cached ARC domain summary by seed paper or domain id. This is cache-only and does not call an LLM."
        )
    )
    def domain_get_summary(
        seed_paper: PaperId = None,
        intent: DomainIntent = "",
        domain_id: DomainId = None,
    ) -> Any:
        return _domain_artifact(
            {
                "seed_paper": seed_paper,
                "intent": intent,
                "domain_id": domain_id,
            },
            artifact="summary",
        )

    @app.tool(
        description=(
            "Get a cached ARC domain graph JSON by seed paper or domain id. This is cache-only and does not call an LLM."
        )
    )
    def domain_get_graph(
        seed_paper: PaperId = None,
        intent: DomainIntent = "",
        domain_id: DomainId = None,
    ) -> Any:
        return _domain_artifact(
            {
                "seed_paper": seed_paper,
                "intent": intent,
                "domain_id": domain_id,
            },
            artifact="graph",
        )

    @app.tool(
        name="llm_domain_get_summary",
        description=(
            "Get a cached ARC domain summary, or call the host LLM provider by starting a domain build when missing."
        ),
    )
    def llm_domain_get_summary(
        seed_paper: PaperId = None,
        intent: DomainIntent = "",
        domain_id: DomainId = None,
        provider: LLMProvider = "auto",
        model: LLMModel = None,
        refresh: Refresh = False,
        workers: Annotated[int, Field(description="Number of parallel arc-paper workers.")] = 8,
        background: Background = False,
    ) -> Any:
        return _domain_artifact_or_start(
            {
                "seed_paper": seed_paper,
                "intent": intent,
                "domain_id": domain_id,
                "provider": provider,
                "model": model,
                "refresh": refresh,
                "workers": workers,
                "background": background,
            },
            artifact="summary",
        )

    @app.tool(
        name="llm_domain_get_graph",
        description=(
            "Get a cached ARC domain graph, or call the host LLM provider by starting a domain build when missing."
        ),
    )
    def llm_domain_get_graph(
        seed_paper: PaperId = None,
        intent: DomainIntent = "",
        domain_id: DomainId = None,
        provider: LLMProvider = "auto",
        model: LLMModel = None,
        refresh: Refresh = False,
        workers: Annotated[int, Field(description="Number of parallel arc-paper workers.")] = 8,
        background: Background = False,
    ) -> Any:
        return _domain_artifact_or_start(
            {
                "seed_paper": seed_paper,
                "intent": intent,
                "domain_id": domain_id,
                "provider": provider,
                "model": model,
                "refresh": refresh,
                "workers": workers,
                "background": background,
            },
            artifact="graph",
        )
