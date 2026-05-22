from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Callable
from uuid import uuid4

from arc_domain_info import service as domain_service
from arc_paper_query import service
from arc_paper_query.batch.db import BatchDB
from arc_paper_query.batch.runner import export_batch, prefetch_batch, run_batch
from arc_paper_query.host import detect_host, select_llm_provider
from arc_paper_query.ids import normalize_paper_id
from pydantic import Field


ToolHandler = Callable[[dict[str, Any]], Any]
SUMMARY_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arc-summary")
DOMAIN_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arc-domain")
SUMMARY_JOBS: dict[str, dict[str, Any]] = {}
DOMAIN_JOBS: dict[str, dict[str, Any]] = {}
SUMMARY_LOCK = Lock()
DOMAIN_LOCK = Lock()

SERVER_INSTRUCTIONS = (
    "Use ARC when a user asks about theoretical-physics papers or arXiv papers: "
    "titles, abstracts, authors, references, citing papers, citation counts, "
    "ar5iv table of contents, sections, equation context, LLM paper summaries, "
    "or research-domain construction from a seed paper. "
    "Paper IDs may be passed with or without the arXiv: prefix, for example "
    "0911.3380, arXiv:0911.3380, or hep-th/0601001. For one paper use paper_id; "
    "for multiple papers use paper_ids."
)

PAPER_ID_DESCRIPTION = (
    "Single paper identifier. arXiv IDs may be written as 0911.3380, "
    "arXiv:0911.3380, or hep-th/0601001."
)
PAPER_IDS_DESCRIPTION = "Multiple paper identifiers. Use this instead of paper_id for batch queries."
REFRESH_DESCRIPTION = "Bypass cached data and refetch source metadata or full text when possible."
ENRICH_REFERENCES_DESCRIPTION = (
    "When true, fetch and cache each referenced paper's INSPIRE metadata through paper-query, "
    "including title, abstract, authors, and identifiers when available."
)
SECTION_DESCRIPTION = "Section heading, section number, or section id to retrieve from the ar5iv full text."
QUERY_DESCRIPTION = "Equation label, symbol, or phrase to find nearby equation context in the paper."
BATCH_NAME_DESCRIPTION = "Name of a summary batch stored in ARC's local SQLite batch database."
DOMAIN_INTENT_DESCRIPTION = "Optional description of the user's scientific interest or desired subfield scope."
DOMAIN_ID_DESCRIPTION = "Optional ARC domain id returned by domain_build or arc-domain-info init."
CITER_LIMIT_DESCRIPTION = "Maximum number of citing papers to return from INSPIRE, clamped to 1..1000."
CITER_SORT_DESCRIPTION = "INSPIRE citer sort order: mostrecent or mostcited."

PaperId = Annotated[str | None, Field(description=PAPER_ID_DESCRIPTION)]
PaperIds = Annotated[list[str] | None, Field(description=PAPER_IDS_DESCRIPTION)]
Refresh = Annotated[bool, Field(description=REFRESH_DESCRIPTION)]
EnrichReferences = Annotated[bool, Field(description=ENRICH_REFERENCES_DESCRIPTION)]
Section = Annotated[str, Field(description=SECTION_DESCRIPTION)]
EquationQuery = Annotated[str, Field(description=QUERY_DESCRIPTION)]
BatchName = Annotated[str, Field(description=BATCH_NAME_DESCRIPTION)]
DomainIntent = Annotated[str, Field(description=DOMAIN_INTENT_DESCRIPTION)]
DomainId = Annotated[str | None, Field(description=DOMAIN_ID_DESCRIPTION)]
CiterLimit = Annotated[int, Field(description=CITER_LIMIT_DESCRIPTION)]
CiterSort = Annotated[str, Field(description=CITER_SORT_DESCRIPTION)]


def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    try:
        handler = TOOL_HANDLERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown ARC MCP tool: {name}") from exc
    return handler(arguments)


def _paper_ids(args: dict[str, Any]):
    return args.get("paper_ids") or args.get("paper_id")


TOOL_HANDLERS: dict[str, ToolHandler] = {
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
    "get_equation_context": lambda args: service.get_equation_context(
        _paper_ids(args),
        str(args["query"]),
        refresh=bool(args.get("refresh", False)),
    ),
    "get_LLM_summary": lambda args: _cached_or_start_summary_job(args),
    "generate_LLM_summary": lambda args: _start_summary_job_response(
        _paper_ids(args),
        provider=str(args.get("provider", "auto")),
        model=args.get("model"),
        refresh=bool(args.get("refresh", False)),
    ),
    "get_LLM_summary_status": lambda args: get_summary_job_status(str(args["job_id"])),
    "store_LLM_summary": lambda args: service.store_llm_summary(str(args["paper_id"]), args["summary"]),
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
    "domain_build": lambda args: _start_domain_job_response(args),
    "domain_status": lambda args: _domain_status_response(args),
    "domain_get_summary": lambda args: _domain_artifact_or_start(args, artifact="summary"),
    "domain_get_graph": lambda args: _domain_artifact_or_start(args, artifact="graph"),
}


def _cached_or_start_summary_job(args: dict[str, Any]) -> dict[str, Any]:
    paper_ids = _paper_ids(args)
    if not bool(args.get("refresh", False)):
        cached = service.get_cached_llm_summary(paper_ids)
        if _all_ok(cached):
            return cached
    return _start_summary_job_response(
        paper_ids,
        provider=str(args.get("provider", "auto")),
        model=args.get("model"),
        refresh=bool(args.get("refresh", False)),
    )


def _start_summary_job_response(
    paper_ids: Any,
    *,
    provider: str,
    model: str | None,
    refresh: bool,
) -> dict[str, Any]:
    job_id = uuid4().hex
    normalized = _normalize_ids(paper_ids)
    with SUMMARY_LOCK:
        SUMMARY_JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "phase": "queued",
            "paper_ids": normalized,
            "provider": provider,
            "model": model,
            "refresh": refresh,
            "sections_total": None,
            "sections_completed": 0,
            "current_section": None,
            "events": [],
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
            "result": None,
            "error": None,
        }
    future = SUMMARY_EXECUTOR.submit(_run_summary_job, job_id, normalized, provider, model, refresh)
    future.add_done_callback(_capture_unhandled_job_error(job_id))
    return {
        "ok": False,
        "status": "summary_job_started",
        "job_id": job_id,
        "paper_ids": normalized,
        "message": "No cached LLM summary was returned immediately; background generation has started.",
        "next": {
            "tool": "get_LLM_summary_status",
            "arguments": {"job_id": job_id},
            "poll_after_seconds": 5,
        },
    }


def get_summary_job_status(job_id: str) -> dict[str, Any]:
    with SUMMARY_LOCK:
        job = SUMMARY_JOBS.get(job_id)
        if not job:
            return {
                "ok": False,
                "status": "summary_job_unknown",
                "error": {"code": "summary_job_unknown", "message": f"Unknown summary job: {job_id}"},
                "errors": [],
                "meta": {},
            }
        return {key: value for key, value in job.items() if key != "future"}


def _start_domain_job_response(args: dict[str, Any]) -> dict[str, Any]:
    seed_paper = str(args["seed_paper"])
    intent = str(args.get("intent", ""))
    domain_id = args.get("domain_id")
    provider = str(args.get("provider", "auto"))
    model = args.get("model")
    refresh = bool(args.get("refresh", False))
    workers = int(args.get("workers", 8))
    job_id = uuid4().hex
    with DOMAIN_LOCK:
        DOMAIN_JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "phase": "queued",
            "seed_paper": normalize_paper_id(seed_paper),
            "intent": intent,
            "domain_id": domain_id,
            "provider": provider,
            "model": model,
            "refresh": refresh,
            "workers": workers,
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
            "result": None,
            "error": None,
        }
    future = DOMAIN_EXECUTOR.submit(
        _run_domain_job,
        job_id,
        seed_paper,
        intent,
        domain_id,
        provider,
        model,
        refresh,
        workers,
    )
    future.add_done_callback(_capture_unhandled_domain_job_error(job_id))
    return {
        "ok": False,
        "status": "domain_job_started",
        "job_id": job_id,
        "message": "Domain build has started in the background.",
        "next": {
            "tool": "domain_status",
            "arguments": {"job_id": job_id},
            "poll_after_seconds": 10,
        },
    }


def _domain_status_response(args: dict[str, Any]) -> dict[str, Any]:
    if args.get("job_id"):
        return get_domain_job_status(str(args["job_id"]))
    return domain_service.status(
        args.get("seed_paper"),
        intent=str(args.get("intent", "")),
        domain_id=args.get("domain_id"),
    )


def _domain_artifact_or_start(args: dict[str, Any], *, artifact: str) -> dict[str, Any]:
    if artifact == "summary":
        result = domain_service.get_domain_summary(
            args.get("seed_paper"),
            intent=str(args.get("intent", "")),
            domain_id=args.get("domain_id"),
        )
    elif artifact == "graph":
        result = domain_service.get_domain_graph(
            args.get("seed_paper"),
            intent=str(args.get("intent", "")),
            domain_id=args.get("domain_id"),
        )
    else:
        raise ValueError(f"Unsupported domain artifact: {artifact}")
    if result.get("ok") or not args.get("seed_paper"):
        return result
    error_code = (result.get("error") or {}).get("code")
    if error_code not in {"domain_summary_not_available", "domain_graph_not_available"}:
        return result
    started = _start_domain_job_response(args)
    started["message"] = f"No cached domain {artifact} was returned immediately; background domain build has started."
    return started


def get_domain_job_status(job_id: str) -> dict[str, Any]:
    with DOMAIN_LOCK:
        job = DOMAIN_JOBS.get(job_id)
        if not job:
            return {
                "ok": False,
                "status": "domain_job_unknown",
                "error": {"code": "domain_job_unknown", "message": f"Unknown domain job: {job_id}"},
                "errors": [],
                "meta": {},
            }
        snapshot = dict(job)
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
    job_id: str,
    seed_paper: str,
    intent: str,
    domain_id: str | None,
    provider: str,
    model: str | None,
    refresh: bool,
    workers: int,
) -> None:
    _record_domain_phase(job_id, "running")
    result = domain_service.build_domain(
        seed_paper,
        intent=intent,
        domain_id=domain_id,
        provider=provider,
        model=model,
        refresh=refresh,
        workers=workers,
    )
    status = "done" if isinstance(result, dict) and result.get("ok") else "failed"
    with DOMAIN_LOCK:
        if job_id in DOMAIN_JOBS:
            DOMAIN_JOBS[job_id]["status"] = status
            DOMAIN_JOBS[job_id]["phase"] = status
            DOMAIN_JOBS[job_id]["updated_at"] = _now_iso()
            DOMAIN_JOBS[job_id]["result"] = result


def _capture_unhandled_domain_job_error(job_id: str) -> Callable[[Future], None]:
    def callback(future: Future) -> None:
        exc = future.exception()
        if not exc:
            return
        with DOMAIN_LOCK:
            if job_id in DOMAIN_JOBS:
                DOMAIN_JOBS[job_id]["status"] = "failed"
                DOMAIN_JOBS[job_id]["phase"] = "failed"
                DOMAIN_JOBS[job_id]["updated_at"] = _now_iso()
                DOMAIN_JOBS[job_id]["error"] = str(exc)

    return callback


def _record_domain_phase(job_id: str, phase: str) -> None:
    with DOMAIN_LOCK:
        if job_id in DOMAIN_JOBS:
            DOMAIN_JOBS[job_id]["phase"] = phase
            DOMAIN_JOBS[job_id]["updated_at"] = _now_iso()


def _run_summary_job(job_id: str, paper_ids: Any, provider: str, model: str | None, refresh: bool) -> None:
    _record_summary_progress(job_id, {"event": "job_started"})
    result = service.generate_llm_summary(
        paper_ids,
        provider=provider,
        model=model,
        refresh=refresh,
        progress_callback=lambda event: _record_summary_progress(job_id, event),
    )
    if _all_ok(result):
        status = "done"
    elif isinstance(result, dict) and result.get("status") == "needs_llm":
        status = "needs_llm"
    else:
        status = "failed"
    with SUMMARY_LOCK:
        if job_id in SUMMARY_JOBS:
            SUMMARY_JOBS[job_id]["status"] = status
            SUMMARY_JOBS[job_id]["phase"] = status
            SUMMARY_JOBS[job_id]["updated_at"] = _now_iso()
            SUMMARY_JOBS[job_id]["result"] = result


def _capture_unhandled_job_error(job_id: str) -> Callable[[Future], None]:
    def callback(future: Future) -> None:
        exc = future.exception()
        if not exc:
            return
        with SUMMARY_LOCK:
            if job_id in SUMMARY_JOBS:
                SUMMARY_JOBS[job_id]["status"] = "failed"
                SUMMARY_JOBS[job_id]["phase"] = "failed"
                SUMMARY_JOBS[job_id]["updated_at"] = _now_iso()
                SUMMARY_JOBS[job_id]["error"] = str(exc)

    return callback


def _record_summary_progress(job_id: str, event: dict[str, Any]) -> None:
    timestamped = {"at": _now_iso(), **event}
    with SUMMARY_LOCK:
        job = SUMMARY_JOBS.get(job_id)
        if not job:
            return
        event_name = str(timestamped.get("event") or "")
        job["phase"] = event_name or job.get("phase")
        job["updated_at"] = timestamped["at"]
        if "sections_total" in timestamped:
            job["sections_total"] = timestamped["sections_total"]
        if "sections_completed" in timestamped:
            job["sections_completed"] = timestamped["sections_completed"]
        if event_name in {"section_started", "section_cached", "section_completed"}:
            job["current_section"] = {
                "paper_id": timestamped.get("paper_id"),
                "section_index": timestamped.get("section_index"),
                "sections_total": timestamped.get("sections_total"),
                "section_id": timestamped.get("section_id"),
                "title": timestamped.get("title"),
            }
        elif event_name.startswith("final_") or event_name in {"done", "failed"}:
            job["current_section"] = None
        events = job.setdefault("events", [])
        events.append(timestamped)
        if len(events) > 100:
            del events[:-100]


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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit("The 'mcp' package is required to run arc-mcp.") from exc

    app = FastMCP("arc", instructions=SERVER_INSTRUCTIONS)
    _register_tools(app)

    app.run()


def _one_or_many(paper_id: str | None = None, paper_ids: list[str] | None = None):
    return paper_ids if paper_ids is not None else paper_id


def _register_tools(app: Any) -> None:
    @app.tool(
        description=(
            "Get the title of one or more arXiv papers from INSPIRE. "
            "Use this when the user asks for a paper title or to identify a paper."
        )
    )
    def get_title(paper_id: PaperId = None, paper_ids: PaperIds = None, refresh: Refresh = False) -> Any:
        return service.get_title(_one_or_many(paper_id, paper_ids), refresh=refresh)

    @app.tool(
        description=(
            "Get the abstract of one or more arXiv papers from INSPIRE. "
            "Use this for paper overview, motivation, or abstract lookup."
        )
    )
    def get_abstract(paper_id: PaperId = None, paper_ids: PaperIds = None, refresh: Refresh = False) -> Any:
        return service.get_abstract(_one_or_many(paper_id, paper_ids), refresh=refresh)

    @app.tool(
        description=(
            "Get authors for one or more arXiv papers from INSPIRE. "
            "Use this when the user asks who wrote a paper."
        )
    )
    def get_authors(paper_id: PaperId = None, paper_ids: PaperIds = None, refresh: Refresh = False) -> Any:
        return service.get_authors(_one_or_many(paper_id, paper_ids), refresh=refresh)

    @app.tool(
        description=(
            "Get normalized INSPIRE metadata for one or more arXiv papers, including title, abstract, "
            "authors, identifiers, year, DOI, INSPIRE record id, and citation count."
        )
    )
    def get_metadata(paper_id: PaperId = None, paper_ids: PaperIds = None, refresh: Refresh = False) -> Any:
        return service.get_metadata(_one_or_many(paper_id, paper_ids), refresh=refresh)

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
        return service.get_citers(_one_or_many(paper_id, paper_ids), refresh=refresh, limit=limit, sort=sort)

    @app.tool(
        description=(
            "Get the INSPIRE citation count for one or more arXiv papers. "
            "Use this when only the number of citing papers is needed."
        )
    )
    def get_citer_count(paper_id: PaperId = None, paper_ids: PaperIds = None, refresh: Refresh = False) -> Any:
        return service.get_citer_count(_one_or_many(paper_id, paper_ids), refresh=refresh)

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
        return service.get_references(_one_or_many(paper_id, paper_ids), refresh=refresh, enrich=enrich)

    @app.tool(
        description=(
            "Get the table of contents from ar5iv full text for one or more arXiv papers. "
            "Use this before selecting sections to read."
        )
    )
    def get_toc(paper_id: PaperId = None, paper_ids: PaperIds = None, refresh: Refresh = False) -> Any:
        return service.get_toc(_one_or_many(paper_id, paper_ids), refresh=refresh)

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
        return service.get_section(_one_or_many(paper_id, paper_ids), section, refresh=refresh)

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
        return service.get_equation_context(_one_or_many(paper_id, paper_ids), query, refresh=refresh)

    @app.tool(
        name="get_LLM_summary",
        description=(
            "Get a cached high-quality LLM summary for one or more arXiv papers. "
            "If no cached summary is immediately available, start a background summary job and return a job id."
        ),
    )
    def get_llm_summary(
        paper_id: PaperId = None,
        paper_ids: PaperIds = None,
        provider: Annotated[str, Field(description="LLM provider: auto, codex-cli, claude-cli, or manual.")] = "auto",
        model: Annotated[str | None, Field(description="Optional model name passed to the selected LLM provider.")] = None,
        refresh: Refresh = False,
    ) -> Any:
        return _cached_or_start_summary_job(
            {
                "paper_id": paper_id,
                "paper_ids": paper_ids,
                "provider": provider,
                "model": model,
                "refresh": refresh,
            }
        )

    @app.tool(
        name="generate_LLM_summary",
        description=(
            "Start a background job to generate and cache a high-quality LLM summary for one or more arXiv papers. "
            "Returns immediately with a job id to avoid MCP client tool-call timeouts."
        ),
    )
    def generate_llm_summary(
        paper_id: PaperId = None,
        paper_ids: PaperIds = None,
        provider: Annotated[str, Field(description="LLM provider: auto, codex-cli, claude-cli, or manual.")] = "auto",
        model: Annotated[str | None, Field(description="Optional model name passed to the selected LLM provider.")] = None,
        refresh: Refresh = False,
    ) -> Any:
        return _start_summary_job_response(
            _one_or_many(paper_id, paper_ids),
            provider=provider,
            model=model,
            refresh=refresh,
        )

    @app.tool(
        name="get_LLM_summary_status",
        description=(
            "Check a background LLM summary job started by get_LLM_summary or generate_LLM_summary. "
            "When status is done, the result contains the generated summary."
        ),
    )
    def get_llm_summary_status(
        job_id: Annotated[str, Field(description="Background summary job id returned by get_LLM_summary.")],
    ) -> Any:
        return get_summary_job_status(job_id)

    @app.tool(
        name="store_LLM_summary",
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
        db = BatchDB.default()
        with open(papers_file, encoding="utf-8") as handle:
            paper_ids = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
        db.create_batch(name, paper_ids, prompt_version)
        return {"ok": True, "data": {"batch": name, "counts": db.status_counts(name)}, "errors": [], "meta": {}}

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
        return {"ok": True, "data": prefetch_batch(name, workers=workers), "errors": [], "meta": {}}

    @app.tool(
        description=(
            "Run LLM summary generation for queued or ready items in a summary batch. "
            "Use max_items for calibration before processing a large batch."
        )
    )
    def summary_batch_run(
        name: BatchName,
        provider: Annotated[str, Field(description="LLM provider: auto, codex-cli, claude-cli, or manual.")] = "auto",
        model: Annotated[str | None, Field(description="Optional model name passed to the selected LLM provider.")] = None,
        concurrency: Annotated[int, Field(description="Number of concurrent LLM summary generation workers.")] = 1,
        max_items: Annotated[int | None, Field(description="Optional cap on items to process in this run.")] = None,
    ) -> Any:
        return {
            "ok": True,
            "data": run_batch(name, provider=provider, model=model, concurrency=concurrency, max_items=max_items),
            "errors": [],
            "meta": {},
        }

    @app.tool(description="Get status counts for a resumable paper-summary batch.")
    def summary_batch_status(name: BatchName) -> Any:
        db = BatchDB.default()
        return {"ok": True, "data": {"batch": name, "counts": db.status_counts(name)}, "errors": [], "meta": {}}

    @app.tool(description="Export completed paper summaries from a batch to a JSONL file.")
    def summary_batch_export(
        name: BatchName,
        output: Annotated[str, Field(description="Output path for exported JSONL summaries.")],
    ) -> Any:
        return {"ok": True, "data": export_batch(name, output=Path(output)), "errors": [], "meta": {}}

    @app.tool(description="Move failed items in a paper-summary batch back to queued status for retry.")
    def summary_batch_retry_failed(name: BatchName) -> Any:
        db = BatchDB.default()
        db.retry_failed(name)
        return {"ok": True, "data": {"batch": name, "counts": db.status_counts(name)}, "errors": [], "meta": {}}

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
        description=(
            "Build a cached ARC research-domain package from a seed arXiv paper and optional user intent. "
            "This starts a background job because foundation discovery, network construction, full-text fetches, "
            "and LLM summary generation can be slow."
        )
    )
    def domain_build(
        seed_paper: Annotated[str, Field(description=PAPER_ID_DESCRIPTION)],
        intent: DomainIntent = "",
        domain_id: DomainId = None,
        provider: Annotated[str, Field(description="LLM provider: auto, codex-cli, claude-cli, or manual.")] = "auto",
        model: Annotated[str | None, Field(description="Optional model name passed to the selected LLM provider.")] = None,
        refresh: Refresh = False,
        workers: Annotated[int, Field(description="Number of parallel paper-query workers.")] = 8,
    ) -> Any:
        return _start_domain_job_response(
            {
                "seed_paper": seed_paper,
                "intent": intent,
                "domain_id": domain_id,
                "provider": provider,
                "model": model,
                "refresh": refresh,
                "workers": workers,
            }
        )

    @app.tool(
        description=(
            "Check a domain build background job by job_id, or inspect cached domain artifacts by seed_paper/domain_id."
        )
    )
    def domain_status(
        job_id: Annotated[str | None, Field(description="Background job id returned by domain_build.")] = None,
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
            "Get a cached ARC domain summary by seed paper or domain id. If no cached summary exists and "
            "seed_paper is supplied, start a background domain build and return a job id."
        )
    )
    def domain_get_summary(
        seed_paper: PaperId = None,
        intent: DomainIntent = "",
        domain_id: DomainId = None,
        provider: Annotated[str, Field(description="LLM provider: auto, codex-cli, claude-cli, or manual.")] = "auto",
        model: Annotated[str | None, Field(description="Optional model name passed to the selected LLM provider.")] = None,
        refresh: Refresh = False,
        workers: Annotated[int, Field(description="Number of parallel paper-query workers.")] = 8,
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
            },
            artifact="summary",
        )

    @app.tool(
        description=(
            "Get a cached ARC domain graph JSON by seed paper or domain id. If no cached graph exists and "
            "seed_paper is supplied, start a background domain build and return a job id."
        )
    )
    def domain_get_graph(
        seed_paper: PaperId = None,
        intent: DomainIntent = "",
        domain_id: DomainId = None,
        provider: Annotated[str, Field(description="LLM provider: auto, codex-cli, claude-cli, or manual.")] = "auto",
        model: Annotated[str | None, Field(description="Optional model name passed to the selected LLM provider.")] = None,
        refresh: Refresh = False,
        workers: Annotated[int, Field(description="Number of parallel paper-query workers.")] = 8,
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
            },
            artifact="graph",
        )
