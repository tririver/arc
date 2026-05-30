import asyncio
import threading
import time

import pytest

from arc_mcp import server


@pytest.fixture(autouse=True)
def _mcp_job_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_MCP_CACHE", str(tmp_path / "arc-mcp"))
    monkeypatch.setenv("ARC_MCP_WORKER_MODE", "thread")


def test_call_tool_dispatches_to_service(monkeypatch):
    monkeypatch.setattr(server.service, "get_title", lambda paper_ids, refresh=False: {"ok": True, "data": paper_ids})

    result = server.call_tool("get_title", {"paper_id": "arXiv:0911.3380"})

    assert result == {"ok": True, "data": "arXiv:0911.3380"}


def test_call_tool_rejects_missing_paper_id():
    result = server.call_tool("get_title", {})

    assert result["ok"] is False
    assert result["error"]["code"] == "paper_ids_required"


def test_call_tool_rejects_both_paper_id_forms():
    result = server.call_tool("get_title", {"paper_id": "0911.3380", "paper_ids": ["astro-ph/0610514"]})

    assert result["ok"] is False
    assert result["error"]["code"] == "paper_ids_ambiguous"


def test_call_tool_md2pdf_starts_background_job_without_waiting(monkeypatch, tmp_path):
    source = tmp_path / "report.md"
    output = tmp_path / "report.pdf"
    source.write_text("# Report\n", encoding="utf-8")
    release_conversion = threading.Event()
    calls = {}

    def convert_markdown_to_pdf(**kwargs):
        calls.update(kwargs)
        release_conversion.wait(timeout=2)
        return {
            "ok": True,
            "data": {"input_path": str(source), "output_path": str(output), "pdf_size_bytes": 8},
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(server.typeset_md2pdf, "convert_markdown_to_pdf", convert_markdown_to_pdf)

    started_at = time.monotonic()
    result = server.call_tool(
        "md2pdf",
        {
            "input": str(source),
            "output": str(output),
            "margin": "2cm",
            "mainfont": "Noto Sans",
            "cjk_mainfont": "Noto Sans CJK SC",
            "resource_path": [str(tmp_path)],
            "texlive_bin": "",
        },
    )
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.5
    assert result["status"] == "job_running"
    assert result["job_type"] == "md2pdf"
    assert result["inline_wait_seconds"] == 0.0
    assert result["background_requested"] is True
    assert result["job"]["input"] == str(source)

    release_conversion.set()
    status = _wait_for_mcp_job(result["job_id"])
    assert status["status"] == "done"
    completed = server.job_result(result["job_id"])["result"]
    assert completed["data"]["output_path"] == str(output)
    assert calls["input_path"] == source
    assert calls["output_path"] == output
    assert calls["margin"] == "2cm"
    assert calls["mainfont"] == "Noto Sans"
    assert calls["cjk_mainfont"] == "Noto Sans CJK SC"
    assert calls["resource_paths"] == [tmp_path]
    assert calls["texlive_bin"] is None


def test_call_tool_translate_starts_background_job_without_waiting(monkeypatch, tmp_path):
    source = tmp_path / "report.md"
    output = tmp_path / "report.zh_CN.md"
    source.write_text("# Report\n", encoding="utf-8")
    release_translation = threading.Event()
    calls = {}

    def translate_markdown(**kwargs):
        calls.update(kwargs)
        release_translation.wait(timeout=2)
        return {
            "ok": True,
            "data": {
                "input_markdown_path": str(source),
                "output_markdown_path": str(output),
                "output_pdf_path": str(output.with_suffix(".pdf")),
            },
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(server.typeset_translate, "translate_markdown", translate_markdown)

    started_at = time.monotonic()
    result = server.call_tool(
        "translate",
        {
            "input": str(source),
            "output": str(output),
            "target_language": "Chinese",
            "target_locale": "zh_CN",
            "provider": "manual",
            "model": "test-model",
            "model_tier": "medium",
            "quality": True,
        },
    )
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.5
    assert result["status"] == "job_running"
    assert result["job_type"] == "translate"
    assert result["inline_wait_seconds"] == 0.0
    assert result["background_requested"] is True
    assert result["job"]["input"] == str(source)

    release_translation.set()
    status = _wait_for_mcp_job(result["job_id"])
    assert status["status"] == "done"
    completed = server.job_result(result["job_id"])["result"]
    assert completed["data"]["output_markdown_path"] == str(output)
    assert calls["input_path"] == source
    assert calls["output_path"] == output
    assert calls["target_language"] == "Chinese"
    assert calls["target_locale"] == "zh_CN"
    assert calls["provider"] == "manual"
    assert calls["model"] == "test-model"
    assert calls["model_tier"] == "medium"
    assert calls["quality"] is True
    assert calls["convert_pdf"] is True


def test_call_tool_batch_translate_starts_background_job_without_waiting(monkeypatch, tmp_path):
    release_batch = threading.Event()
    calls = {}

    def batch_translate_project(**kwargs):
        calls.update(kwargs)
        release_batch.wait(timeout=2)
        return {
            "ok": True,
            "data": {"project_dir": str(tmp_path), "candidate_count": 2, "translated_count": 2},
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(server.typeset_translate, "batch_translate_project", batch_translate_project)

    started_at = time.monotonic()
    result = server.call_tool("batch_translate", {"project_dir": str(tmp_path)})
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.5
    assert result["status"] == "job_running"
    assert result["job_type"] == "batch_translate"
    assert result["inline_wait_seconds"] == 0.0
    assert result["background_requested"] is True
    assert result["job"]["project_dir"] == str(tmp_path)

    release_batch.set()
    status = _wait_for_mcp_job(result["job_id"])
    assert status["status"] == "done"
    completed = server.job_result(result["job_id"])["result"]
    assert completed["data"]["translated_count"] == 2
    assert calls["project_dir"] == tmp_path
    assert calls["target_language"] == "Chinese"
    assert calls["target_locale"] == "zh_CN"
    assert calls["model_tier"] == "low"


def test_call_tool_extracts_paper_ids(monkeypatch):
    monkeypatch.setattr(server.service, "extract_paper_ids", lambda text: {"ok": True, "data": [text]})

    result = server.call_tool("extract_paper_ids", {"text": "0911.3380"})

    assert result == {"ok": True, "data": ["0911.3380"]}


def test_call_tool_searches_full_text(monkeypatch):
    def search_full_text(ids, *, query, refresh=False, limit=20, context=1, case_sensitive=False):
        return {
            "ok": True,
            "data": {
                "ids": ids,
                "query": query,
                "refresh": refresh,
                "limit": limit,
                "context": context,
                "case_sensitive": case_sensitive,
            },
        }

    monkeypatch.setattr(server.service, "search_full_text", search_full_text)

    result = server.call_tool(
        "search_full_text",
        {
            "paper_id": "0911.3380",
            "query": "scalar trispectrum",
            "refresh": True,
            "limit": 5,
            "context": 2,
            "case_sensitive": True,
        },
    )

    assert result["data"]["ids"] == "0911.3380"
    assert result["data"]["query"] == "scalar trispectrum"
    assert result["data"]["refresh"] is True
    assert result["data"]["limit"] == 5
    assert result["data"]["context"] == 2
    assert result["data"]["case_sensitive"] is True


def test_call_tool_dispatches_unified_parse(monkeypatch):
    monkeypatch.setattr(
        server.service,
        "parse_source",
        lambda source_path=None, *, source="auto", source_id=None, paper_id=None, html_path=None, tex_path=None, pdf_path=None, refresh=False: {
            "ok": True,
            "data": {
                "source_path": source_path,
                "source": source,
                "source_id": source_id,
                "paper_id": paper_id,
                "html_path": html_path,
                "tex_path": tex_path,
                "pdf_path": pdf_path,
                "refresh": refresh,
            },
        },
    )

    result = server.call_tool(
        "parse",
        {"tex_path": "note.tex", "pdf_path": "book.pdf", "source_id": "lecture-9", "refresh": True},
    )

    assert result["data"]["tex_path"] == "note.tex"
    assert result["data"]["pdf_path"] == "book.pdf"
    assert result["data"]["source_id"] == "lecture-9"
    assert result["data"]["refresh"] is True


def test_call_tool_dispatches_mark_parsed_equation(monkeypatch):
    monkeypatch.setattr(
        server.service,
        "mark_parsed_equation",
        lambda source_id, equation_id, *, status="problematic", reason="": {
            "ok": True,
            "data": {
                "source_id": source_id,
                "target_id": equation_id,
                "status": status,
                "reason": reason,
            },
        },
    )

    result = server.call_tool(
        "mark_parsed_equation",
        {
            "source_id": "lecture-9",
            "equation_id": "eq_00001",
            "status": "problematic",
            "reason": "Bad sign",
        },
    )

    assert result["data"]["source_id"] == "lecture-9"
    assert result["data"]["target_id"] == "eq_00001"
    assert result["data"]["status"] == "problematic"
    assert result["data"]["reason"] == "Bad sign"


def test_call_tool_search_full_text_defaults_to_one_context_line(monkeypatch):
    def search_full_text(ids, *, query, refresh=False, limit=20, context=1, case_sensitive=False):
        return {"ok": True, "data": {"context": context}}

    monkeypatch.setattr(server.service, "search_full_text", search_full_text)

    result = server.call_tool("search_full_text", {"paper_id": "0911.3380", "query": "scalar exchange"})

    assert result["data"]["context"] == 1


def test_call_tool_search_full_text_allows_missing_paper_id(monkeypatch):
    def search_full_text(ids, *, query, refresh=False, limit=20, context=1, case_sensitive=False):
        return {"ok": True, "data": {"ids": ids, "query": query}}

    monkeypatch.setattr(server.service, "search_full_text", search_full_text)

    result = server.call_tool("search_full_text", {"query": "scalar exchange"})

    assert result["ok"] is True
    assert result["data"] == {"ids": None, "query": "scalar exchange"}


def test_call_tool_paper_ids_safe_dir_name():
    result = server.call_tool("paper_ids_safe_dir_name", {"paper_ids": ["0911.3380", "astro-ph/0610514"]})

    assert result["ok"] is True
    assert result["data"] == "0911.3380_x_astro-ph_0610514"


def test_call_tool_llm_infer_main_references_short_circuits_ids(monkeypatch):
    monkeypatch.setattr(
        server.service,
        "extract_paper_ids",
        lambda text: {"ok": True, "data": ["arXiv:0911.3380"], "errors": [], "meta": {}},
    )
    monkeypatch.setattr(
        server.service,
        "llm_infer_main_references",
        lambda text, provider="auto", model=None, refresh=False: {
            "ok": True,
            "data": ["arXiv:0911.3380"],
            "errors": [],
            "meta": {
                "provider": "local-parser",
                "llm_used": False,
                "text": text,
                "refresh": refresh,
            },
        },
    )

    result = server.call_tool("llm_infer_main_references", {"text": "0911.3380"})

    assert result["ok"] is True
    assert result["data"] == ["arXiv:0911.3380"]
    assert result["meta"]["llm_used"] is False
    assert result["meta"]["text"] == "0911.3380"


def test_call_tool_llm_infer_main_references_uses_background_manager(monkeypatch):
    monkeypatch.setenv("ARC_MCP_INLINE_WAIT_SEC", "1")
    monkeypatch.setattr(
        server.service,
        "extract_paper_ids",
        lambda text: {"ok": True, "data": [], "errors": [], "meta": {}},
    )

    def infer(text, provider="auto", model=None, refresh=False):
        return {
            "ok": True,
            "data": ["arXiv:0911.3380"],
            "errors": [],
            "meta": {"text": text, "provider": provider, "model": model, "refresh": refresh},
        }

    monkeypatch.setattr(server.service, "llm_infer_main_references", infer)

    result = server.call_tool(
        "llm_infer_main_references",
        {"text": "CMB trispectrum", "provider": "manual", "model": "test-model", "refresh": True},
    )

    assert result["ok"] is True
    assert result["data"] == ["arXiv:0911.3380"]
    assert result["meta"]["job"]["job_type"] == "main_reference_inference"
    assert result["meta"]["text"] == "CMB trispectrum"
    assert result["meta"]["provider"] == "manual"
    assert result["meta"]["model"] == "test-model"
    assert result["meta"]["refresh"] is True


def test_call_tool_passes_reference_enrichment(monkeypatch):
    def get_references(paper_ids, refresh=False, enrich=False):
        return {"ok": True, "data": {"paper_ids": paper_ids, "refresh": refresh, "enrich": enrich}}

    monkeypatch.setattr(server.service, "get_references", get_references)

    result = server.call_tool("get_references", {"paper_id": "0911.3380", "enrich": True})

    assert result["data"]["paper_ids"] == "0911.3380"
    assert result["data"]["enrich"] is True


def test_call_tool_passes_metadata_and_citer_options(monkeypatch):
    monkeypatch.setattr(
        server.service,
        "get_metadata",
        lambda paper_ids, refresh=False: {"ok": True, "data": {"paper_ids": paper_ids, "refresh": refresh}},
    )

    def get_citers(paper_ids, refresh=False, limit=1000, sort="mostrecent"):
        return {"ok": True, "data": {"paper_ids": paper_ids, "limit": limit, "sort": sort}}

    monkeypatch.setattr(server.service, "get_citers", get_citers)

    metadata = server.call_tool("get_metadata", {"paper_id": "0911.3380"})
    citers = server.call_tool("get_citers", {"paper_id": "0911.3380", "limit": 7, "sort": "mostcited"})

    assert metadata["data"]["paper_ids"] == "0911.3380"
    assert citers["data"]["limit"] == 7
    assert citers["data"]["sort"] == "mostcited"


def test_call_tool_rejects_unknown_tool():
    try:
        server.call_tool("missing", {})
    except ValueError as exc:
        assert "Unknown ARC MCP tool" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_fastmcp_tools_have_discovery_metadata():
    from mcp.server.fastmcp import FastMCP

    app = FastMCP("arc", instructions=server.SERVER_INSTRUCTIONS)
    server._register_tools(app)

    tools = asyncio.run(app.list_tools())
    by_name = {tool.name: tool for tool in tools}

    assert "md2pdf" in by_name
    assert "Markdown" in by_name["md2pdf"].description
    assert "XeLaTeX" in by_name["md2pdf"].description
    assert "background" in by_name["md2pdf"].description
    assert "translate" in by_name
    assert "batch_translate" in by_name
    assert "background" in by_name["translate"].description
    assert "target_language" in by_name["translate"].inputSchema["properties"]
    assert by_name["translate"].inputSchema["properties"]["model_tier"]["default"] == "low"
    assert "project_dir" in by_name["batch_translate"].inputSchema["properties"]
    assert "input" in by_name["md2pdf"].inputSchema["properties"]
    assert "output" in by_name["md2pdf"].inputSchema["properties"]
    assert "natural-language text" in by_name["extract_paper_ids"].description
    assert "0911.3380" in by_name["paper_ids_safe_dir_name"].description
    assert "cached parsed ar5iv text" in by_name["search_full_text"].description
    assert "parse" in by_name
    assert "canonical parsed JSON" in by_name["parse"].description
    assert "tex_path" in by_name["parse"].inputSchema["properties"]
    assert "pdf_path" in by_name["parse"].inputSchema["properties"]
    assert "mark_parsed_equation" in by_name
    assert "problematic" in by_name["mark_parsed_equation"].description
    assert "equation_id" in by_name["mark_parsed_equation"].inputSchema["properties"]
    assert "web search" in by_name["llm_infer_main_references"].description
    assert "arXiv papers" in by_name["get_title"].description
    assert "LLM summary" in by_name["llm_generate_summary"].description
    assert "INSPIRE metadata" in by_name["get_metadata"].description
    assert by_name["get_title"].inputSchema["properties"]["paper_id"]["description"].startswith("Single paper")
    assert "query" in by_name["search_full_text"].inputSchema["properties"]
    assert "limit" in by_name["get_citers"].inputSchema["properties"]
    assert "sort" in by_name["get_citers"].inputSchema["properties"]
    assert "enrich" in by_name["get_references"].inputSchema["properties"]
    assert by_name["get_section"].inputSchema["properties"]["section"]["description"].startswith("Section")
    assert "job_status" in by_name
    assert "cancel_job" in by_name
    assert "doctor_cache" in by_name
    assert "llm_domain_build" in by_name
    assert "domain_status" in by_name
    assert "domain_get_summary" in by_name
    assert "provider" not in by_name["domain_get_summary"].inputSchema["properties"]
    assert "llm_summary_batch_run" in by_name
    assert by_name["llm_domain_build"].inputSchema["properties"]["seed_paper"]["description"].startswith("Single paper")
    assert "background" in by_name["llm_generate_summary"].inputSchema["properties"]
    assert "background" in by_name["llm_infer_main_references"].inputSchema["properties"]
    assert "background" in by_name["llm_get_summary"].inputSchema["properties"]
    assert "background" in by_name["llm_domain_build"].inputSchema["properties"]
    assert "background" in by_name["llm_domain_get_summary"].inputSchema["properties"]
    assert "background" in by_name["llm_domain_get_graph"].inputSchema["properties"]
    assert "background" in by_name["llm_summary_batch_run"].inputSchema["properties"]
    assert "model_tier" in by_name["llm_generate_summary"].inputSchema["properties"]
    assert "model_tier" in by_name["llm_get_summary"].inputSchema["properties"]
    assert "model_tier" in by_name["llm_domain_build"].inputSchema["properties"]
    assert "model_tier" in by_name["llm_summary_batch_run"].inputSchema["properties"]
    assert "summary_batch_create" in by_name
    assert "summary_batch_prefetch" in by_name
    assert "summary_batch_status" in by_name
    assert "summary_batch_export" in by_name
    assert "summary_batch_retry_failed" in by_name
    provider_description = by_name["llm_generate_summary"].inputSchema["properties"]["provider"]["description"]
    assert "built-in provider" in provider_description
    assert "configured provider id" not in provider_description


def test_call_tool_dispatches_summary_batch_tools(monkeypatch, tmp_path):
    papers_file = tmp_path / "papers.txt"
    papers_file.write_text("0911.3380\n# comment\nastro-ph/0610514\n", encoding="utf-8")
    output_file = tmp_path / "summaries.jsonl"
    created = {}

    class FakeBatchDB:
        @classmethod
        def default(cls):
            return cls()

        def create_batch(self, name, paper_ids, prompt_version):
            created["create"] = {
                "name": name,
                "paper_ids": paper_ids,
                "prompt_version": prompt_version,
            }

        def status_counts(self, name):
            return {"queued": 2, "name": name}

        def retry_failed(self, name):
            created["retry_failed"] = name

    monkeypatch.setattr(server, "BatchDB", FakeBatchDB)
    monkeypatch.setattr(server, "prefetch_batch", lambda name, workers=4: {"name": name, "workers": workers})
    monkeypatch.setattr(server, "export_batch", lambda name, output: {"name": name, "output": str(output)})

    create = server.call_tool(
        "summary_batch_create",
        {"name": "batch-a", "papers_file": str(papers_file), "prompt_version": "paper-summary-v2"},
    )
    prefetch = server.call_tool("summary_batch_prefetch", {"name": "batch-a", "workers": 3})
    status = server.call_tool("summary_batch_status", {"name": "batch-a"})
    export = server.call_tool("summary_batch_export", {"name": "batch-a", "output": str(output_file)})
    retry = server.call_tool("summary_batch_retry_failed", {"name": "batch-a"})

    assert create["ok"] is True
    assert created["create"]["paper_ids"] == ["0911.3380", "astro-ph/0610514"]
    assert created["create"]["prompt_version"] == "paper-summary-v2"
    assert prefetch["data"] == {"name": "batch-a", "workers": 3}
    assert status["data"]["counts"] == {"queued": 2, "name": "batch-a"}
    assert export["data"] == {"name": "batch-a", "output": str(output_file)}
    assert retry["data"]["counts"] == {"queued": 2, "name": "batch-a"}
    assert created["retry_failed"] == "batch-a"


def test_get_llm_summary_starts_background_job_when_uncached(monkeypatch):
    monkeypatch.setenv("ARC_MCP_INLINE_WAIT_SEC", "0.001")

    def generate_summary(paper_ids, provider="auto", model=None, model_tier=None, refresh=False, progress_callback=None):
        time.sleep(0.02)
        if progress_callback:
            progress_callback(
                {
                    "event": "section_started",
                    "paper_id": paper_ids,
                    "section_index": 1,
                    "sections_total": 1,
                    "sections_completed": 0,
                    "section_id": "S1",
                    "title": "Intro",
                }
            )
            progress_callback(
                {
                    "event": "section_completed",
                    "paper_id": paper_ids,
                    "section_index": 1,
                    "sections_total": 1,
                    "sections_completed": 1,
                    "section_id": "S1",
                    "title": "Intro",
                }
            )
        return {
            "ok": True,
            "data": {"paper_id": paper_ids, "provider": provider},
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(
        server.service,
        "get_cached_llm_summary",
        lambda paper_ids: {"ok": False, "error": {"code": "summary_not_available"}},
    )
    monkeypatch.setattr(server.service, "generate_llm_summary", generate_summary)

    started = server.call_tool("llm_get_summary", {"paper_id": "0911.3380"})

    assert started["status"] == "job_running"
    assert started["job"]["paper_ids"] == "arXiv:0911.3380"
    status = _wait_for_job(started["job_id"])
    assert status["status"] == "done"
    assert status["sections_total"] == 1
    assert status["sections_completed"] == 1
    assert status["phase"] == "done"
    assert [event["event"] for event in status["events"]][-2:] == ["section_started", "section_completed"]
    assert server.job_result(started["job_id"])["result"]["data"]["paper_id"] == "arXiv:0911.3380"


def test_get_llm_summary_returns_cached_without_background_job(monkeypatch):
    monkeypatch.setattr(
        server.service,
        "get_cached_llm_summary",
        lambda paper_ids: {"ok": True, "data": {"title": "Cached"}, "errors": [], "meta": {"cache": "hit"}},
    )

    result = server.call_tool("llm_get_summary", {"paper_id": "0911.3380"})

    assert result["ok"] is True
    assert result["data"]["title"] == "Cached"


def test_llm_generate_summary_returns_inline_result_when_fast(monkeypatch):
    monkeypatch.setenv("ARC_MCP_INLINE_WAIT_SEC", "1")
    monkeypatch.setattr(
        server.service,
        "generate_llm_summary",
        lambda paper_ids, provider="auto", model=None, model_tier=None, refresh=False, progress_callback=None: {
            "ok": True,
            "data": {"paper_id": paper_ids},
            "errors": [],
            "meta": {},
        },
    )

    result = server.call_tool("llm_generate_summary", {"paper_id": "0911.3380"})

    assert result["ok"] is True
    assert result["data"]["paper_id"] == "arXiv:0911.3380"
    assert result["meta"]["job"]["status"] == "done"


def test_llm_generate_summary_background_true_returns_job_immediately(monkeypatch):
    monkeypatch.setenv("ARC_MCP_INLINE_WAIT_SEC", "10")

    def generate_summary(paper_ids, provider="auto", model=None, model_tier=None, refresh=False, progress_callback=None):
        time.sleep(0.05)
        return {
            "ok": True,
            "data": {"paper_id": paper_ids, "provider": provider},
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(server.service, "generate_llm_summary", generate_summary)

    started = server.call_tool("llm_generate_summary", {"paper_id": "0911.3380", "background": True})

    assert started["status"] == "job_running"
    assert started["background_requested"] is True
    assert started["inline_wait_seconds"] == 0.0
    assert started["job"]["background"] is True
    status = _wait_for_job(started["job_id"])
    assert status["status"] == "done"
    assert server.job_result(started["job_id"])["result"]["data"]["paper_id"] == "arXiv:0911.3380"


def test_llm_generate_summary_rejects_auto_provider_with_exact_model_before_background_job(monkeypatch):
    monkeypatch.setenv("ARC_MCP_INLINE_WAIT_SEC", "0")

    result = server.call_tool(
        "llm_generate_summary",
        {"paper_id": "0911.3380", "provider": "auto", "model": "gpt-5.5", "background": True},
    )

    assert result["ok"] is False
    assert "Exact model requires explicit provider" in result["error"]["message"]
    assert "job_id" not in result


def test_domain_build_starts_background_job(monkeypatch):
    monkeypatch.setenv("ARC_MCP_INLINE_WAIT_SEC", "0.001")

    def build_domain(
        seed_paper,
        intent="",
        domain_id=None,
        provider="auto",
        model=None,
        model_tier=None,
        refresh=False,
        workers=8,
    ):
        time.sleep(0.02)
        return {
            "ok": True,
            "data": {
                "seed_paper": seed_paper,
                "intent": intent,
                "domain_id": domain_id or "domain-test",
                "provider": provider,
                "model_tier": model_tier,
                "workers": workers,
            },
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(server.domain_service, "build_domain", build_domain)

    started = server.call_tool(
        "llm_domain_build",
        {"seed_paper": "0911.3380", "intent": "inflation", "model_tier": "high", "workers": 2},
    )

    assert started["status"] == "job_running"
    assert started["job"]["model_tier"] == "high"
    status = _wait_for_domain_job(started["job_id"])
    assert status["status"] == "done"
    assert status["seed_paper"] == "arXiv:0911.3380"
    assert "domain_status" in status
    result = server.job_result(started["job_id"])["result"]
    assert result["data"]["intent"] == "inflation"
    assert result["data"]["model_tier"] == "high"
    assert result["data"]["workers"] == 2


def test_domain_build_background_true_returns_job_immediately(monkeypatch):
    monkeypatch.setenv("ARC_MCP_INLINE_WAIT_SEC", "10")

    def build_domain(seed_paper, intent="", domain_id=None, provider="auto", model=None, refresh=False, workers=8):
        time.sleep(0.05)
        return {
            "ok": True,
            "data": {"seed_paper": seed_paper, "intent": intent, "provider": provider},
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(server.domain_service, "build_domain", build_domain)

    started = server.call_tool(
        "llm_domain_build",
        {"seed_paper": "0911.3380", "intent": "inflation", "background": True},
    )

    assert started["status"] == "job_running"
    assert started["background_requested"] is True
    assert started["job"]["background"] is True
    status = _wait_for_domain_job(started["job_id"])
    assert status["status"] == "done"
    assert server.job_result(started["job_id"])["result"]["data"]["intent"] == "inflation"


def test_domain_status_without_job_delegates_to_domain_service(monkeypatch):
    monkeypatch.setattr(
        server.domain_service,
        "status",
        lambda seed_paper=None, intent="", domain_id=None: {
            "ok": True,
            "data": {"seed_paper": seed_paper, "intent": intent, "domain_id": domain_id},
        },
    )

    result = server.call_tool("domain_status", {"seed_paper": "0911.3380", "intent": "inflation"})

    assert result["data"]["seed_paper"] == "0911.3380"
    assert result["data"]["intent"] == "inflation"


def test_domain_get_summary_starts_build_when_missing(monkeypatch):
    monkeypatch.setattr(
        server.domain_service,
        "get_domain_summary",
        lambda seed_paper=None, intent="", domain_id=None: {
            "ok": False,
            "error": {"code": "domain_summary_not_available", "message": "missing"},
            "errors": [],
            "meta": {},
        },
    )

    def build_domain(seed_paper, intent="", domain_id=None, provider="auto", model=None, refresh=False, workers=8):
        return {
            "ok": True,
            "data": {"seed_paper": seed_paper, "intent": intent, "provider": provider},
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(server.domain_service, "build_domain", build_domain)

    started = server.call_tool(
        "llm_domain_get_summary",
        {"seed_paper": "0911.3380", "intent": "inflation", "provider": "manual"},
    )

    assert started["ok"] is True
    assert started["meta"]["job"]["status"] == "done"
    assert started["data"]["provider"] == "manual"


def test_domain_get_summary_starts_build_when_cached_summary_invalid(monkeypatch):
    monkeypatch.setattr(
        server.domain_service,
        "get_domain_summary",
        lambda seed_paper=None, intent="", domain_id=None: {
            "ok": False,
            "error": {"code": "domain_summary_invalid", "message": "deterministic fallback summary is invalid"},
            "errors": [],
            "meta": {},
        },
    )

    def build_domain(seed_paper, intent="", domain_id=None, provider="auto", model=None, refresh=False, workers=8):
        return {
            "ok": True,
            "data": {"seed_paper": seed_paper, "intent": intent, "provider": provider},
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(server.domain_service, "build_domain", build_domain)

    started = server.call_tool(
        "llm_domain_get_summary",
        {"seed_paper": "0911.3380", "intent": "inflation", "provider": "manual"},
    )

    assert started["ok"] is True
    assert started["meta"]["job"]["status"] == "done"
    assert started["data"]["provider"] == "manual"


def test_domain_get_summary_is_cache_only(monkeypatch):
    monkeypatch.setattr(
        server.domain_service,
        "get_domain_summary",
        lambda seed_paper=None, intent="", domain_id=None: {
            "ok": False,
            "error": {"code": "domain_summary_not_available", "message": "missing"},
            "errors": [],
            "meta": {},
        },
    )

    result = server.call_tool("domain_get_summary", {"seed_paper": "0911.3380", "intent": "inflation"})

    assert result["ok"] is False
    assert result["error"]["code"] == "domain_summary_not_available"


def test_cancel_job_requires_explicit_tool_call(monkeypatch):
    monkeypatch.setenv("ARC_MCP_INLINE_WAIT_SEC", "0.001")

    def generate_summary(paper_ids, provider="auto", model=None, model_tier=None, refresh=False, progress_callback=None):
        time.sleep(0.05)
        return {"ok": True, "data": {}, "errors": [], "meta": {}}

    monkeypatch.setattr(
        server.service,
        "get_cached_llm_summary",
        lambda paper_ids: {"ok": False, "error": {"code": "summary_not_available"}},
    )
    monkeypatch.setattr(server.service, "generate_llm_summary", generate_summary)

    started = server.call_tool("llm_get_summary", {"paper_id": "0911.3380"})
    cancelled = server.call_tool("cancel_job", {"job_id": started["job_id"]})

    assert cancelled["status"] in {"cancel_requested", "cancelled"}


def _wait_for_job(job_id):
    deadline = time.time() + 2
    while time.time() < deadline:
        status = server.get_summary_job_status(job_id)
        if status["status"] in {"done", "failed", "needs_llm", "cancelled"}:
            return status
        time.sleep(0.01)
    raise AssertionError("summary job did not finish")


def _wait_for_domain_job(job_id):
    deadline = time.time() + 2
    while time.time() < deadline:
        status = server.get_domain_job_status(job_id)
        if status["status"] in {"done", "failed", "needs_llm", "cancelled"}:
            return status
        time.sleep(0.01)
    raise AssertionError("domain job did not finish")


def _wait_for_mcp_job(job_id):
    deadline = time.time() + 2
    while time.time() < deadline:
        status = server.job_status(job_id)
        if status["status"] in {"done", "failed", "needs_llm", "cancelled"}:
            return status
        time.sleep(0.01)
    raise AssertionError("MCP job did not finish")
