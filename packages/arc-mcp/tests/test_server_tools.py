import asyncio
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

    assert "arXiv papers" in by_name["get_title"].description
    assert "LLM summary" in by_name["llm_generate_summary"].description
    assert "INSPIRE metadata" in by_name["get_metadata"].description
    assert by_name["get_title"].inputSchema["properties"]["paper_id"]["description"].startswith("Single paper")
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
    assert "background" in by_name["llm_get_summary"].inputSchema["properties"]
    assert "background" in by_name["llm_domain_build"].inputSchema["properties"]
    assert "background" in by_name["llm_domain_get_summary"].inputSchema["properties"]
    assert "background" in by_name["llm_domain_get_graph"].inputSchema["properties"]
    assert "background" in by_name["llm_summary_batch_run"].inputSchema["properties"]


def test_get_llm_summary_starts_background_job_when_uncached(monkeypatch):
    monkeypatch.setenv("ARC_MCP_INLINE_WAIT_SEC", "0.001")

    def generate_summary(paper_ids, provider="auto", model=None, refresh=False, progress_callback=None):
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
        lambda paper_ids, provider="auto", model=None, refresh=False, progress_callback=None: {
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

    def generate_summary(paper_ids, provider="auto", model=None, refresh=False, progress_callback=None):
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


def test_domain_build_starts_background_job(monkeypatch):
    monkeypatch.setenv("ARC_MCP_INLINE_WAIT_SEC", "0.001")

    def build_domain(seed_paper, intent="", domain_id=None, provider="auto", model=None, refresh=False, workers=8):
        time.sleep(0.02)
        return {
            "ok": True,
            "data": {
                "seed_paper": seed_paper,
                "intent": intent,
                "domain_id": domain_id or "domain-test",
                "provider": provider,
                "workers": workers,
            },
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(server.domain_service, "build_domain", build_domain)

    started = server.call_tool("llm_domain_build", {"seed_paper": "0911.3380", "intent": "inflation", "workers": 2})

    assert started["status"] == "job_running"
    status = _wait_for_domain_job(started["job_id"])
    assert status["status"] == "done"
    assert status["seed_paper"] == "arXiv:0911.3380"
    assert "domain_status" in status
    result = server.job_result(started["job_id"])["result"]
    assert result["data"]["intent"] == "inflation"
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

    def generate_summary(paper_ids, provider="auto", model=None, refresh=False, progress_callback=None):
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
