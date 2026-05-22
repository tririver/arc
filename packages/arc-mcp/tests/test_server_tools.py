import asyncio
import time

from arc_mcp import server


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
    assert "LLM summary" in by_name["generate_LLM_summary"].description
    assert "INSPIRE metadata" in by_name["get_metadata"].description
    assert by_name["get_title"].inputSchema["properties"]["paper_id"]["description"].startswith("Single paper")
    assert "limit" in by_name["get_citers"].inputSchema["properties"]
    assert "sort" in by_name["get_citers"].inputSchema["properties"]
    assert "enrich" in by_name["get_references"].inputSchema["properties"]
    assert by_name["get_section"].inputSchema["properties"]["section"]["description"].startswith("Section")
    assert "get_LLM_summary_status" in by_name
    assert "doctor_cache" in by_name
    assert "domain_build" in by_name
    assert "domain_status" in by_name
    assert "domain_get_summary" in by_name
    assert "provider" in by_name["domain_get_summary"].inputSchema["properties"]
    assert by_name["domain_build"].inputSchema["properties"]["seed_paper"]["description"].startswith("Single paper")


def test_get_llm_summary_starts_background_job_when_uncached(monkeypatch):
    def generate_summary(paper_ids, provider="auto", model=None, refresh=False, progress_callback=None):
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

    started = server.call_tool("get_LLM_summary", {"paper_id": "0911.3380"})

    assert started["status"] == "summary_job_started"
    assert started["paper_ids"] == "arXiv:0911.3380"
    status = _wait_for_job(started["job_id"])
    assert status["status"] == "done"
    assert status["sections_total"] == 1
    assert status["sections_completed"] == 1
    assert status["phase"] == "done"
    assert [event["event"] for event in status["events"]][-2:] == ["section_started", "section_completed"]
    assert status["result"]["data"]["paper_id"] == "arXiv:0911.3380"


def test_get_llm_summary_returns_cached_without_background_job(monkeypatch):
    monkeypatch.setattr(
        server.service,
        "get_cached_llm_summary",
        lambda paper_ids: {"ok": True, "data": {"title": "Cached"}, "errors": [], "meta": {"cache": "hit"}},
    )

    result = server.call_tool("get_LLM_summary", {"paper_id": "0911.3380"})

    assert result["ok"] is True
    assert result["data"]["title"] == "Cached"


def test_domain_build_starts_background_job(monkeypatch):
    def build_domain(seed_paper, intent="", domain_id=None, provider="auto", model=None, refresh=False, workers=8):
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

    started = server.call_tool("domain_build", {"seed_paper": "0911.3380", "intent": "inflation", "workers": 2})

    assert started["status"] == "domain_job_started"
    status = _wait_for_domain_job(started["job_id"])
    assert status["status"] == "done"
    assert status["seed_paper"] == "arXiv:0911.3380"
    assert "domain_status" in status
    assert status["result"]["data"]["intent"] == "inflation"
    assert status["result"]["data"]["workers"] == 2


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
        "domain_get_summary",
        {"seed_paper": "0911.3380", "intent": "inflation", "provider": "manual"},
    )

    assert started["status"] == "domain_job_started"
    status = _wait_for_domain_job(started["job_id"])
    assert status["status"] == "done"
    assert status["result"]["data"]["provider"] == "manual"


def _wait_for_job(job_id):
    deadline = time.time() + 2
    while time.time() < deadline:
        status = server.get_summary_job_status(job_id)
        if status["status"] != "running":
            return status
        time.sleep(0.01)
    raise AssertionError("summary job did not finish")


def _wait_for_domain_job(job_id):
    deadline = time.time() + 2
    while time.time() < deadline:
        status = server.get_domain_job_status(job_id)
        if status["status"] != "running":
            return status
        time.sleep(0.01)
    raise AssertionError("domain job did not finish")
