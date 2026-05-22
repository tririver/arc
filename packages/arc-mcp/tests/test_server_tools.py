import asyncio
import time

from arc_mcp import server


def test_call_tool_dispatches_to_service(monkeypatch):
    monkeypatch.setattr(server.service, "get_title", lambda paper_ids, refresh=False: {"ok": True, "data": paper_ids})

    result = server.call_tool("get_title", {"paper_id": "arXiv:0911.3380"})

    assert result == {"ok": True, "data": "arXiv:0911.3380"}


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
    assert by_name["get_title"].inputSchema["properties"]["paper_id"]["description"].startswith("Single paper")
    assert by_name["get_section"].inputSchema["properties"]["section"]["description"].startswith("Section")
    assert "get_LLM_summary_status" in by_name
    assert "doctor_cache" in by_name


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


def _wait_for_job(job_id):
    deadline = time.time() + 2
    while time.time() < deadline:
        status = server.get_summary_job_status(job_id)
        if status["status"] != "running":
            return status
        time.sleep(0.01)
    raise AssertionError("summary job did not finish")
