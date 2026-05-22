import asyncio

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
