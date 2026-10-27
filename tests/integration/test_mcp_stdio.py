"""Integration tests of the MCP stdio adapters against the bundled demo server.

These spawn a real MCP server child process (``tests/fixtures/demo_mcp_server.py``) and
exercise the spec → ``MCPStdioTool`` path end to end: connect, list functions, call a tool
and close cleanly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from selenita.core.mcp import McpServerSpec, close_mcp_tools, open_mcp_tools

_SERVER = str(Path(__file__).parent.parent / "fixtures" / "demo_mcp_server.py")


@pytest.mark.asyncio
async def test_open_tools_from_spec_connects_calls_and_closes() -> None:
    spec = McpServerSpec(name="demo", command=sys.executable, args=(_SERVER,))
    tools = await open_mcp_tools([spec])
    assert len(tools) == 1
    tool = tools[0]
    try:
        functions = [f.name for f in tool.functions]
        assert {"add", "today", "shout"} <= set(functions)
        result = await tool.call_tool("add", a=2, b=3)
        joined = "".join(getattr(item, "text", "") or "" for item in result)
        assert "5" in joined
    finally:
        await close_mcp_tools(tools)


@pytest.mark.asyncio
async def test_open_tools_reports_failing_server_but_survives(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    spec = McpServerSpec(name="broken", command=sys.executable, args=("-c", "raise SystemExit(1)"))
    with caplog.at_level(logging.WARNING):
        tools = await open_mcp_tools([spec])
    assert tools == []
    assert any("failed to start" in record.message for record in caplog.records)
