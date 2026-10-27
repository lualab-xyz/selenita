"""Tiny stdio MCP server used by selenita tests and manual smokes (tools epic).

Run it with: ``.venv/bin/python tests/fixtures/demo_mcp_server.py`` — it speaks MCP over
stdio, so an MCP client (e.g. ``agent_framework.MCPStdioTool``) launches it as a child
process. Tools are deliberately harmless: pure functions over their arguments.
"""

from __future__ import annotations

from datetime import date

from mcp.server.mcpserver import MCPServer

server = MCPServer(name="selenita-demo", instructions="Demo tools for selenita tests.")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool()
def today() -> str:
    """Return today's date as YYYY-MM-DD."""
    return date.today().isoformat()


@server.tool()
def shout(text: str) -> str:
    """Uppercase the given text."""
    return text.upper()


if __name__ == "__main__":  # pragma: no cover - process entry point
    server.run()
