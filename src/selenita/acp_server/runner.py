"""Entry points of the ACP server (stdio transport; TCP NDJSON has its own task)."""

from __future__ import annotations

import logging

from selenita.acp_server.agent import run_acp_server

logger = logging.getLogger(__name__)


async def run_acp_server_stdio() -> None:
    """ACP server over stdio: JSON-RPC NDJSON on stdin/stdout (ACP v1 spec)."""
    logger.info("acp server: starting stdio transport")
    await run_acp_server()
    logger.info("acp server: stdio transport closed")


async def run_acp_server_tcp(port: int) -> None:
    """ACP server over TCP NDJSON (custom ACP v1 transport).

    Pending implementation (task ``acp-tcp-ndjson``): same JSON-RPC framing and protocol
    lifespan, over a TCP socket with an optional port.
    """
    raise NotImplementedError("TCP NDJSON transport (--port) is not implemented yet")
