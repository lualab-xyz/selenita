"""MCP (stdio) integration: server specs → MAF tools (tools epic, D013).

Sources of MCP servers (MVP scope):
- Terminal modes (CLI/TUI): ``SELENITA_MCP_SERVERS`` — JSON list of ``{"name"?,
  "command", "args"?, "env"?}`` — and the ``--mcp-cmd`` flag (one server, shlex-split).
- ACP mode: the ``mcp_servers`` the ACP client declares in ``session/new``.

Degrades gracefully (D008): a spec that cannot be parsed or a server that fails to start
is skipped with a warning; the conversation never dies because of MCP.
"""

from __future__ import annotations

import json
import logging
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import agent_framework as af

from selenita.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class McpServerSpec:
    """One stdio MCP server: ``command args…`` with optional extra environment."""

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)


def parse_specs_json(raw: str) -> list[McpServerSpec]:
    """Parse ``SELENITA_MCP_SERVERS`` (JSON list of server objects) into specs.

    Raises:
        ValueError: malformed JSON or a server without a usable ``command``.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("must be a JSON list of server objects")
    specs: list[McpServerSpec] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or not item.get("command"):
            raise ValueError(f"server #{index} needs at least {{'command'}}")
        args = item.get("args") or []
        env = item.get("env") or {}
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ValueError(f"server #{index}: 'args' must be a list of strings")
        if not isinstance(env, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in env.items()
        ):
            raise ValueError(f"server #{index}: 'env' must be a mapping of strings")
        name = str(item.get("name") or Path(str(item["command"])).name)
        specs.append(
            McpServerSpec(
                name=name, command=str(item["command"]), args=tuple(args), env=dict(env)
            )
        )
    return specs


def parse_spec_cmd(raw: str, name: str | None = None) -> McpServerSpec:
    """Parse the ``--mcp-cmd`` flag: shlex-split command line of a single server.

    Raises:
        ValueError: the command line is empty.
    """
    parts = shlex.split(raw)
    if not parts:
        raise ValueError("empty command line")
    return McpServerSpec(
        name=name or Path(parts[0]).name, command=parts[0], args=tuple(parts[1:])
    )


def collect_terminal_specs(settings: Settings) -> list[McpServerSpec]:
    """Specs for the terminal modes (env JSON + ``--mcp-cmd``), degrading to fewer specs."""
    specs: list[McpServerSpec] = []
    if settings.mcp_servers:
        try:
            specs.extend(parse_specs_json(settings.mcp_servers))
        except ValueError as exc:
            logger.warning("ignoring SELENITA_MCP_SERVERS: %s", exc)
    if settings.mcp_cmd:
        try:
            specs.append(parse_spec_cmd(settings.mcp_cmd))
        except ValueError as exc:
            logger.warning("ignoring --mcp-cmd: %s", exc)
    return specs


def specs_from_acp(mcp_servers: Any) -> list[McpServerSpec]:
    """Translate the ``mcp_servers`` of ``session/new`` (SDK objects or plain dicts).

    Anything unusable (missing command, or a non-stdio transport) is skipped with a
    warning: stdio is the MVP transport.
    """
    specs: list[McpServerSpec] = []
    for server in list(mcp_servers or []):
        if isinstance(server, dict):
            name = server.get("name")
            command = server.get("command")
            args = server.get("args") or []
            env = server.get("env") or {}
        else:
            name = getattr(server, "name", None)
            command = getattr(server, "command", None)
            args = getattr(server, "args", None) or []
            env = getattr(server, "env", None) or {}
        if not command:
            logger.warning("skipping MCP server %r: not stdio or missing command", name)
            continue
        specs.append(
            McpServerSpec(
                name=str(name or Path(str(command)).name),
                command=str(command),
                args=tuple(str(arg) for arg in args),
                env={str(key): str(value) for key, value in dict(env).items()},
            )
        )
    return specs


async def open_mcp_tools(specs: Sequence[McpServerSpec]) -> list[af.MCPStdioTool]:
    """Start every stdio server and return its MAF tool; failures are skipped (D008)."""
    tools: list[af.MCPStdioTool] = []
    for spec in specs:
        tool = af.MCPStdioTool(
            name=spec.name, command=spec.command, args=list(spec.args), env=dict(spec.env)
        )
        try:
            await tool.connect()
        except Exception as exc:
            logger.warning("MCP server %r failed to start (skipping): %s", spec.name, exc)
            continue
        logger.info("MCP server connected: %s (%s)", spec.name, spec.command)
        tools.append(tool)
    return tools


async def close_mcp_tools(tools: Sequence[af.MCPStdioTool]) -> None:
    """Stop every connected MCP tool; close errors are logged, never raised."""
    for tool in tools:
        try:
            await tool.close()
        except Exception as exc:
            logger.warning("error closing MCP tool %s: %s", getattr(tool, "name", "?"), exc)
