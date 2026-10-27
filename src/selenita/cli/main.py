"""selenita CLI: single entry point that dispatches the three modes.

    selenita --acp [--port N]       ACP server (JSON-RPC NDJSON over stdio; TCP pending)
    selenita "prompt" [-s ID]       non-interactive CLI mode (one shot)
    selenita [-s ID]                interactive TUI mode (basic REPL)

The CLI only parses and delegates: logic lives in ``selenita.core`` (D012 — all entry
point modes target the same core; MAF powers it without touching this layer).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
from dataclasses import replace

from selenita.acp_server import runner
from selenita.cli import terminal, tui
from selenita.config import Settings, setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="selenita",
        description="ACP (Agent Client Protocol) agent powered by Microsoft Agent Framework.",
        epilog=(
            "Modes: --acp starts the ACP server; a positional prompt runs a single "
            "non-interactive turn; with no prompt the interactive TUI opens. "
            "Sessions (-s/--session-id) are shared between the CLI and the TUI."
        ),
    )
    parser.add_argument(
        "--acp",
        action="store_true",
        help="Run selenita as an ACP server (JSON-RPC NDJSON over stdio).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"selenita {importlib.metadata.version('selenita')}",
        help="Print the installed selenita version and exit.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="N",
        help="Serve the ACP protocol over TCP NDJSON too, on port N (not implemented yet).",
    )
    parser.add_argument(
        "-s",
        "--session-id",
        default=None,
        metavar="SESS_ID",
        help="Reuse this session (sess_*); when missing, start with an empty history.",
    )
    parser.add_argument("prompt", nargs="?", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "-p",
        "--prompt",
        dest="prompt_flag",
        default=None,
        help="Prompt as a parameter (alternative to the positional prompt).",
    )
    parser.add_argument(
        "--mcp-cmd",
        dest="mcp_cmd",
        default=None,
        metavar="COMMAND",
        help=(
            "Attach one stdio MCP server in terminal modes, e.g. "
            '--mcp-cmd "npx -y @modelcontextprotocol/server-filesystem /tmp". '
            "In ACP mode the client declares servers per session instead."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``selenita`` console script; returns the exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    setup_logging(settings)

    if args.acp:
        if args.prompt or args.prompt_flag:
            parser.error("--acp does not accept a prompt: turns arrive via the ACP protocol")
        if args.session_id is not None:
            parser.error("--acp does not accept --session-id: ids are created by the ACP client")
        if args.mcp_cmd:
            parser.error(
                "--mcp-cmd only applies to terminal modes: in ACP the client declares "
                "MCP servers per session (mcpServers of session/new)"
            )
        if args.port is not None:
            try:
                asyncio.run(runner.run_acp_server_tcp(args.port))
            except NotImplementedError as exc:
                parser.error(str(exc))
            return 0
        asyncio.run(runner.run_acp_server_stdio())
        return 0

    if args.prompt is not None and args.prompt_flag is not None:
        parser.error("use only one prompt form: positional or --prompt/-p")
    prompt = args.prompt if args.prompt is not None else args.prompt_flag
    if args.mcp_cmd:
        settings = replace(settings, mcp_cmd=args.mcp_cmd)
    if prompt is not None:
        return terminal.run_terminal_prompt(prompt, args.session_id, settings)
    return tui.run_interactive(args.session_id, settings=settings)


if __name__ == "__main__":  # pragma: no cover - enables ``python -m selenita``
    raise SystemExit(main())
