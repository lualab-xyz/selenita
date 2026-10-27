"""Built-in tools of selenita (tools epic) + their MAF registration.

The model can call a handful of its own tools (mirroring what other coding agents
offer): read/write/edit files, run a shell command, fetch a URL and — only in the
interactive TUI — ask the user a question. Approval is decided by the permission
policy in ``core/permissions.py``: read-only tools (``selenita_read_file``,
``selenita_web_fetch``) and ``ask_user`` (the user's own input needs no permission)
run freely; the mutating ones (``write_file``, ``edit_file``, ``run_command``) ask
the user through the adapter; ``ask_user`` is only injected in the TUI (D: "no se
inyecta esa tool" in non-interactive modes).

``selenita_edit_file`` follows the exact-match convention used by other agents
(reference: opencode's edit tool): ``old_string`` must match exactly once, so a
slip (wrong quote/dash/whitespace, duplicated block) fails loudly instead of
corrupting the file; ``replace_all`` opts into replacing every occurrence. Smart
quotes, dashes and unicode spaces are normalized when matching, and CRLF line
endings are tolerated.
"""

from __future__ import annotations

import logging
import re
import subprocess
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import agent_framework as af
from agent_framework.exceptions import ToolException

logger = logging.getLogger(__name__)

# Output caps: keep the model context small no matter what a file/URL/command returns.
_READ_MAX_CHARS = 100_000
_FETCH_MAX_BYTES = 300_000
_RUN_MAX_CHARS = 40_000
_DEFAULT_RUN_TIMEOUT = 120  # seconds

DETAIL_LIMIT = 100  # chars of a tool detail line before it is truncated with "…"

# AskUser callback: (question, options) -> the user's answer (TUI only).
AskUserCallback = Callable[[str, Sequence[str] | None], Awaitable[str]]


def summarize_tool_call(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """One compact line describing a tool call (permission prompts and ``$`` blocks).

    The command tools show their command verbatim, anything else shows ``key=value``
    pairs; overlong details are cut at ``_DETAIL_LIMIT`` with an ellipsis so the
    prompt stays on a single row.
    """
    if tool_name == "selenita_run_command":
        detail = str(arguments.get("command", ""))
    else:
        detail = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    if len(detail) <= DETAIL_LIMIT:
        return detail
    return detail[:DETAIL_LIMIT] + "…"


# Human label of each tool in the ``$`` execution blocks (Copilot-CLI-style).
TOOL_TITLES = {
    "selenita_read_file": "Read file",
    "selenita_write_file": "Write file",
    "selenita_edit_file": "Edit file",
    "selenita_run_command": "Shell",
    "selenita_web_fetch": "WebFetch",
    "selenita_ask_user": "Ask user",
}


def tool_title(tool_name: str) -> str:
    """Display title of a tool call; unknown tools (e.g. MCP) keep their raw name."""
    return TOOL_TITLES.get(tool_name, tool_name)


# ---------------------------------------------------------------------------
# Individual tools (each one is safe to unit-test in isolation)
# ---------------------------------------------------------------------------


def selenita_read_file(path: str, max_chars: int = _READ_MAX_CHARS) -> str:
    """Read a UTF-8 text file (truncated when huge); returns its contents."""
    target = Path(path)
    if not target.is_file():
        raise ToolException(f"File not found: {path}")
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolException(f"{path} is not valid UTF-8 text: {exc}") from exc
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n… (truncated to {max_chars} chars)"
    return text


def selenita_write_file(path: str, content: str) -> str:
    """Create or overwrite a whole file with ``content`` (parents created as needed)."""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ToolException(f"Could not write {path}: {exc}") from exc
    return f"Wrote {len(content)} chars to {path}."


def selenita_edit_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Replace ``old_string`` with ``new_string`` in ``path``.

    ``old_string`` must match exactly once unless ``replace_all`` is set: ambiguous or
    missing matches fail loudly instead of editing the wrong spot.
    """
    if not old_string:
        raise ToolException(
            "old_string must not be empty; use selenita_write_file to create or overwrite a file."
        )
    if old_string == new_string:
        raise ToolException("new_string must differ from old_string.")
    target = Path(path)
    try:
        # newline="" disables universal-newline translation so \r\n files keep
        # their exact bytes and _find_matches can locate CRLF-tolerant spans.
        with target.open("r", encoding="utf-8", newline="") as fh:
            content = fh.read()
    except OSError as exc:
        raise ToolException(f"Could not read {path}: {exc}") from exc
    matches = _find_matches(content, old_string)
    if not matches:
        raise ToolException(
            f"old_string not found in {path}. Copy the exact text to replace, minding "
            "quotes, dashes and whitespace."
        )
    if len(matches) > 1 and not replace_all:
        raise ToolException(
            f"old_string matched {len(matches)} times in {path}: add more surrounding "
            "context, or set replace_all=True to replace every occurrence."
        )
    chosen = matches if replace_all else matches[:1]
    result = _apply_replacements(content, chosen, new_string)
    try:
        with target.open("w", encoding="utf-8", newline="") as fh:
            fh.write(result)
    except OSError as exc:
        raise ToolException(f"Could not write {path}: {exc}") from exc
    return f"Edited {path}: {len(chosen)} replacement(s) applied."


def selenita_run_command(command: str, timeout: int = _DEFAULT_RUN_TIMEOUT) -> str:
    """Run ``command`` through the shell and return exit code plus captured output."""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ToolException(
            f"Command timed out after {timeout}s: {command!r}. "
            "No output was captured from the timed-out process."
        ) from None
    except OSError as exc:
        raise ToolException(f"Could not run command: {exc}") from exc
    stdout = _cap(proc.stdout, _RUN_MAX_CHARS)
    stderr = _cap(proc.stderr, _RUN_MAX_CHARS)
    return f"exit code: {proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"


def selenita_web_fetch(url: str, max_bytes: int = _FETCH_MAX_BYTES) -> str:
    """Fetch ``url`` and return its body as text (capped; non-text bytes are lossy)."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 (user-approved)
            data = response.read(max_bytes + 1)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ToolException(f"web_fetch failed for {url}: {exc}") from exc
    truncated = len(data) > max_bytes
    text = data[:max_bytes].decode("utf-8", errors="replace")
    if truncated:
        text += f"\n… (body truncated to {max_bytes} bytes)"
    return text


def _make_ask_user(
    ask_user_callback: AskUserCallback | None,
) -> Callable[..., Awaitable[str]]:
    """Build the ``ask_user`` tool around the adapter's callback (TUI only)."""

    async def selenita_ask_user(question: str, options: list[str] | None = None) -> str:
        if ask_user_callback is None:
            return "No interactive user is available: ask_user only works in the TUI."
        return await ask_user_callback(question, options)

    return selenita_ask_user


# ---------------------------------------------------------------------------
# Toolkit: which tools are offered, and how approvals are collected
# ---------------------------------------------------------------------------


@dataclass
class Toolkit:
    """Which built-in tools are injected into the agent, plus permission wiring.

    Adapters build a toolkit and pass it to ``create_core``:
    - the TUI offers everything and wires the permission/ask_user prompts;
    - the one-shot CLI offers mutating tools only when stdin is a TTY (no ask_user);
    - the ACP server currently offers the read-only tools only (permission channel is
      a later phase, D phase).
    """

    include_mutating: bool = True
    include_ask_user: bool = False
    asker: Any | None = None  # ApprovalPolicy.Asker (avoid a circular import)
    ask_user_callback: AskUserCallback | None = None


def make_builtin_tools(toolkit: Toolkit) -> list[af.FunctionTool]:
    """Build the MAF ``FunctionTool`` list described by ``toolkit``."""
    tools: list[af.FunctionTool] = [
        af.FunctionTool(
            name="selenita_read_file",
            description="Read a UTF-8 text file from the workspace and return its "
            "contents (truncated when very large). Use before editing a file.",
            func=selenita_read_file,
        ),
        af.FunctionTool(
            name="selenita_web_fetch",
            description="Fetch a URL over HTTPS and return its body as text "
            "(truncated when very large). Use to read documentation or web pages.",
            func=selenita_web_fetch,
        ),
    ]
    if toolkit.include_mutating:
        tools += [
            af.FunctionTool(
                name="selenita_write_file",
                description="Create or overwrite an entire file with the given "
                "content. Use for new files or full rewrites; for a small change use "
                "selenita_edit_file instead.",
                func=selenita_write_file,
            ),
            af.FunctionTool(
                name="selenita_edit_file",
                description="Replace an exact old_string with new_string inside a "
                "file. old_string must match exactly once unless replace_all=True; "
                "include enough surrounding context to make it unique.",
                func=selenita_edit_file,
            ),
            af.FunctionTool(
                name="selenita_run_command",
                description="Run a shell command in the workspace and return its exit "
                "code plus captured stdout/stderr. It can take a while; a timeout "
                "guard is enforced.",
                func=selenita_run_command,
            ),
        ]
    if toolkit.include_ask_user:
        tools.append(
            af.FunctionTool(
                name="selenita_ask_user",
                description="Ask the user a question and return their answer. Use when "
                "you need a decision or information that only the user can provide. "
                "Only available in the interactive TUI.",
                func=_make_ask_user(toolkit.ask_user_callback),
            )
        )
    return tools


# ---------------------------------------------------------------------------
# edit_file matching helpers (exact-match, opencode-style)
# ---------------------------------------------------------------------------


def _normalize_for_match(value: str) -> str:
    """Map smart quotes/dashes/unicode spaces to their plain ASCII counterparts."""
    return (
        value.replace("‘", "'")
        .replace("’", "'")
        .replace("‚", "'")
        .replace("‛", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("„", '"')
        .replace("‟", '"')
        .replace("‐", "-")
        .replace("‑", "-")
        .replace("‒", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("―", "-")
        .replace("−", "-")
        .replace("\u00a0", " ")
        .replace("\u2002", " ")
        .replace("\u2003", " ")
        .replace("\u2004", " ")
        .replace("\u2005", " ")
        .replace("\u2006", " ")
        .replace("\u2007", " ")
        .replace("\u2008", " ")
        .replace("\u2009", " ")
        .replace("\u200a", " ")
        .replace("\u202f", " ")
        .replace("\u205f", " ")
        .replace("\u3000", " ")
    )


def _find_matches(content: str, search: str) -> list[tuple[int, int]]:
    """(start, end) spans of ``search`` in ``content``, tolerant of quotes/CRLF.

    First tries exact substring occurrences (handles in-line and no-trailing-newline
    edits), then falls back to a line-by-line match that normalizes smart quotes,
    dashes and unicode spaces and tolerates CRLF line endings.
    """
    exact = _raw_occurrences(content, search)
    if exact:
        return exact
    return _line_occurrences(content, search)


def _raw_occurrences(content: str, search: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = content.find(search)
    while start != -1:
        spans.append((start, start + len(search)))
        start = content.find(search, start + len(search))
    return spans


def _line_occurrences(content: str, search: str) -> list[tuple[int, int]]:
    """Normalized, CRLF-tolerant match of ``search`` across whole lines."""
    trailing_newline = search.endswith("\n")
    expected = [normal for normal in search.split("\n")]
    if trailing_newline:
        expected.pop()

    lines: list[tuple[int, int, int, bool, str]] = []  # start, end, content_end, newline, text
    for match in re.finditer(r"[^\n]*(?:\n|$)", content):
        line = match[0]
        if not line:
            continue
        newline = line.endswith("\n")
        text = line[:-1] if newline else line
        content_end = match.start() + len(text) - (1 if text.endswith("\r") else 0)
        lines.append((match.start(), match.start() + len(line), content_end, newline, text))

    spans: list[tuple[int, int]] = []
    for index, entry in enumerate(lines):
        window = lines[index : index + len(expected)]
        if len(window) != len(expected):
            continue
        if any(
            _normalize_for_match(item[4].rstrip())
            != _normalize_for_match(exp.rstrip())
            for item, exp in zip(window, expected, strict=True)
        ):
            continue
        last = window[-1]
        if trailing_newline and not last[3]:
            continue
        end = last[2] if not trailing_newline else last[1]
        spans.append((entry[0], end))
    # Deduplicate overlapping candidates (a shorter match inside a longer one).
    unique: list[tuple[int, int]] = []
    for start, end in spans:
        if any(s < end and start < e for s, e in unique):
            continue
        unique.append((start, end))
    return unique


def _apply_replacements(
    content: str, spans: Sequence[tuple[int, int]], new_string: str
) -> str:
    """Apply ``spans`` (already in document order) with ``new_string`` in one pass."""
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(content[cursor:start])
        pieces.append(new_string)
        cursor = end
    pieces.append(content[cursor:])
    return "".join(pieces)


def _cap(text: str, limit: int) -> str:
    """Truncate ``text`` at ``limit`` chars with an explicit notice."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… (truncated to {limit} chars)"


__all__ = [
    "AskUserCallback",
    "Toolkit",
    "make_builtin_tools",
    "selenita_edit_file",
    "selenita_read_file",
    "selenita_run_command",
    "selenita_web_fetch",
    "selenita_write_file",
    "summarize_tool_call",
]
