"""Interactive TUI mode (``selenita`` with no arguments).

Dependency-free minimalist REPL:
- Welcome screen with an ASCII moon logo, session metadata and the last message of the
  session (truncated when long, with an explicit notice).
- Cyan ``❯ `` prompt for the user; agent answers carry the magenta ``¤`` marker and
  indentation; model reasoning streams in a muted (dim + italic) style with a ``·``.
- Alt+Enter (and Shift+Enter when the terminal distinguishes it) inserts a newline drawn
  with a double indent so multi-line text reads like a block.
- "exit"/"quit" or Ctrl+D/Ctrl+C end the session.

For testability the line reader is injectable (``reader``); over a pipe a line-by-line
mode without raw keyboard is used.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import logging
import os
import select
import shutil
import signal
import subprocess
import sys
import termios
import tty
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import IO, TextIO

from selenita import __version__
from selenita.config import Settings
from selenita.core import AgentCore, create_core
from selenita.core.permissions import (
    DECISION_ALWAYS,
    DECISION_APPROVE,
    DECISION_REJECT,
    ToolEvent,
)
from selenita.core.tools import Toolkit, summarize_tool_call
from selenita.sessions.store import SessionStore, validate_session_id

logger = logging.getLogger(__name__)

_EXIT_WORDS = {"exit", "quit"}

# Colors/styles: prompt marks the user's own input, ¤ marks the agent's answer, and the
# reasoning stream is muted (dim + italic) so it reads as "thinking" underneath.
_COLOR_PROMPT = "\x1b[36m❯ \x1b[0m"  # cyan: differential color of the user prompt
_COLOR_AGENT = "\x1b[35m¤\x1b[0m"  # magenta: marker of the agent answers
_COLOR_REASON = "\x1b[36m·\x1b[0m"  # cyan dot marker for the reasoning block
_COLOR_WARN = "\x1b[33m⚠\x1b[0m"  # yellow: permission prompts
_COLOR_ASK = "\x1b[33m?\x1b[0m"  # yellow: ask_user questions
_COLOR_TOOL = "\x1b[36m$\x1b[0m"  # cyan: tool-execution block marker
_TOOL_MARKER = "$"
_AGENT_MARKER = "¤"
_REASON_MARKER = "·"
_REASON_STYLE = "\x1b[2m\x1b[3m"  # dim + italic for the reasoning stream and tool details
_MUTED = _REASON_STYLE  # same muted look for tool-call detail lines in permission prompts
_PLAIN_PROMPT = "❯ "
_BOLD = "\x1b[1m"  # "Selenita" title in the welcome screen
_DIM = "\x1b[2m"  # metadata and help in muted gray
_CYAN = "\x1b[36m"
_RESET = "\x1b[0m"

# ASCII moon logo shown at the top of the welcome screen (last line fuses with the info).
_LOGO = ("  ▄██▀", " ██▒", " ██▒", "  ▀██▄")

# Truncation of the last message in the welcome screen
_KEEP_LINES = 6  # lines shown when truncating
_REVEAL_THRESHOLD = 8  # truncation kicks in from this count on

# Window (s) to tell Alt+Enter (\x1b \r) apart from an incomplete key sequence
_ALT_ENTER_TIMEOUT = 0.08


def run_interactive(
    session_id: str | None,
    *,
    settings: Settings | None = None,
    core: AgentCore | None = None,
    input_stream: IO[str] | None = None,
    out: TextIO | None = None,
    is_tty: bool | None = None,
    reader: Callable[[], str | None] | None = None,
) -> int:
    """REPL loop of the interactive mode. Injections: core, input, output and line reader."""
    settings = settings or Settings.from_env()
    out = out or sys.stdout
    is_tty = sys.stdin.isatty() if is_tty is None else is_tty
    reader = reader or _default_reader(input_stream or sys.stdin, is_tty, out)
    core = core or _build_tui_core(settings, out, reader, is_tty)

    try:
        sid = validate_session_id(session_id) if session_id else core.new_session_id()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _print_welcome_rows = _print_welcome(out, sid, core.store, is_tty, settings)
    if is_tty:
        _pin_header(out, _print_welcome_rows)

    # One event loop for the whole session: MAF/httpx clients stay bound to it. A
    # per-turn asyncio.run() closes the loop, poisons pooled connections and breaks the
    # second turn with "Event loop is closed".
    loop = asyncio.new_event_loop()
    resized = {"flag": False}

    def _on_resize(_signum: int, _frame: object) -> None:
        resized["flag"] = True

    # SIGWINCH re-pins the header (scroll region bottom follows the window height);
    # outside the main thread / unsupported platforms, resizes are ignored.
    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGWINCH, _on_resize)

    try:
        while True:
            if resized["flag"]:
                resized["flag"] = False
                if is_tty:
                    _pin_header(out, _print_welcome_rows)
            out.write(_COLOR_PROMPT if is_tty else _PLAIN_PROMPT)
            out.flush()
            line = reader()
            if line is None:
                break
            text = line.strip()
            if text.lower() in _EXIT_WORDS:
                break
            if not text:
                continue
            # In raw mode the Enter echo already moved the line down; over a pipe the
            # gap must be written, twice (line end + blank line) before the answer.
            out.write("\n" if is_tty else "\n\n")
            out.flush()
            printer = _StreamPrinter(out, is_tty=is_tty)
            # Live tool blocks ($ Shell / detail / ✓) feed straight from the middleware.
            core.on_tool_event = printer.tool
            # Consume the whole turn inside a single run_until_complete: stepping the
            # generator with per-chunk run_until_complete calls copies the context each
            # time and breaks MAF's telemetry tokens ("created in a different Context").
            async def _consume(text=text, printer=printer) -> None:
                async for chunk in core.respond(sid, text):
                    printer.write(chunk.kind, chunk.text)
                    out.flush()

            loop.run_until_complete(_consume())
            out.write("\n\n")
            out.flush()
    finally:
        # Release the MCP tools while the loop is still alive; then close the loop itself.
        loop.run_until_complete(core.aclose())
        loop.close()

    # Give the terminal back a full scroll region: whatever runs after selenita must
    # not be trapped inside (or break) the pinned header band.
    if is_tty:
        out.write("\x1b[r")
    out.write("See you soon.\n")
    return 0


class _StreamPrinter:
    """Renders streamed core chunks keeping markers and line indentation stable.

    The first chunk of each kind prints its marker (``¤`` answers, ``·`` reasoning in
    TTYs); any further line of the same kind is indented two spaces, so streamed pieces
    behave like the old whole-block rendering.
    """

    def __init__(self, out: TextIO, *, is_tty: bool) -> None:
        self._out = out
        self._tty = is_tty
        self._kind: str | None = None
        self._line_start = True

    def write(self, kind: str, text: str) -> None:
        if not text:
            return
        # Separate the reasoning phase from the answer phase: close the line if the
        # thought ended mid-line and always leave a blank line before the answer.
        if self._kind == "reasoning" and kind == "answer":
            if not self._line_start:
                self._out.write("\n")
                self._line_start = True
            self._out.write("\n")
        if self._kind == "tool_result" and kind in ("answer", "reasoning"):
            # A tool block closes with its ✓/✗ row; leave one blank line before the
            # next agent text, be it reasoning or the answer.
            if not self._line_start:
                self._out.write("\n")
                self._line_start = True
            self._out.write("\n")
        kind_started = self._kind != kind
        self._kind = kind
        segments = text.split("\n")
        last_index = len(segments) - 1
        for index, segment in enumerate(segments):
            if index > 0:
                self._out.write("\n")
                self._line_start = True
            # Skip the prefix on the phantom line a trailing "\n" opens: the indent is
            # only worth drawing once actual content arrives (or a real new line follows).
            if segment or index < last_index:
                if self._line_start:
                    self._out.write(self._prefix(kind, kind_started))
                    self._line_start = False
                if segment:
                    self._out.write(self._styled(kind, segment))

    def _prefix(self, kind: str, kind_started: bool) -> str:
        """Marker of the kind at its first line, plain indent afterwards.

        Colors only when the stream goes to a TTY; over a pipe the plain ``¤ `` marker
        keeps the output copy-paste friendly (matching the whole-block renderer).
        """
        if kind == "reasoning":
            marker = _COLOR_REASON if self._tty else _REASON_MARKER
        else:
            marker = _COLOR_AGENT if self._tty else _AGENT_MARKER
        return f"{marker} " if kind_started else "  "

    def _styled(self, kind: str, text: str) -> str:
        """Reasoning wrapped in the muted style (TTY only); answers unstyled."""
        if kind == "reasoning" and self._tty:
            return f"{_REASON_STYLE}{text}{_RESET}"
        return text

    def tool(self, event: ToolEvent) -> None:
        """One compact block per tool call (``$`` name + muted detail) and its result.

        Called live from the middleware (LlmCore tool loop), so the block appears
        between the streamed texts that surround the execution, Copilot-CLI-style:
        ``$ Shell`` / command line / ``✓ 0.1s``.
        """
        if event.kind == "call":
            if not self._line_start:
                self._out.write("\n")  # close a streamed line still open
            self._out.write("\n")  # blank gap between streamed text and the tool block
            mark = f"{_COLOR_TOOL} " if self._tty else f"{_TOOL_MARKER} "
            self._out.write(f"{mark}{event.title}\n")
            if event.detail:
                indent = f"  {event.detail}\n"
                if self._tty:
                    indent = f"  {_MUTED}{event.detail}{_RESET}\n"
                self._out.write(indent)
            self._kind = "tool_call"
            self._line_start = True
            self._out.flush()
            return
        # result event: status row right below the call block
        if event.status == "done" and event.duration is None:
            # No execution to time (e.g. ask_user: the wait is the user's own and the
            # selector already shows the choice): no status row at all. The block is
            # still closed so the next streamed text gets its blank gap.
            self._kind = "tool_result"
            self._line_start = True
            self._out.flush()
            return
        glyph = "✓" if event.status == "done" else "✗"
        label = event.status or ""
        if event.status == "done" and event.duration is not None:
            label = f"{event.duration:.1f}s"
        if self._tty:
            self._out.write(f"  {_MUTED}{glyph} {label}{_RESET}\n")
        else:
            self._out.write(f"  {glyph} {label}\n")
        self._kind = "tool_result"
        self._line_start = True
        self._out.flush()


async def _turn(core: AgentCore, session_id: str, user_text: str) -> str:
    """One turn against the core, recomposed as full text (kept for tests/one-shot use)."""
    chunks: list[str] = []
    async for chunk in core.respond(session_id, user_text):
        chunks.append(chunk.text)
    return "".join(chunks)


def _build_tui_core(
    settings: Settings,
    out: TextIO,
    reader: Callable[[], str | None],
    is_tty: bool,
) -> AgentCore:
    """Core offering every built-in tool, wired to interactive prompts (tools epic).

    ``ask_user`` and the permission asker run inside the MAF event loop; the blocking
    reader is pushed to a worker thread so the loop stays responsive while waiting.
    """
    warn = _COLOR_WARN if is_tty else "⚠"
    ask_mark = _COLOR_ASK if is_tty else "?"
    core_holder: dict[str, AgentCore] = {}

    async def asker(tool_name: str, arguments: dict[str, object]) -> str:
        # With the live $ blocks wired, the tool name and its arguments are already on
        # screen (the middleware prints them just before asking); stay minimal then.
        live_blocks = core_holder["core"].on_tool_event is not None
        if live_blocks:
            # The $ block already showed tool name and arguments: minimal inline prompt.
            out.write(f"{warn} needs permission ")
            prompt = "[y]es · [a]lways · [n]o > "
        else:  # event-less cores (tests, echo): the classic full prompt
            detail = summarize_tool_call(tool_name, arguments)
            detail_line = f"  {_MUTED}{detail}{_RESET}\n" if is_tty else f"  {detail}\n"
            out.write(f"\n{warn} {tool_name} needs your permission\n")
            out.write(detail_line)
            prompt = "  [y]es · [a]lways · [n]o > "
        out.write(prompt)
        out.flush()
        answer = (await asyncio.to_thread(reader) or "").strip().lower()
        # The raw-TTY reader echoes Enter itself (line closed); over a pipe the line is
        # left open and must be closed here (a "\n" would add a blank line on raw TTYs).
        if not is_tty:
            out.write("\n")
        out.flush()
        if answer in ("a", "always"):
            return DECISION_ALWAYS
        if answer in ("y", "yes", ""):
            return DECISION_APPROVE
        return DECISION_REJECT

    async def ask_user(question: str, options: Sequence[str] | None = None) -> str:
        if is_tty and options:
            picked = await asyncio.to_thread(_ask_user_interactive, out, question, list(options))
            return picked or ""
        out.write(f"\n{ask_mark} {question}\n")
        if options:
            for index, option in enumerate(options, start=1):
                out.write(f"   {index}. {option}\n")
        out.write("  your answer > ")
        out.flush()
        answer = (await asyncio.to_thread(reader) or "").strip()
        out.write("\n")
        out.flush()
        return answer

    toolkit = Toolkit(
        include_mutating=True,
        include_ask_user=True,
        asker=asker,
        ask_user_callback=ask_user,
    )
    core_holder["core"] = create_core(settings, toolkit=toolkit)
    return core_holder["core"]


_SELECT_MAX = 70  # selector entries longer than this are truncated with "…"
_CUSTOM_ANSWER_LABEL = "Write your own answer…"


def _selector_pick(
    out: TextIO,
    options: Sequence[str],
    keys: Iterator[str],
    *,
    styled: bool = False,
) -> str | None:
    """Core of the navigable single-choice picker (kept termios-free to unit-test).

    Renders one line per entry (the last one is "Write your own answer…") and reacts to
    ``_raw_keys`` events: ``up``/``down`` move the ``❯`` marker with wrap,
    ``char:<digit>`` jumps straight to that entry (beyond the options → the custom
    entry), ``enter`` picks the highlighted one. Picking the custom entry starts an
    **inline editor on that very line** (Copilot-CLI-style): ``char:`` appends,
    ``backspace`` erases, ``enter`` confirms (empty → back to navigation) and
    ``ctrl-c``/``ctrl-d`` cancel the whole question. Returns the picked string or
    ``None``. Entries (and typed text) longer than ``_SELECT_MAX`` are truncated.
    """
    entries = [*options, _CUSTOM_ANSWER_LABEL]
    total = len(entries)
    current = 0
    editing = False
    buffer = ""

    def entry_line(index: int) -> str:
        marker = "❯ " if index == current else "  "
        if editing and index == total - 1:
            if buffer:
                shown = buffer if len(buffer) <= _SELECT_MAX else buffer[:_SELECT_MAX] + "…"
            else:  # empty editor shows the placeholder, muted in TTYs
                shown = f"{_MUTED}{entries[-1]}{_RESET}" if styled else entries[-1]
            return f"{marker}{shown}"
        shown = entries[index]
        if len(shown) > _SELECT_MAX:
            shown = shown[:_SELECT_MAX] + "…"
        return f"{marker}{shown}"

    def draw(redraw: bool) -> None:
        if redraw:
            out.write(f"\r\x1b[{total}A")
        for index in range(total):
            out.write(f"\x1b[2K{entry_line(index)}\r\n")
        out.flush()

    draw(redraw=False)
    for key in keys:
        if editing:
            if key.startswith("char:"):
                buffer += key[5:]
                draw(redraw=True)
            elif key == "backspace":
                buffer = buffer[:-1]
                draw(redraw=True)
            elif key == "enter":
                if buffer.strip():
                    return buffer.strip()
                editing = False  # empty answer → back to navigation
                draw(redraw=True)
            elif key in ("ctrl-c", "ctrl-d"):
                return None
            continue  # arrows while editing are ignored (stay on the editor)
        if key == "up":
            current = (current - 1) % total
            draw(redraw=True)
        elif key == "down":
            current = (current + 1) % total
            draw(redraw=True)
        elif key.startswith("char:"):
            digit = key[5:]
            if digit in "123456789":
                index = int(digit) - 1
                if index < len(options):
                    return options[index]
                current = total - 1  # any digit beyond the options = custom entry
                draw(redraw=True)
        elif key == "enter":
            if current < len(options):
                return options[current]
            editing = True  # custom entry: start the inline editor on its own line
            draw(redraw=True)
        elif key in ("ctrl-c", "ctrl-d"):
            return None
    return None


def _ask_user_interactive(
    out: TextIO,
    question: str,
    options: Sequence[str],
) -> str | None:
    """Blocking navigable question prompt for a TTY (Copilot-CLI-style selector).

    Sets raw mode, prints the question and hands the rest to ``_selector_pick`` driven
    by ``_raw_keys``; the terminal is restored before returning. All line breaks use
    ``\\r\\n`` (raw mode turns OPOST off, a bare ``\\n`` keeps the cursor column). A
    gap ``\\r\\n`` closes the block so the agent's resumed answer starts one blank
    line below. Never prompts for permission — ``ask_user`` is the user's own input.
    """
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        out.write(f"{_COLOR_ASK} {question}\r\n")
        out.flush()
        picked = _selector_pick(out, options, _raw_keys(fd), styled=True)
        return picked
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def _print_welcome(
    out: TextIO,
    session_id: str,
    store: SessionStore,
    is_tty: bool,
    settings: Settings | None = None,
) -> int:
    """Welcome: moon logo, session/context metadata, last message and help.

    Returns the number of rows the header occupies, so the interactive loop can pin
    them (scroll region) and re-pin after a resize. Context shows the working
    directory, the git branch (when the cwd is a repository) and the model.
    """
    settings = settings or Settings.from_env()
    history = store.load(session_id)
    reset = _RESET if is_tty else ""
    if is_tty:
        for art in _LOGO[:-1]:
            out.write(f"{_CYAN}{art}{_RESET}\n")
        out.write(f"{_CYAN}{_LOGO[-1]}{_RESET} ")
        rows = len(_LOGO)
    else:
        out.write("● ")
        rows = 1
    bold = _BOLD if is_tty else ""
    out.write(f"{bold}Selenita v{__version__}{reset}")
    out.write(f" · session: {session_id} · history: {len(history)} messages\n")
    out.write("\n")  # breathing room between the title and the context line
    rows += 1
    out.write(f"{_DIM if is_tty else ''}{_context_line(settings)}{reset}\n")
    rows += 2
    if history:
        last = history[-1]
        lines, total = _truncated_tail(last.text)
        out.write(f"{reset}last message ({last.role}):\n")
        rows += 1
        if total is not None:
            out.write(f"  {reset}[… truncated — last {_KEEP_LINES} of {total} lines …]{reset}")
            out.write("\n")
            rows += 1
        for line in lines:
            out.write(f"  {line}\n")
        rows += len(lines)
    # Full-width divider on terminals (it follows the window width); fixed on pipes,
    # where the output should not depend on any terminal geometry.
    width = shutil.get_terminal_size(fallback=(80, 24)).columns if is_tty else 30
    out.write(f"{_DIM if is_tty else ''}{'─' * width}{reset}\n")
    out.write(f'{_DIM if is_tty else ""}Alt+Enter = newline · "exit" or Ctrl+D = quit{reset}\n\n')
    return rows + 3  # divider, help, trailing blank


def _context_line(settings: Settings) -> str:
    """``Working directory: … · Branch: … · Model: …`` (branch only inside a git repo)."""
    cwd = Path.cwd()
    segments = [f"Working directory: {cwd}"]
    branch = _git_branch(cwd)
    if branch is not None:
        segments.append(f"Branch: {branch}")
    if settings.agent_mode == "echo":
        segments.append("Model: echo")
    elif settings.llm_provider == "ollama":
        segments.append(f"Model: {settings.ollama_model}")
    else:
        segments.append(f"Model: {settings.openai_model}")
    return " · ".join(segments)


def _git_branch(cwd: Path) -> str | None:
    """Current abbrev-ref of the repo at ``cwd``, or None outside one."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False, cwd=cwd,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return branch or None


def _pin_header(out: TextIO, used_rows: int) -> None:
    """Pin the (already printed) header via an ANSI scroll region.

    ``CSI top;bottom r`` makes the terminal scroll only the rows below the header, so
    the welcome block stays fixed while content streams. Two quirks are handled:
    DECSTBM homes the cursor (xterm behavior), so it is restored right below the
    header; and when the window is too short for a header-only band the region is
    reset and the header behaves like a normal scrolling block again.
    """
    lines = shutil.get_terminal_size(fallback=(80, 24)).lines
    top = used_rows + 1
    if lines - used_rows < 4:  # too short to leave a usable band: unpin
        out.write("\x1b[r")
        return
    out.write(f"\x1b[{top};{lines}r\x1b[{top};1H")


def _render_agent_block(text: str, is_tty: bool) -> str:
    """Agent answer: ¤ marker on the first line and two-space indent afterwards."""
    marker = f"{_COLOR_AGENT} " if is_tty else f"{_AGENT_MARKER} "
    lines = text.splitlines() or [""]
    return "\n".join(
        f"{marker}{line}" if index == 0 else f"  {line}"
        for index, line in enumerate(lines)
    )


def _truncated_tail(text: str) -> tuple[list[str], int | None]:
    """Tail lines of a text; above the threshold returns (tail, total); never truncated if None."""
    lines = text.splitlines() or [""]
    if len(lines) <= _REVEAL_THRESHOLD:
        return lines, None
    return lines[-_KEEP_LINES:], len(lines)


# ---------------------------------------------------------------------------
# Line readers: raw TTY (real keyboard) and line-by-line mode (pipe/tests)
# ---------------------------------------------------------------------------


def _default_reader(
    input_stream: IO[str], is_tty: bool, out: TextIO
) -> Callable[[], str | None]:
    """Pick the adequate reader: raw over a terminal, line-by-line over a pipe."""
    if is_tty:
        return lambda: _read_line_tty(out)
    return _make_piped_reader(input_stream)


def _make_piped_reader(input_stream: IO[str]) -> Callable[[], str | None]:
    """Line reader for non-interactive stdin; EOF → ``None``."""

    def read_line() -> str | None:
        line = input_stream.readline()
        if line == "":
            return None
        return line[:-1] if line.endswith("\n") else line

    return read_line


def _read_line_tty(out: TextIO) -> str | None:
    """Read one line with the terminal in raw mode.

    Enter submits; Alt+Enter inserts ``\\n``; Backspace erases; Ctrl+C/Ctrl+D end (``None``).
    Keyed text must be echoed by this reader itself (setraw turns echo off).
    """
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    buffer = ""
    try:
        tty.setraw(fd)
        for key in _raw_keys(fd):
            # In raw mode OPOST is off: without "\r" the cursor moves down but
            # keeps its column, so anything printed after it gets misaligned.
            if key == "enter":
                out.write("\r\n")
                out.flush()
                return buffer
            if key.startswith("char:"):
                buffer += key[5:]
                out.write(key[5:])
            elif key == "newline":
                buffer += "\n"
                # Double visual indent: every newline draws "\r\n  " so the
                # multi-line prompt reads like a block (spaces are not buffered).
                out.write("\r\n  ")
            elif key == "backspace":
                buffer = buffer[:-1]
                out.write("\b \b")
            elif key in ("ctrl-c", "ctrl-d"):
                out.write("^C\r\n" if key == "ctrl-c" else "\r\n")
                out.flush()
                return None
            out.flush()
        return None  # terminal EOF (should not happen in raw mode)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def _raw_keys(fd: int) -> Iterator[str]:
    """Yield the terminal raw keys as logical events.

    ``char:c`` a printable character, ``newline`` Enter with a modifier (Alt+Enter and the
    standard Shift+Enter/Ctrl+Enter sequences with key code 13), ``enter``, ``backspace``,
    ``up``, ``down``, ``ctrl-c``, ``ctrl-d``; other escape sequences (F-keys…) are consumed whole.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    while True:
        byte = os.read(fd, 1)
        if byte in (b"\r", b"\n"):
            yield "enter"
            continue
        if byte == b"\x1b":
            following = _peek_byte(fd)
            if following in (b"\r", b"\n"):
                yield "newline"
                continue
            seq = _read_escape_body(fd, following)
            if seq in (b"[A", b"[B"):  # plain vertical arrows
                yield "up" if seq == b"[A" else "down"
                continue
            if _is_modified_enter(seq):
                yield "newline"
            continue
        if byte == b"\x7f":
            yield "backspace"
            continue
        if byte == b"\x03":
            yield "ctrl-c"
            continue
        if byte == b"\x04":
            yield "ctrl-d"
            continue
        if not byte:
            return
        char = decoder.decode(byte)
        if char:
            yield f"char:{char}"


def _read_escape_body(fd: int, first_byte: bytes) -> bytes:
    """Consume the POST-ESC body of a sequence and return it whole.

    - CSI (``ESC [ params final``, e.g. ``ESC[13;2u``): ``[`` is NOT the final byte even
      though it falls in 0x40-0x7E; params are 0x20-0x3F and it closes with 0x40-0x7E.
    - SS3 (``ESC O x``): 'O' is followed by exactly one byte (F1-F4).
    - Any other case: a single literal byte that is not part of a modified Enter.
    """
    if first_byte == b"[":
        seq = bytearray(b"[")
        while True:
            nxt = os.read(fd, 1)
            if not nxt:
                break
            seq += nxt
            if 0x40 <= nxt[0] <= 0x7E:  # CSI final byte
                break
        return bytes(seq)
    if first_byte == b"O":  # SS3 of F1-F4: exactly one more byte
        return first_byte + os.read(fd, 1)
    return first_byte


def _is_modified_enter(seq: bytes) -> bool:
    """Is ``seq`` an Enter with a modifier (Alt/Shift/Ctrl) in CSI-u or xterm notation?

    Accepted Shift+Enter shapes: ``ESC[13;2u`` (kitty/CSI-u) and ``ESC[27;2;13~``
    (xterm modifyOtherKeys). Unmodified Enter never arrives in these shapes.
    """
    if len(seq) < 4 or not seq.startswith(b"[") or seq[-1:] not in (b"u", b"~"):
        return False
    try:
        parts = [int(part) for part in seq[1:-1].decode("ascii").split(";")]
    except (UnicodeDecodeError, ValueError):
        return False
    if seq.endswith(b"u"):  # kitty: keycode;modifiers
        return parts[:1] == [13] and len(parts) >= 2
    return len(parts) >= 3 and parts[0] == 27 and parts[2] == 13  # xterm: 27;mod;key


def _peek_byte(fd: int) -> bytes:
    """Read the next byte if it arrives in time (Alt+Enter); else ``b""``."""
    if not select.select([fd], [], [], _ALT_ENTER_TIMEOUT)[0]:
        return b""
    return os.read(fd, 1)
