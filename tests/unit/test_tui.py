"""Unit tests of the interactive TUI mode (injected readers, no real keyboard)."""

from __future__ import annotations

import io
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from selenita.cli import tui
from selenita.config import Settings
from selenita.core.echo import EchoCore
from selenita.core.permissions import ToolEvent
from selenita.sessions.store import InMemorySessionStore, SessionMessage

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Logical text (ANSI styling removed).

    The stream printer colors the answer marker on every channel, so markers/gaps are
    asserted on the unstyled text to keep the needles style-agnostic.
    """
    return _ANSI_RE.sub("", text)


def _reader(lines: list[str | None]):
    """Fake reader that returns the lines in order and then EOF (None)."""
    it = iter(lines)
    return lambda: next(it, None)


def test_truncated_tail_short_only() -> None:
    lines, total = tui._truncated_tail("a\nb\nc")
    assert lines == ["a", "b", "c"]
    assert total is None  # not truncated


def test_truncated_tail_long_shows_kept_tail_and_total() -> None:
    text = "\n".join(f"l{i}" for i in range(1, 21))  # 20 lines
    lines, total = tui._truncated_tail(text)
    assert total == 20
    assert lines == [f"l{i}" for i in range(15, 21)]  # last 6


def test_render_agent_block_marks_and_indents() -> None:
    assert tui._render_agent_block("hello", is_tty=False) == "¤ hello"
    assert tui._render_agent_block("hello\nworld", is_tty=False) == "¤ hello\n  world"
    assert tui._render_agent_block("", is_tty=False) == "¤ "  # empty answer is possible
    assert tui._render_agent_block("a\nb\nc", is_tty=False) == "¤ a\n  b\n  c"


def test_stream_printer_blank_line_between_reasoning_and_answer() -> None:
    out = io.StringIO()
    printer = tui._StreamPrinter(out, is_tty=False)
    printer.write("reasoning", "thinking stuff")
    printer.write("answer", "the answer")
    assert _plain(out.getvalue()) == "· thinking stuff\n\n¤ the answer"


def test_stream_printer_blank_line_after_reasoning_that_ended_in_newline() -> None:
    out = io.StringIO()
    printer = tui._StreamPrinter(out, is_tty=False)
    printer.write("reasoning", "thinking\n")
    printer.write("answer", "the answer")
    assert _plain(out.getvalue()) == "· thinking\n\n¤ the answer"


def test_stream_printer_tool_blocks_call_and_result() -> None:
    """A tool call prints $ name + details, and the result row lands right below."""
    out = io.StringIO()
    printer = tui._StreamPrinter(out, is_tty=False)
    printer.tool(ToolEvent("call", "selenita_write_file", "Write file", "path='f.txt'"))
    printer.tool(ToolEvent("result", "selenita_write_file", "Write file", status="done",
                           duration=0.02))
    printer.write("answer", "Listo")
    assert _plain(out.getvalue()) == "\n$ Write file\n  path='f.txt'\n  ✓ 0.0s\n\n¤ Listo"


def test_stream_printer_tool_result_denied_and_failed() -> None:
    out = io.StringIO()
    printer = tui._StreamPrinter(out, is_tty=False)
    printer.tool(ToolEvent("call", "selenita_run_command", "Shell", "rm prueba.txt"))
    printer.tool(ToolEvent("result", "selenita_run_command", "Shell", status="denied",
                           duration=0.001))
    printer.tool(ToolEvent("call", "selenita_web_fetch", "WebFetch", detail=None))
    printer.tool(ToolEvent("result", "selenita_web_fetch", "WebFetch",
                           status="failed: boom", duration=0.5))
    text = _plain(out.getvalue())
    assert "  ✗ denied\n" in text
    assert "$ WebFetch\n" in text  # no detail line when there is nothing to show
    assert "  ✗ failed: boom\n" in text


def test_stream_printer_answer_starts_fresh_marker_after_tool_block() -> None:
    """After a tool block the answer marker is drawn again (new block, not an indent)."""
    out = io.StringIO()
    printer = tui._StreamPrinter(out, is_tty=False)
    printer.write("answer", "Creo un ejemplo:")
    printer.tool(ToolEvent("call", "selenita_write_file", "Write file", "path='f.txt'"))
    printer.tool(ToolEvent("result", "selenita_write_file", "Write file", status="done",
                           duration=0.0))
    printer.write("answer", "Listo")
    text = _plain(out.getvalue())
    assert "¤ Creo un ejemplo:" in text
    assert "Write file" in text
    assert text.count("¤") == 2  # marker re-drawn after the tool block


def test_stream_printer_reasoning_gets_gap_after_tool_result() -> None:
    """The blank gap after a tool block applies to the reasoning phase too."""
    out = io.StringIO()
    printer = tui._StreamPrinter(out, is_tty=False)
    printer.tool(ToolEvent("call", "selenita_write_file", "Write file", "path='f.txt'"))
    printer.tool(ToolEvent("result", "selenita_write_file", "Write file", status="done",
                           duration=0.02))
    printer.write("reasoning", "thinking stuff")
    printer.write("answer", "Listo")
    text = _plain(out.getvalue())
    assert "  ✓ 0.0s\n\n· thinking stuff\n\n¤ Listo" in text


def test_stream_printer_skips_status_row_when_no_duration() -> None:
    """Timed-less done results (ask_user: the wait is the user's own) print no row."""
    out = io.StringIO()
    printer = tui._StreamPrinter(out, is_tty=False)
    printer.tool(ToolEvent("call", "selenita_ask_user", "Ask user", detail=None))
    printer.tool(ToolEvent("result", "selenita_ask_user", "Ask user", status="done"))
    printer.write("reasoning", "The user picked one.")
    text = _plain(out.getvalue())
    assert text == "\n$ Ask user\n\n· The user picked one."


def test_welcome_divider_matches_terminal_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """The header divider spans the terminal width on TTYs (fixed width on pipes)."""
    monkeypatch.setattr(
        shutil, "get_terminal_size", lambda fallback=None: SimpleNamespace(columns=120, lines=40)
    )
    out = io.StringIO()
    tui._print_welcome(out, "sess_divider", InMemorySessionStore(), is_tty=True)
    assert ("─" * 120 + "\n") in _plain(out.getvalue())

    out = io.StringIO()
    tui._print_welcome(out, "sess_divider", InMemorySessionStore(), is_tty=False)
    assert ("─" * 30 + "\n") in _plain(out.getvalue())


def test_welcome_shows_context_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Header context: working directory, git branch (only in repos) and model."""
    settings = Settings(agent_mode="llm", openai_model="glm-model", openai_base_url=None)
    out = io.StringIO()
    tui._print_welcome(out, "sess_ctx", InMemorySessionStore(), is_tty=False, settings=settings)
    text = _plain(out.getvalue())
    assert f"Working directory: {Path.cwd()}" in text
    assert "Model: glm-model" in text
    # /workspace is a git repository, so the branch is part of the header context.
    assert "Branch: " in text


def test_welcome_pins_header_with_scroll_region(monkeypatch: pytest.MonkeyPatch) -> None:
    """On TTYs the header rows are pinned below a scroll region (and reset if short)."""
    monkeypatch.setattr(
        shutil, "get_terminal_size", lambda fallback=None: SimpleNamespace(columns=100, lines=40)
    )
    monkeypatch.setattr(tui, "_git_branch", lambda _cwd: "feature/x")
    out = io.StringIO()
    rows = tui._print_welcome(out, "sess_pin", InMemorySessionStore(), is_tty=True)
    out = io.StringIO()
    tui._pin_header(out, rows)
    # DECSTBM homes the cursor: the pin restores it right below the header.
    assert out.getvalue() == f"\x1b[{rows + 1};40r\x1b[{rows + 1};1H"

    monkeypatch.setattr(
        shutil, "get_terminal_size", lambda fallback=None: SimpleNamespace(columns=100, lines=7)
    )
    out = io.StringIO()
    tui._pin_header(out, rows)
    assert out.getvalue() == "\x1b[r"  # too short for a header band: region reset


def test_git_branch_is_none_outside_repositories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert tui._git_branch(tmp_path) is None


def test_interactive_loop_prompts_echos_and_exits_by_word() -> None:
    out = io.StringIO()
    core = EchoCore(store=InMemorySessionStore())
    code = tui.run_interactive(
        None, core=core, out=out, is_tty=False, reader=_reader(["hello", "", "exit"])
    )
    assert code == 0
    text = _plain(out.getvalue())
    assert text.startswith("● ")
    assert "· session:" in text and "history: 0 messages" in text
    assert "❯ " in text
    assert "¤ hello\n\n" in text  # answer with marker and gap before the next prompt
    assert 'Alt+Enter = newline · "exit" or Ctrl+D = quit' in text
    assert "See you soon." in text


def test_interactive_exits_on_quit_word() -> None:
    out = io.StringIO()
    code = tui.run_interactive(None, out=out, is_tty=False, reader=_reader(["quit"]))
    assert code == 0
    assert "See you soon." in out.getvalue()


def test_interactive_salir_is_not_an_exit_word() -> None:
    out = io.StringIO()
    core = EchoCore(store=InMemorySessionStore())
    code = tui.run_interactive(
        None, core=core, out=out, is_tty=False, reader=_reader(["salir", "exit"])
    )
    assert code == 0
    text = _plain(out.getvalue())
    # "salir" is turned like any other message; the session only ends with exit/quit.
    assert "¤ salir\n\n" in text
    assert "See you soon." in text


def test_interactive_registers_turns_in_core_store() -> None:
    out = io.StringIO()
    store = InMemorySessionStore()
    core = EchoCore(store=store)
    tui.run_interactive(
        "sess_tui1", core=core, out=out, is_tty=False, reader=_reader(["hola", "quit"])
    )
    assert [(m.role, m.text) for m in store.load("sess_tui1")] == [
        ("user", "hola"),
        ("agent", "hola"),
    ]


def test_interactive_exits_on_eof() -> None:
    out = io.StringIO()
    code = tui.run_interactive(None, out=out, is_tty=False, reader=_reader([]))
    assert code == 0
    assert "See you soon." in out.getvalue()


def test_interactive_invalid_session_id_exits_with_error() -> None:
    out = io.StringIO()
    code = tui.run_interactive("not-valid", out=out, is_tty=False, reader=_reader([]))
    assert code == 2


def test_interactive_shows_last_message_of_resumed_session() -> None:
    out = io.StringIO()
    store = InMemorySessionStore()
    store.append("sess_prev", SessionMessage(role="user", text="previous question"))
    store.append("sess_prev", SessionMessage(role="agent", text="previous answer"))
    core = EchoCore(store=store)
    tui.run_interactive(
        "sess_prev", core=core, out=out, is_tty=False, reader=_reader(["exit"])
    )
    welcome = out.getvalue()
    assert "history: 2 messages" in welcome
    assert "last message (agent):" in welcome
    assert "previous answer" in welcome


def test_interactive_truncates_long_last_message() -> None:
    out = io.StringIO()
    store = InMemorySessionStore()
    long_text = "\n".join(f"line {i}" for i in range(1, 26))
    store.append("sess_long", SessionMessage(role="agent", text=long_text))
    core = EchoCore(store=store)
    tui.run_interactive("sess_long", core=core, out=out, is_tty=False, reader=_reader(["exit"]))
    welcome = out.getvalue()
    assert "truncated" in welcome
    assert "[… truncated — last 6 of 25 lines …]" in welcome
    assert "line 1" not in welcome  # the head is gone
    assert "line 25" in welcome  # the tail is there


def test_piped_reader_strips_eol_and_reports_eof() -> None:
    stream = io.StringIO("one\ntwo with  spaces   \n")
    reader = tui._make_piped_reader(stream)
    assert reader() == "one"
    assert reader() == "two with  spaces   "
    assert reader() is None  # EOF


# ---------------------------------------------------------------------------
# Raw key recognition: Enter with a modifier (Alt/Shift/Ctrl)
# ---------------------------------------------------------------------------


def test_is_modified_enter_recognizes_the_standard_sequences() -> None:
    assert tui._is_modified_enter(b"[13;2u")  # Shift+Enter (kitty CSI-u)
    assert tui._is_modified_enter(b"[13;3u")  # Alt+Enter (kitty)
    assert tui._is_modified_enter(b"[27;2;13~")  # Shift+Enter (xterm modifyOtherKeys)
    assert not tui._is_modified_enter(b"")  # empty
    assert not tui._is_modified_enter(b"[D")  # left arrow
    assert not tui._is_modified_enter(b"[13~")  # F3 (that is NOT modified Enter!)
    assert not tui._is_modified_enter(b"[13;2x")  # unknown final byte
    assert not tui._is_modified_enter(b"[OP")  # SS3 F1-F4
    assert not tui._is_modified_enter(b"[1;2A")  # Shift+arrow


def test_raw_keys_maps_key_sequences_with_a_real_pty() -> None:
    """Over a real pty: chars, Alt/Shift+Enter, arrows and a discarded unknown sequence."""
    import os

    master, slave = os.openpty()
    events: list[str] = []
    keys = tui._raw_keys(master)
    try:
        # One single write per sequence with a drain after every expected event: the
        # silent sequences (arrows, SS3) are consumed whole without emitting an event.
        os.write(slave, b"a")
        events.append(next(keys))  # char:a
        os.write(slave, b"\x1b[13;2u")  # Shift+Enter (kitty CSI-u)
        events.append(next(keys))  # newline
        os.write(slave, b"\x1b\r")  # Alt+Enter
        events.append(next(keys))  # newline
        os.write(slave, b"\x1b[1;2A")  # Shift+arrow: no event
        os.write(slave, b"\x1b[A")  # plain Up arrow → "up"
        events.append(next(keys))  # up
        os.write(slave, b"\x1bOD")  # F2/SS3: no event, consumed whole
        os.write(slave, b"\r")
        events.append(next(keys))  # enter (after draining the two above)
        os.write(slave, b"\x04")
        events.append(next(keys))  # ctrl-d
    finally:
        keys.close()
        os.close(slave)
        os.close(master)
    assert events == ["char:a", "newline", "newline", "up", "enter", "ctrl-d"]


def test_selector_arrows_move_and_enter_picks() -> None:
    """Arrow down highlights the second entry; Enter picks it."""
    out = io.StringIO()
    keys = iter(["down", "enter"])
    picked = tui._selector_pick(out, ["Python", "JavaScript"], keys)
    assert picked == "JavaScript"
    rendered = _plain(out.getvalue())
    assert "Python" in rendered
    assert "JavaScript" in rendered


def test_selector_up_wraps_to_custom_entry_placeholder() -> None:
    """Up from the top wraps to the "Write your own answer…" entry (editor pending)."""
    out = io.StringIO()
    keys = iter(["up", "enter"])  # enter on the custom entry opens the inline editor
    picked = tui._selector_pick(out, ["a", "b"], keys)
    assert picked is None  # keys exhausted while editing: nothing confirmed
    rendered = _plain(out.getvalue())
    assert "Write your own answer…" in rendered
    assert "❯ Write your own answer…" in rendered  # editor starts on the custom line


def test_selector_digit_jumps_to_that_option() -> None:
    keys = iter(["char:2", "enter"])
    picked = tui._selector_pick(io.StringIO(), ["a", "b", "c"], keys)
    assert picked == "b"


def test_selector_digit_beyond_options_lands_on_custom_entry() -> None:
    keys = iter(["char:9"])
    picked = tui._selector_pick(io.StringIO(), ["a", "b"], keys)
    assert picked is None  # custom entry highlighted, no keys for the editor


def test_selector_custom_types_text_inline() -> None:
    """Picking the custom entry edits text in place; Enter returns it trimmed."""
    out = io.StringIO()
    keys = iter(
        [
            "down", "down", "enter",  # into the custom entry → inline editor
            "char:m", "char:y",  # typing "my"
            "backspace",  # erase "y" → "m"
            "char: ", "char:i",  # append " i"
            "enter",  # confirm
        ]
    )
    picked = tui._selector_pick(out, ["a", "b"], keys)
    assert picked == "m i"


def test_selector_custom_empty_enter_returns_to_navigation() -> None:
    """Empty editor + Enter does NOT answer; the placeholder line is drawn again."""
    out = io.StringIO()
    keys = iter(["char:9", "enter", "enter"])  # editor opened, empty enter → nav again
    picked = tui._selector_pick(out, ["a", "b"], keys)
    assert picked is None  # keys exhausted, still nothing picked
    assert out.getvalue().count("Write your own answer…") >= 2  # placeholder drawn again


def test_selector_custom_ctrl_c_cancels() -> None:
    keys = iter(["char:9", "enter", "char:x", "ctrl-c"])
    picked = tui._selector_pick(io.StringIO(), ["a"], keys)
    assert picked is None


def test_selector_cancel_returns_none() -> None:
    picked = tui._selector_pick(io.StringIO(), ["a"], iter(["ctrl-c"]))
    assert picked is None
