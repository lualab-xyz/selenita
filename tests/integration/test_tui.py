"""Integration tests of the interactive TUI mode.

Two variants:
- **Pipe** (process stdin): line mode, no raw keyboard — full REPL flow.
- **Real pty** (``os.openpty``): raw keyboard with Alt+Enter for multiple lines.
"""

from __future__ import annotations

import asyncio
import os
import re
import select
import subprocess
import sys
import time

_TIMEOUT = 30

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Text without ANSI styling: the TUI colors the answer marker on every channel."""
    return _ANSI_RE.sub("", text)


async def _run_piped(*args: str, stdin_text: str) -> tuple[int, str, str]:
    # Echo mode is forced: a .env with LLM credentials in the repo would otherwise make
    # the default agent mode "llm" and turn these tests into real (non-hermetic) calls.
    env = {**os.environ, "SELENITA_AGENT_MODE": "echo"}
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "selenita", *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    out, err = await asyncio.wait_for(proc.communicate(stdin_text.encode()), _TIMEOUT)
    return proc.returncode or 0, out.decode(), err.decode()


async def test_interactive_repl_over_pipe() -> None:
    code, out, _ = await _run_piped(
        "--session-id", "sess_tui_int", stdin_text="hello from a pipe\nbye\nexit\n"
    )
    out = _plain(out)
    assert code == 0
    assert out.count("❯") >= 3  # three prompt renders (two turns + exit)
    assert "¤ hello from a pipe\n\n" in out  # answer with marker + gap before the prompt
    assert "See you soon." in out
    assert "history:" in out  # the welcome shows the session state


async def test_interactive_repl_exits_on_eof() -> None:
    code, out, _ = await _run_piped(stdin_text="single turn\n")
    assert code == 0
    assert "single turn\n\n" in out
    assert "See you soon." in out


async def test_interactive_invalid_session_id_fails_fast() -> None:
    code, _, err = await _run_piped("--session-id", "bad id", stdin_text="")
    assert code == 2
    assert "invalid session_id" in err


# ---------------------------------------------------------------------------
# Raw mode over a pty: real keyboard (Alt+Enter → new line; Enter → submits)
# ---------------------------------------------------------------------------


def _spawn_tui(args: list[str]) -> tuple[subprocess.Popen[int], int]:
    """Launches the TUI over a fresh pty; returns (process, master_fd)."""
    master, slave = os.openpty()
    proc = subprocess.Popen(
        [sys.executable, "-m", "selenita", *args],
        stdin=slave, stdout=slave, stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    os.close(slave)
    return proc, master


def _close_tui(proc: subprocess.Popen[int], master: int) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5)
    os.close(master)


def _drain(fd: int, seconds: float = 5.0) -> str:
    """Reads everything available from an fd during a fixed window (for slow output)."""
    chunks: list[str] = []
    while select.select([fd], [], [], 0.2)[0]:
        data = os.read(fd, 4096)
        if not data:
            break
        chunks.append(data.decode("utf-8", errors="replace"))
    return "".join(chunks)


def _wait_for_output(fd: int, needle: str, timeout: float = 15.0) -> str:
    """Accumulates output until it contains ``needle`` or the timeout hits (avoids flakiness)."""
    collected: list[str] = []
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if select.select([fd], [], [], 0.2)[0]:
            try:
                data = os.read(fd, 4096)
            except OSError:  # pty slave gone (EIO): the child exited
                break
            if not data:
                break
            collected.append(_plain(data.decode("utf-8", errors="replace")))
            if needle in "".join(collected):
                break
    return "".join(collected)


def test_interactive_raw_tty_alt_enter_and_submit() -> None:
    proc, master = _spawn_tui(["--session-id", "sess_tui_pty"])
    try:
        _wait_for_output(master, "❯")  # wait for the welcome (avoids startup races)
        time.sleep(0.3)  # make sure the child already enabled raw mode before typing
        os.write(master, b"hello")  # type without submitting
        os.write(master, b"\x1b\r")  # Alt+Enter → new line
        os.write(master, b"world")
        os.write(master, b"\r")  # Enter → submits the "hello\nworld" turn
        # answer + gap + next prompt
        output = _wait_for_output(master, "¤ hello\r\n  world\r\n\r\n❯")
        assert "hello\r\n  world" in output  # Alt+Enter echo: \r\n + double indent
        _wait_for_output(master, "❯")  # next turn active (raw mode already restored)
        time.sleep(0.3)  # margin so the reader re-enters raw mode before Ctrl+D
        os.write(master, b"\x04")  # Ctrl+D → exit
        _wait_for_output(master, "See you soon.")
        proc.wait(timeout=_TIMEOUT)
        assert proc.returncode == 0
    finally:
        _close_tui(proc, master)


def test_interactive_raw_tty_shift_enter_kitty_sequence() -> None:
    """Shift+Enter in CSI-u notation (kitty/xterm.js): also inserts a new line."""
    proc, master = _spawn_tui(["--session-id", "sess_tui_pty2"])
    try:
        _wait_for_output(master, "❯")
        time.sleep(0.3)
        os.write(master, b"two")
        os.write(master, b"\x1b[13;2u")  # Shift+Enter (CSI-u)
        os.write(master, b" lines")
        os.write(master, b"\r")
        output = _wait_for_output(master, "¤ two\r\n   lines\r\n\r\n❯")
        assert "two\r\n   lines" in output  # echo: double indent (2) + user space (1) = 3
        time.sleep(0.3)
        os.write(master, b"\x04")
        _wait_for_output(master, "See you soon.")
        proc.wait(timeout=_TIMEOUT)
        assert proc.returncode == 0
    finally:
        _close_tui(proc, master)
