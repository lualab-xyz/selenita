"""Integration tests of the non-interactive CLI mode: ``selenita "prompt" [-s ID]``."""

from __future__ import annotations

import asyncio
import os
import re
import sys

import pytest

from selenita.cli import terminal
from selenita.config import Settings
from selenita.core.llm import LlmCore
from selenita.sessions.store import InMemorySessionStore
from tests._fakes import FakeChatClient

_TIMEOUT = 30


async def _run(*args: str) -> tuple[int, str, str]:
    # Echo mode is forced: a .env with LLM credentials in the repo would otherwise make
    # the default agent mode "llm" and turn these tests into real (non-hermetic) calls.
    env = {**os.environ, "SELENITA_AGENT_MODE": "echo"}
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "selenita", *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
    )
    out, err = await asyncio.wait_for(proc.communicate(), _TIMEOUT)
    return proc.returncode or 0, out.decode(), err.decode()


async def test_one_shot_echoes_and_reports_session_id() -> None:
    code, out, _ = await _run("hello echo", "--session-id", "sess_fix01")
    assert code == 0
    assert out.splitlines() == ["hello echo", "[session: sess_fix01]"]


async def test_one_shot_without_session_id_creates_one() -> None:
    code, out, _ = await _run("try without id")
    assert code == 0
    assert out.splitlines()[0] == "try without id"
    match = re.fullmatch(r"\[session: (sess_[0-9a-f]{32})\]", out.splitlines()[1])
    assert match, out


async def test_one_shot_persists_across_runs() -> None:
    """Two runs over the same session: the JSONL history grows (same core logic)."""
    import json
    import os
    from pathlib import Path

    sid = "sess_hist01"
    code1, _, _ = await _run("first", "--session-id", sid)
    assert code1 == 0
    code2, out, _ = await _run("second", "--session-id", sid)
    assert code2 == 0
    assert out.splitlines() == ["second", f"[session: {sid}]"]

    store_path = Path(os.environ["SELENITA_SESSIONS_DIR"]) / f"{sid}.jsonl"
    records = [json.loads(line) for line in store_path.read_text(encoding="utf-8").splitlines()]
    assert [(r["role"], r["text"]) for r in records] == [
        ("user", "first"),
        ("agent", "first"),
        ("user", "second"),
        ("agent", "second"),
    ]


async def test_one_shot_rejects_invalid_session_id() -> None:
    code, _, err = await _run("hello", "--session-id", "bad id")
    assert code == 2
    assert "invalid session_id" in err


def test_one_shot_reasoning_goes_to_stderr_and_answer_to_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A core streaming reasoning chunks sends them to stderr, one line per chunk."""
    fake = FakeChatClient("hello answer", reasoning_chunks=["thinking quietly"])
    core = LlmCore(client=fake, store=InMemorySessionStore())
    monkeypatch.setattr(terminal, "create_core", lambda _settings, *, toolkit=None: core)

    code = terminal.run_terminal_prompt("hello", "sess_cli_r", Settings.from_env())

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == "hello answer\n[session: sess_cli_r]\n"
    assert captured.err == "thinking quietly\n\n"  # close line + blank gap, mirroring the TUI


@pytest.mark.parametrize(
    "args,fragment",
    [
        (["--acp", "hello"], "prompt"),
        (["--acp", "--session-id", "sess_fix01"], "--session-id"),
    ],
)
async def test_flag_combinations_rejected(args: list[str], fragment: str) -> None:
    code, _, err = await _run(*args)
    assert code == 2
    # The parser.error strings still live in src (Spanish; reported): fragment keeps
    # the assertion language-neutral while pinning which flag combination failed.
    assert fragment in err
