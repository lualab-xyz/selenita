"""Unit tests of the built-in tools (tools epic): no network, tiny fixtures in tmp_path."""

from __future__ import annotations

from pathlib import Path

import agent_framework as af
import pytest
from agent_framework.exceptions import ToolException

from selenita.core.tools import (
    Toolkit,
    make_builtin_tools,
    selenita_edit_file,
    selenita_read_file,
    selenita_run_command,
    selenita_web_fetch,
    selenita_write_file,
    summarize_tool_call,
)

# ---------------------------------------------------------------------------
# selenita_read_file
# ---------------------------------------------------------------------------


def test_read_file_returns_utf8_text(tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("hola mundo", encoding="utf-8")

    assert selenita_read_file(str(target)) == "hola mundo"


def test_read_file_missing_raises_tool_exception(tmp_path: Path) -> None:
    with pytest.raises(ToolException, match="File not found"):
        selenita_read_file(str(tmp_path / "nope.txt"))


def test_read_file_truncates_huge_files(tmp_path: Path) -> None:
    target = tmp_path / "big.txt"
    target.write_text("a" * 5000, encoding="utf-8")

    out = selenita_read_file(str(target), max_chars=100)

    assert out.startswith("a" * 100)
    assert "truncated to 100" in out
    assert len(out) < 5000


def test_read_file_binary_raises_tool_exception(tmp_path: Path) -> None:
    target = tmp_path / "bin.dat"
    target.write_bytes(b"\xff\xfe\x00binary")

    with pytest.raises(ToolException, match="not valid UTF-8"):
        selenita_read_file(str(target))


# ---------------------------------------------------------------------------
# selenita_write_file
# ---------------------------------------------------------------------------


def test_write_file_creates_parents_and_content(tmp_path: Path) -> None:
    target = tmp_path / "dir" / "nested" / "file.txt"

    msg = selenita_write_file(str(target), "línea 1\nlínea 2")

    assert msg.startswith("Wrote 15 chars")
    assert target.read_text(encoding="utf-8") == "línea 1\nlínea 2"


def test_write_file_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("old", encoding="utf-8")

    selenita_write_file(str(target), "new")

    assert target.read_text(encoding="utf-8") == "new"


# ---------------------------------------------------------------------------
# selenita_edit_file
# ---------------------------------------------------------------------------


def _sample(path: Path, text: str = "def foo():\n    return 1\n") -> None:
    path.write_text(text, encoding="utf-8")


def test_edit_file_replaces_unique_match(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    _sample(target, "a = 1\na = 2\n")

    msg = selenita_edit_file(str(target), "a = 2", "a = 3")

    assert "1 replacement(s)" in msg
    assert target.read_text(encoding="utf-8") == "a = 1\na = 3\n"


def test_edit_file_no_match_raises(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    _sample(target)

    with pytest.raises(ToolException, match="not found"):
        selenita_edit_file(str(target), "def bar():", "def baz():")


def test_edit_file_multiple_matches_raises_without_replace_all(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    _sample(target, "x = 1\nx = 2\nx = 3\n")

    with pytest.raises(ToolException, match="matched 3 times"):
        selenita_edit_file(str(target), "x = ", "y = ")


def test_edit_file_replace_all_replaces_every_occurrence(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    _sample(target, "x = 1\nx = 2\nx = 3\n")

    msg = selenita_edit_file(str(target), "x = ", "y = ", replace_all=True)

    assert "3 replacement(s)" in msg
    assert target.read_text(encoding="utf-8") == "y = 1\ny = 2\ny = 3\n"


def test_edit_file_empty_old_string_raises(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    _sample(target)

    with pytest.raises(ToolException, match="must not be empty"):
        selenita_edit_file(str(target), "", "x")


def test_edit_file_identical_strings_raise(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    _sample(target)

    with pytest.raises(ToolException, match="must differ"):
        selenita_edit_file(str(target), "def foo():", "def foo():")


def test_edit_file_normalizes_smart_quotes_and_dashes(tmp_path: Path) -> None:
    """The model may emit Unicode punctuation the file stores as ASCII."""
    target = tmp_path / "doc.md"
    target.write_text("It's “quoted” - done\n", encoding="utf-8")

    selenita_edit_file(str(target), "It’s “quoted” – done", "It is quoted - done")

    assert target.read_text(encoding="utf-8") == "It is quoted - done\n"


def test_edit_file_tolerates_crlf_line_endings(tmp_path: Path) -> None:
    target = tmp_path / "win.txt"
    target.write_bytes(b"first\r\nsecond\r\nthird\r\n")

    selenita_edit_file(str(target), "first\nsecond", "first and second")

    assert target.read_bytes() == b"first and second\r\nthird\r\n"


# ---------------------------------------------------------------------------
# selenita_run_command
# ---------------------------------------------------------------------------


def test_run_command_returns_exit_code_and_output() -> None:
    out = selenita_run_command("echo hello-tool")

    assert out.startswith("exit code: 0")
    assert "hello-tool" in out


def test_run_command_captures_stderr_and_nonzero_exit() -> None:
    out = selenita_run_command("echo boo >&2; exit 3")

    assert out.startswith("exit code: 3")
    assert "boo" in out


def test_run_command_timeout_raises_tool_exception() -> None:
    with pytest.raises(ToolException, match="timed out"):
        selenita_run_command("sleep 5", timeout=1)


def test_run_command_caps_output() -> None:
    out = selenita_run_command("python3 -c 'print(\"a\" * 100_000)'", timeout=30)

    assert "truncated to 40000" in out
    assert "a" * 40_000 in out


# ---------------------------------------------------------------------------
# selenita_web_fetch (urllib monkeypatched: no network)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._body[:size]


def test_web_fetch_returns_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(url: str, timeout: int) -> _FakeResponse:
        assert url == "https://example.com"
        return _FakeResponse(b"<html>hi</html>")

    monkeypatch.setattr("selenita.core.tools.urllib.request.urlopen", fake_urlopen)

    assert selenita_web_fetch("https://example.com") == "<html>hi</html>"


def test_web_fetch_truncates_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "selenita.core.tools.urllib.request.urlopen",
        lambda url, timeout: _FakeResponse(b"b" * 1000),
    )

    out = selenita_web_fetch("https://example.com", max_bytes=100)

    assert out.startswith("b" * 100)
    assert "truncated to 100" in out


def test_web_fetch_failure_raises_tool_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    def boom(url: str, timeout: int) -> _FakeResponse:
        raise urllib.error.URLError("nope")

    monkeypatch.setattr("selenita.core.tools.urllib.request.urlopen", boom)

    with pytest.raises(ToolException, match="web_fetch failed"):
        selenita_web_fetch("https://example.com")


# ---------------------------------------------------------------------------
# Toolkit / make_builtin_tools
# ---------------------------------------------------------------------------


def test_make_builtin_tools_default_toolkit_offers_read_only_and_mutating() -> None:
    tools = make_builtin_tools(Toolkit())

    assert [t.name for t in tools] == [
        "selenita_read_file",
        "selenita_web_fetch",
        "selenita_write_file",
        "selenita_edit_file",
        "selenita_run_command",
    ]


def test_make_builtin_tools_read_only_skips_mutating() -> None:
    tools = make_builtin_tools(Toolkit(include_mutating=False))

    assert [t.name for t in tools] == ["selenita_read_file", "selenita_web_fetch"]


def test_make_builtin_tools_tui_adds_ask_user() -> None:
    tools = make_builtin_tools(Toolkit(include_ask_user=True))

    assert "selenita_ask_user" in [t.name for t in tools]


@pytest.mark.asyncio
async def test_ask_user_without_callback_says_not_available() -> None:
    tools = make_builtin_tools(Toolkit(include_ask_user=True))
    ask_user = next(t for t in tools if t.name == "selenita_ask_user")

    result = await ask_user.func("question?")

    assert "only works in the TUI" in result


@pytest.mark.asyncio
async def test_ask_user_delegates_to_callback_with_options() -> None:
    seen: list[tuple[str, list[str] | None]] = []

    async def callback(question: str, options: list[str] | None) -> str:
        seen.append((question, options))
        return "option B"

    tools = make_builtin_tools(Toolkit(include_ask_user=True, ask_user_callback=callback))
    ask_user = next(t for t in tools if t.name == "selenita_ask_user")

    result = await ask_user.func("pick one", ["A", "B"])

    assert result == "option B"
    assert seen == [("pick one", ["A", "B"])]


def test_builtin_tools_are_maf_function_tools() -> None:
    for tool in make_builtin_tools(Toolkit()):
        assert isinstance(tool, af.FunctionTool)


def test_edit_helpers_normalize_unicode_only_for_matching() -> None:
    """Regression: the real file text must never be mutated, only the match key."""
    target = Path("/tmp/selenita-match-escapes.txt")
    target.write_text("plain ascii", encoding="utf-8")

    selenita_edit_file(str(target), "plain\u00a0ascii", "plain space ascii")

    assert target.read_text(encoding="utf-8") == "plain space ascii"
    target.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# summarize_tool_call (compact permission detail line)
# ---------------------------------------------------------------------------


def test_summarize_tool_call_highlights_the_command() -> None:
    """selenita_run_command shows the command itself, not key=value noise."""
    text = summarize_tool_call(
        "selenita_run_command", {"command": "git status", "cwd": "/workspace", "timeout": 120}
    )
    assert text == "git status"


def test_summarize_tool_call_shows_key_value_pairs() -> None:
    text = summarize_tool_call("selenita_write_file", {"path": "a.txt", "content": "x"})
    assert text == "path='a.txt', content='x'"


def test_summarize_tool_call_truncates_overlong_detail() -> None:
    text = summarize_tool_call("selenita_run_command", {"command": "x" * 500})
    assert text.endswith("…")
    assert len(text) == 101  # 100 kept chars + the ellipsis
