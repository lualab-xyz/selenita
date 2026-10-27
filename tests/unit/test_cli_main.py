"""Unit tests of the CLI dispatch (parser + cross validations)."""

from __future__ import annotations

import importlib.metadata

import pytest

from selenita.cli.main import _build_parser, main


def test_parser_accepts_the_three_modes() -> None:
    acp = _build_parser().parse_args(["--acp"])
    assert acp.acp and acp.port is None and acp.prompt is None and acp.session_id is None

    one_shot = _build_parser().parse_args(["hello", "--session-id", "sess_abc12"])
    assert one_shot.prompt == "hello" and one_shot.session_id == "sess_abc12"

    interactive = _build_parser().parse_args([])
    assert interactive.prompt is None and interactive.acp is False


def test_parser_accepts_prompt_flag() -> None:
    flagged = _build_parser().parse_args(["-p", "hello", "--session-id", "sess_flag1"])
    assert flagged.prompt_flag == "hello" and flagged.prompt is None
    long_flag = _build_parser().parse_args(["--prompt", "bye"])
    assert long_flag.prompt_flag == "bye"


def test_version_flag_prints_the_installed_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.startswith(f"selenita {importlib.metadata.version('selenita')}")


@pytest.mark.parametrize(
    "argv",
    [
        ["--acp", "-p", "hello"],
        ["--acp", "--prompt", "hello"],
        ["hello", "-p", "hello"],
    ],
)
def test_invalid_flag_combinations_fail_fast(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SELENITA_LOG_LEVEL", "error")
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2  # usage on stderr through parser.error
