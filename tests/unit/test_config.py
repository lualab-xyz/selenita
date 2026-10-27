"""Unit tests of the ``.env`` loader and Settings integration (round 3 feedback)."""

import logging
from pathlib import Path

from selenita.config import Settings, load_dotenv, setup_logging


def test_load_dotenv_reads_keys_quotes_and_export(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# selenita environment",
                "",
                "SELENITA_AGENT_MODE=llm",
                'SELENITA_LLM_PROVIDER="openai-chat"',
                "export SELENITA_OPENAI_MODEL='default'",
                "MALFORMED_LINE",
                "=NO_KEY",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env: dict[str, str] = {}
    load_dotenv(env_file, env)
    assert env["SELENITA_AGENT_MODE"] == "llm"
    assert env["SELENITA_LLM_PROVIDER"] == "openai-chat"
    assert env["SELENITA_OPENAI_MODEL"] == "default"
    assert "MALFORMED_LINE" not in env
    assert "NO_KEY" not in env


def test_load_dotenv_never_overrides_existing_keys(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SELENITA_AGENT_MODE=echo\n", encoding="utf-8")
    env = {"SELENITA_AGENT_MODE": "llm"}
    load_dotenv(env_file, env)
    assert env["SELENITA_AGENT_MODE"] == "llm"


def test_load_dotenv_missing_file_is_noop(tmp_path: Path) -> None:
    env: dict[str, str] = {"A": "1"}
    assert load_dotenv(tmp_path / "missing.env", env) is env


def test_settings_from_env_reads_dotenv(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "SELENITA_LLM_PROVIDER=openai-chat\nSELENITA_OPENAI_MODEL=fancy\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    for key in ("SELENITA_LLM_PROVIDER", "SELENITA_OPENAI_MODEL"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings.from_env()
    assert settings.llm_provider == "openai-chat"
    assert settings.openai_model == "fancy"


def test_settings_from_env_prefers_real_environment(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("SELENITA_LLM_PROVIDER=ollama\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SELENITA_LLM_PROVIDER", "openai")
    settings = Settings.from_env()
    assert settings.llm_provider == "openai"


def test_setup_logging_silences_third_party_http_loggers() -> None:
    setup_logging(Settings(log_level="info"))
    for name in ("httpx", "httpx2", "httpcore", "httpcore2"):
        assert logging.getLogger(name).level == logging.WARNING


def test_setup_logging_debug_keeps_third_party_http_loggers() -> None:
    setup_logging(Settings(log_level="debug"))
    for name in ("httpx", "httpx2", "httpcore", "httpcore2"):
        assert logging.getLogger(name).level == logging.NOTSET
