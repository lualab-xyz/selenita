"""Common test configuration: isolated environment for all tests (unit + integration).

- ``SELENITA_SESSIONS_DIR`` points to a temporary directory (no test persists in home).
- ``SELENITA_LOG_LEVEL=error`` to keep subprocess output clean.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELENITA_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("SELENITA_LOG_LEVEL", "error")
