"""Unit tests of the session store (JSONL on disk + in memory)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from selenita.sessions.store import (
    InMemorySessionStore,
    JsonlSessionStore,
    SessionMessage,
    new_session_id,
    validate_session_id,
)


class TestSessionIds:
    def test_new_session_id_has_expected_shape(self) -> None:
        sid = new_session_id()
        assert sid.startswith("sess_")
        assert validate_session_id(sid) == sid

    def test_validate_rejects_wrong_shapes(self) -> None:
        for bad in ("", "no_prefix", "sess_", "../etc/passwd", "sess_../x", "user_session"):
            with pytest.raises(ValueError, match="invalid session_id"):
                validate_session_id(bad)

    def test_message_rejects_unknown_roles(self) -> None:
        with pytest.raises(ValueError, match="invalid role"):
            SessionMessage(role="system", text="x")


class TestInMemoryStore:
    def test_roundtrip_and_isolation(self) -> None:
        store = InMemorySessionStore()
        assert store.exists("sess_aaaa") is False
        assert store.load("sess_aaaa") == []

        store.append("sess_aaaa", SessionMessage(role="user", text="hello"))
        store.append("sess_aaaa", SessionMessage(role="agent", text="hello"))
        history = store.load("sess_aaaa")
        assert [m.role for m in history] == ["user", "agent"]
        assert [m.text for m in history] == ["hello", "hello"]

        # Independent session and list returned by copy (does not mutate the store)
        store.load("sess_aaaa").append(SessionMessage(role="user", text="injected"))
        assert len(store.load("sess_aaaa")) == 2
        assert store.exists("sess_bbbb") is False


class TestJsonlStore:
    def test_append_creates_dir_and_load_roundtrip(self, tmp_path: Path) -> None:
        store = JsonlSessionStore(tmp_path / "sessions")
        store.append("sess_uid1", SessionMessage(role="user", text="jalapeño ñ"))
        store.append("sess_uid1", SessionMessage(role="agent", text="jalapeño ñ"))

        assert (tmp_path / "sessions" / "sess_uid1.jsonl").is_file()
        history = store.load("sess_uid1")
        assert [m.text for m in history] == ["jalapeño ñ", "jalapeño ñ"]

        # A fresh store (another process) sees the same history: it persists across runs.
        reopened = JsonlSessionStore(tmp_path / "sessions")
        assert reopened.exists("sess_uid1") is True
        assert [m.text for m in reopened.load("sess_uid1")] == ["jalapeño ñ", "jalapeño ñ"]

    def test_load_missing_session_returns_empty(self, tmp_path: Path) -> None:
        store = JsonlSessionStore(tmp_path)
        assert store.exists("sess_none") is False
        assert store.load("sess_none") == []

    def test_invalid_ids_never_reach_the_filesystem(self, tmp_path: Path) -> None:
        store = JsonlSessionStore(tmp_path)
        with pytest.raises(ValueError, match="invalid session_id"):
            store.load("../../escape")
        with pytest.raises(ValueError, match="invalid session_id"):
            store.append("sess_white/../etc", SessionMessage(role="user", text="x"))

    def test_file_format_is_jsonl_one_record_per_line(self, tmp_path: Path) -> None:
        store = JsonlSessionStore(tmp_path)
        store.append("sess_fmt0", SessionMessage(role="user", text="line"))
        lines = (tmp_path / "sess_fmt0.jsonl").read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        assert record["role"] == "user"
        assert record["text"] == "line"
        assert "created_at" in record
