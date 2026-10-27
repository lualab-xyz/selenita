"""Unit tests of the MCP stdio spec parsing and adapters (tools epic, D013)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from selenita.config import Settings
from selenita.core import mcp
from selenita.core.mcp import (
    McpServerSpec,
    close_mcp_tools,
    collect_terminal_specs,
    open_mcp_tools,
    parse_spec_cmd,
    parse_specs_json,
    specs_from_acp,
)


def test_parse_specs_json_full_shape() -> None:
    raw = '[{"name":"fs","command":"npx","args":["-y","srv","/tmp"],"env":{"A":"1"}}]'
    assert parse_specs_json(raw) == [
        McpServerSpec(name="fs", command="npx", args=("-y", "srv", "/tmp"), env={"A": "1"})
    ]


def test_parse_specs_json_defaults_and_minimal() -> None:
    specs = parse_specs_json('[{"command":"/usr/bin/mcp-server-demo"}]')
    assert specs == [McpServerSpec(name="mcp-server-demo", command="/usr/bin/mcp-server-demo")]


@pytest.mark.parametrize(
    "raw",
    ['{"command": "x"}', "not json", '[{"command": "x", "args": "y"}]', '["plain"]', ""],
)
def test_parse_specs_json_rejects_malformed(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_specs_json(raw)


def test_parse_specs_json_empty_list_is_valid() -> None:
    assert parse_specs_json("[]") == []


def test_parse_spec_cmd_shlex_splits_and_names_after_command() -> None:
    spec = parse_spec_cmd('npx  -y @modelcontextprotocol/server-x "/tmp/my dir"')
    assert spec == McpServerSpec(
        name="npx", command="npx", args=("-y", "@modelcontextprotocol/server-x", "/tmp/my dir")
    )
    assert parse_spec_cmd("python srv.py", name="py") == McpServerSpec(
        "py", "python", ("srv.py",)
    )


def test_parse_spec_cmd_empty_raises() -> None:
    with pytest.raises(ValueError):
        parse_spec_cmd("   ")


def test_collect_terminal_specs_combines_and_degrades(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        specs = collect_terminal_specs(Settings(mcp_servers="{bad", mcp_cmd=""))
    assert specs == []
    assert any("SELENITA_MCP_SERVERS" in record.message for record in caplog.records)

    good = Settings(mcp_servers='[{"command":"c1"}]', mcp_cmd="c2 --flag")
    specs = collect_terminal_specs(good)
    assert [(s.name, s.args) for s in specs] == [("c1", ()), ("c2", ("--flag",))]


def test_specs_from_acp_accepts_objects_dicts_and_skips_others(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sdk_obj = SimpleNamespace(name="fs", command="npx", args=["-y"], env={"K": "V"})
    with caplog.at_level(logging.WARNING):
        specs = specs_from_acp(
            [sdk_obj, {"name": "py", "command": "python"}, {"name": "http-only"}, None]
        )
    assert [s.name for s in specs] == ["fs", "py"]
    assert specs[0].args == ("-y",) and specs[0].env == {"K": "V"}
    assert specs[1].args == ()
    assert any("not stdio" in record.message for record in caplog.records)
    assert specs_from_acp(None) == []


# --- open/close with a stubbed MCP tool class -------------------------------------


class _StubTool:
    installed: list[_StubTool] = []

    def __init__(self, *, name: str, command: str, args: list[str], env: dict[str, str]) -> None:
        self.name, self.command, self.args, self.env = name, command, args, env
        self.connected = False
        self.fail_connect = name == "bad"
        _StubTool.installed.append(self)

    async def connect(self) -> None:
        if self.fail_connect:
            raise RuntimeError("boom")
        self.connected = True

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_open_mcp_tools_skips_failing_servers_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(mcp.af, "MCPStdioTool", _StubTool)
    _StubTool.installed = []
    specs = [McpServerSpec("good", "g"), McpServerSpec("bad", "b"), McpServerSpec("ok", "o")]
    with caplog.at_level(logging.WARNING):
        tools = await open_mcp_tools(specs)
    assert [t.name for t in tools] == ["good", "ok"]
    assert tools[0].connected is True
    assert [t.name for t in _StubTool.installed] == ["good", "bad", "ok"]
    assert tools[0].args == [] and tools[0].env == {}
    assert any("failed to start" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_close_mcp_tools_swallows_close_errors() -> None:
    class Broken:
        name = "broken"

        async def close(self) -> None:
            raise RuntimeError("nope")

    class Quiet:
        name = "quiet"
        closed = False

        async def close(self) -> None:
            self.closed = True

    quiet = Quiet()
    await close_mcp_tools([Broken(), quiet])  # must not raise
    assert quiet.closed is True
