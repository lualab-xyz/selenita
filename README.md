# selenita

**English** · [Español](README.es.md)

An AI coding agent that implements the **[Agent Client Protocol (ACP)](https://agentclientprotocol.com/) v1**,
built on **[Microsoft Agent Framework](https://github.com/microsoft/agent-framework) (Python)**
and the official ACP Python SDK (`agent-client-protocol`).

It plugs into any ACP client (e.g. [Zed](https://zed.dev)) and is also usable straight
from the terminal.

## Install (alpha channel)

Alpha builds are published as tagged pre-releases (`v0.1.0aN`, PEP 440) of the public
distribution repo [lualab-xyz/selenita](https://github.com/lualab-xyz/selenita):

```bash
# Linux/macOS — one-liner (downloads the release wheel, verifies it against
# SHA256SUMS, installs into ~/.selenita/venv and links `selenita` in ~/.local/bin)
curl -fsSL https://raw.githubusercontent.com/lualab-xyz/selenita/alpha-1/install.sh | bash

# Windows (PowerShell)
irm https://raw.githubusercontent.com/lualab-xyz/selenita/alpha-1/install.ps1 | iex

# Or manually, on any platform with Python >= 3.11
pip install https://github.com/lualab-xyz/selenita/releases/download/v0.1.0a1/selenita-0.1.0a1-py3-none-any.whl
```

Releases are tag-driven: pushing `v0.1.0aN` to the public repo builds the wheel + sdist,
smoke-tests the installed CLI on the supported platforms (linux-x64, windows-x64) and
attaches the wheel, sdist, installers and SHA256SUMS to the release.

## Usage (LLM core live, with reasoning streaming and echo fallback)

All three entry points target the same agent core (one core, three adapters):

```bash
# 1) ACP server over stdio (add command=selenita, args=["--acp"] to your ACP client config)
selenita --acp

# 2) One-shot CLI (creates a session and prints its session-id)
selenita "hello"           # → answer + [session: sess_xxx]
selenita --prompt "hello"  # same thing (-p)

# 2b) One-shot CLI resuming a persisted session (JSONL under ~/.selenita/sessions)
selenita "more things" --session-id sess_8e1f...

# 3) Interactive TUI on the same session (cyan ❯ prompt, agent answers marked with ¤,
#    model reasoning streams muted with ·, Alt+Enter = newline; "exit"/quit or Ctrl+D/Ctrl+C quit;
#    welcome shows the last message of the session, truncated when long)
selenita
selenita --session-id sess_8e1f...
```

Note: Alt+Enter and Shift+Enter insert a newline when the terminal distinguishes them
(`ESC CR`, kitty `CSI 13;<mod>u` or xterm `CSI 27;<mod>;13~`); in many emulators
Shift+Enter equals Enter and only Alt+Enter works.

## Configuration

By environment variables (`SELENITA_*`), and optionally by a `.env` file in the current
working directory — real environment variables always win:

```bash
# .env (not committed; see .gitignore)
SELENITA_AGENT_MODE=llm            # or "echo" to force the echo core
SELENITA_LLM_PROVIDER=openai       # "openai" (Responses API) | "openai-chat" | "ollama"
SELENITA_OPENAI_BASE_URL=https://your-llm-endpoint.example/v1
SELENITA_OPENAI_MODEL=default
SELENITA_OPENAI_API_KEY=sk-your-key
```

> The Responses client always runs stateless (`store=False`, full inline history — see
> design D014); the default base URL points at the only gateway route that supports
> Responses tool loops. `previous_response_id` stays as a future optimization.

Missing LLM configuration degrades gracefully: selenita starts in **echo mode** and logs
a warning; every turn still works.

```bash
# ACP server over TCP, NDJSON (custom transport, not implemented yet)
selenita --acp --port 8765

# MCP in terminal mode: connect one stdio MCP server (single-server shortcut)
selenita "Use the add tool to compute 2 + 3" --mcp-cmd "npx -y @modelcontextprotocol/server-filesystem ."

# ... or several servers via env (name optional):
# SELENITA_MCP_SERVERS='[{"name":"demo","command":"python","args":["mcp_server.py"]}]'
```

## Stack

| Piece | Choice |
|---|---|
| Language | Python 3.11+ (asyncio) |
| Agent core | Microsoft Agent Framework (`agent-framework-core`) |
| Protocol | ACP v1 via the official Python SDK (`agent-client-protocol`) |
| MCP | MAF MCPStdioTool (servers passed by the ACP client and/or configured locally via env/`--mcp-cmd`) |
| LLM | OpenAI (Responses API) and Ollama, configurable via env (`SELENITA_LLM_PROVIDER`) |
| Tests | pytest + pytest-asyncio, with LLM and MCP mocked from day 1 |
| Quality | ruff + mypy (typed src) |

## Status

🚧 **Under construction** — live LLM core (streamed reasoning and answers), echo fallback,
`--acp` server, one-shot CLI and interactive TUI on the same core (74 tests). Living
documentation under [`/.context/`](.context/):

- [`design.md`](.context/design.md) — architecture and protocol mapping (internal, ES).
- [`todo.md`](.context/todo.md) — backlog by epics (internal, ES).
- [`task.md`](.context/task.md) — current task.
- [`decisions.md`](.context/decisions.md) — project rules and decisions (internal, ES).
- [`changelog.md`](.context/changelog.md) — integration history (internal, ES).

Workflow: `feature/task-name` branches, frequent checkpoint commits, never directly on `main`/`dev`.