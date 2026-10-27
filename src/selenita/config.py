"""Environment-based configuration (decisions D006/D007/D008 in ``.context/decisions.md``).

Plus a tiny dependency-free ``.env`` loader: it reads ``KEY=VALUE`` lines from ``.env``
so exported variables survive between shells. Real environment variables always win
(the file only fills gaps).
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(
    path: Path, env: MutableMapping[str, str] | None = None
) -> MutableMapping[str, str]:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into ``env`` (or ``os.environ``).

    Rules: ``#`` comments and blank lines are skipped, optional surrounding quotes are
    stripped, an optional leading ``export`` is ignored, and existing keys are preserved.
    """
    env = os.environ if env is None else env
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, UnicodeDecodeError):
        return env
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].strip()  # tolerate ``export KEY=VALUE``
        key, sep, value = stripped.partition("=")
        if not sep or not (key := key.strip()) or key in env:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        env[key] = value
    return env


@dataclass(frozen=True)
class Settings:
    """Snapshot of the process effective configuration."""

    sessions_dir: Path = field(default_factory=lambda: Path.home() / ".selenita" / "sessions")
    log_level: str = "info"
    # --- Agent mode and LLM (D002/D006) -------------------------------------------------
    # SELENITA_AGENT_MODE: "llm" (real core on MAF) | "echo" (echo, for trials).
    agent_mode: str = "llm"
    # SELENITA_LLM_PROVIDER: "openai" (Responses API) | "openai-chat" | "ollama" (D006).
    llm_provider: str = "openai"
    # Default gateway supports Responses tool loops (D014); the `default` route returns
    # empty text after tool calls, so selenita ships pointing at `openrouter`.
    openai_base_url: str | None = "http://host.docker.internal:3000/api/ai-proxy/openrouter"
    openai_model: str = "default"
    openai_api_key: str | None = None
    ollama_model: str = "llama3.2"
    ollama_host: str | None = None
    # --- MCP stdio servers (D013) -------------------------------------------------------
    # SELENITA_MCP_SERVERS: JSON list of {"name"?, "command", "args"?, "env"?}.
    mcp_servers: str | None = None
    # Carries the --mcp-cmd flag (single server); set via dataclasses.replace in the CLI.
    mcp_cmd: str | None = None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        """Build settings from the environment (a dict can be injected in tests too).

        When reading the real environment, ``.env`` from the current directory is loaded
        first (filling gaps only); tests injecting a dict skip the file for determinism.
        """
        if environ is None:
            load_dotenv(Path.cwd() / ".env")
        env = os.environ if environ is None else environ
        return cls(
            sessions_dir=Path(
                env.get("SELENITA_SESSIONS_DIR", str(Path.home() / ".selenita" / "sessions"))
            ),
            log_level=env.get("SELENITA_LOG_LEVEL", "info").lower(),
            agent_mode=env.get("SELENITA_AGENT_MODE", "llm").lower(),
            llm_provider=env.get("SELENITA_LLM_PROVIDER", "openai").lower(),
            openai_base_url=env.get("SELENITA_OPENAI_BASE_URL"),
            openai_model=env.get("SELENITA_OPENAI_MODEL", env.get("OPENAI_MODEL", "default")),
            openai_api_key=env.get("SELENITA_OPENAI_API_KEY") or env.get("OPENAI_API_KEY"),
            ollama_model=env.get("SELENITA_OLLAMA_MODEL", env.get("OLLAMA_MODEL", "llama3.2")),
            ollama_host=env.get("SELENITA_OLLAMA_HOST") or env.get("OLLAMA_HOST"),
            mcp_servers=env.get("SELENITA_MCP_SERVERS"),
        )


# Third-party loggers that flood stderr with per-request lines (above all the vendored
# httpx2 "HTTP Request: POST …"). They stay quiet unless SELENITA_LOG_LEVEL=debug.
_NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "agent_framework")


def setup_logging(settings: Settings) -> None:
    """Configure logging to stderr exclusively.

    In ACP mode stdout belongs to the protocol (NDJSON) — nothing else may exit there
    (D008: stdout is reserved for the protocol when running as an ACP server).
    Third-party loggers (HTTP clients and MAF's per-function INFO lines) are capped at
    WARNING so the console only shows our own messages; ``SELENITA_LOG_LEVEL=debug``
    re-enables them for troubleshooting.
    """
    level = logging.getLevelNamesMapping().get(settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if level > logging.DEBUG:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
    else:  # debug mode: third-party HTTP traces pass through again (idempotent)
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.NOTSET)
