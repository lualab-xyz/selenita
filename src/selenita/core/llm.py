"""Real LLM core (LlmCore) on top of Microsoft Agent Framework (D002).

Sessions and persistence remain ours (D009: one JSONL file per ``sess_id``): every turn
rebuilds the message list from the store and runs a MAF streaming run. The typed chunks
this generator emits are what the three entry points translate (ACP, CLI, TUI — D012).

Providers (D006): ``openai`` → MAF OpenAIChatClient, which in agent-framework 1.x is the
**Responses API** client; ``openai-chat`` → OpenAIChatCompletionClient (legacy); and
``ollama`` → OllamaChatClient. Credentials come from the environment, never hardcoded.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator, Callable, Sequence

import agent_framework as af

from selenita.config import Settings
from selenita.sessions.store import JsonlSessionStore, SessionStore, new_session_id

from .chunks import ChunkKind, CoreChunk
from .clients import PlaintextReasoningClient
from .echo import _persist  # reuse the echo's graceful-degradation help to write the store
from .mcp import McpServerSpec, close_mcp_tools, collect_terminal_specs, open_mcp_tools
from .permissions import ApprovalMiddleware, ApprovalPolicy, Asker, ToolEvent
from .tools import Toolkit, make_builtin_tools

logger = logging.getLogger(__name__)

_UNREACHABLE_MSG = "⚠ Could not reach the model. Please check the LLM configuration."

_INSTRUCTIONS = (
    "You are selenita, a coding agent integrated with the Agent Client Protocol (ACP). "
    "Be useful, concrete and brief (max ~10 lines). When asked for code, show a minimal "
    "working block."
)


class LlmConfigError(Exception):
    """Missing or invalid LLM configuration (unknown provider, missing API key…)."""


def _deltas(update: object) -> list[CoreChunk]:
    """Streaming deltas of one update, typed by content kind.

    MAF tags incremental pieces with ``merge_mode="append"`` and completed-item
    events (e.g. the Responses API ``response.output_text.done``, which carries the
    whole accumulated text) with ``merge_mode="replace"``: re-emitting those would
    duplicate the stream. A missing merge_mode means delta (append semantics).

    Args:
        update: ``AgentResponseUpdate`` emitted by ``Agent.run(stream=True)``.

    Returns:
        Typed chunks of this update (possibly empty).
    """
    contents = getattr(update, "contents", None) or []
    chunks: list[CoreChunk] = []
    for content in contents:
        ctype = getattr(content, "type", None)
        kind: ChunkKind | None = None
        if ctype == "text":
            kind, text = "answer", getattr(content, "text", "") or ""
        elif ctype == "text_reasoning":
            kind, text = "reasoning", getattr(content, "text", "") or ""
        else:
            continue
        if not text or getattr(content, "merge_mode", None) == "replace":
            continue
        chunks.append(CoreChunk(kind=kind, text=text))
    return chunks


def _build_chat_client(settings: Settings) -> af.SupportsChatGetResponse:
    """Chat client according to ``SELENITA_LLM_PROVIDER`` (D006)."""
    provider = settings.llm_provider
    try:
        if provider == "openai":
            # PlaintextReasoningClient keeps MAF's stateless Responses tool loop
            # working on gateways that return plain-text reasoning (D014).
            return PlaintextReasoningClient(  # type: ignore[call-arg]
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        if provider == "openai-chat":
            from agent_framework.openai import OpenAIChatCompletionClient

            return OpenAIChatCompletionClient(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        if provider == "ollama":
            from agent_framework_ollama import OllamaChatClient

            return OllamaChatClient(model=settings.ollama_model, host=settings.ollama_host)
    except LlmConfigError:
        raise
    except Exception as exc:  # SDK errors while validating credentials/base URL
        raise LlmConfigError(f"{provider}: {exc}") from exc
    raise LlmConfigError(f"unknown provider: {provider!r} (openai|openai-chat|ollama)")


class LlmCore:
    """MAF core: Agent with fixed instructions, per-session history and optional MCP tools.

    MCP tools (D013) connect lazily inside the event loop on the first turn: specs are
    collected at build time (env/``--mcp-cmd``, or given directly for ACP sessions) and
    every failing server is skipped with a warning (D008). ``aclose`` stops them.
    """

    def __init__(
        self,
        client: af.SupportsChatGetResponse,
        store: SessionStore,
        mcp_specs: Sequence[McpServerSpec] = (),
        *,
        stateless: bool = False,
        toolkit: Toolkit | None = None,
    ) -> None:
        # No MAF Session per turn (session=None): our store provides the history.
        self._client = client
        self._store = store
        self._mcp_specs = tuple(mcp_specs)
        self._mcp_tools: Sequence[af.MCPStdioTool] = ()
        self._tools_attached = False
        self._stateless = stateless  # Responses: store=False → full inline history (D014)
        # Live tool-execution feedback: adapters (TUI/CLI) set a callback rendered as
        # compact ``$`` blocks; None keeps events silent (e.g. ACP transport).
        self.on_tool_event: Callable[[ToolEvent], None] | None = None
        # Built-in tools (tools epic): read/write/edit/run/web_fetch + optional ask_user,
        # gated by the ApprovalMiddleware so mutating calls ask the user first.
        self._toolkit = toolkit
        self._builtin_tools: Sequence[af.FunctionTool] = (
            make_builtin_tools(toolkit) if toolkit is not None else ()
        )
        self._approval_policy = ApprovalPolicy(toolkit.asker) if toolkit is not None else None
        middleware = (
            [ApprovalMiddleware(self._approval_policy, emit=self._emit_tool_event)]
            if self._approval_policy is not None
            else []
        )
        self._middleware = middleware
        self._agent = self._make_agent(list(self._builtin_tools))
        # Per-session MCP (ACP agent mode): ``session/new`` may declare ``mcpServers``;
        # each declaring session gets its own agent variant (client-owned servers are
        # session-scoped and must not leak into other sessions).
        self._session_agents: dict[str, af.Agent] = {}
        self._session_mcp: dict[str, list[af.MCPStdioTool]] = {}

    def _make_agent(self, tools: list[object]) -> af.Agent:
        """Build the MAF agent for a given tool set (same client + instructions + gate)."""
        return af.Agent(
            client=self._client,
            name="selenita",
            instructions=_INSTRUCTIONS,
            tools=tools,
            middleware=self._middleware,
        )

    @classmethod
    def build(cls, settings: Settings, *, toolkit: Toolkit | None = None) -> LlmCore:
        """Build from settings; raises LlmConfigError when configuration is incomplete."""
        if not settings.openai_api_key and settings.llm_provider.startswith("openai"):
            raise LlmConfigError(
                "missing OPENAI_API_KEY/SELENITA_OPENAI_API_KEY for the openai provider"
            )
        return cls(
            client=_build_chat_client(settings),
            store=JsonlSessionStore.from_settings(settings),
            mcp_specs=collect_terminal_specs(settings),
            stateless=settings.llm_provider == "openai",
            toolkit=toolkit,
        )

    @property
    def store(self) -> SessionStore:
        """Underlying session store."""
        return self._store

    def new_session_id(self) -> str:
        """Fresh session identifier (``sess_*``)."""
        return new_session_id()

    @property
    def approval_policy(self) -> ApprovalPolicy | None:
        """Permission policy wiring the tool gate (None when no built-in tools)."""
        return self._approval_policy

    def set_permission_asker(self, asker: Asker | None) -> None:
        """Wire (or clear) the asker that prompts the user for tool permissions."""
        if self._approval_policy is not None:
            self._approval_policy.asker = asker

    def _emit_tool_event(self, event: ToolEvent) -> None:
        """Forward one tool-execution event to the adapter's callback (if any)."""
        callback = self.on_tool_event
        if callback is not None:
            callback(event)

    async def aclose(self) -> None:
        """Stop every MCP tool (base and per-session); idempotent, safe to call twice."""
        self._tools_attached = True  # a closed core never tries to connect again
        await close_mcp_tools(self._mcp_tools)
        self._mcp_tools = ()
        session_tools = self._session_mcp
        self._session_mcp = {}
        self._session_agents = {}
        for tools in session_tools.values():
            await close_mcp_tools(tools)

    async def attach_session_mcp(self, session_id: str, specs: Sequence[McpServerSpec]) -> None:
        """Attach client-declared MCP tools to ONE session (ACP ``session/new``).

        Builds a dedicated agent variant for the session (base builtin tools + env-mode
        MCP tools + the session's servers). A failing server is skipped with a warning;
        with no usable server the session simply runs on the shared agent.
        """
        await self._ensure_tools()  # env/CLI-provided servers first (if any)
        tools = await open_mcp_tools(list(specs))
        if not tools:
            logger.info("no usable MCP server for session %s: running without them", session_id)
            return
        self._session_mcp[session_id] = list(tools)
        self._session_agents[session_id] = self._make_agent(
            [*self._builtin_tools, *self._mcp_tools, *tools]
        )
        logger.info(
            "attached %d MCP tool(s) to session %s", len(tools), session_id
        )

    async def _ensure_tools(self) -> None:
        """Attach the env/``--mcp-cmd`` tools on first use (runs inside the event loop).

        The base agent is rebuilt only when at least one server connected; servers
        already attempted are never retried within the lifetime of this core.
        """
        if self._tools_attached or not self._mcp_specs:
            self._tools_attached = True
            return
        self._tools_attached = True
        tools = await open_mcp_tools(list(self._mcp_specs))
        if tools:
            self._mcp_tools = tools
            self._agent = self._make_agent([*self._builtin_tools, *tools])

    async def respond(self, session_id: str, user_text: str) -> AsyncIterator[CoreChunk]:
        """One turn against the LLM: store history → streaming run → persistence."""
        if not user_text.strip():
            return
        await self._ensure_tools()
        # Per-session agent variant when the session declared MCP servers; otherwise the
        # shared one (stateless runs: same client, history comes from the store).
        agent = self._session_agents.get(session_id) or self._agent
        _persist(self._store, session_id, user_text, "user")
        messages = self._build_messages(session_id)
        answer_parts: list[str] = []
        try:
            options = af.ChatOptions(store=False) if self._stateless else None
            stream = agent.run(messages, stream=True, options=options)
            if inspect.isawaitable(stream):  # get_response may return a coroutine depending client
                stream = await stream
            async for update in stream:
                for chunk in _deltas(update):
                    if chunk.kind == "answer":
                        answer_parts.append(chunk.text)
                    yield chunk
        except Exception as exc:  # network/API: the turn must not die ugly (D008, degradation)
            logger.error("error in LLM turn (session %s): %s", session_id, exc)
            yield CoreChunk(kind="answer", text=_UNREACHABLE_MSG)
            return
        if answer_parts:
            _persist(self._store, session_id, "".join(answer_parts), "agent")

    def _build_messages(self, session_id: str) -> list[af.Message]:
        """Full store history as MAF Messages (user/agent → user/assistant)."""
        role_map = {"user": "user", "agent": "assistant"}
        return [
            af.Message(role=role_map.get(entry.role, "user"), contents=[entry.text])
            for entry in self._store.load(session_id)
        ]
