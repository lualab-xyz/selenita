"""ACP test harness: real JSON-RPC NDJSON client against a server subprocess.

Reusable by any protocol integration test (task ``acp-jsonrpc-harness``):
it launches ``python -m selenita --acp``, writes NDJSON requests to stdin and separates,
line by line, the responses of the current ``id`` from the ``session/update``
notifications of each exchange.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

TEST_TIMEOUT = float(os.environ.get("SELENITA_TEST_TIMEOUT", "30"))


class RpcExchange:
    """Result of one request: server response + notifications of the exchange."""

    def __init__(self, result: dict, notifications: list[dict],
                 permission_requests: list[dict] | None = None) -> None:
        self.result = result
        self.notifications = notifications
        # Server→client requests seen while this exchange was in flight
        # (currently ``session/request_permission``).
        self.permission_requests: list[dict] = permission_requests or []

    def chunk_texts(self, session_update: str = "agent_message_chunk") -> list[str]:
        """Chunk texts in the session/update notifications of this exchange."""
        texts: list[str] = []
        for notif in self.notifications:
            if notif["method"] != "session/update":
                continue
            update = notif["params"]["update"]
            if update.get("sessionUpdate") == session_update:
                texts.append(update["content"].get("text", ""))
        return texts

    def methods(self) -> list[str]:
        """Ordered list of the methods notified during the exchange (flow debugging)."""
        return [n["method"] for n in self.notifications]


class AcpTestClient:
    """Minimal ACP client (client side, stdio transport) for integration tests.

    It answers server→client requests: ``session/request_permission`` decisions are
    taken from ``permission_decisions`` (``allow_once`` | ``allow_always`` |
    ``reject_once`` | ``cancelled``); when the list runs out it defaults to
    ``allow_once`` so tool e2es stay readable. Any other server request is answered
    with a JSON-RPC "not implemented" error.
    """

    def __init__(self, process: asyncio.subprocess.Process,
                 permission_decisions: list[str] | None = None) -> None:
        self._proc = process
        self._next_id = 0
        self._decisions = list(permission_decisions or [])

    @classmethod
    async def launch(cls, permission_decisions: list[str] | None = None) -> AcpTestClient:
        """Launches the ``python -m selenita --acp`` server with stdio pipes."""
        return await cls.launch_env(
            # Echo mode is forced so a .env in the repo cannot flip the agent mode to "llm".
            {"SELENITA_LOG_LEVEL": "error", "SELENITA_AGENT_MODE": "echo"},
            permission_decisions,
        )

    @classmethod
    async def launch_env(cls, env: dict, permission_decisions: list[str] | None = None,
                         stderr_file: str | None = None) -> AcpTestClient:
        """Launches ``python -m selenita --acp`` with a custom environment (LLM stub e2es)."""
        # The file handle is closed with the subprocess transport (short-lived test file).
        stderr: object = open(stderr_file, "wb") if stderr_file else asyncio.subprocess.DEVNULL  # noqa: SIM115
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "selenita",
            "--acp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr,
            env={**os.environ, **env},
        )
        return cls(process, permission_decisions)

    async def exchange(self, method: str, params: dict | None = None) -> RpcExchange:
        """Sends a request and returns its response along with the received notifications."""
        self._next_id += 1
        request_id = self._next_id
        request: dict = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            request["params"] = params

        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(request, separators=(",", ":")).encode() + b"\n")
        await self._proc.stdin.drain()

        notifications: list[dict] = []
        permission_requests: list[dict] = []
        while True:
            line = await asyncio.wait_for(self._proc.stdout.readline(), TEST_TIMEOUT)
            line = line.strip()
            if not line:
                raise AssertionError("the server closed stdout before answering")
            data = json.loads(line)
            if data.get("id") == request_id and ("result" in data or "error" in data):
                if "error" in data:
                    raise AssertionError(f"response with ACP error: {data['error']}")
                return RpcExchange(result=data["result"],
                                   notifications=notifications,
                                   permission_requests=permission_requests)
            if "id" in data and "method" in data:
                # Server→client request: answer it before reading the next line.
                permission_requests.append(data)
                await self._answer_server_request(data)
                continue
            if "method" in data:
                notifications.append(data)

    async def _answer_server_request(self, data: dict) -> None:
        """Builds and writes the client-side response of a server→client request."""
        request_id = data["id"]
        method = data["method"]
        if method == "session/request_permission":
            decision = self._decisions.pop(0) if self._decisions else "allow_once"
            if decision == "cancelled":
                outcome: dict = {"outcome": "cancelled"}
            else:
                outcome = {"outcome": "selected", "optionId": decision}
            await self._send({"jsonrpc": "2.0", "id": request_id,
                              "result": {"outcome": outcome}})
        else:
            error = {"code": -32601, "message": "not implemented in test client"}
            await self._send({"jsonrpc": "2.0", "id": request_id, "error": error})

    async def _send(self, message: dict) -> None:
        """Writes one JSON-RPC message line to the server's stdin."""
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(message, separators=(",", ":")).encode() + b"\n")
        await self._proc.stdin.drain()

    # -- shortcuts of the standard ACP flow -------------------------------

    async def initialize(self) -> dict:
        return (await self.exchange("initialize", {"protocolVersion": 1})).result

    async def new_session(self, cwd: str | None = None) -> dict:
        params = {"cwd": cwd or os.getcwd(), "mcpServers": []}
        return (await self.exchange("session/new", params)).result

    async def prompt(self, session_id: str, texts: list[str]) -> RpcExchange:
        prompt_blocks = [{"type": "text", "text": t} for t in texts]
        return await self.exchange(
            "session/prompt", {"sessionId": session_id, "prompt": prompt_blocks}
        )

    async def aclose(self) -> None:
        """Closes stdin so the server ends after processing the last message."""
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        try:
            await asyncio.wait_for(self._proc.wait(), 5)
        except TimeoutError:
            self._proc.kill()
            await self._proc.wait()
