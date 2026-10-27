"""Fake OpenAI ``/responses`` server (SSE) for deterministic LLM e2e tests.

It reproduces the wire format that MAF's Responses client consumes
(`agent_framework_openai.OpenAIChatClient` streams ``POST /responses`` and parses
``response.output_item.*`` events), so integration tests can drive real tool loops
— including permission flows — without a live model.

Turns are queued: each scripted turn becomes one model response (a function call or
an assistant message). When the queue runs dry the server answers with a plain
fallback message, making unexpected tool loops visible in the transcript.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from openai.types.responses import Response

__all__ = ["FakeResponsesServer", "FunctionCallTurn", "MessageTurn"]


@dataclass
class MessageTurn:
    """One model response: an assistant text message."""

    text: str


@dataclass
class FunctionCallTurn:
    """One model response: the model asks to run one tool."""

    name: str
    arguments: str
    call_id: str = ""


@dataclass
class _TurnState:
    """Runtime state while one scripted turn is being streamed."""

    response_id: str = ""
    items: list[dict] = field(default_factory=list)


class FakeResponsesServer:
    """Minimal HTTP server serving scripted ``POST /responses`` SSE streams."""

    def __init__(self) -> None:
        self.turns: list[MessageTurn | FunctionCallTurn] = []
        self.requests: list[dict] = []  # parsed request bodies, for assertions
        self._server: asyncio.Server | None = None
        self._seq = 0
        self._call_seq = 0
        self._resp_seq = 0

    @property
    def base_url(self) -> str:
        assert self._server is not None, "server not started"
        host, port = self._server.sockets[0].getsockname()[:2]
        return f"http://{host}:{int(port)}/v1"

    def queue(self, *turns: MessageTurn | FunctionCallTurn) -> None:
        self.turns.extend(turns)

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    # ------------------------------------------------------------------
    # HTTP handling
    # ------------------------------------------------------------------

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            content_length = 0
            while True:
                header = await reader.readline()
                if not header or header in (b"\r\n", b"\n"):
                    break
                if header.startswith(b"Content-Length:"):
                    content_length = int(header.split(b":")[1].strip())
            body = await reader.readexactly(content_length) if content_length else b""
            if body:
                self.requests.append(json.loads(body))
                payload = self._sse_response(self.requests[-1])
            else:
                payload = self._sse_response({})
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Cache-Control: no-store\r\n"
                b"Connection: close\r\n\r\n"
            )
            writer.write(payload)
            await writer.drain()
        finally:
            writer.close()

    def _sse_response(self, request: dict) -> bytes:
        """Builds the full SSE document of one scripted turn."""
        events = [self._response_event("response.created", self._response(None))]
        events.append(self._response_event("response.in_progress", self._response(None)))
        turn = self._next_turn(request)
        if isinstance(turn, FunctionCallTurn):
            events.extend(self._function_events(turn))
        else:
            events.extend(self._message_events(turn))
        events.append(self._response_event("response.completed", self._response(turn)))
        return b"".join(
            f"event: {ev['type']}\ndata: {json.dumps(ev, separators=(',', ':'))}\n\n".encode()
            for ev in events
        )

    def _next_turn(self, request: dict) -> MessageTurn | FunctionCallTurn:
        if not self.turns:
            return MessageTurn("[stub] unexpected extra model call (queued turns exhausted)")
        return self.turns.pop(0)

    def _function_events(self, turn: FunctionCallTurn) -> list[dict]:
        self._call_seq += 1
        item_id = f"fc_{self._call_seq}"
        call_id = turn.call_id or f"call_{self._call_seq}"
        item = {
            "id": item_id,
            "type": "function_call",
            "call_id": call_id,
            "name": turn.name,
            "arguments": turn.arguments,
            "status": "completed",
        }
        return [
            self._event("response.output_item.added", {
                "output_index": 0,
                "item": {**item, "arguments": "", "status": "in_progress"},
            }),
            self._event("response.function_call_arguments.delta", {
                "item_id": item_id, "output_index": 0, "delta": turn.arguments,
            }),
            self._event("response.function_call_arguments.done", {
                "item_id": item_id, "output_index": 0, "arguments": turn.arguments,
            }),
            self._event("response.output_item.done", {"output_index": 0, "item": item}),
        ]

    def _message_events(self, turn: MessageTurn) -> list[dict]:
        self._call_seq += 1
        item_id = f"msg_{self._call_seq}"
        item = {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": turn.text, "annotations": []}],
        }
        return [
            self._event("response.output_item.added", {
                "output_index": 0,
                "item": {**item, "status": "in_progress", "content": []},
            }),
            self._event("response.output_text.delta", {
                "item_id": item_id, "output_index": 0, "content_index": 0,
                "delta": turn.text,
            }),
            self._event("response.output_text.done", {
                "item_id": item_id, "output_index": 0, "content_index": 0,
                "text": turn.text,
            }),
            self._event("response.output_item.done", {"output_index": 0, "item": item}),
        ]

    # ------------------------------------------------------------------
    # Event builders (validated with the openai SDK shapes)
    # ------------------------------------------------------------------

    def _event(self, ev_type: str, extra: dict) -> dict:
        self._seq += 1
        return {"type": ev_type, "sequence_number": self._seq, **extra}

    def _response_event(self, ev_type: str, response: dict) -> dict:
        return self._event(ev_type, {"response": response})

    def _response(self, turn: MessageTurn | FunctionCallTurn | None) -> dict:
        self._resp_seq += 1
        if isinstance(turn, MessageTurn):
            output = [{
                "id": f"msg_{self._resp_seq}", "type": "message", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": turn.text, "annotations": []}],
            }]
        elif isinstance(turn, FunctionCallTurn):
            output = [{
                "id": f"fc_{self._resp_seq}", "type": "function_call",
                "call_id": turn.call_id or f"call_{self._resp_seq}", "name": turn.name,
                "arguments": turn.arguments, "status": "completed",
            }]
        else:
            output = []
        response = {
            "id": f"resp_{self._resp_seq}",
            "object": "response",
            "created_at": int(time.time()),
            "model": "stub",
            "status": "completed",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
        Response.model_validate(response)  # fail fast on shape drift
        return response
