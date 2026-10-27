"""Shared test fakes: network-free MAF chat clients (D008/D002).

``FakeChatClient`` implements the minimal protocol that ``Agent`` consumes
(``get_response`` + ``additional_properties``), with scheduled replies and a record of
the received messages, so the LLM core wiring is tested without touching the network.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence

import agent_framework as af


class FakeChatClient:
    """Fake chat client: returns the scheduled response (text or callable)."""

    def __init__(
        self,
        replay: str | Callable[[Sequence[af.Message]], str],
        *,
        streaming_chunks: list[str] | None = None,
        replace_replay: str | None = None,
        reasoning_chunks: list[str] | None = None,
        function_call: af.Content | None = None,
    ) -> None:
        self.last_messages: list[af.Message] = []
        self.last_stream = False
        self.last_options: object = None
        self._replay = replay
        self.streaming_chunks = streaming_chunks
        # Scheduled reasoning pieces streamed BEFORE the answer text (streaming only).
        self.reasoning_chunks = reasoning_chunks
        # If set, after the deltas an update is emitted with the whole text
        # (merge_mode="replace"), like the Responses client does with output_text.done.
        self.replace_replay = replace_replay
        # When set, the FIRST request streams this function_call content; only after
        # the messages carry a function_result for it (the MAF loop re-request) does
        # the fake reply with text. Mirrors the Responses client single-shot loop.
        self.function_call = function_call

    @property
    def additional_properties(self) -> dict[str, object]:  # part of the MAF protocol
        return {}

    def get_response(
        self,
        messages: Sequence[af.Message],
        *,
        stream: bool = False,
        options: object = None,
        **kwargs: object,
    ) -> af.ChatResponse | af.ResponseStream:
        # MAF consumes get_response synchronously: in streaming it returns a
        # ResponseStream ready to iterate (RawAgent does not await it).
        self.last_messages = list(messages)
        self.last_stream = stream
        self.last_options = options
        if stream:
            # Tool loop schedule: request #1 carries a function_call, later
            # requests (already containing a function_result) get plain text.
            if self.function_call is not None and not self._has_function_result(messages):
                stream = self._stream_call()
            else:
                stream = self._stream(
                    self.streaming_chunks
                    if self.streaming_chunks is not None
                    else [self._text_of(messages)]
                )
            return af.ResponseStream(stream, finalizer=_aggregate_chat_response)
        return af.ChatResponse(
            messages=[af.Message(role="assistant", contents=[self._text_of(messages)])]
        )

    @staticmethod
    def _has_function_result(messages: Sequence[af.Message]) -> bool:
        """True when the conversation already carries a function_result content."""
        for message in messages:
            for content in message.contents or []:
                if getattr(content, "type", None) == "function_result":
                    return True
        return False

    async def _stream_call(self) -> AsyncIterator[af.ChatResponseUpdate]:
        """Stream a single function_call update (like the Responses client does)."""
        yield af.ChatResponseUpdate(
            contents=[self.function_call], role="assistant"
        )

    def _text_of(self, messages: Sequence[af.Message]) -> str:
        """Reply following the fake's schedule (a callable receives the history)."""
        if callable(self._replay):
            return self._replay(messages)
        return self._replay

    async def _stream(self, chunks: list[str]) -> AsyncIterator[af.ChatResponseUpdate]:
        for reasoning in self.reasoning_chunks or []:
            yield af.ChatResponseUpdate(
                contents=[af.Content(type="text_reasoning", text=reasoning)],
                role="assistant",
            )
        for chunk in chunks:
            yield af.ChatResponseUpdate(
                contents=[af.Content(type="text", text=chunk)], role="assistant"
            )
        if self.replace_replay is not None:
            yield af.ChatResponseUpdate(
                contents=[
                    af.Content(
                        type="text", text=self.replace_replay, merge_mode="replace"
                    )
                ],
                role="assistant",
            )


def _aggregate_chat_response(updates: Sequence[af.ChatResponseUpdate]) -> af.ChatResponse:
    """Finalizer mirroring the real Responses client ``output_text.done`` merge.

    Accumulates streamed contents into a single assistant message: text deltas merge
    with a trailing ``merge_mode="replace"`` chunk replacing them, while function_call
    (and other) contents are kept verbatim. Without this the MAF function-invocation
    layer (``FunctionInvocationLayer``) cannot work, because it asks the client's
    stream for the aggregated ``ChatResponse`` before processing tool calls.
    """
    contents: list[af.Content] = []
    for update in updates:
        for content in update.contents or []:
            if content.type == "text" and content.merge_mode == "replace":
                contents = [c for c in contents if c.type != "text"]
            contents.append(content)
    messages: list[af.Message] = []
    if contents:
        messages.append(af.Message(role="assistant", contents=contents))
    return af.ChatResponse(messages=messages, usage_details=None)


class ToolLoopClient(af.FunctionInvocationLayer, FakeChatClient):
    """``FakeChatClient`` wrapped in MAF's function-invocation layer.

    This reproduces the layering of the real OpenAI Responses client
    (``FunctionInvocationLayer`` + raw client) so unit tests can exercise the whole
    tool loop without a network: the model emits a ``function_call``, MAF routes it
    through function middleware (our permission gate), runs the function, streams
    back a ``function_result``, re-requests the model, and returns the final text.
    """
