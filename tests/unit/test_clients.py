"""Tests of the custom chat-client subclass (D014: stateless Responses + plain-text reasoning)."""

from __future__ import annotations

import agent_framework as af

from selenita.core.clients import PlaintextReasoningClient


def _client() -> PlaintextReasoningClient:
    return PlaintextReasoningClient(
        model="default", api_key="sk-test", base_url="http://localhost:1"
    )


def test_reasoning_item_rebuilt_from_plain_text() -> None:
    client = _client()
    contents = [
        af.Content(type="text_reasoning", text="first part", merge_mode="append"),
        af.Content(type="text_reasoning", text="second part", merge_mode="append"),
    ]
    item = client._prepare_provider_reasoning_item_for_openai("rs_tmp_123", contents)
    assert item is not None
    assert item["type"] == "reasoning"
    assert item["id"] == "rs_tmp_123"
    assert item["summary"] == [{"type": "summary_text", "text": "first part second part"}]
    assert item["content"] == [{"type": "reasoning_text", "text": "first part second part"}]


def test_reasoning_item_truncated_to_summary_limit() -> None:
    client = _client()
    contents = [af.Content(type="text_reasoning", text="x" * 500, merge_mode="append")]
    item = client._prepare_provider_reasoning_item_for_openai("rs_tmp_1", contents)
    assert item is not None
    assert item["content"] == [{"type": "reasoning_text", "text": "x" * 200}]


def test_reasoning_item_none_when_no_plain_text() -> None:
    client = _client()
    assert client._prepare_provider_reasoning_item_for_openai("rs_tmp_1", []) is None
    contents = [af.Content(type="function_call", name="f", arguments="{}")]
    assert client._prepare_provider_reasoning_item_for_openai("rs_tmp_1", contents) is None


def test_is_openai_chat_client_subclass() -> None:
    from agent_framework.openai import OpenAIChatClient

    assert isinstance(_client(), OpenAIChatClient)
