"""OpenAI Responses client tweaks for stateless providers (D014).

selenita always runs the MAF Responses client in ``store: false`` mode — full
inline history, no ``previous_response_id`` — so it works against any gateway,
including the openrouter route that rejects response-id continuation (and the
default route that returns empty text after tool loops).

MAF validates stateless replay by trying to rebuild every reasoning item that
precedes a tool call (Responses requires reasoning to be re-sent inline).
Proxies that strip ``encrypted_content`` and return plain-text reasoning with
temporary ids (e.g. the AI gateway on the openrouter route) can't be rebuilt by
the base class, which makes MAF raise ``ChatClientInvalidRequestException`` on
the tool-loop step. ``PlaintextReasoningClient`` implements the official
extension hook ``_prepare_provider_reasoning_item_for_openai`` to reconstruct
those items from the plain text MAF already holds in memory — the model keeps
reasoning, and the tool loop completes.
"""

from __future__ import annotations

from collections.abc import Sequence

import agent_framework as af
from agent_framework.openai import OpenAIChatClient

__all__ = ["PlaintextReasoningClient"]


class PlaintextReasoningClient(OpenAIChatClient):
    """OpenAI Responses client that can replay plain-text reasoning items.

    The base client only rebuilds reasoning when the provider returned
    ``encrypted_content``. This subclass fills the provider hook with the plain
    text it already stored, producing a valid Responses ``reasoning`` item, so
    stateless tool loops keep working on proxies that don't encrypt reasoning.
    """

    def _prepare_provider_reasoning_item_for_openai(
        self,
        reasoning_id: str,
        contents: Sequence[af.Content],
    ) -> dict[str, object] | None:
        parts: list[str] = []
        for content in contents:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                parts.append(text)
        text = " ".join(parts)
        if not text:
            return None
        return {
            "type": "reasoning",
            "id": reasoning_id,
            "summary": [{"type": "summary_text", "text": text[:200]}],
            "content": [{"type": "reasoning_text", "text": text[:200]}],
        }
