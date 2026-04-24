"""Portkey-backed LLM engine (OpenAI-compatible chat completions)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from portkey_ai import Portkey


def _assistant_text(msg: Any) -> str:
    """Return assistant text; Qwen3.x on DeepInfra may put output in ``reasoning_content``."""
    if msg is None:
        return ""
    content = getattr(msg, "content", None) or ""
    if isinstance(content, str) and content.strip():
        return content
    reasoning = getattr(msg, "reasoning_content", None) or ""
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return content if isinstance(content, str) else str(content or "")


@dataclass
class PortkeyEngine:
    """Thin wrapper around the Portkey Python SDK.

    Accepts either a plain string prompt or a list of chat messages and
    returns the assistant's text reply.  The ``response_format`` parameter
    is accepted for API compatibility with the original AgentFlow codebase
    but is **ignored** -- we always return plain text and let the caller
    parse structured fields via regex (which is what the original repo does).
    """

    model: str
    api_key: str = field(default_factory=lambda: os.environ.get("PORTKEY_API_KEY", ""))
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 0.9

    def __post_init__(self):
        self.client = Portkey(api_key=self.api_key)

    def __call__(
        self,
        prompt: str | list[dict[str, str]],
        *,
        response_format: Any = None,
        **kwargs: Any,
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
            messages = [{"role": "user", "content": prompt[0]}]
        else:
            messages = prompt

        create_kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", self.top_p),
        )

        resp = self.client.chat.completions.create(**create_kwargs)
        return _assistant_text(resp.choices[0].message)
