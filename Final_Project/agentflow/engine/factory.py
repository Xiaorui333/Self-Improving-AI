"""Factory for creating LLM engine instances."""

from __future__ import annotations

import os
from typing import Any, Union

from .local_engine import LocalEngine
from .portkey_engine import PortkeyEngine

_DEFAULT_API_KEY = os.environ.get("PORTKEY_API_KEY", "")

# Prefix used by config.py to mark locally-served models.
_LOCAL_PREFIX = "local://"


def create_llm_engine(
    model_string: str,
    *,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> Union[PortkeyEngine, LocalEngine]:
    """Create an LLM engine for the given model string.

    * ``local://<url>`` → LocalEngine (OpenAI-compatible self-hosted endpoint,
      used for the GRPO-trained Qwen3.5-0.8B model served on Modal).
    * anything else → PortkeyEngine (cloud API via Portkey proxy).
    """
    if model_string.startswith(_LOCAL_PREFIX):
        base_url = model_string[len(_LOCAL_PREFIX):]
        return LocalEngine(
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return PortkeyEngine(
        model=model_string,
        api_key=api_key or _DEFAULT_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )
