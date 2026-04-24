"""LocalEngine — calls any OpenAI-compatible HTTP endpoint.

Used to wire the GRPO-trained Qwen3.5-0.8B model (served on Modal) into the
full AgentFlow pipeline as a drop-in replacement for PortkeyEngine.

Model string convention: ``local://<base_url>``

``base_url`` is the **origin only** (no ``/v1``): the engine POSTs to
``<base_url>/chat/completions``. This avoids Modal edge issues with ``/v1/*``
paths when using the OpenAI Python SDK.

Example: ``local://https://fridaliu2024--serve-grpo-model-web.modal.run``
"""

from __future__ import annotations

from typing import Any


class LocalEngine:
    """POST JSON to ``{base_url}/chat/completions`` (OpenAI response shape)."""

    def __init__(
        self,
        base_url: str,
        model: str = "qwen3-grpo",
        *,
        api_key: str = "local",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        top_p: float = 0.9,
    ) -> None:
        try:
            import httpx  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "httpx package required for LocalEngine: pip install httpx"
            ) from exc

        self._httpx = httpx
        raw = base_url.strip()
        if raw.startswith("local://"):
            raw = raw[len("local://") :]
        # Accept legacy ``.../v1`` — strip so we always hit ``/chat/completions``.
        self._base = raw.rstrip("/").removesuffix("/v1")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    # ------------------------------------------------------------------
    def __call__(
        self,
        prompt: str | list[dict[str, str]],
        *,
        response_format: Any = None,
        **kwargs: Any,
    ) -> str:
        if isinstance(prompt, str):
            messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
            messages = [{"role": "user", "content": prompt[0]}]
        else:
            messages = prompt  # type: ignore[assignment]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
        }
        url = f"{self._base}/chat/completions"
        with self._httpx.Client(timeout=600.0) as client:
            r = client.post(url, json=payload, headers=self._headers)
            r.raise_for_status()
            data = r.json()
        try:
            return (data["choices"][0]["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Bad chat response: {data!r}") from exc
