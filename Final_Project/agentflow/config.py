"""Centralised configuration constants for the AgentFlow project."""

from __future__ import annotations

import os

PORTKEY_API_KEY = "vz5IftR67A0EnNT579WHyhd0Wlb9"

# Origin of the Modal-served GRPO model (no ``/v1`` — LocalEngine POSTs to /chat/completions).
#   export GRPO_MODEL_URL=https://fridaliu2024--serve-grpo-model-web.modal.run
_GRPO_URL = os.environ.get(
    "GRPO_MODEL_URL",
    "http://localhost:8000",   # fallback: local server with POST /chat/completions
)

MODELS = {
    "qwen2.5-7b": "@deepinfra/Qwen/Qwen2.5-7B-Instruct",
    "qwen3.5-0.8b": "@deepinfra/Qwen/Qwen3.5-0.8B",
    "qwen3.5-2b": "@deepinfra/Qwen/Qwen3.5-2B",
    "qwen3.5-4b": "@deepinfra/Qwen/Qwen3.5-4B",
    "qwen3.5-9b": "@deepinfra/Qwen/Qwen3.5-9B",
    "qwen3.5-27b": "@deepinfra/Qwen/Qwen3.5-27B",
    # GRPO-trained Qwen3.5-0.8B served on Modal (Phase 6 full pipeline)
    "qwen3.5-0.8b-grpo": f"local://{_GRPO_URL}",
}

BENCHMARKS = [
    "bamboogle",
    "2wiki",
    "hotpotqa",
    "musique",
    "gaia",
    "aime24",
    "amc23",
    "gameof24",
    "gpqa",
    "medqa",
]

DEFAULT_MAX_STEPS = 10
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096
