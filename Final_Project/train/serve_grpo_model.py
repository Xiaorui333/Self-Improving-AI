"""Modal web endpoint: serves the GRPO-trained Qwen3.5-0.8B as an
OpenAI-compatible API so AgentFlow can use it as the Planner.

Deploy:
    modal run train/run_train.py --serve

Once deployed, copy the printed URL and set:
    export GRPO_MODEL_URL=https://<your-endpoint-url>/v1

Then run benchmarks with the trained Planner:
    python3.11 benchmarks/run_benchmark.py --model qwen3.5-0.8b-grpo --benchmarks all --sample_size 20
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Modal setup — same volume as training so checkpoint is available
# ---------------------------------------------------------------------------

vol = modal.Volume.from_name("grpo-agentflow-vol", create_if_missing=True)
VOLUME_ROOT = "/vol/flow_grpo_agentflow"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTHONUNBUFFERED": "1"})
    .pip_install("torch", "torchvision")
    .pip_install([
        "transformers",
        "peft",
        "accelerate",
        "fastapi",
        "uvicorn",
        "pillow",
    ])
)

app = modal.App("serve-grpo-model")


# ---------------------------------------------------------------------------
# Model loader (runs once on container start)
# ---------------------------------------------------------------------------

def _load_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_name = "Qwen/Qwen3.5-0.8B"

    # Find best checkpoint from training run
    best_ckpt_marker = os.path.join(VOLUME_ROOT, "best_checkpoint.txt")
    if os.path.exists(best_ckpt_marker):
        with open(best_ckpt_marker) as f:
            ckpt_path = f.read().strip()
        print(f"Loading best checkpoint: {ckpt_path}")
        base = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(base, ckpt_path).merge_and_unload()
    else:
        print(f"No checkpoint found at {best_ckpt_marker}, loading base model")
        model = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(base_name)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# FastAPI ASGI app
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="L40S",
    timeout=3600,
    volumes={VOLUME_ROOT: vol},
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    from fastapi import Body, FastAPI, HTTPException
    from pydantic import BaseModel

    import torch

    fastapi_app = FastAPI(title="AgentFlow GRPO Model API")

    # Load model at startup (once per container)
    model, tokenizer = _load_model()

    # ---- Pydantic schemas (OpenAI-compatible) ----------------------------

    class Message(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str = "qwen3-grpo"
        messages: list[Message]
        temperature: float = 0.0
        max_tokens: int = 2048
        top_p: float = 0.9
        stream: bool = False

    class ChatCompletionChoice(BaseModel):
        index: int = 0
        message: Message
        finish_reason: str = "stop"

    class ChatCompletionUsage(BaseModel):
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0

    class ChatCompletionResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: list[ChatCompletionChoice]
        usage: ChatCompletionUsage

    # ---- Endpoints -------------------------------------------------------

    @fastapi_app.get("/health")
    def health():
        return {"status": "ok", "model": "qwen3.5-0.8b-grpo"}

    @fastapi_app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": "qwen3-grpo", "object": "model"}],
        }

    def _do_chat_completion_sync(body: dict[str, Any]) -> ChatCompletionResponse:
        msgs_raw = body.get("messages") or []
        msgs = [{"role": m["role"], "content": m.get("content") or ""} for m in msgs_raw]
        temperature = float(body.get("temperature", 0.0))
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 2048)
        top_p = float(body.get("top_p", 0.9))
        do_sample = temperature > 0.0

        prompt_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": min(max_tokens, 2048),
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        completion_ids = out[0][prompt_len:]
        content = tokenizer.decode(completion_ids, skip_special_tokens=True)
        n_prompt = prompt_len
        n_completion = len(completion_ids)

        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time()*1000)}",
            created=int(time.time()),
            model="qwen3-grpo",
            choices=[
                ChatCompletionChoice(
                    message=Message(role="assistant", content=content),
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=n_prompt,
                completion_tokens=n_completion,
                total_tokens=n_prompt + n_completion,
            ),
        )

    # JSON body — use ``Body(...)`` so Modal/FastAPI does not treat params as query strings.
    @fastapi_app.post("/chat/completions")
    def chat_completions_openai_compat(body: dict = Body(...)):
        try:
            return _do_chat_completion_sync(body)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @fastapi_app.post("/v1/chat/completions")
    def chat_completions_v1(body: dict = Body(...)):
        try:
            return _do_chat_completion_sync(body)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return fastapi_app
