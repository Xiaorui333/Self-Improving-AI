from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from .db import transaction


@dataclass
class GenerationConfig:
    run_name: str
    provider: str
    model_name: str
    temperature: float = 0.2
    top_p: float | None = 0.95
    max_new_tokens: int = 512
    seed: int | None = 42


def create_run(conn: sqlite3.Connection, config: GenerationConfig) -> int:
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO generation_runs(
                run_name, provider, model_name, temperature, top_p, max_new_tokens, seed, config_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                config.run_name,
                config.provider,
                config.model_name,
                config.temperature,
                config.top_p,
                config.max_new_tokens,
                config.seed,
                json.dumps({}, ensure_ascii=True),
            ),
        )
    row = conn.execute(
        "SELECT id FROM generation_runs WHERE run_name = ?",
        (config.run_name,),
    ).fetchone()
    assert row is not None
    return int(row["id"])


def fetch_pending_problems(
    conn: sqlite3.Connection,
    run_id: int,
    dataset_key: str | None = None,
    split_name: str | None = None,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    params: list[Any] = [run_id]
    where = [
        "g.id IS NULL",
    ]
    if dataset_key:
        where.append("d.name = ?")
        params.append(dataset_key)
    if split_name:
        where.append("ds.split_name = ?")
        params.append(split_name)
    where_sql = " AND ".join(where)
    limit_sql = f"LIMIT {int(limit)}" if limit else ""

    query = f"""
        SELECT
            p.id AS problem_id,
            p.prompt AS prompt,
            p.starter_code AS starter_code,
            p.entry_point AS entry_point,
            p.problem_uid AS problem_uid
        FROM problems p
        JOIN dataset_splits ds ON ds.id = p.dataset_split_id
        JOIN datasets d ON d.id = ds.dataset_id
        LEFT JOIN generations g
            ON g.problem_id = p.id
            AND g.run_id = ?
        WHERE {where_sql}
        ORDER BY p.id ASC
        {limit_sql}
    """
    rows = conn.execute(query, tuple(params)).fetchall()
    return rows


def _compose_prompt(row: sqlite3.Row) -> str:
    prompt = row["prompt"] or ""
    starter = row["starter_code"] or ""
    if starter.strip():
        return f"{prompt}\n\n{starter}"
    return prompt


class TransformersGenerator:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        temperature: float,
        top_p: float | None,
        max_new_tokens: int,
        seed: int | None,
    ) -> dict[str, Any]:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        start = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p if top_p is not None else 1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        completion_ids = outputs[0][inputs["input_ids"].shape[1] :]
        completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        completion_tokens = int(completion_ids.shape[0])
        return {
            "completion": completion,
            "finish_reason": "length" if completion_tokens >= max_new_tokens else "stop",
            "latency_ms": elapsed_ms,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "raw": {},
        }


class OpenAIGenerator:
    def __init__(self, model_name: str) -> None:
        self.client = OpenAI()
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        temperature: float,
        top_p: float | None,
        max_new_tokens: int,
        seed: int | None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": prompt,
            "temperature": temperature,
            "top_p": top_p if top_p is not None else 1.0,
            "max_output_tokens": max_new_tokens,
        }
        if seed is not None:
            request_kwargs["seed"] = seed
        try:
            response = self.client.responses.create(**request_kwargs)
        except TypeError as exc:
            # Older/newer SDK variants may not support seed for responses.create.
            if "seed" not in str(exc):
                raise
            request_kwargs.pop("seed", None)
            response = self.client.responses.create(**request_kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        usage = getattr(response, "usage", None)
        return {
            "completion": getattr(response, "output_text", ""),
            "finish_reason": "stop",
            "latency_ms": elapsed_ms,
            "usage": {
                "prompt_tokens": getattr(usage, "input_tokens", None),
                "completion_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
            "raw": response.model_dump(),
        }


def run_generation(
    conn: sqlite3.Connection,
    config: GenerationConfig,
    dataset_key: str | None = None,
    split_name: str | None = None,
    limit: int | None = None,
) -> int:
    run_id = create_run(conn, config)
    rows = fetch_pending_problems(
        conn=conn,
        run_id=run_id,
        dataset_key=dataset_key,
        split_name=split_name,
        limit=limit,
    )
    if config.provider == "transformers":
        backend = TransformersGenerator(config.model_name)
    elif config.provider == "openai":
        backend = OpenAIGenerator(config.model_name)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")

    inserted = 0
    for row in rows:
        prompt_text = _compose_prompt(row)
        result = backend.generate(
            prompt=prompt_text,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_new_tokens,
            seed=config.seed,
        )
        with transaction(conn):
            conn.execute(
                """
                INSERT INTO generations(
                    run_id,
                    problem_id,
                    prompt_text,
                    completion_text,
                    finish_reason,
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    model_response_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(row["problem_id"]),
                    prompt_text,
                    result["completion"],
                    result["finish_reason"],
                    result["latency_ms"],
                    result["usage"]["prompt_tokens"],
                    result["usage"]["completion_tokens"],
                    result["usage"]["total_tokens"],
                    json.dumps(result["raw"], ensure_ascii=True),
                ),
            )
        inserted += 1
    return inserted

