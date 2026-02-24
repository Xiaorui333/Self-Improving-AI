from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .db import transaction


@dataclass
class KBBuildConfig:
    datasets: tuple[str, ...] = ("apps", "codecontests")
    splits: tuple[str, ...] = ("train",)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    max_steps_per_problem: int = 8
    max_step_chars: int = 400
    limit_problems: int | None = None


def _solution_to_steps(solution: str, max_steps: int, max_chars: int) -> list[str]:
    lines = [ln.rstrip() for ln in solution.splitlines() if ln.strip()]
    if not lines:
        return []
    steps: list[str] = []
    chunk: list[str] = []
    for ln in lines:
        chunk.append(ln)
        is_break = ln.strip().startswith(("def ", "class ", "for ", "while ", "if ", "try:", "except "))
        if len(chunk) >= 8 or is_break:
            step = "\n".join(chunk).strip()[:max_chars]
            if step:
                steps.append(step)
            chunk = []
        if len(steps) >= max_steps:
            break
    if chunk and len(steps) < max_steps:
        step = "\n".join(chunk).strip()[:max_chars]
        if step:
            steps.append(step)
    return steps


class TextEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        all_vecs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            toks = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            out = self.model(**toks)
            hidden = out.last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            vec = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_vecs.append(vec.cpu().numpy().astype(np.float32))
        if not all_vecs:
            return np.zeros((0, 384), dtype=np.float32)
        return np.concatenate(all_vecs, axis=0)


def build_kb_steps(conn: sqlite3.Connection, cfg: KBBuildConfig) -> dict[str, int]:
    where_dataset = ",".join("?" for _ in cfg.datasets)
    where_split = ",".join("?" for _ in cfg.splits)
    limit_sql = f"LIMIT {int(cfg.limit_problems)}" if cfg.limit_problems else ""
    rows = conn.execute(
        f"""
        SELECT
            d.name AS dataset_name,
            ds.split_name AS split_name,
            p.problem_uid,
            p.prompt,
            p.canonical_solution
        FROM problems p
        JOIN dataset_splits ds ON ds.id = p.dataset_split_id
        JOIN datasets d ON d.id = ds.dataset_id
        WHERE d.name IN ({where_dataset})
          AND ds.split_name IN ({where_split})
          AND p.canonical_solution IS NOT NULL
          AND TRIM(p.canonical_solution) != ''
        ORDER BY p.id ASC
        {limit_sql}
        """,
        (*cfg.datasets, *cfg.splits),
    ).fetchall()

    items: list[tuple[str, str, str, int, str, str]] = []
    for row in rows:
        solution = str(row["canonical_solution"])
        steps = _solution_to_steps(solution, cfg.max_steps_per_problem, cfg.max_step_chars)
        for step_idx, step_text in enumerate(steps):
            items.append(
                (
                    str(row["dataset_name"]),
                    str(row["split_name"]),
                    str(row["problem_uid"]),
                    step_idx,
                    step_text,
                    json.dumps({"prompt_head": str(row["prompt"])[:200]}, ensure_ascii=True),
                )
            )

    with transaction(conn):
        conn.execute(
            "DELETE FROM kb_steps WHERE source_dataset IN ({})".format(where_dataset),
            cfg.datasets,
        )
        conn.executemany(
            """
            INSERT INTO kb_steps(
                source_dataset,
                source_split,
                problem_uid,
                step_index,
                step_text,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            items,
        )

    inserted = len(items)
    if inserted == 0:
        return {"steps": 0, "embedded": 0}

    embedder = TextEmbedder(cfg.embedding_model)
    texts = [it[4] for it in items]
    vecs = embedder.encode(texts, batch_size=cfg.batch_size)
    if vecs.shape[0] != inserted:
        raise RuntimeError("Embedding row count mismatch")

    step_rows = conn.execute(
        """
        SELECT id FROM kb_steps
        WHERE source_dataset IN ({})
        ORDER BY id ASC
        """.format(where_dataset),
        cfg.datasets,
    ).fetchall()
    ids = [int(r["id"]) for r in step_rows][-inserted:]
    with transaction(conn):
        for idx, step_id in enumerate(ids):
            vec = vecs[idx]
            conn.execute(
                """
                UPDATE kb_steps
                SET embedding_model = ?, embedding_blob = ?, embedding_dim = ?
                WHERE id = ?
                """,
                (cfg.embedding_model, vec.tobytes(), int(vec.shape[0]), step_id),
            )
    return {"steps": inserted, "embedded": inserted}


class KBRetriever:
    def __init__(self, conn: sqlite3.Connection, embedding_model: str) -> None:
        self.conn = conn
        self.embedder = TextEmbedder(embedding_model)
        rows = conn.execute(
            """
            SELECT id, step_text, embedding_blob, embedding_dim
            FROM kb_steps
            WHERE embedding_model = ? AND embedding_blob IS NOT NULL
            """,
            (embedding_model,),
        ).fetchall()
        self.step_ids: list[int] = []
        self.step_texts: list[str] = []
        vectors: list[np.ndarray] = []
        for r in rows:
            dim = int(r["embedding_dim"] or 0)
            blob = r["embedding_blob"]
            if not blob or dim <= 0:
                continue
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0] != dim:
                continue
            self.step_ids.append(int(r["id"]))
            self.step_texts.append(str(r["step_text"]))
            vectors.append(vec)
        self.matrix = np.stack(vectors, axis=0) if vectors else np.zeros((0, 1), dtype=np.float32)

    def max_similarity(self, text: str, top_k: int = 5) -> tuple[float, list[str]]:
        if self.matrix.shape[0] == 0:
            return 0.0, []
        q = self.embedder.encode([text])[0]
        sims = self.matrix @ q
        idx = np.argsort(-sims)[: max(1, top_k)]
        top_sims = [float(sims[i]) for i in idx]
        top_texts = [self.step_texts[i] for i in idx]
        return (max(top_sims) if top_sims else 0.0), top_texts

