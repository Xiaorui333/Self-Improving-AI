from __future__ import annotations

import json
import math
import random
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .evaluate import execute_problem, sanitize_humaneval_candidate
from .generate import OpenAIGenerator, TransformersGenerator
from .kb import KBRetriever


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RPMMCTSConfig:
    run_name: str
    provider: str
    model_name: str
    dataset_key: str
    split_name: str
    limit: int | None = None
    offset: int | None = None
    temperature: float = 0.2
    top_p: float = 0.95
    step_max_new_tokens: int = 128
    max_depth: int = 6
    # Paper hyperparameters
    num_simulations: int = 5       # rollout iterations
    branching_factor: int = 3      # b
    ucb_beta: float = 0.5          # β
    alpha_select: float = 0.5      # α (KB weight in SelectionScore)
    gamma_backup: float = 0.5      # γ (exec weight in Q-value backup)
    theta_succ: float = 0.95       # success threshold for early exit
    similarity_threshold: float = 0.85  # cosine sim pruning threshold
    reflection_iters: int = 2      # max reflection / truncation rounds
    timeout_sec: int = 10
    seed: int = 42
    kb_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Node
# Paper: root = problem description; all other nodes = single algorithmic step.
# Stored fields align with paper's N(s), N(s,a), Q(s,a) semantics.
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """
    A node in the MCTS tree.

    text        : cumulative state s = problem + all steps from root to here
    action_text : the single new step a this node adds over its parent
    kb_reward   : K(s,a) — cosine sim of (s,a) against KB
    visits      : N(s,a)
    value_sum   : sum of rewards seen through this node (for computing Q)
    is_pruned   : True if filtered out by similarity pruning (kept in DB for analysis)
    """
    text: str
    depth: int
    kb_reward: float
    parent: "Node | None" = None
    visits: int = 0
    value_sum: float = 0.0
    children: list["Node"] = field(default_factory=list)
    last_feedback: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    action_text: str = ""
    is_pruned: bool = False

    def q(self) -> float:
        """Q(s,a) = empirical mean cumulative reward."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _persist_node(conn: sqlite3.Connection, node: Node, run_id: int, problem_id: int) -> None:
    """Upsert a node into mcts_nodes (insert on first visit, update stats on backprop)."""
    conn.execute(
        """
        INSERT INTO mcts_nodes(
            id, run_id, problem_id, parent_id,
            state_text, action_text,
            visits, q_value, kb_reward,
            is_terminal, is_pruned, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, '{}')
        ON CONFLICT(id) DO UPDATE SET
            visits    = excluded.visits,
            q_value   = excluded.q_value,
            is_pruned = excluded.is_pruned
        """,
        (
            node.id,
            run_id,
            problem_id,
            node.parent.id if node.parent else None,
            node.text,
            node.action_text,
            node.visits,
            node.q(),
            node.kb_reward,
            int(node.is_pruned),
        ),
    )


def _update_node_db(conn: sqlite3.Connection, node: Node) -> None:
    """Update just visits and Q-value after backpropagation."""
    conn.execute(
        "UPDATE mcts_nodes SET visits = ?, q_value = ? WHERE id = ?",
        (node.visits, node.q(), node.id),
    )


# ---------------------------------------------------------------------------
# DB setup helpers
# ---------------------------------------------------------------------------

def _create_run(conn: sqlite3.Connection, cfg: RPMMCTSConfig) -> int:
    conn.execute(
        """
        INSERT INTO generation_runs(
            run_name, provider, model_name, temperature, top_p, max_new_tokens, seed, config_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cfg.run_name,
            "rpm_mcts",
            cfg.model_name,
            cfg.temperature,
            cfg.top_p,
            cfg.step_max_new_tokens,
            cfg.seed,
            json.dumps(
                {
                    "branching_factor": cfg.branching_factor,
                    "num_simulations": cfg.num_simulations,
                    "max_depth": cfg.max_depth,
                    "ucb_beta": cfg.ucb_beta,
                    "alpha_select": cfg.alpha_select,
                    "gamma_backup": cfg.gamma_backup,
                    "theta_succ": cfg.theta_succ,
                    "similarity_threshold": cfg.similarity_threshold,
                    "reflection_iters": cfg.reflection_iters,
                },
                ensure_ascii=True,
            ),
        ),
    )
    row = conn.execute(
        "SELECT id FROM generation_runs WHERE run_name = ?", (cfg.run_name,)
    ).fetchone()
    assert row is not None
    return int(row["id"])


def _fetch_problems(conn: sqlite3.Connection, cfg: RPMMCTSConfig) -> list[sqlite3.Row]:
    limit_sql = f"LIMIT {int(cfg.limit)}" if cfg.limit else ""
    offset_sql = f"OFFSET {int(cfg.offset)}" if cfg.offset else ""
    return conn.execute(
        f"""
        SELECT
            p.id AS problem_id,
            p.problem_uid,
            p.prompt,
            p.entry_point,
            p.test_spec_json
        FROM problems p
        JOIN dataset_splits ds ON ds.id = p.dataset_split_id
        JOIN datasets d ON d.id = ds.dataset_id
        WHERE d.name = ? AND ds.split_name = ?
        ORDER BY p.id ASC
        {limit_sql}
        {offset_sql}
        """,
        (cfg.dataset_key, cfg.split_name),
    ).fetchall()


def _make_backend(provider: str, model_name: str) -> Any:
    if provider == "transformers":
        return TransformersGenerator(model_name)
    if provider == "openai":
        return OpenAIGenerator(model_name)
    raise ValueError(f"Unsupported provider: {provider}")


# ---------------------------------------------------------------------------
# Phase 1 – Selection
# Paper §Selection: UCB + α·K(s,a) = SelectionScore
# ---------------------------------------------------------------------------

def _ucb(parent: Node, child: Node, cfg: RPMMCTSConfig) -> float:
    """
    Paper Eq (3)+(4):
        SelectionScore(s,a) = UCB(s,a) + α · K(s,a)
        UCB(s,a)            = Q(s,a) + β · sqrt(log N(s) / (1 + N(s,a)))
    """
    explore = cfg.ucb_beta * math.sqrt(math.log(parent.visits + 1.0) / (child.visits + 1e-6))
    return child.q() + explore + cfg.alpha_select * child.kb_reward


def _select_leaf(root: Node, cfg: RPMMCTSConfig) -> tuple[Node, list[Node]]:
    """Traverse from root to the highest-SelectionScore leaf (paper §Selection)."""
    node = root
    path: list[Node] = [node]
    while node.children and node.depth < cfg.max_depth:
        # Only consider non-pruned children for selection
        candidates = [c for c in node.children if not c.is_pruned]
        if not candidates:
            break
        node = max(candidates, key=lambda c: _ucb(path[-1], c, cfg))
        path.append(node)
    return node, path


# ---------------------------------------------------------------------------
# Phase 2 – Expansion
# Paper §Expansion:
#   - Generate b sibling steps SEQUENTIALLY (each sibling gets prior siblings as context)
#   - Apply cosine-similarity filter; prune near-duplicates
#   - Persist both kept AND pruned nodes in mcts_nodes
# ---------------------------------------------------------------------------

def _similarity_filter(
    candidates: list[tuple[str, str]],   # (action_text, full_state_text)
    retriever: KBRetriever,
    threshold: float,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (kept, pruned) pairs. Filters by cosine sim among action texts."""
    if len(candidates) <= 1:
        return candidates, []
    action_texts = [c[0] for c in candidates]
    embeds = retriever.embedder.encode(action_texts)
    # L2-normalise so dot product == cosine similarity
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeds = embeds / norms
    kept_idx: list[int] = []
    pruned_idx: list[int] = []
    for i in range(len(candidates)):
        keep = True
        for j in kept_idx:
            sim = float(np.dot(embeds[i], embeds[j]))
            if sim >= threshold:
                keep = False
                break
        (kept_idx if keep else pruned_idx).append(i)
    return [candidates[i] for i in kept_idx], [candidates[i] for i in pruned_idx]


def _expand(
    node: Node,
    prompt: str,
    dataset_name: str,
    entry_point: str | None,
    backend: Any,
    retriever: KBRetriever,
    cfg: RPMMCTSConfig,
    rng: random.Random,
    conn: sqlite3.Connection,
    run_id: int,
    problem_id: int,
) -> list[Node]:
    """
    Expansion uses dataset-specific mode:
    - HumanEval(+): code-mode (each child is a full code candidate).
    - Others: step-mode (each child is one next algorithmic step).
    """
    raw_candidates: list[tuple[str, str]] = []  # (action, full_state)
    generated_siblings: list[str] = []           # prior siblings used as context
    is_humaneval = dataset_name.startswith("humaneval")

    for b in range(cfg.branching_factor):
        seed = cfg.seed + rng.randint(0, 10_000) + b

        if is_humaneval and entry_point:
            # Code-mode: instruction goes BEFORE the stub so the model completes
            # the function body directly. Putting instructions after the stub
            # causes completion-mode models to echo them into the function body.
            sibling_ctx = ""
            if generated_siblings:
                sibling_ctx = (
                    "Previously generated alternatives (do not repeat):\n"
                    + "\n---\n".join(generated_siblings)
                    + "\n\n"
                )
            instruction = (
                f"Complete the following Python function. "
                f"Return ONLY the function body for def {entry_point}. "
                "No tests, asserts, main blocks, markdown, or extra functions.\n\n"
            )
            base = instruction + sibling_ctx + prompt
        else:
            output_guardrail = (
                "\n\n[Output format constraint]\n"
                "Return exactly ONE concise next algorithmic step in plain English. "
                "Do not output Python code, markdown fences, tests, or explanations.\n"
            )
            sibling_ctx = ""
            if generated_siblings:
                sibling_ctx = (
                    "\n\n[Previously generated alternative steps (do not repeat):\n"
                    + "\n---\n".join(generated_siblings)
                    + "\n]"
                )
            base = (f"{prompt}\n{node.text}" if node.text else prompt) + output_guardrail + sibling_ctx

        result = backend.generate(
            prompt=base,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.step_max_new_tokens,
            seed=seed,
        )
        action = (result.get("completion") or "").rstrip()
        if action:
            if is_humaneval:
                # Code-mode: each node stores one complete candidate implementation.
                full_state = action
            else:
                # Step-mode: state is cumulative concatenation of steps.
                full_state = (node.text + "\n\n" + action).rstrip() if node.text else action
            raw_candidates.append((action, full_state))
            generated_siblings.append(action)

    # Similarity filtering (paper: cosine-sim pruning at threshold 0.85)
    kept, pruned = _similarity_filter(raw_candidates, retriever, cfg.similarity_threshold)

    # Persist pruned nodes (is_pruned=True; tracked for analysis but not traversed)
    for action, full_state in pruned:
        kb_score, _ = retriever.max_similarity(action, top_k=3)
        pruned_node = Node(
            text=full_state,
            depth=node.depth + 1,
            kb_reward=kb_score,
            parent=node,
            action_text=action,
            is_pruned=True,
        )
        _persist_node(conn, pruned_node, run_id, problem_id)

    # Create, persist, and return kept child nodes
    children: list[Node] = []
    for action, full_state in kept:
        kb_score, _ = retriever.max_similarity(action, top_k=3)
        child = Node(
            text=full_state,
            depth=node.depth + 1,
            kb_reward=kb_score,
            parent=node,
            action_text=action,
            # value_sum starts at 0: KB reward guides selection via UCB (+α·K(s,a)),
            # NOT Q computation. Seeding with kb_score inflated Q and caused _best_leaf
            # to override execution-proven completions with high-KB but unverified nodes.
            value_sum=0.0,
        )
        _persist_node(conn, child, run_id, problem_id)
        children.append(child)

    node.children.extend(children)
    return children


# ---------------------------------------------------------------------------
# Phase 3 – Evaluation & Reflection
# Paper §Evaluation and Reflection:
#   - Run sandbox on current path's code
#   - If fails: locate erroneous step, truncate to good prefix
#   - INJECT verified correct prefix steps back into MCTS tree as new nodes
# ---------------------------------------------------------------------------

def _truncate_to_good_prefix(text: str) -> str:
    """Remove the last algorithmic step (the erroneous one) from the path."""
    blocks = [blk for blk in text.split("\n\n") if blk.strip()]
    if len(blocks) <= 1:
        return ""
    return "\n\n".join(blocks[:-1]).rstrip()


def _reflect_and_score(
    dataset_name: str,
    prompt: str,
    entry_point: str | None,
    test_spec_json: str,
    node: Node,
    parent_node: Node,
    cfg: RPMMCTSConfig,
    retriever: KBRetriever,
    conn: sqlite3.Connection,
    run_id: int,
    problem_id: int,
) -> tuple[str, float, str, list[Node]]:
    """
    Evaluation+reflection mode:
    - HumanEval(+): code-mode execution only (no step truncation/injection).
    - Others: step-mode with truncation + dynamic node injection.

    Returns: (code_text_executed, exec_score, feedback, injected_nodes)
    """
    text = node.text
    feedback = ""
    injected: list[Node] = []
    last_executed_code = text
    is_humaneval = dataset_name.startswith("humaneval")

    # HumanEval(+) code-mode: evaluate code candidates directly.
    if is_humaneval and entry_point:
        code_text, _ = sanitize_humaneval_candidate(
            prompt_text=prompt,
            completion_text=text,
            entry_point=entry_point,
        )
        res = execute_problem(
            dataset_name=dataset_name,
            prompt_text=prompt,
            completion_text=code_text,
            entry_point=entry_point,
            test_spec_json=test_spec_json,
            timeout_sec=cfg.timeout_sec,
        )
        if int(res["passed"]) == 1:
            return code_text, 1.0, "", injected
        feedback = str(res.get("stderr") or "")[:240]
        if "SyntaxError" in feedback or "IndentationError" in feedback:
            return code_text, 0.0, feedback, injected
        return code_text, 0.2, feedback, injected

    for _ in range(cfg.reflection_iters + 1):
        code_text = text
        last_executed_code = code_text

        res = execute_problem(
            dataset_name=dataset_name,
            prompt_text=prompt,
            completion_text=code_text,
            entry_point=entry_point,
            test_spec_json=test_spec_json,
            timeout_sec=cfg.timeout_sec,
        )
        if int(res["passed"]) == 1:
            return code_text, 1.0, "", injected

        stderr = str(res.get("stderr") or "")
        feedback = stderr[:240]

        # Locate erroneous step: truncate the last step (paper's LDB-style step isolation)
        good_prefix = _truncate_to_good_prefix(text)
        if not good_prefix:
            break

        # --- Dynamic Node Injection (paper: "retain verified correct steps in tree") ---
        # The good_prefix is a verified-correct partial path. If it's not already a node,
        # inject it permanently into the tree so future iterations can branch from it.
        if good_prefix != node.text and good_prefix != parent_node.text:
            existing = next((c for c in parent_node.children if c.text == good_prefix), None)
            if existing is None:
                # Derive the action text = the newly-verified step beyond parent
                if parent_node.text and good_prefix.startswith(parent_node.text):
                    # Keep leading indentation; remove separator newlines + trailing whitespace only.
                    injected_action = good_prefix[len(parent_node.text):].lstrip("\n").rstrip()
                else:
                    injected_action = good_prefix.rstrip()

                kb_score, _ = retriever.max_similarity(injected_action, top_k=3)
                injected_node = Node(
                    text=good_prefix,
                    depth=parent_node.depth + 1,
                    kb_reward=kb_score,
                    parent=parent_node,
                    action_text=injected_action,
                    value_sum=0.0,  # no KB seeding; let Q accumulate from actual rewards
                )
                parent_node.children.append(injected_node)
                _persist_node(conn, injected_node, run_id, problem_id)
                injected.append(injected_node)

        text = good_prefix  # retry with truncated path

    # Reward: 0.0 for syntax errors, 0.2 for logic failures (partial credit)
    if "SyntaxError" in feedback or "IndentationError" in feedback:
        return last_executed_code, 0.0, feedback, injected
    return last_executed_code, 0.2, feedback, injected


# ---------------------------------------------------------------------------
# Phase 4 – Backpropagation
# Paper §Backpropagation: Q(s,a) = γ·r_exec + (1-γ)·r_LLM
# We use kb_reward as r_LLM proxy (no separate LLM scorer).
# ---------------------------------------------------------------------------

def _backpropagate(path: list[Node], reward: float, conn: sqlite3.Connection) -> None:
    """Update N(s,a) and Q(s,a) for every node along the path back to root."""
    for p in path:
        p.visits += 1
        p.value_sum += reward
        _update_node_db(conn, p)


# ---------------------------------------------------------------------------
# Main MCTS loop
# ---------------------------------------------------------------------------

def _search_one(
    prompt: str,
    dataset_name: str,
    entry_point: str | None,
    test_spec_json: str,
    backend: Any,
    retriever: KBRetriever,
    cfg: RPMMCTSConfig,
    rng: random.Random,
    conn: sqlite3.Connection,
    run_id: int,
    problem_id: int,
) -> tuple[str, float, list[dict[str, Any]]]:
    """
    Full RPM-MCTS loop for a single problem.
    Persists root node immediately; all subsequent nodes via _expand / _reflect.
    """
    # Root node represents the problem (state s, no steps yet)
    root = Node(text="", depth=0, kb_reward=0.0, action_text="<root>")
    _persist_node(conn, root, run_id, problem_id)

    traces: list[dict[str, Any]] = []
    best_code = ""
    best_exec = -1.0

    for sim_idx in range(cfg.num_simulations):

        # --- Phase 1: Selection ---
        node, path = _select_leaf(root, cfg)

        # --- Phase 2: Expansion ---
        if node.depth < cfg.max_depth:
            children = _expand(
                node,
                prompt,
                dataset_name,
                entry_point,
                backend,
                retriever,
                cfg,
                rng,
                conn,
                run_id,
                problem_id,
            )
            if children:
                # Pick child with highest KB prior for simulation (paper: "K(s,a) guides selection")
                node = max(children, key=lambda c: c.kb_reward)
                path.append(node)

        # --- Phase 3: Evaluation & Reflection ---
        parent_for_injection = path[-2] if len(path) >= 2 else root
        code_text, exec_score, feedback, injected_nodes = _reflect_and_score(
            dataset_name=dataset_name,
            prompt=prompt,
            entry_point=entry_point,
            test_spec_json=test_spec_json,
            node=node,
            parent_node=parent_for_injection,
            cfg=cfg,
            retriever=retriever,
            conn=conn,
            run_id=run_id,
            problem_id=problem_id,
        )

        # --- Phase 4: Backpropagation ---
        # Paper Eq (5): Q(s,a) = γ·r_exec + (1-γ)·r_LLM
        reward = cfg.gamma_backup * exec_score + (1.0 - cfg.gamma_backup) * node.kb_reward
        _backpropagate(path, reward, conn)
        node.last_feedback = feedback

        traces.append(
            {
                "sim": sim_idx,
                "depth": node.depth,
                "kb_reward": node.kb_reward,
                "exec_score": exec_score,
                "reward": reward,
                "feedback": feedback,
                "injected_nodes": len(injected_nodes),
                "path_len": len(path),
            }
        )

        if exec_score > best_exec:
            best_exec = exec_score
            best_code = code_text

        # Early exit as soon as an execution pass is found.
        if exec_score == 1.0:
            print(f"    ✓ early exit at sim {sim_idx} (exec_score={exec_score:.2f})", flush=True)
            return best_code, best_exec, traces

    # Final output must come from evaluated code only.
    # Avoid leaf-text fallback because state text can be non-executable/mixed.
    return best_code, best_exec, traces


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_rpm_mcts(conn: sqlite3.Connection, cfg: RPMMCTSConfig) -> dict[str, Any]:
    run_id = _create_run(conn, cfg)
    problems = _fetch_problems(conn, cfg)
    retriever = KBRetriever(conn, cfg.kb_embedding_model)
    backend = _make_backend(cfg.provider, cfg.model_name)
    rng = random.Random(cfg.seed)

    generated = 0
    total = len(problems)
    for idx, row in enumerate(problems):
        print(f"[RPM-MCTS] [{idx+1}/{total}] {row['problem_uid']}", flush=True)
        prompt = str(row["prompt"] or "")
        problem_id = int(row["problem_id"])

        completion, score, traces = _search_one(
            prompt=prompt,
            dataset_name=cfg.dataset_key,
            entry_point=(row["entry_point"] or None),
            test_spec_json=str(row["test_spec_json"] or "{}"),
            backend=backend,
            retriever=retriever,
            cfg=cfg,
            rng=rng,
            conn=conn,
            run_id=run_id,
            problem_id=problem_id,
        )

        # Store best completion
        conn.execute(
            """
            INSERT INTO generations(
                run_id, problem_id, prompt_text, completion_text,
                finish_reason, latency_ms, prompt_tokens, completion_tokens, total_tokens,
                model_response_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                problem_id,
                prompt,
                completion,
                "mcts",
                None, None, None, None,
                json.dumps({"best_score": score}, ensure_ascii=True),
            ),
        )

        # Store per-simulation trace events
        for t in traces:
            conn.execute(
                """
                INSERT INTO mcts_traces(run_id, problem_id, event_type, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, problem_id, "simulation", json.dumps(t, ensure_ascii=True)),
            )

        # Commit per-problem for safety (avoids losing all work on crash/suspend)
        conn.commit()
        generated += 1
        n_injected = sum(t.get("injected_nodes", 0) for t in traces)
        print(
            f"  → best_score={score:.3f}  sims={len(traces)}"
            f"  injected_nodes={n_injected}",
            flush=True,
        )

    return {"run_id": run_id, "generated": generated}
