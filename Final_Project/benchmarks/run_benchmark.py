#!/usr/bin/env python3
"""Unified benchmark runner for AgentFlow.

Usage:
    python3.11 benchmarks/run_benchmark.py --model qwen2.5-7b --benchmarks bamboogle --sample_size 5
    python3.11 benchmarks/run_benchmark.py --model qwen3.5-0.8b --benchmarks all --sample_size 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.config import BENCHMARKS, MODELS, PORTKEY_API_KEY
from agentflow.solver import Solver
from benchmarks.score import score_sample

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def load_benchmark(name: str, sample_size: int | None = None) -> list[dict]:
    path = os.path.join(DATA_DIR, name, "data.json")
    with open(path) as f:
        data = json.load(f)
    if sample_size and sample_size < len(data):
        random.seed(42)
        data = random.sample(data, sample_size)
    return data


def run_single_benchmark(
    solver: Solver,
    benchmark: str,
    data: list[dict],
    out_dir: str,
) -> dict:
    """Run solver on every sample, score, and save results."""
    os.makedirs(out_dir, exist_ok=True)

    all_scores: list[dict[str, float]] = []
    for i, sample in enumerate(data):
        question = sample.get("query") or sample.get("question", "")
        pid = sample.get("pid", i)

        out_file = os.path.join(out_dir, f"output_{pid}.json")
        if os.path.exists(out_file):
            with open(out_file) as f:
                existing = json.load(f)
            pred = existing.get("direct_output", "")
        else:
            print(f"\n{'='*60}")
            print(f"[{benchmark}] Sample {i+1}/{len(data)} (pid={pid})")
            print(f"{'='*60}")
            try:
                result = solver.solve(question)
                pred = result.get("direct_output", "")
                result["pid"] = pid
                result["gold_answer"] = sample.get("answer", "")
                with open(out_file, "w") as f:
                    json.dump(result, f, indent=2, default=str)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                pred = ""

        scores = score_sample(benchmark, pred, sample)
        all_scores.append(scores)
        print(f"  Scores: {scores}")

    # Aggregate
    if not all_scores:
        return {}
    agg: dict[str, float] = {}
    for key in all_scores[0]:
        vals = [s[key] for s in all_scores]
        agg[key] = sum(vals) / len(vals)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-7b", choices=list(MODELS.keys()),
                        help="Model key. Use 'qwen3.5-0.8b-grpo' for the GRPO-trained "
                             "Planner (requires GRPO_MODEL_URL env var to be set).")
    parser.add_argument(
        "--benchmarks", default="all",
        help="Comma-separated benchmark names, or 'all'",
    )
    parser.add_argument("--sample_size", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    model_string = MODELS[args.model]
    benchmarks = BENCHMARKS if args.benchmarks == "all" else args.benchmarks.split(",")

    print(f"Model: {args.model} ({model_string})")
    print(f"Benchmarks: {benchmarks}")
    print(f"Sample size: {args.sample_size}")

    solver = Solver(
        model=model_string,
        api_key=PORTKEY_API_KEY,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )

    summary: dict[str, dict] = {}
    for bm in benchmarks:
        data_path = os.path.join(DATA_DIR, bm, "data.json")
        if not os.path.exists(data_path):
            print(f"\nSkipping {bm}: data not found at {data_path}")
            continue

        data = load_benchmark(bm, args.sample_size)
        out_dir = os.path.join(RESULTS_DIR, args.model, bm)
        print(f"\n{'#'*60}")
        print(f"# Benchmark: {bm} ({len(data)} samples)")
        print(f"{'#'*60}")

        t0 = time.time()
        agg = run_single_benchmark(solver, bm, data, out_dir)
        elapsed = time.time() - t0

        agg["n_samples"] = len(data)
        agg["elapsed_s"] = round(elapsed, 1)
        summary[bm] = agg
        print(f"\n>>> {bm} aggregated: {agg}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, args.model, "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n\nFull summary saved to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
