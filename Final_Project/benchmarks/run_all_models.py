#!/usr/bin/env python3
"""Run AgentFlow on all benchmarks with all models (Phase 2 + 3).

Usage:
    # All models, 5 samples each
    python3.11 benchmarks/run_all_models.py --sample_size 5

    # Single model
    python3.11 benchmarks/run_all_models.py --models qwen2.5-7b --sample_size 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.config import BENCHMARKS, MODELS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
PYTHON = sys.executable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", default="all",
        help="Comma-separated model keys (e.g. qwen2.5-7b,qwen3.5-0.8b) or 'all'",
    )
    parser.add_argument("--benchmarks", default="all")
    parser.add_argument("--sample_size", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=5)
    args = parser.parse_args()

    models = list(MODELS.keys()) if args.models == "all" else args.models.split(",")
    benchmarks = BENCHMARKS if args.benchmarks == "all" else args.benchmarks.split(",")

    all_results: dict[str, dict] = {}

    for model_key in models:
        if model_key not in MODELS:
            print(f"Unknown model: {model_key}, skipping")
            continue

        print(f"\n{'#'*70}")
        print(f"# Model: {model_key}")
        print(f"{'#'*70}")

        cmd = [
            PYTHON, "benchmarks/run_benchmark.py",
            "--model", model_key,
            "--benchmarks", ",".join(benchmarks),
            "--sample_size", str(args.sample_size),
            "--max_steps", str(args.max_steps),
        ]
        subprocess.run(cmd)

        summary_path = os.path.join(RESULTS_DIR, model_key, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                all_results[model_key] = json.load(f)

    # Print combined table
    if all_results:
        print(f"\n\n{'='*90}")
        print("COMBINED RESULTS")
        print(f"{'='*90}")

        header = f"{'Model':<18}"
        for bm in benchmarks:
            header += f" {bm:>10}"
        print(header)
        print("-" * len(header))

        for model_key in models:
            if model_key not in all_results:
                continue
            row = f"{model_key:<18}"
            for bm in benchmarks:
                bm_data = all_results[model_key].get(bm, {})
                score = bm_data.get("accuracy", bm_data.get("f1", bm_data.get("em", 0)))
                row += f" {score:>10.3f}"
            print(row)

        # Save combined
        combined_path = os.path.join(RESULTS_DIR, "combined_results.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {combined_path}")


if __name__ == "__main__":
    main()
