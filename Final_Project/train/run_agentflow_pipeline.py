#!/usr/bin/env python3
"""Orchestrate the full AgentFlow + Flow-GRPO + serve + benchmark pipeline.

Task list (matches README Phase 6b):
  1. Train Planner (modal run train/run_train.py --agentflow [--smoke-test])
  2. Deploy API (modal deploy train/serve_grpo_model.py)
  3. Set GRPO_MODEL_URL (auto-detected from deploy output when possible)
  4. Run benchmarks/run_benchmark.py with --model qwen3.5-0.8b-grpo

Usage:
  cd Final_Project
  python3 train/run_agentflow_pipeline.py                    # smoke train + deploy + tiny benchmark
  python3 train/run_agentflow_pipeline.py --full-train       # full train (~hours), then deploy + benchmark
  python3 train/run_agentflow_pipeline.py --skip-train       # only deploy + benchmark (checkpoint exists)
  python3 train/run_agentflow_pipeline.py --benchmarks all --sample-size 20

Requires: modal CLI, network; for benchmarks: pip install openai
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _modal_bin() -> str:
    """Prefer the ``modal`` CLI on PATH (correct Modal + Python), not ``python -m modal``."""
    import shutil
    p = shutil.which("modal")
    return p if p else sys.executable


def _modal_cmd(*parts: str) -> list[str]:
    exe = _modal_bin()
    if exe.endswith("modal") or os.path.basename(exe) == "modal":
        return [exe, *parts]
    return [sys.executable, "-m", "modal", *parts]


def _python_for_benchmarks() -> str:
    """Pick a Python that has portkey_ai + openai (same env as AgentFlow)."""
    import shutil
    override = os.environ.get("PYTHON", "").strip()
    if override:
        return override
    # Prefer 3.11 (project default) before generic ``python3``.
    candidates = ("python3.11", "python3.12", "python3", sys.executable)
    for name in candidates:
        exe = name if os.path.isabs(name) else shutil.which(name)
        if not exe:
            continue
        chk = subprocess.run(
            [exe, "-c", "import portkey_ai, openai"],
            capture_output=True,
        )
        if chk.returncode == 0:
            return exe
    return sys.executable


def _ensure_benchmark_deps(py: str) -> None:
    """Install openai + portkey-ai if missing (needed for LocalEngine + Portkey)."""
    r = subprocess.run(
        [py, "-c", "import portkey_ai, openai"],
        capture_output=True,
    )
    if r.returncode == 0:
        return
    print(">>> Installing openai + portkey-ai for benchmark runner...\n", flush=True)
    subprocess.run(
        [py, "-m", "pip", "install", "-q", "openai>=1.0.0", "portkey-ai"],
        check=True,
    )


def _run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    return subprocess.run(
        cmd,
        cwd=cwd or ROOT,
        capture_output=capture,
        text=True,
    )


def _parse_modal_url(text: str) -> str | None:
    """Extract first https://*.modal.run URL from Modal CLI output."""
    m = re.search(r"(https://[a-zA-Z0-9_.-]+\.modal\.run)", text)
    return m.group(1) if m else None


def main() -> int:
    p = argparse.ArgumentParser(description="AgentFlow GRPO full pipeline")
    p.add_argument(
        "--full-train",
        action="store_true",
        help="Run full AgentFlow GRPO training (~3h) instead of smoke (~2 min)",
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; use existing checkpoint on Modal volume",
    )
    p.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Skip modal deploy (use existing deployment + GRPO_MODEL_URL)",
    )
    p.add_argument(
        "--benchmarks",
        default="bamboogle",
        help="Comma list or 'all' (passed to run_benchmark.py)",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=2,
        help="Benchmark sample size (use 20 for report-quality runs)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="AgentFlow max_steps per benchmark item",
    )
    args = p.parse_args()

    os.chdir(ROOT)

    # --- 1. Train ---
    if not args.skip_train:
        train_cmd = _modal_cmd("run", "train/run_train.py", "--agentflow")
        if not args.full_train:
            train_cmd.append("--smoke-test")
        r = _run(train_cmd)
        if r.returncode != 0:
            print("ERROR: training failed.", file=sys.stderr)
            return 1
    else:
        print(">>> Skipping training (--skip-train)")

    # --- 2. Deploy ---
    grpo_url = os.environ.get("GRPO_MODEL_URL", "").strip()
    if not args.skip_deploy:
        r = _run(
            _modal_cmd("deploy", "train/serve_grpo_model.py"),
            capture=True,
        )
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            print(out, file=sys.stderr)
            print("ERROR: modal deploy failed.", file=sys.stderr)
            return 1
        parsed = _parse_modal_url(out)
        if parsed:
            grpo_url = parsed.rstrip("/").removesuffix("/v1")
            print(f"\n>>> Detected GRPO_MODEL_URL={grpo_url}\n")
        elif not grpo_url:
            print(
                "\nCould not auto-detect endpoint URL from deploy output.\n"
                "Set manually, e.g.:\n"
                "  export GRPO_MODEL_URL=https://<workspace>--serve-grpo-model-web.modal.run/v1\n"
                "Then re-run with: --skip-train --skip-deploy\n",
                file=sys.stderr,
            )
            return 1
    else:
        print(">>> Skipping deploy (--skip-deploy); using GRPO_MODEL_URL from environment")
        if not grpo_url:
            print("ERROR: GRPO_MODEL_URL is not set.", file=sys.stderr)
            return 1

    os.environ["GRPO_MODEL_URL"] = grpo_url

    # --- 3. Benchmarks (reload config picks up env) ---
    py = _python_for_benchmarks()
    _ensure_benchmark_deps(py)
    bench_cmd = [
        py,
        "benchmarks/run_benchmark.py",
        "--model",
        "qwen3.5-0.8b-grpo",
        "--benchmarks",
        args.benchmarks,
        "--sample_size",
        str(args.sample_size),
        "--max_steps",
        str(args.max_steps),
    ]
    r = _run(bench_cmd)
    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
