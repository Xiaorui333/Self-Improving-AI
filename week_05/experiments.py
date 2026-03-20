"""MuJoCo A2C Experiments
========================
Part 2/3: Compare A2C with TD(0), MC, and GAE on three MuJoCo robotics environments.
Part 4:   Hyperparameter grid search over num_envs, policy_lr, value_lr, γ, λ.

Usage:
    python experiments.py --compare          # algorithm comparison + plots
    python experiments.py --grid-search      # hyperparameter grid search
    python experiments.py --all              # both (default)

Requirements: gymnasium[mujoco], torch, numpy, matplotlib, seaborn
"""

import json
import os
import time
from itertools import product

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from a2c import (
    GaussianPolicy,
    ValueEstimator,
    VectorizedEnvWrapper,
    a2c,
)

# ── Configuration ───────────────────────────────────────────────────────────────

RESULTS_DIR = "results"
PLOT_DIR = "plots"

ENVS = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]

ALGORITHMS = {
    "A2C TD(0) (λ=0)":    0.0,
    "A2C MC (λ=1)":       1.0,
    "A2C GAE (λ=0.95)":   0.95,
}

COMPARISON_CONFIG = dict(
    num_envs=4,
    policy_lr=3e-4,
    value_lr=1e-3,
    gamma=0.99,
    epochs=150,
    rollout_traj_len=256,
    train_v_iters=10,
    entropy_coeff=0.01,
    hidden_sizes=(64, 64),
    seed=42,
)

GRID = dict(
    num_envs=[4, 8],
    policy_lr=[3e-4, 1e-3],
    value_lr=[1e-3, 1e-2],
    gamma=[0.95, 0.99],
    lam=[0.0, 0.95, 1.0],
)

GRID_SEARCH_EPOCHS = 30

# ── Helpers ─────────────────────────────────────────────────────────────────────


def smooth(values: list[float], window: int = 20) -> np.ndarray:
    """Simple moving-average smoother for learning curves."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def run_experiment(
    env_name: str,
    num_envs: int = 8,
    policy_lr: float = 3e-4,
    value_lr: float = 1e-3,
    gamma: float = 0.99,
    lam: float = 0.95,
    epochs: int = 500,
    rollout_traj_len: int = 2048,
    train_v_iters: int = 10,
    entropy_coeff: float = 0.01,
    hidden_sizes: tuple[int, ...] = (64, 64),
    seed: int = 42,
    verbose: bool = True,
) -> list[float]:
    """Run a single A2C experiment on a continuous-action MuJoCo environment."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    base_env = gym.make(env_name)
    env = VectorizedEnvWrapper(base_env, num_envs=num_envs)

    policy = GaussianPolicy(env, lr=policy_lr, hidden_sizes=hidden_sizes)
    critic = ValueEstimator(env, lr=value_lr, hidden_sizes=hidden_sizes)

    returns = a2c(
        env, policy, critic,
        gamma=gamma, lam=lam, epochs=epochs,
        train_v_iters=train_v_iters,
        rollout_traj_len=rollout_traj_len,
        entropy_coeff=entropy_coeff,
        verbose=verbose,
    )
    return returns


# ── Part 2 & 3: Algorithm Comparison ───────────────────────────────────────────


def run_comparison() -> None:
    """Train TD(0) / MC / GAE on three MuJoCo envs and plot each env's curves."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="colorblind")

    for env_name in ENVS:
        print(f"\n{'=' * 64}")
        print(f"  Environment: {env_name}")
        print(f"{'=' * 64}")

        fig, ax = plt.subplots(figsize=(10, 6))

        for algo_name, lam_val in ALGORITHMS.items():
            print(f"\n--- {algo_name} ---")
            t0 = time.time()

            returns = run_experiment(env_name, lam=lam_val, **COMPARISON_CONFIG)

            elapsed = time.time() - t0
            print(f"Finished in {elapsed:.0f}s  |  Final avg return: {returns[-1]:.1f}")

            result_file = os.path.join(
                RESULTS_DIR, f"{env_name}_{algo_name.replace(' ', '_')}.json"
            )
            with open(result_file, "w") as f:
                json.dump(returns, f)

            smoothed = smooth(returns)
            ax.plot(smoothed, label=algo_name, linewidth=1.5)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Average Episode Return", fontsize=12)
        ax.set_title(f"A2C Learning Curves — {env_name}", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        sns.despine()

        plot_file = os.path.join(PLOT_DIR, f"{env_name}_comparison.png")
        fig.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved plot → {plot_file}")


# ── Part 4: Hyperparameter Grid Search ─────────────────────────────────────────


def run_grid_search() -> None:
    """Exhaustive grid search over (num_envs, policy_lr, value_lr, γ, λ) per env."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    param_names = list(GRID.keys())
    param_values = list(GRID.values())
    combos = list(product(*param_values))
    total = len(combos)

    for env_name in ENVS:
        print(f"\n{'=' * 64}")
        print(f"  Grid Search: {env_name}  ({total} configurations)")
        print(f"{'=' * 64}")

        best_score = -np.inf
        best_config: dict | None = None
        all_results: list[dict] = []

        for idx, combo in enumerate(combos):
            config = dict(zip(param_names, combo))
            tag = "  ".join(f"{k}={v}" for k, v in config.items())
            print(f"\n[{idx + 1}/{total}] {tag}")

            t0 = time.time()
            try:
                returns = run_experiment(
                    env_name,
                    num_envs=config["num_envs"],
                    policy_lr=config["policy_lr"],
                    value_lr=config["value_lr"],
                    gamma=config["gamma"],
                    lam=config["lam"],
                    epochs=GRID_SEARCH_EPOCHS,
                    rollout_traj_len=256,
                    train_v_iters=10,
                    entropy_coeff=0.01,
                    hidden_sizes=(64, 64),
                    seed=42,
                    verbose=False,
                )
                tail = returns[int(0.8 * len(returns)):]
                score = float(np.mean(tail))
            except Exception as e:
                print(f"  FAILED: {e}")
                score = float("-inf")
                returns = []

            elapsed = time.time() - t0
            print(f"  Score (mean last 20%): {score:.1f}   ({elapsed:.0f}s)")

            all_results.append({
                "config": {k: v for k, v in config.items()},
                "score": score,
            })

            if score > best_score:
                best_score = score
                best_config = config

        # ── Save results ──
        gs_file = os.path.join(RESULTS_DIR, f"{env_name}_grid_search.json")
        with open(gs_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'─' * 48}")
        print(f"  Best configuration for {env_name}:")
        for k, v in best_config.items():
            print(f"    {k:>15s} = {v}")
        print(f"    {'score':>15s} = {best_score:.1f}")
        print(f"  Results saved → {gs_file}")

        # ── Heatmap: score vs (lam, gamma) marginalised over other params ──
        _plot_grid_search_heatmap(env_name, all_results)


def _plot_grid_search_heatmap(env_name: str, results: list[dict]) -> None:
    """Create a γ × λ heatmap (best score over other params) from grid search results."""
    gammas = sorted(set(r["config"]["gamma"] for r in results))
    lams = sorted(set(r["config"]["lam"] for r in results))

    heatmap = np.full((len(gammas), len(lams)), -np.inf)
    for r in results:
        gi = gammas.index(r["config"]["gamma"])
        li = lams.index(r["config"]["lam"])
        if r["score"] > heatmap[gi, li]:
            heatmap[gi, li] = r["score"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        heatmap, ax=ax, annot=True, fmt=".0f",
        xticklabels=[str(l) for l in lams],
        yticklabels=[str(g) for g in gammas],
        cmap="viridis",
    )
    ax.set_xlabel("λ", fontsize=12)
    ax.set_ylabel("γ", fontsize=12)
    ax.set_title(f"Grid Search Best Scores — {env_name}", fontsize=13)

    plot_file = os.path.join(PLOT_DIR, f"{env_name}_grid_heatmap.png")
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved → {plot_file}")


# ── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MuJoCo A2C Experiments")
    parser.add_argument("--compare", action="store_true",
                        help="Run algorithm comparison (Parts 2 & 3)")
    parser.add_argument("--grid-search", action="store_true",
                        help="Run hyperparameter grid search (Part 4)")
    parser.add_argument("--all", action="store_true",
                        help="Run everything")
    args = parser.parse_args()

    if args.all or (not args.compare and not args.grid_search):
        run_comparison()
        run_grid_search()
    else:
        if args.compare:
            run_comparison()
        if args.grid_search:
            run_grid_search()
