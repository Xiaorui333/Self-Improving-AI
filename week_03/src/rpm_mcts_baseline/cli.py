from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

from .dataset_catalog import DEFAULT_SPECS
from .db import connect, init_db

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rpm-baseline",
        description="Part 1 baseline pipeline for RPM-MCTS with SQLite-first data management.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("artifacts/rpm_mcts.sqlite"),
        help="SQLite database path.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="Initialize SQLite schema.")

    ingest_parser = sub.add_parser("ingest", help="Ingest datasets into SQLite.")
    ingest_parser.add_argument(
        "--datasets",
        nargs="+",
        default=["humaneval_plus", "mbpp_plus", "apps", "codecontests"],
        choices=sorted(DEFAULT_SPECS.keys()),
        help="Dataset keys to ingest.",
    )
    ingest_parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional cap for debugging/smoke tests.",
    )

    gen_parser = sub.add_parser(
        "generate-baseline",
        help="Run baseline single-shot code generation and store results in SQLite.",
    )
    gen_parser.add_argument("--run-name", required=True, help="Unique run name.")
    gen_parser.add_argument(
        "--provider",
        choices=["transformers", "openai"],
        default="transformers",
        help="Model provider backend.",
    )
    gen_parser.add_argument(
        "--model-name",
        required=True,
        help="HF model id for transformers, or OpenAI model name.",
    )
    gen_parser.add_argument("--dataset", default=None, help="Optional dataset key filter.")
    gen_parser.add_argument("--split", default=None, help="Optional split filter.")
    gen_parser.add_argument("--limit", type=int, default=None, help="Optional max problems.")
    gen_parser.add_argument("--temperature", type=float, default=0.2)
    gen_parser.add_argument("--top-p", type=float, default=0.95)
    gen_parser.add_argument("--max-new-tokens", type=int, default=512)
    gen_parser.add_argument("--seed", type=int, default=42)

    eval_parser = sub.add_parser(
        "evaluate-baseline",
        help="Execute generated code against tests and store pass/fail outcomes in SQLite.",
    )
    eval_parser.add_argument("--run-name", required=True, help="Generation run name to evaluate.")
    eval_parser.add_argument("--timeout-sec", type=int, default=8, help="Per-sample timeout.")
    eval_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate samples even if evaluation rows already exist.",
    )
    eval_parser.add_argument("--limit", type=int, default=None, help="Optional max samples.")

    sub.add_parser(
        "report-pass1",
        help="Read stored pass@1 metrics for all runs/datasets/splits.",
    )

    kb_parser = sub.add_parser(
        "build-kb",
        help="Build KB step traces from train splits, embed, and store in SQLite.",
    )
    kb_parser.add_argument(
        "--datasets",
        nargs="+",
        default=["apps", "codecontests"],
        choices=sorted(DEFAULT_SPECS.keys()),
    )
    kb_parser.add_argument("--splits", nargs="+", default=["train"])
    kb_parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    kb_parser.add_argument("--batch-size", type=int, default=32)
    kb_parser.add_argument("--limit-problems", type=int, default=None)

    mcts_parser = sub.add_parser(
        "run-rpm-mcts",
        help="Run RPM-MCTS generation with KB reward, similarity filtering, and reflection loop.",
    )
    mcts_parser.add_argument("--run-name", required=True)
    mcts_parser.add_argument("--provider", choices=["transformers", "openai"], default="transformers")
    mcts_parser.add_argument("--model-name", required=True)
    mcts_parser.add_argument("--dataset", required=True)
    mcts_parser.add_argument("--split", required=True)
    mcts_parser.add_argument("--limit", type=int, default=None)
    mcts_parser.add_argument("--offset", type=int, default=None, help="Skip first N problems (resume support).")
    mcts_parser.add_argument("--temperature", type=float, default=0.2)
    mcts_parser.add_argument("--top-p", type=float, default=0.95)
    mcts_parser.add_argument("--step-max-new-tokens", type=int, default=128)
    mcts_parser.add_argument("--max-depth", type=int, default=6)
    mcts_parser.add_argument("--num-simulations", type=int, default=5)
    mcts_parser.add_argument("--branching-factor", type=int, default=3)
    mcts_parser.add_argument("--beta", type=float, default=0.5, help="UCB exploration constant beta.")
    mcts_parser.add_argument(
        "--alpha-select",
        type=float,
        default=0.5,
        help="KB reward weight in selection/initial value term.",
    )
    mcts_parser.add_argument(
        "--gamma-backup",
        type=float,
        default=0.5,
        help="KB reward weight in backup reward.",
    )
    # Backward-compat alias for older scripts; maps to gamma_backup if provided.
    mcts_parser.add_argument("--gamma", type=float, default=None, help=argparse.SUPPRESS)
    mcts_parser.add_argument("--theta-succ", type=float, default=0.95)
    mcts_parser.add_argument("--similarity-threshold", type=float, default=0.85)
    mcts_parser.add_argument("--reflection-iters", type=int, default=2)
    mcts_parser.add_argument("--timeout-sec", type=int, default=10)
    mcts_parser.add_argument("--seed", type=int, default=42)
    mcts_parser.add_argument(
        "--kb-embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "init-db":
        init_db(args.db_path)
        print(f"Initialized database: {args.db_path}")
        return

    init_db(args.db_path)
    conn = connect(args.db_path)

    if args.command == "ingest":
        from .ingest import ingest_dataset

        for dataset_key in args.datasets:
            spec = DEFAULT_SPECS[dataset_key]
            stats = ingest_dataset(
                conn=conn,
                spec=spec,
                limit_per_split=args.limit_per_split,
            )
            stats_text = ", ".join(f"{k}={v}" for k, v in stats.items())
            print(f"Ingested {dataset_key}: {stats_text}")
        return

    if args.command == "generate-baseline":
        from .generate import GenerationConfig, run_generation

        config = GenerationConfig(
            run_name=args.run_name,
            provider=args.provider,
            model_name=args.model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        inserted = run_generation(
            conn=conn,
            config=config,
            dataset_key=args.dataset,
            split_name=args.split,
            limit=args.limit,
        )
        print(f"Generated {inserted} completions under run '{args.run_name}'.")
        return

    if args.command == "evaluate-baseline":
        from .evaluate import EvaluationConfig, evaluate_run

        result = evaluate_run(
            conn=conn,
            cfg=EvaluationConfig(
                run_name=args.run_name,
                timeout_sec=args.timeout_sec,
                overwrite=args.overwrite,
                limit=args.limit,
            ),
        )
        counters = result["counters"]
        print(
            "Evaluation completed: "
            f"evaluated={counters['evaluated']}, "
            f"passed={counters['passed']}, "
            f"failed={counters['failed']}, "
            f"timeout={counters['timeout']}, "
            f"unsupported={counters['unsupported']}"
        )
        for metric in result["metrics"]:
            print(
                "pass@1 "
                f"{metric['dataset_name']}/{metric['split_name']}="
                f"{metric['pass@1']:.4f} "
                f"({metric['passed']}/{metric['total']})"
            )
        return

    if args.command == "report-pass1":
        rows = conn.execute(
            """
            SELECT
                gr.run_name,
                m.dataset_name,
                m.split_name,
                m.metric_value,
                m.metadata_json,
                m.computed_at
            FROM run_metrics m
            JOIN generation_runs gr ON gr.id = m.run_id
            WHERE m.metric_name = 'pass@1'
            ORDER BY m.computed_at DESC, gr.run_name
            """
        ).fetchall()
        if not rows:
            print("No pass@1 metrics found.")
            return
        for row in rows:
            print(
                f"{row['run_name']} "
                f"{row['dataset_name']}/{row['split_name']} "
                f"pass@1={float(row['metric_value']):.4f} "
                f"metadata={row['metadata_json']} "
                f"computed_at={row['computed_at']}"
            )
        return

    if args.command == "build-kb":
        from .kb import KBBuildConfig, build_kb_steps

        stats = build_kb_steps(
            conn=conn,
            cfg=KBBuildConfig(
                datasets=tuple(args.datasets),
                splits=tuple(args.splits),
                embedding_model=args.embedding_model,
                batch_size=args.batch_size,
                limit_problems=args.limit_problems,
            ),
        )
        print(f"KB built: steps={stats['steps']}, embedded={stats['embedded']}")
        return

    if args.command == "run-rpm-mcts":
        from .rpm_mcts import RPMMCTSConfig, run_rpm_mcts

        gamma_backup = args.gamma_backup if args.gamma is None else args.gamma
        stats = run_rpm_mcts(
            conn=conn,
            cfg=RPMMCTSConfig(
                run_name=args.run_name,
                provider=args.provider,
                model_name=args.model_name,
                dataset_key=args.dataset,
                split_name=args.split,
                limit=args.limit,
                offset=args.offset,
                temperature=args.temperature,
                top_p=args.top_p,
                step_max_new_tokens=args.step_max_new_tokens,
                max_depth=args.max_depth,
                num_simulations=args.num_simulations,
                branching_factor=args.branching_factor,
                ucb_beta=args.beta,
                alpha_select=args.alpha_select,
                gamma_backup=gamma_backup,
                theta_succ=args.theta_succ,
                similarity_threshold=args.similarity_threshold,
                reflection_iters=args.reflection_iters,
                timeout_sec=args.timeout_sec,
                seed=args.seed,
                kb_embedding_model=args.kb_embedding_model,
            ),
        )
        print(f"RPM-MCTS completed: run_id={stats['run_id']}, generated={stats['generated']}")
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

