from __future__ import annotations

import argparse
from pathlib import Path

from .dataset_catalog import DEFAULT_SPECS
from .db import connect, init_db


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
        default=["humaneval", "mbpp"],
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

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

