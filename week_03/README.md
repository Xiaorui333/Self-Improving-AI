# RPM-MCTS Baseline (Part 1)

This project implements **Part 1** of your plan for replicating RPM-MCTS:

- single SQLite database for all data
- no intermediate JSON/CSV files in the pipeline
- minimal dependencies
- no RL framework and no agent framework

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Commands

Initialize DB:

```bash
rpm-baseline --db-path /absolute/path/to/rpm_mcts.sqlite init-db
```

Ingest default baseline datasets (HumanEval + MBPP):

```bash
rpm-baseline --db-path /absolute/path/to/rpm_mcts.sqlite ingest
```

Optional smoke test with a small subset:

```bash
rpm-baseline --db-path /absolute/path/to/rpm_mcts.sqlite ingest --limit-per-split 20
```

Run baseline generation with Transformers:

```bash
rpm-baseline \
  --db-path /absolute/path/to/rpm_mcts.sqlite \
  generate-baseline \
  --run-name baseline_codegen_v1 \
  --provider transformers \
  --model-name bigcode/starcoderbase-1b \
  --dataset humaneval \
  --split test \
  --limit 50
```

Run baseline generation with OpenAI:

```bash
export OPENAI_API_KEY=...
rpm-baseline \
  --db-path /absolute/path/to/rpm_mcts.sqlite \
  generate-baseline \
  --run-name baseline_openai_v1 \
  --provider openai \
  --model-name gpt-4.1-mini \
  --dataset humaneval \
  --split test \
  --limit 50
```

## SQLite-First Data Flow

All intermediate data are table-backed:

- `datasets`, `dataset_splits`: dataset inventory
- `raw_samples`: raw row payload from source datasets
- `problems`: normalized benchmark tasks
- `generation_runs`: run-level config and metadata
- `generations`: per-problem model output and token/latency metadata
- `evaluations`: reserved for execution/test results (Part 1.5 / Part 2)

## Notes on Exact Paper Alignment

The code is intentionally designed so dataset specs are explicit and easy to modify in `src/rpm_mcts_baseline/dataset_catalog.py`.

If the paper uses a different benchmark set/split definition, update that file and rerun ingestion; old split rows are replaced idempotently for those splits.

