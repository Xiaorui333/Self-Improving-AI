from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    version TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dataset_splits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    split_name TEXT NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dataset_id, split_name),
    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS raw_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_split_id INTEGER NOT NULL,
    sample_index INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dataset_split_id, sample_index),
    FOREIGN KEY(dataset_split_id) REFERENCES dataset_splits(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS problems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_split_id INTEGER NOT NULL,
    problem_uid TEXT NOT NULL,
    prompt TEXT NOT NULL,
    starter_code TEXT,
    entry_point TEXT,
    canonical_solution TEXT,
    test_spec_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dataset_split_id, problem_uid),
    FOREIGN KEY(dataset_split_id) REFERENCES dataset_splits(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS generation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    temperature REAL NOT NULL,
    top_p REAL,
    max_new_tokens INTEGER NOT NULL,
    seed INTEGER,
    config_json TEXT NOT NULL DEFAULT '{}',
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    problem_id INTEGER NOT NULL,
    prompt_text TEXT NOT NULL,
    completion_text TEXT NOT NULL,
    finish_reason TEXT,
    latency_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    model_response_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES generation_runs(id) ON DELETE CASCADE,
    FOREIGN KEY(problem_id) REFERENCES problems(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL UNIQUE,
    status TEXT NOT NULL,
    passed INTEGER NOT NULL DEFAULT 0,
    score REAL,
    stdout TEXT,
    stderr TEXT,
    traceback TEXT,
    eval_json TEXT NOT NULL DEFAULT '{}',
    evaluated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(generation_id) REFERENCES generations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS run_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    dataset_name TEXT NOT NULL,
    split_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    computed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, dataset_name, split_name, metric_name),
    FOREIGN KEY(run_id) REFERENCES generation_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_raw_samples_split ON raw_samples(dataset_split_id);
CREATE INDEX IF NOT EXISTS idx_problems_split ON problems(dataset_split_id);
CREATE INDEX IF NOT EXISTS idx_generations_run ON generations(run_id);
CREATE INDEX IF NOT EXISTS idx_generations_problem ON generations(problem_id);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON run_metrics(run_id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: str | Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def upsert_dataset(
    conn: sqlite3.Connection,
    name: str,
    source: str,
    version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    metadata_json = json.dumps(metadata or {}, ensure_ascii=True)
    conn.execute(
        """
        INSERT INTO datasets(name, source, version, metadata_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            source=excluded.source,
            version=excluded.version,
            metadata_json=excluded.metadata_json
        """,
        (name, source, version, metadata_json),
    )
    row = conn.execute("SELECT id FROM datasets WHERE name = ?", (name,)).fetchone()
    assert row is not None
    return int(row["id"])


def upsert_split(
    conn: sqlite3.Connection,
    dataset_id: int,
    split_name: str,
    sample_count: int = 0,
    metadata: dict[str, Any] | None = None,
) -> int:
    metadata_json = json.dumps(metadata or {}, ensure_ascii=True)
    conn.execute(
        """
        INSERT INTO dataset_splits(dataset_id, split_name, sample_count, metadata_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(dataset_id, split_name) DO UPDATE SET
            sample_count=excluded.sample_count,
            metadata_json=excluded.metadata_json
        """,
        (dataset_id, split_name, sample_count, metadata_json),
    )
    row = conn.execute(
        "SELECT id FROM dataset_splits WHERE dataset_id = ? AND split_name = ?",
        (dataset_id, split_name),
    ).fetchone()
    assert row is not None
    return int(row["id"])


def upsert_run_metric(
    conn: sqlite3.Connection,
    run_id: int,
    dataset_name: str,
    split_name: str,
    metric_name: str,
    metric_value: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    metadata_json = json.dumps(metadata or {}, ensure_ascii=True)
    conn.execute(
        """
        INSERT INTO run_metrics(
            run_id, dataset_name, split_name, metric_name, metric_value, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, dataset_name, split_name, metric_name)
        DO UPDATE SET
            metric_value=excluded.metric_value,
            metadata_json=excluded.metadata_json,
            computed_at=CURRENT_TIMESTAMP
        """,
        (
            run_id,
            dataset_name,
            split_name,
            metric_name,
            metric_value,
            metadata_json,
        ),
    )
