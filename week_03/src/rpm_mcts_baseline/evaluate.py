from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import transaction, upsert_run_metric


PASS_MARKER = "__RPM_BASELINE_PASS__"


@dataclass
class EvaluationConfig:
    run_name: str
    timeout_sec: int = 8
    overwrite: bool = False
    limit: int | None = None


def _parse_json(payload: str) -> dict[str, Any]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _build_program(row: sqlite3.Row) -> tuple[str | None, str | None]:
    dataset_name = str(row["dataset_name"])
    completion = row["completion_text"] or ""
    prompt_text = row["prompt_text"] or ""
    entry_point = row["entry_point"] or ""
    test_spec = _parse_json(row["test_spec_json"] or "{}")

    if dataset_name.startswith("humaneval"):
        test_code = test_spec.get("test")
        if not isinstance(test_code, str) or not test_code.strip():
            return None, "missing_humaneval_test_code"
        if not entry_point:
            return None, "missing_entry_point"
        # HumanEval-style tests define check(candidate). We run it with the expected entry point.
        program = (
            f"{prompt_text}{completion}\n\n"
            f"{test_code}\n\n"
            f"check({entry_point})\n"
            f"print({PASS_MARKER!r})\n"
        )
        return program, None

    if dataset_name.startswith("mbpp"):
        test_list = test_spec.get("test_list")
        if not isinstance(test_list, list) or not test_list:
            return None, "missing_mbpp_test_list"
        assertions = "\n".join(str(item) for item in test_list)
        program = f"{completion}\n\n{assertions}\nprint({PASS_MARKER!r})\n"
        return program, None

    return None, f"unsupported_dataset:{dataset_name}"


def _run_python(program_text: str, timeout_sec: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rpm_eval_") as tmp_dir:
        test_file = Path(tmp_dir) / "candidate_eval.py"
        test_file.write_text(program_text, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "timeout",
                "passed": 0,
                "score": 0.0,
                "stdout": (exc.stdout or ""),
                "stderr": (exc.stderr or ""),
                "traceback": "timeout",
                "eval_json": {"timeout_sec": timeout_sec},
            }

    passed = int(proc.returncode == 0 and PASS_MARKER in (proc.stdout or ""))
    status = "passed" if passed else "failed"
    return {
        "status": status,
        "passed": passed,
        "score": float(passed),
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
        "traceback": "",
        "eval_json": {"returncode": proc.returncode},
    }


def _get_run_id(conn: sqlite3.Connection, run_name: str) -> int:
    row = conn.execute(
        "SELECT id FROM generation_runs WHERE run_name = ?",
        (run_name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Run not found: {run_name}")
    return int(row["id"])


def _fetch_targets(conn: sqlite3.Connection, cfg: EvaluationConfig, run_id: int) -> list[sqlite3.Row]:
    where_eval = "e.id IS NULL"
    if cfg.overwrite:
        where_eval = "1=1"
    limit_sql = f"LIMIT {int(cfg.limit)}" if cfg.limit else ""
    query = f"""
        SELECT
            g.id AS generation_id,
            g.prompt_text,
            g.completion_text,
            p.entry_point,
            p.test_spec_json,
            d.name AS dataset_name,
            ds.split_name
        FROM generations g
        JOIN problems p ON p.id = g.problem_id
        JOIN dataset_splits ds ON ds.id = p.dataset_split_id
        JOIN datasets d ON d.id = ds.dataset_id
        LEFT JOIN evaluations e ON e.generation_id = g.id
        WHERE g.run_id = ? AND {where_eval}
        ORDER BY g.id ASC
        {limit_sql}
    """
    return conn.execute(query, (run_id,)).fetchall()


def _upsert_evaluation(conn: sqlite3.Connection, generation_id: int, result: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO evaluations(
            generation_id, status, passed, score, stdout, stderr, traceback, eval_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(generation_id) DO UPDATE SET
            status=excluded.status,
            passed=excluded.passed,
            score=excluded.score,
            stdout=excluded.stdout,
            stderr=excluded.stderr,
            traceback=excluded.traceback,
            eval_json=excluded.eval_json,
            evaluated_at=CURRENT_TIMESTAMP
        """,
        (
            generation_id,
            result["status"],
            int(result["passed"]),
            float(result["score"]),
            result["stdout"],
            result["stderr"],
            result["traceback"],
            json.dumps(result["eval_json"], ensure_ascii=True),
        ),
    )


def _compute_and_store_pass_at_1(conn: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            d.name AS dataset_name,
            ds.split_name AS split_name,
            COUNT(*) AS total,
            SUM(CASE WHEN e.passed = 1 THEN 1 ELSE 0 END) AS passed_count
        FROM generations g
        JOIN evaluations e ON e.generation_id = g.id
        JOIN problems p ON p.id = g.problem_id
        JOIN dataset_splits ds ON ds.id = p.dataset_split_id
        JOIN datasets d ON d.id = ds.dataset_id
        WHERE g.run_id = ?
        GROUP BY d.name, ds.split_name
        ORDER BY d.name, ds.split_name
        """,
        (run_id,),
    ).fetchall()

    metrics: list[dict[str, Any]] = []
    for row in rows:
        total = int(row["total"])
        passed_count = int(row["passed_count"] or 0)
        value = (passed_count / total) if total else 0.0
        upsert_run_metric(
            conn=conn,
            run_id=run_id,
            dataset_name=str(row["dataset_name"]),
            split_name=str(row["split_name"]),
            metric_name="pass@1",
            metric_value=value,
            metadata={"passed": passed_count, "total": total},
        )
        metrics.append(
            {
                "dataset_name": str(row["dataset_name"]),
                "split_name": str(row["split_name"]),
                "pass@1": value,
                "passed": passed_count,
                "total": total,
            }
        )
    return metrics


def evaluate_run(conn: sqlite3.Connection, cfg: EvaluationConfig) -> dict[str, Any]:
    run_id = _get_run_id(conn, cfg.run_name)
    targets = _fetch_targets(conn, cfg=cfg, run_id=run_id)

    counters = {
        "evaluated": 0,
        "passed": 0,
        "failed": 0,
        "timeout": 0,
        "unsupported": 0,
    }
    with transaction(conn):
        for row in targets:
            generation_id = int(row["generation_id"])
            program_text, prep_error = _build_program(row)
            if prep_error:
                result = {
                    "status": "unsupported",
                    "passed": 0,
                    "score": 0.0,
                    "stdout": "",
                    "stderr": prep_error,
                    "traceback": "",
                    "eval_json": {"reason": prep_error},
                }
                counters["unsupported"] += 1
            else:
                result = _run_python(program_text=program_text, timeout_sec=cfg.timeout_sec)
                if result["status"] == "passed":
                    counters["passed"] += 1
                elif result["status"] == "timeout":
                    counters["timeout"] += 1
                else:
                    counters["failed"] += 1
            _upsert_evaluation(conn, generation_id=generation_id, result=result)
            counters["evaluated"] += 1

        metrics = _compute_and_store_pass_at_1(conn=conn, run_id=run_id)

    return {
        "run_id": run_id,
        "run_name": cfg.run_name,
        "counters": counters,
        "metrics": metrics,
    }

