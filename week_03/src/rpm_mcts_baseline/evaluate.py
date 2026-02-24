from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
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


def _extract_entrypoint_function_block(code: str, entry_point: str) -> str | None:
    """Extract top-level `def entry_point(...)` block and ignore all other code."""
    lines = code.splitlines()

    # Preferred path: AST gives precise function boundaries when parseable.
    try:
        import ast

        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
                start = node.lineno
                if node.decorator_list:
                    start = min(start, *(d.lineno for d in node.decorator_list))
                end = node.end_lineno or node.lineno
                return "\n".join(lines[start - 1 : end]).rstrip()
    except SyntaxError:
        pass

    # Fallback path for non-parseable completions: scan text for top-level def.
    start_idx: int | None = None
    pattern = re.compile(rf"^(?:async\s+def|def)\s+{re.escape(entry_point)}\s*\(")
    for i, ln in enumerate(lines):
        if pattern.match(ln):
            start_idx = i
            break
    if start_idx is None:
        return None

    end_idx = len(lines)
    stop_re = re.compile(r"^(?:async\s+def|def|class)\s+|^if __name__\s*==")
    for j in range(start_idx + 1, len(lines)):
        if stop_re.match(lines[j]):
            end_idx = j
            break
    return "\n".join(lines[start_idx:end_idx]).rstrip()


def _reindent_completion(prompt: str, completion: str) -> str:
    """Re-add lost indentation to the first line when upstream text was stripped.

    If a completion ever arrives with leading indentation removed from the first
    line, this helper restores that first-line indentation using the prompt
    context while preserving relative indentation of subsequent lines.

    Example (prompt ends with ``    \"\"\"\\n``, expected indent = 4):
        raw model output : ``    return x\\n``
        after .strip()   : ``return x\\n``          ← first-line indent lost
        after this func  : ``    return x\\n``       ← restored
    """
    if not completion or completion[0] in (" ", "\t"):
        return completion  # already indented — nothing to fix

    # If the completion itself starts a new def/class at column 0 (e.g. a nested
    # helper the model generated before the return statement), reindenting would
    # corrupt all subsequent lines.  Let it pass through; _truncate_to_function_body
    # will handle it, and the assembler will surface any real syntax issues.
    first_token = completion.lstrip().split()[0] if completion.strip() else ""
    if first_token in ("def", "class", "async"):
        return completion

    # Detect expected indent from the last non-empty line of the prompt.
    # Only fix the FIRST line to preserve relative indentation in following lines.
    for line in reversed(prompt.split("\n")):
        if line.strip():
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                lines = completion.split("\n")
                lines[0] = " " * indent + lines[0]
                return "\n".join(lines)
            break
    return completion


def _extract_code_from_markdown_fences(text: str) -> str:
    """Extract code content from markdown fences if present."""
    fenced = re.findall(r"```(?:[^\n`]*)\n(.*?)```", text, flags=re.DOTALL)
    if not fenced:
        return text
    return "\n\n".join(block.strip("\n") for block in fenced if block.strip())


def _has_top_level_entry_def(code: str, entry_point: str) -> bool:
    """Return True if code contains top-level `def <entry_point>(...)`."""
    pattern = rf"(?m)^def\s+{re.escape(entry_point)}\s*\("
    return re.search(pattern, code) is not None


def _indent_body_lines(body: str, spaces: int = 4) -> str:
    """Indent non-empty lines by `spaces` spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + ln if ln.strip() else ln for ln in body.splitlines())


def _remove_forbidden_solution_lines(code: str) -> str:
    """Drop obvious non-solution artifacts from generated candidate text."""
    forbidden = (
        r"^\s*assert\b",
        r"^\s*if\s+__name__\s*==",
        r"^\s*def\s+check\b",
        r"^\s*def\s+check_solution\b",
        r"^\s*check_solution\s*\(",
        r"^\s*doctest\.",
        r"^\s*main\s*\(",
    )
    out: list[str] = []
    for ln in code.splitlines():
        if any(re.search(pat, ln) for pat in forbidden):
            continue
        out.append(ln)
    return "\n".join(out)


def sanitize_humaneval_candidate(
    prompt_text: str,
    completion_text: str,
    entry_point: str,
) -> tuple[str, str]:
    """Return (solution_code, candidate_completion) for HumanEval assembly.

    3-stage sanitizer:
      1) extract code from markdown fences
      2) if top-level def entry_point exists, use completion as full solution code
      3) else treat completion as body and indent all non-empty lines by 4 spaces
    """
    cleaned = _extract_code_from_markdown_fences(completion_text).strip("\n")
    cleaned = _remove_forbidden_solution_lines(cleaned)
    fn_block = _extract_entrypoint_function_block(cleaned, entry_point)
    if fn_block:
        solution_code = fn_block
        candidate_completion = fn_block
    else:
        # Treat as body: normalize indentation first, then indent to function scope.
        body = textwrap.dedent(cleaned).strip("\n")
        body = _indent_body_lines(body, spaces=4)
        solution_code = f"{prompt_text}{body}"
        candidate_completion = body
    return solution_code, candidate_completion


def _build_program(row: sqlite3.Row) -> tuple[str | None, str | None]:
    dataset_name = str(row["dataset_name"])
    completion = row["completion_text"] or ""
    prompt_text = row["problem_prompt"] or row["prompt_text"] or ""
    entry_point = row["entry_point"] or ""
    test_spec = _parse_json(row["test_spec_json"] or "{}")

    if dataset_name.startswith("humaneval"):
        test_code = test_spec.get("test")
        if not isinstance(test_code, str) or not test_code.strip():
            return None, "missing_humaneval_test_code"
        if not entry_point:
            return None, "missing_entry_point"

        solution_code, _ = sanitize_humaneval_candidate(
            prompt_text=prompt_text,
            completion_text=completion,
            entry_point=entry_point,
        )

        # HumanEval-style tests define check(candidate). We run it with the expected entry point.
        program = (
            f"{solution_code}\n\n"
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
        # Reindent in case completion was stripped (e.g. from MCTS)
        completion = _reindent_completion(completion, completion)
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


def execute_problem(
    dataset_name: str,
    prompt_text: str,
    completion_text: str,
    entry_point: str | None,
    test_spec_json: str,
    timeout_sec: int,
) -> dict[str, Any]:
    row_like = {
        "dataset_name": dataset_name,
        "problem_prompt": prompt_text,
        "prompt_text": prompt_text,
        "completion_text": completion_text,
        "entry_point": entry_point or "",
        "test_spec_json": test_spec_json,
    }
    program_text, prep_error = _build_program(row_like)  # type: ignore[arg-type]
    if prep_error:
        return {
            "status": "unsupported",
            "passed": 0,
            "score": 0.0,
            "stdout": "",
            "stderr": prep_error,
            "traceback": "",
            "eval_json": {"reason": prep_error},
        }
    assert program_text is not None
    return _run_python(program_text=program_text, timeout_sec=timeout_sec)


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
            p.prompt AS problem_prompt,
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

