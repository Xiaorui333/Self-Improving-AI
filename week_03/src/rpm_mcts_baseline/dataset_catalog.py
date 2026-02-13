from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    hf_path: str
    hf_name: str | None
    splits: tuple[str, ...]
    id_field: str
    prompt_field: str
    starter_code_field: str | None = None
    entry_point_field: str | None = None
    canonical_solution_field: str | None = None
    test_field: str | None = None


# NOTE:
# Confirm benchmark names/splits with the RPM-MCTS paper section before final runs.
# These defaults are common code-generation baselines and can be overridden in CLI.
DEFAULT_SPECS: dict[str, DatasetSpec] = {
    "humaneval": DatasetSpec(
        key="humaneval",
        hf_path="openai_humaneval",
        hf_name=None,
        splits=("test",),
        id_field="task_id",
        prompt_field="prompt",
        entry_point_field="entry_point",
        canonical_solution_field="canonical_solution",
        test_field="test",
    ),
    "mbpp": DatasetSpec(
        key="mbpp",
        hf_path="mbpp",
        hf_name=None,
        splits=("test",),
        id_field="task_id",
        prompt_field="text",
        test_field="test_list",
    ),
}


def as_jsonable_record(record: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = value
        elif isinstance(value, dict):
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe

