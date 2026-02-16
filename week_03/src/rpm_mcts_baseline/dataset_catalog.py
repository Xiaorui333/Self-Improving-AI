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
    source_split_map: dict[str, str] | None = None
    split_filters: dict[str, dict[str, Any]] | None = None
    raw_keep_fields: tuple[str, ...] | None = None


# NOTE:
# APPS uses a script-free mirror ("likaixin/APPS-verified") so this stays compatible
# with datasets versions where script-based loaders are disabled.
DEFAULT_SPECS: dict[str, DatasetSpec] = {
    "humaneval_plus": DatasetSpec(
        key="humaneval_plus",
        hf_path="evalplus/humanevalplus",
        hf_name=None,
        splits=("test",),
        id_field="task_id",
        prompt_field="prompt",
        entry_point_field="entry_point",
        canonical_solution_field="canonical_solution",
        test_field="test",
        raw_keep_fields=("task_id", "prompt", "canonical_solution", "entry_point", "test"),
    ),
    "mbpp_plus": DatasetSpec(
        key="mbpp_plus",
        hf_path="evalplus/mbppplus",
        hf_name=None,
        splits=("test",),
        id_field="task_id",
        prompt_field="prompt",
        canonical_solution_field="code",
        test_field="test_list",
        raw_keep_fields=(
            "task_id",
            "prompt",
            "code",
            "test_list",
            "test_imports",
            "test",
            "source_file",
        ),
    ),
    "apps": DatasetSpec(
        key="apps",
        hf_path="likaixin/APPS-verified",
        hf_name=None,
        splits=(
            "train",
            "test_introductory",
            "test_interview",
            "test_competition",
        ),
        id_field="problem_id",
        prompt_field="question",
        starter_code_field="starter_code",
        test_field="input_output",
        raw_keep_fields=(
            "id",
            "problem_id",
            "question",
            "input_output",
            "difficulty",
            "url",
            "starter_code",
        ),
        source_split_map={
            "train": "train",
            "test_introductory": "train",
            "test_interview": "train",
            "test_competition": "train",
        },
        split_filters={
            "train": {"difficulty": "interview"},
            "test_introductory": {"difficulty": "introductory"},
            "test_interview": {"difficulty": "interview"},
            "test_competition": {"difficulty": "competition"},
        },
    ),
    "codecontests": DatasetSpec(
        key="codecontests",
        hf_path="deepmind/code_contests",
        hf_name=None,
        splits=("train", "test"),
        id_field="name",
        prompt_field="description",
        test_field="public_tests",
        raw_keep_fields=("name", "description", "public_tests", "difficulty", "source"),
    ),
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

