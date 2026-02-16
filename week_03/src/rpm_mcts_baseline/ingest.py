from __future__ import annotations

import json
import sqlite3
from typing import Any

from datasets import load_dataset

from .dataset_catalog import DatasetSpec, as_jsonable_record
from .db import transaction, upsert_dataset, upsert_split


def _get_str(record: dict[str, Any], field: str | None) -> str | None:
    if not field:
        return None
    value = record.get(field)
    if value is None:
        return None
    return str(value)


def _resolve_problem_uid(record: dict[str, Any], spec: DatasetSpec, index: int) -> str:
    value = record.get(spec.id_field)
    if value is None:
        return f"{spec.key}-{index}"
    return str(value)


def _apply_split_filter(records: list[dict[str, Any]], split_filter: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not split_filter:
        return records
    kept: list[dict[str, Any]] = []
    for record in records:
        matches = True
        for key, expected in split_filter.items():
            if record.get(key) != expected:
                matches = False
                break
        if matches:
            kept.append(record)
    return kept


def _project_raw_record(record: dict[str, Any], spec: DatasetSpec) -> dict[str, Any]:
    if not spec.raw_keep_fields:
        return record
    projected: dict[str, Any] = {}
    for field in spec.raw_keep_fields:
        if field in record:
            projected[field] = record[field]
    return projected


def ingest_dataset(
    conn: sqlite3.Connection,
    spec: DatasetSpec,
    source: str = "huggingface",
    version: str | None = None,
    limit_per_split: int | None = None,
) -> dict[str, int]:
    dataset_id = upsert_dataset(
        conn=conn,
        name=spec.key,
        source=source,
        version=version,
        metadata={
            "hf_path": spec.hf_path,
            "hf_name": spec.hf_name,
        },
    )

    ingested: dict[str, int] = {}

    source_split_map = spec.source_split_map or {}
    split_filters = spec.split_filters or {}
    source_cache: dict[str, list[dict[str, Any]]] = {}

    for split_name in spec.splits:
        source_split = source_split_map.get(split_name, split_name)
        if source_split not in source_cache:
            ds = load_dataset(path=spec.hf_path, name=spec.hf_name, split=source_split)
            source_cache[source_split] = [as_jsonable_record(dict(sample)) for sample in ds]

        records = source_cache[source_split]
        records = _apply_split_filter(records, split_filters.get(split_name))
        if limit_per_split is not None:
            records = records[: max(0, limit_per_split)]

        split_id = upsert_split(
            conn=conn,
            dataset_id=dataset_id,
            split_name=split_name,
            sample_count=len(records),
        )

        with transaction(conn):
            # Keep ingestion idempotent for repeated runs.
            conn.execute(
                """
                DELETE FROM run_metrics
                WHERE run_id IN (
                    SELECT DISTINCT g.run_id
                    FROM generations g
                    JOIN problems p ON p.id = g.problem_id
                    WHERE p.dataset_split_id = ?
                )
                """,
                (split_id,),
            )
            conn.execute(
                """
                DELETE FROM evaluations
                WHERE generation_id IN (
                    SELECT g.id
                    FROM generations g
                    JOIN problems p ON p.id = g.problem_id
                    WHERE p.dataset_split_id = ?
                )
                """,
                (split_id,),
            )
            conn.execute(
                """
                DELETE FROM generations
                WHERE problem_id IN (
                    SELECT id FROM problems WHERE dataset_split_id = ?
                )
                """,
                (split_id,),
            )
            conn.execute("DELETE FROM raw_samples WHERE dataset_split_id = ?", (split_id,))
            conn.execute("DELETE FROM problems WHERE dataset_split_id = ?", (split_id,))

            for idx, record in enumerate(records):
                raw_record = _project_raw_record(record, spec)
                conn.execute(
                    """
                    INSERT INTO raw_samples(dataset_split_id, sample_index, record_json)
                    VALUES (?, ?, ?)
                    """,
                    (split_id, idx, json.dumps(raw_record, ensure_ascii=True)),
                )

                problem_uid = _resolve_problem_uid(record, spec, idx)
                prompt = _get_str(record, spec.prompt_field) or ""
                starter_code = _get_str(record, spec.starter_code_field)
                entry_point = _get_str(record, spec.entry_point_field)
                canonical_solution = _get_str(record, spec.canonical_solution_field)

                test_payload = {}
                if spec.test_field and spec.test_field in record:
                    test_payload = {spec.test_field: record.get(spec.test_field)}

                metadata = {"source_index": idx}

                conn.execute(
                    """
                    INSERT INTO problems(
                        dataset_split_id,
                        problem_uid,
                        prompt,
                        starter_code,
                        entry_point,
                        canonical_solution,
                        test_spec_json,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        split_id,
                        problem_uid,
                        prompt,
                        starter_code,
                        entry_point,
                        canonical_solution,
                        json.dumps(test_payload, ensure_ascii=True),
                        json.dumps(metadata, ensure_ascii=True),
                    ),
                )
        ingested[split_name] = len(records)
    return ingested

