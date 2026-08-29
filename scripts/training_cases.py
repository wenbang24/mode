"""Shared prompt-safety training-case contract and deterministic selection."""

from __future__ import annotations

import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable


EXPECTED_ROWS = 10_000
CASE_COLUMNS = (
    "case_id",
    "source_dataset",
    "source_index",
    "prompt",
    "prompt_harm_label",
    "adversarial",
    "subcategory",
)
VALID_LABELS = {"harmful", "unharmful"}


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def case_schema():
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("case_id", pa.string(), nullable=False),
            pa.field("source_dataset", pa.string(), nullable=False),
            pa.field("source_index", pa.int64(), nullable=False),
            pa.field("prompt", pa.string(), nullable=False),
            pa.field("prompt_harm_label", pa.string(), nullable=False),
            pa.field("adversarial", pa.bool_()),
            pa.field("subcategory", pa.string()),
        ]
    )


def validate_cases(
    rows: Iterable[dict[str, Any]], expected_rows: int | None = EXPECTED_ROWS
) -> list[dict[str, Any]]:
    rows = list(rows)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows:,} rows, found {len(rows):,}")

    case_ids: set[str] = set()
    source_indices: set[int] = set()
    prompts: set[str] = set()
    for index, row in enumerate(rows):
        if tuple(row) != CASE_COLUMNS:
            raise ValueError(
                f"row {index} has columns {tuple(row)!r}; expected {CASE_COLUMNS!r}"
            )
        case_id = row["case_id"]
        source_dataset = row["source_dataset"]
        source_index = row["source_index"]
        prompt = row["prompt"]
        label = row["prompt_harm_label"]
        adversarial = row["adversarial"]
        subcategory = row["subcategory"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"row {index} has an invalid case_id")
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        case_ids.add(case_id)
        if not isinstance(source_dataset, str) or not source_dataset.strip():
            raise ValueError(f"row {index} has an invalid source_dataset")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError(f"row {index} has an invalid source_index")
        if source_index in source_indices:
            raise ValueError(f"duplicate source_index: {source_index}")
        source_indices.add(source_index)
        if not isinstance(prompt, str) or not (normalized := normalize_prompt(prompt)):
            raise ValueError(f"row {index} has an empty prompt")
        if normalized in prompts:
            raise ValueError(f"duplicate normalized prompt at row {index}")
        prompts.add(normalized)
        if label not in VALID_LABELS:
            raise ValueError(f"row {index} has an invalid prompt_harm_label")
        if adversarial is not None and not isinstance(adversarial, bool):
            raise ValueError(f"row {index} has an invalid adversarial value")
        if subcategory is not None and (
            not isinstance(subcategory, str) or not subcategory.strip()
        ):
            raise ValueError(f"row {index} has an invalid subcategory")
    return rows


def read_cases(
    path: Path, expected_rows: int | None = EXPECTED_ROWS
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pq.read_table(path)
    if not table.schema.equals(case_schema(), check_metadata=False):
        raise ValueError(f"{path} does not match the common training-case schema")
    return validate_cases(table.to_pylist(), expected_rows)


def write_cases(
    path: Path,
    rows: Iterable[dict[str, Any]],
    expected_rows: int | None = EXPECTED_ROWS,
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    rows = validate_cases(rows, expected_rows)
    table = pa.Table.from_pylist(rows, schema=case_schema())
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    read_cases(path, expected_rows)
    return path.stat().st_size


def deduplicate_consistent(
    rows: Iterable[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], int]:
    rows = list(rows)
    labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels[normalize_prompt(row["prompt"])].add(row["prompt_harm_label"])
    conflicts = {prompt for prompt, values in labels.items() if len(values) > 1}

    indices = list(range(len(rows)))
    random.Random(f"{seed}:dedupe").shuffle(indices)
    seen: set[str] = set()
    selected = []
    for index in indices:
        prompt = normalize_prompt(rows[index]["prompt"])
        if prompt not in conflicts and prompt not in seen:
            seen.add(prompt)
            selected.append(index)
    return [rows[index] for index in sorted(selected)], len(conflicts)


def stratified_sample(
    rows: Iterable[dict[str, Any]],
    sample_size: int,
    seed: int,
    key: Callable[[dict[str, Any]], Hashable],
) -> tuple[list[dict[str, Any]], dict[Hashable, int], Counter]:
    rows = list(rows)
    if sample_size > len(rows):
        raise ValueError(
            f"requested {sample_size:,} rows but only {len(rows):,} are eligible"
        )
    strata: dict[Hashable, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[key(row)].append(index)
    counts = Counter({name: len(indices) for name, indices in strata.items()})
    ideal = {name: sample_size * count / len(rows) for name, count in counts.items()}
    quotas = {name: int(value) for name, value in ideal.items()}
    remaining = sample_size - sum(quotas.values())
    for name in sorted(strata, key=lambda item: (-(ideal[item] % 1), repr(item)))[
        :remaining
    ]:
        quotas[name] += 1

    rng = random.Random(seed)
    selected = []
    for name in sorted(strata, key=repr):
        pool = list(strata[name])
        rng.shuffle(pool)
        selected.extend(pool[: quotas[name]])
    rng.shuffle(selected)
    if len(selected) != sample_size:
        raise AssertionError("stratified sampling returned the wrong row count")
    return [rows[index] for index in selected], quotas, counts
