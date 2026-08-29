# /// script
# dependencies = [
#     "huggingface-hub==1.28.0",
#     "marimo",
#     "pyarrow==25.0.1",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def title(mo):
    mo.md(r"""
    # Aegis 2.0 train, validation, and test selector

    Selects 10,000 training cases from the pinned official
    [`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0)
    training Parquet and converts its validation and test Parquets to the common
    prompt-safety schema. Blank/`REDACTED` prompts, conflicting normalized labels,
    duplicates, and cross-split overlaps are removed with test > validation > train
    precedence.
    """)
    return


@app.cell
def imports():
    import marimo as mo
    from collections import Counter
    from pathlib import Path

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from scripts.training_cases import (
        CASE_COLUMNS,
        deduplicate_consistent,
        normalize_prompt,
        read_cases,
        stratified_sample,
        validate_cases,
        write_cases,
    )

    return (
        CASE_COLUMNS,
        Counter,
        Path,
        deduplicate_consistent,
        hf_hub_download,
        mo,
        normalize_prompt,
        pq,
        read_cases,
        stratified_sample,
        validate_cases,
        write_cases,
    )


@app.cell
def configuration(Path):
    WORKSPACE = Path(__file__).resolve().parent
    SOURCE_REPO = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
    SOURCE_REVISION = "cd1abe041ba6f595fea47a67f650bcc0a809ea81"
    SOURCE_FILES = {
        "train": "aegis_v2_train.parquet",
        "valid": "aegis_v2_val.parquet",
        "test": "aegis_v2_test.parquet",
    }
    SOURCE_SPLITS = {"train": "train", "valid": "validation", "test": "test"}
    OUTPUT_PATHS = {
        "train": WORKSPACE / "aegis2_train_10000_seed42.parquet",
        "valid": WORKSPACE / "aegis2_validation.parquet",
        "test": WORKSPACE / "aegis2_test.parquet",
    }
    EXPECTED_SPLIT_SIZES = {"train": 10_000, "valid": 1_189, "test": 1_914}
    EXPECTED_LABEL_COUNTS = {
        "train": {"harmful": 5_046, "unharmful": 4_954},
        "valid": {"harmful": 641, "unharmful": 548},
        "test": {"harmful": 1_030, "unharmful": 884},
    }
    EXPECTED_ELIGIBLE_ROWS = {"train": 22_850, "valid": 1_200, "test": 1_914}
    EXPECTED_CONFLICTS = {"train": 167, "valid": 0, "test": 1}
    SEED = 42
    return (
        EXPECTED_CONFLICTS,
        EXPECTED_ELIGIBLE_ROWS,
        EXPECTED_LABEL_COUNTS,
        EXPECTED_SPLIT_SIZES,
        OUTPUT_PATHS,
        SEED,
        SOURCE_FILES,
        SOURCE_REPO,
        SOURCE_REVISION,
        SOURCE_SPLITS,
    )


@app.cell
def load_sources(
    Path,
    SOURCE_FILES,
    SOURCE_REPO,
    SOURCE_REVISION,
    hf_hub_download,
    pq,
):
    source_rows = {}
    for _split, _filename in SOURCE_FILES.items():
        _source_file = Path(
            hf_hub_download(
                repo_id=SOURCE_REPO,
                repo_type="dataset",
                filename=_filename,
                revision=SOURCE_REVISION,
            )
        )
        source_rows[_split] = pq.read_table(
            _source_file,
            columns=["id", "prompt", "prompt_label", "violated_categories"],
        ).to_pylist()
    return (source_rows,)


@app.cell
def eligibility(SOURCE_REPO, SOURCE_SPLITS, source_rows):
    candidate_rows = {}
    exclusion_counts = {}
    for _split, _rows in source_rows.items():
        _excluded_prompts = 0
        _ignored_labels = 0
        for _source_index, _row in enumerate(_rows):
            _prompt = (_row["prompt"] or "").strip()
            _source_label = (_row["prompt_label"] or "").strip().casefold()
            if not _prompt or _prompt.casefold() == "redacted":
                _excluded_prompts += 1
                continue
            if _source_label not in {"safe", "unsafe"}:
                _ignored_labels += 1
                continue
            candidate_rows.setdefault(_split, []).append(
                {
                    "case_id": f"aegis2:{_split}:{_row['id']}",
                    "source_dataset": f"{SOURCE_REPO}/{SOURCE_SPLITS[_split]}",
                    "source_index": _source_index,
                    "prompt": _prompt,
                    "prompt_harm_label": {
                        "unsafe": "harmful",
                        "safe": "unharmful",
                    }[_source_label],
                    "adversarial": None,
                    "subcategory": (_row["violated_categories"] or "").strip()
                    or None,
                }
            )
        exclusion_counts[_split] = {
            "blank_or_redacted": _excluded_prompts,
            "unknown_label": _ignored_labels,
        }
    return candidate_rows, exclusion_counts


@app.cell
def deduplicate_and_isolate(
    EXPECTED_CONFLICTS,
    EXPECTED_ELIGIBLE_ROWS,
    SEED,
    candidate_rows,
    deduplicate_consistent,
    normalize_prompt,
):
    eligible_rows = {}
    conflicting_prompt_keys = {}
    for _split, _rows in candidate_rows.items():
        eligible_rows[_split], conflicting_prompt_keys[_split] = (
            deduplicate_consistent(_rows, seed=SEED)
        )
    assert conflicting_prompt_keys == EXPECTED_CONFLICTS
    assert {
        _split: len(_rows) for _split, _rows in eligible_rows.items()
    } == EXPECTED_ELIGIBLE_ROWS

    _test_keys = {
        normalize_prompt(_row["prompt"]) for _row in eligible_rows["test"]
    }
    validation_rows = [
        _row
        for _row in eligible_rows["valid"]
        if normalize_prompt(_row["prompt"]) not in _test_keys
    ]
    _held_out_keys = _test_keys | {
        normalize_prompt(_row["prompt"]) for _row in validation_rows
    }
    training_pool = [
        _row
        for _row in eligible_rows["train"]
        if normalize_prompt(_row["prompt"]) not in _held_out_keys
    ]
    assert len(training_pool) == 22_537
    return eligible_rows, training_pool, validation_rows


@app.cell
def select_training(SEED, stratified_sample, training_pool):
    training_rows, training_quotas, _training_counts = stratified_sample(
        training_pool,
        10_000,
        SEED,
        key=lambda row: row["prompt_harm_label"],
    )
    _repeat_rows, _repeat_quotas, _ = stratified_sample(
        training_pool,
        10_000,
        SEED,
        key=lambda row: row["prompt_harm_label"],
    )
    assert _repeat_quotas == training_quotas
    assert [row["case_id"] for row in _repeat_rows] == [
        row["case_id"] for row in training_rows
    ]
    return (training_rows,)


@app.cell
def validate_splits(
    Counter,
    EXPECTED_LABEL_COUNTS,
    EXPECTED_SPLIT_SIZES,
    eligible_rows,
    normalize_prompt,
    training_rows,
    validate_cases,
    validation_rows,
):
    selected_rows = {
        "train": training_rows,
        "valid": validation_rows,
        "test": eligible_rows["test"],
    }
    for _split, _rows in selected_rows.items():
        validate_cases(_rows, EXPECTED_SPLIT_SIZES[_split])
        assert dict(
            Counter(_row["prompt_harm_label"] for _row in _rows)
        ) == EXPECTED_LABEL_COUNTS[_split]
    _prompt_keys = {
        _split: {normalize_prompt(_row["prompt"]) for _row in _rows}
        for _split, _rows in selected_rows.items()
    }
    assert not (_prompt_keys["train"] & _prompt_keys["valid"])
    assert not (_prompt_keys["train"] & _prompt_keys["test"])
    assert not (_prompt_keys["valid"] & _prompt_keys["test"])
    return (selected_rows,)


@app.cell
def save_and_validate(
    CASE_COLUMNS,
    EXPECTED_SPLIT_SIZES,
    OUTPUT_PATHS,
    read_cases,
    selected_rows,
    write_cases,
):
    output_bytes = {}
    validated_rows = {}
    for _split, _rows in selected_rows.items():
        _expected = EXPECTED_SPLIT_SIZES[_split]
        output_bytes[_split] = write_cases(
            OUTPUT_PATHS[_split], _rows, expected_rows=_expected
        )
        validated_rows[_split] = read_cases(
            OUTPUT_PATHS[_split], expected_rows=_expected
        )
        assert tuple(validated_rows[_split][0]) == CASE_COLUMNS
        assert [row["case_id"] for row in validated_rows[_split]] == [
            row["case_id"] for row in _rows
        ]
    return output_bytes, validated_rows


@app.cell
def summary(
    Counter,
    OUTPUT_PATHS,
    exclusion_counts,
    mo,
    output_bytes,
    source_rows,
    validated_rows,
):
    summary_rows = [
        {
            "split": _split,
            "source rows": len(source_rows[_split]),
            "excluded prompts": exclusion_counts[_split]["blank_or_redacted"],
            "output rows": len(_rows),
            "labels": dict(
                Counter(_row["prompt_harm_label"] for _row in _rows)
            ),
            "output": OUTPUT_PATHS[_split].name,
            "MB": round(output_bytes[_split] / 1_000_000, 2),
        }
        for _split, _rows in validated_rows.items()
    ]
    mo.vstack(
        [
            mo.md("## Selection complete"),
            mo.ui.table(summary_rows, selection=None),
            mo.md("## Preview"),
            mo.ui.table(validated_rows["train"][:20], selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
