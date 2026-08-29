# /// script
# dependencies = [
#     "datasets==5.0.1",
#     "marimo",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def title(mo):
    mo.md(r"""
    # WildGuard train, validation, and test selector

    Selects 10,000 training and 1,000 validation cases from `WildGuardTrain`,
    converts every prompt-labeled `WildGuardTest` case, and removes normalized
    cross-split overlaps with test > validation > train precedence. Training and
    validation preserve the joint distribution of prompt harmfulness, response
    harmfulness, and response refusal.
    """)
    return


@app.cell
def imports():
    import os
    from collections import Counter
    from pathlib import Path

    import marimo as mo
    from datasets import load_dataset

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
        load_dataset,
        mo,
        normalize_prompt,
        os,
        read_cases,
        stratified_sample,
        validate_cases,
        write_cases,
    )


@app.cell
def configuration(Path):
    WORKSPACE = Path(__file__).resolve().parent
    DATASET_ID = "allenai/wildguardmix"
    DATASET_CONFIGS = {"train": "wildguardtrain", "test": "wildguardtest"}
    DATASET_SPLITS = {"train": "train", "test": "test"}
    DATASET_REVISION = "d29c47f41c8b51348b5c8e8c81c039b3132b66d1"
    ENV_PATH = WORKSPACE / "scripts" / ".env"
    OUTPUT_PATHS = {
        "train": WORKSPACE / "wildguardtrain_10000_seed42.parquet",
        "valid": WORKSPACE / "wildguardtrain_validation_1000_seed42.parquet",
        "test": WORKSPACE / "wildguardtest.parquet",
    }
    EXPECTED_SPLIT_SIZES = {"train": 10_000, "valid": 1_000, "test": 1_699}
    EXPECTED_LABEL_COUNTS = {
        "train": {"harmful": 5_720, "unharmful": 4_280},
        "valid": {"harmful": 571, "unharmful": 429},
        "test": {"unharmful": 945, "harmful": 754},
    }
    EXPECTED_JOINT_QUOTAS = {
        "train": {
            ("harmful", "harmful", "compliance"): 2_185,
            ("harmful", "unharmful", "compliance"): 622,
            ("harmful", "unharmful", "refusal"): 2_913,
            ("unharmful", "harmful", "compliance"): 6,
            ("unharmful", "unharmful", "compliance"): 2_157,
            ("unharmful", "unharmful", "refusal"): 2_117,
        },
        "valid": {
            ("harmful", "harmful", "compliance"): 218,
            ("harmful", "unharmful", "compliance"): 62,
            ("harmful", "unharmful", "refusal"): 291,
            ("unharmful", "harmful", "compliance"): 1,
            ("unharmful", "unharmful", "compliance"): 216,
            ("unharmful", "unharmful", "refusal"): 212,
        },
    }
    STRATUM_COLUMNS = (
        "prompt_harm_label",
        "response_harm_label",
        "response_refusal_label",
    )
    VALID_LABELS = {
        "prompt_harm_label": ("harmful", "unharmful"),
        "response_harm_label": ("harmful", "unharmful"),
        "response_refusal_label": ("refusal", "compliance"),
    }
    SEED = 42
    return (
        DATASET_CONFIGS,
        DATASET_ID,
        DATASET_REVISION,
        DATASET_SPLITS,
        ENV_PATH,
        EXPECTED_JOINT_QUOTAS,
        EXPECTED_LABEL_COUNTS,
        EXPECTED_SPLIT_SIZES,
        OUTPUT_PATHS,
        SEED,
        STRATUM_COLUMNS,
        VALID_LABELS,
    )


@app.cell
def load_sources(
    DATASET_CONFIGS,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLITS,
    ENV_PATH,
    load_dataset,
    os,
):
    def _read_env_value(path, key):
        if not path.is_file():
            return None
        for _raw_line in path.read_text(encoding="utf-8").splitlines():
            _line = _raw_line.strip()
            if not _line or _line.startswith("#"):
                continue
            _candidate, _separator, _value = _line.partition("=")
            if _separator and _candidate.strip() == key:
                _value = _value.strip()
                if (
                    len(_value) >= 2
                    and _value[0] == _value[-1]
                    and _value[0] in {'"', "'"}
                ):
                    _value = _value[1:-1]
                return _value or None
        return None

    _hf_token = os.environ.get("HF_TOKEN") or _read_env_value(
        ENV_PATH, "HF_TOKEN"
    )
    if not _hf_token:
        raise RuntimeError(
            f"HF_TOKEN is required in the environment or {ENV_PATH}."
        )
    source_rows = {
        _split: load_dataset(
            DATASET_ID,
            _config,
            split=DATASET_SPLITS[_split],
            revision=DATASET_REVISION,
            token=_hf_token,
        )
        for _split, _config in DATASET_CONFIGS.items()
    }
    return (source_rows,)


@app.cell
def eligibility(
    SEED,
    STRATUM_COLUMNS,
    VALID_LABELS,
    deduplicate_consistent,
    normalize_prompt,
    source_rows,
):
    def _has_non_empty_text(row, column):
        _value = row.get(column)
        return isinstance(_value, str) and bool(_value.strip())

    _training_candidates = [
        {**_row, "source_index": _source_index}
        for _source_index, _row in enumerate(source_rows["train"])
        if _has_non_empty_text(_row, "prompt")
        and _has_non_empty_text(_row, "response")
        and all(
            _row.get(_column) in VALID_LABELS[_column]
            for _column in STRATUM_COLUMNS
        )
    ]
    _test_candidates = [
        {**_row, "source_index": _source_index}
        for _source_index, _row in enumerate(source_rows["test"])
        if _has_non_empty_text(_row, "prompt")
        and _row.get("prompt_harm_label") in VALID_LABELS["prompt_harm_label"]
    ]
    eligible_training_rows, training_conflicts = deduplicate_consistent(
        _training_candidates, SEED
    )
    test_rows, test_conflicts = deduplicate_consistent(_test_candidates, SEED)
    assert len(eligible_training_rows) == 18_621
    assert len(test_rows) == 1_699
    assert training_conflicts == test_conflicts == 0
    assert len(source_rows["test"]) - len(_test_candidates) == 26

    _test_keys = {normalize_prompt(_row["prompt"]) for _row in test_rows}
    training_pool = [
        _row
        for _row in eligible_training_rows
        if normalize_prompt(_row["prompt"]) not in _test_keys
    ]
    assert len(training_pool) == 18_620
    return test_rows, training_pool


@app.cell
def select_training_and_validation(
    EXPECTED_JOINT_QUOTAS,
    SEED,
    STRATUM_COLUMNS,
    normalize_prompt,
    stratified_sample,
    training_pool,
):
    def joint_key(row):
        return tuple(row[_column] for _column in STRATUM_COLUMNS)

    training_source_rows, training_quotas, _ = stratified_sample(
        training_pool, 10_000, SEED, joint_key
    )
    _training_keys = {
        normalize_prompt(_row["prompt"]) for _row in training_source_rows
    }
    _validation_pool = [
        _row
        for _row in training_pool
        if normalize_prompt(_row["prompt"]) not in _training_keys
    ]
    validation_source_rows, validation_quotas, _ = stratified_sample(
        _validation_pool, 1_000, SEED, joint_key
    )
    assert training_quotas == EXPECTED_JOINT_QUOTAS["train"]
    assert validation_quotas == EXPECTED_JOINT_QUOTAS["valid"]

    _repeat_train, _repeat_train_quotas, _ = stratified_sample(
        training_pool, 10_000, SEED, joint_key
    )
    _repeat_valid, _repeat_valid_quotas, _ = stratified_sample(
        _validation_pool, 1_000, SEED, joint_key
    )
    assert _repeat_train_quotas == training_quotas
    assert _repeat_valid_quotas == validation_quotas
    assert [row["source_index"] for row in _repeat_train] == [
        row["source_index"] for row in training_source_rows
    ]
    assert [row["source_index"] for row in _repeat_valid] == [
        row["source_index"] for row in validation_source_rows
    ]
    return training_source_rows, validation_source_rows


@app.cell
def build_cases(
    DATASET_CONFIGS,
    DATASET_ID,
    test_rows,
    training_source_rows,
    validation_source_rows,
):
    def _case(split, row):
        _source_split = "test" if split == "test" else "train"
        return {
            "case_id": f"wildguard:{split}:{row['source_index']}",
            "source_dataset": (
                f"{DATASET_ID}/{DATASET_CONFIGS[_source_split]}"
            ),
            "source_index": row["source_index"],
            "prompt": row["prompt"].strip(),
            "prompt_harm_label": row["prompt_harm_label"],
            "adversarial": row["adversarial"],
            "subcategory": row["subcategory"] or None,
        }

    selected_rows = {
        "train": [_case("train", _row) for _row in training_source_rows],
        "valid": [_case("valid", _row) for _row in validation_source_rows],
        "test": [_case("test", _row) for _row in test_rows],
    }
    return (selected_rows,)


@app.cell
def validate_splits(
    Counter,
    EXPECTED_LABEL_COUNTS,
    EXPECTED_SPLIT_SIZES,
    normalize_prompt,
    selected_rows,
    validate_cases,
):
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
    return


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
def summary(Counter, OUTPUT_PATHS, mo, output_bytes, source_rows, validated_rows):
    summary_rows = [
        {
            "split": _split,
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
            mo.md(
                f"## Selection complete\n\n"
                f"WildGuardTrain source rows: **{len(source_rows['train']):,}**  \n"
                f"WildGuardTest source rows: **{len(source_rows['test']):,}**"
            ),
            mo.ui.table(summary_rows, selection=None),
            mo.md("## Preview"),
            mo.ui.table(validated_rows["train"][:20], selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
