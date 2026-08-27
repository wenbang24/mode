# /// script
# dependencies = [
#     "datasets==5.0.1",
#     "marimo",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def title(mo):
    mo.md(r"""
    # WildGuardTrain 10,000-case selector

    Selects 10,000 prompt-response rows from `allenai/wildguardmix` (`wildguardtrain`)
    after normalized prompt de-duplication, while preserving the joint distribution of:
    - prompt harmfulness
    - response harmfulness
    - response action (`refusal` = blocked, `compliance` = allowed)

    The saved Parquet contains only the common prompt-level fields used by PIGuard,
    AdaSteer, and GuardAgent.
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
        deduplicate_consistent,
        read_cases,
        stratified_sample,
        validate_cases,
        write_cases,
    )

    return (
        Counter,
        Path,
        deduplicate_consistent,
        load_dataset,
        mo,
        os,
        read_cases,
        stratified_sample,
        validate_cases,
        write_cases,
    )


@app.cell
def config(Path):
    DATASET_ID = "allenai/wildguardmix"
    DATASET_CONFIG = "wildguardtrain"
    DATASET_SPLIT = "train"
    DATASET_REVISION = "d29c47f41c8b51348b5c8e8c81c039b3132b66d1"
    SOURCE_DATASET = f"{DATASET_ID}/{DATASET_CONFIG}"
    SAMPLE_SIZE = 10_000
    SEED = 42
    ENV_PATH = Path("scripts/.env")
    OUTPUT_PATH = Path("wildguardtrain_10000_seed42.parquet")
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
    EXPECTED_ELIGIBLE_ROWS = 18_621
    EXPECTED_PROMPT_LABEL_COUNTS = {"harmful": 5_719, "unharmful": 4_281}
    EXPECTED_JOINT_QUOTAS = {
        ("harmful", "harmful", "compliance"): 2_185,
        ("harmful", "unharmful", "compliance"): 622,
        ("harmful", "unharmful", "refusal"): 2_912,
        ("unharmful", "harmful", "compliance"): 7,
        ("unharmful", "unharmful", "compliance"): 2_157,
        ("unharmful", "unharmful", "refusal"): 2_117,
    }
    RESPONSE_ACTION = {"refusal": "blocked", "compliance": "allowed"}
    return (
        DATASET_CONFIG,
        DATASET_ID,
        DATASET_REVISION,
        DATASET_SPLIT,
        ENV_PATH,
        EXPECTED_ELIGIBLE_ROWS,
        EXPECTED_JOINT_QUOTAS,
        EXPECTED_PROMPT_LABEL_COUNTS,
        OUTPUT_PATH,
        RESPONSE_ACTION,
        SAMPLE_SIZE,
        SEED,
        SOURCE_DATASET,
        STRATUM_COLUMNS,
        VALID_LABELS,
    )


@app.cell
def load_data(
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
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

    wildguardtrain = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
        token=_hf_token,
    )
    return (wildguardtrain,)


@app.cell
def eligibility(
    SEED,
    STRATUM_COLUMNS,
    VALID_LABELS,
    deduplicate_consistent,
    mo,
    wildguardtrain,
):
    def _has_non_empty_response(row):
        _response = row.get("response")
        return isinstance(_response, str) and bool(_response.strip())

    def _is_eligible(row):
        _prompt = row.get("prompt")
        return (
            isinstance(_prompt, str)
            and bool(_prompt.strip())
            and _has_non_empty_response(row)
            and all(
                row.get(_column) in VALID_LABELS[_column]
                for _column in STRATUM_COLUMNS
            )
        )

    response_row_count = sum(
        _has_non_empty_response(_row) for _row in wildguardtrain
    )
    _eligible_rows = [
        {**_row, "source_index": _source_index}
        for _source_index, _row in enumerate(wildguardtrain)
        if _is_eligible(_row)
    ]
    eligible_cases, conflicting_prompt_keys = deduplicate_consistent(
        _eligible_rows, SEED
    )
    eligibility_summary = [
        {"population": "All WildGuardTrain rows", "rows": len(wildguardtrain)},
        {"population": "Non-empty responses", "rows": response_row_count},
        {
            "population": "Eligible complete labels",
            "rows": len(_eligible_rows),
        },
        {
            "population": "Conflicting normalized prompt keys removed",
            "rows": conflicting_prompt_keys,
        },
        {
            "population": "Unique normalized-prompt pool",
            "rows": len(eligible_cases),
        },
    ]
    eligibility_view = mo.vstack(
        [mo.md("## Eligibility"), mo.ui.table(eligibility_summary)]
    )
    eligibility_view
    return conflicting_prompt_keys, eligible_cases


@app.cell
def quotas(
    SAMPLE_SIZE,
    SEED,
    STRATUM_COLUMNS,
    eligible_cases,
    stratified_sample,
):
    def joint_key(row):
        return tuple(row[_column] for _column in STRATUM_COLUMNS)

    selected_source_rows, joint_quotas, source_counts = stratified_sample(
        eligible_cases,
        SAMPLE_SIZE,
        SEED,
        joint_key,
    )
    _repeat_rows, _repeat_quotas, _repeat_counts = stratified_sample(
        eligible_cases,
        SAMPLE_SIZE,
        SEED,
        joint_key,
    )
    assert [row["source_index"] for row in selected_source_rows] == [
        row["source_index"] for row in _repeat_rows
    ]
    assert joint_quotas == _repeat_quotas
    assert source_counts == _repeat_counts
    return joint_key, joint_quotas, selected_source_rows, source_counts


@app.cell
def selection(SOURCE_DATASET, selected_source_rows, validate_cases):
    training_cases = [
        {
            "case_id": f"wildguardtrain:{_row['source_index']}",
            "source_dataset": SOURCE_DATASET,
            "source_index": _row["source_index"],
            "prompt": _row["prompt"],
            "prompt_harm_label": _row["prompt_harm_label"],
            "adversarial": _row["adversarial"],
            "subcategory": _row["subcategory"] or None,
        }
        for _row in selected_source_rows
    ]
    validate_cases(training_cases)
    return (training_cases,)


@app.cell
def validation(
    Counter,
    EXPECTED_ELIGIBLE_ROWS,
    EXPECTED_JOINT_QUOTAS,
    EXPECTED_PROMPT_LABEL_COUNTS,
    conflicting_prompt_keys,
    eligible_cases,
    joint_key,
    joint_quotas,
    mo,
    selected_source_rows,
    training_cases,
):
    selected_counts = Counter(joint_key(_row) for _row in selected_source_rows)
    prompt_label_counts = Counter(
        _row["prompt_harm_label"] for _row in training_cases
    )

    assert len(eligible_cases) == EXPECTED_ELIGIBLE_ROWS
    assert conflicting_prompt_keys == 0
    assert dict(prompt_label_counts) == EXPECTED_PROMPT_LABEL_COUNTS
    assert dict(selected_counts) == EXPECTED_JOINT_QUOTAS
    assert joint_quotas == EXPECTED_JOINT_QUOTAS

    validation_summary = [
        {"check": "Selected rows", "result": f"{len(training_cases):,}"},
        {
            "check": "Distinct source rows",
            "result": f"{len({row['source_index'] for row in training_cases}):,}",
        },
        {
            "check": "Distinct normalized prompts",
            "result": f"{len(training_cases):,}",
        },
        {"check": "Prompt labels", "result": dict(prompt_label_counts)},
        {"check": "Joint quotas", "result": "exact"},
        {"check": "Seed repeatability", "result": "identical"},
    ]
    validation_view = mo.vstack(
        [mo.md("## Validation"), mo.ui.table(validation_summary)]
    )
    validation_view
    return (selected_counts,)


@app.cell
def distributions(
    RESPONSE_ACTION,
    SAMPLE_SIZE,
    eligible_cases,
    mo,
    selected_counts,
    source_counts,
):
    joint_distribution = []
    for _key in sorted(source_counts):
        _source_count = source_counts[_key]
        _selected_count = selected_counts[_key]
        joint_distribution.append(
            {
                "prompt_harm": _key[0],
                "response_harm": _key[1],
                "response_action": RESPONSE_ACTION[_key[2]],
                "source_rows": _source_count,
                "source_pct": round(
                    100 * _source_count / len(eligible_cases), 4
                ),
                "selected_rows": _selected_count,
                "selected_pct": round(100 * _selected_count / SAMPLE_SIZE, 4),
            }
        )

    distribution_view = mo.vstack(
        [
            mo.md("## Preserved joint distribution"),
            mo.ui.table(joint_distribution),
        ]
    )
    distribution_view
    return


@app.cell
def save_parquet(OUTPUT_PATH, mo, read_cases, training_cases, write_cases):
    bytes_written = write_cases(OUTPUT_PATH, training_cases)
    saved_cases = read_cases(OUTPUT_PATH)
    parquet_row_count = len(saved_cases)

    save_summary = [
        {"output": str(OUTPUT_PATH.resolve())},
        {"rows": f"{parquet_row_count:,}"},
        {"bytes": f"{bytes_written:,}"},
    ]
    save_view = mo.vstack(
        [mo.md("## Saved Parquet"), mo.ui.table(save_summary)]
    )
    save_view
    return


@app.cell
def preview(mo, training_cases):
    preview_rows = training_cases[:25]
    preview_view = mo.vstack(
        [mo.md("## Selected-row preview"), mo.ui.table(preview_rows)]
    )
    preview_view
    return


if __name__ == "__main__":
    app.run()
