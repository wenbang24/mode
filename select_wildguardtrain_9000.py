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
    Selects 9,000 prompt-response rows with unique prompts from `allenai/wildguardmix`
    (`wildguardtrain`) while preserving the joint distribution of:
    - prompt harmfulness
    - response harmfulness
    - response action (`refusal` = blocked, `compliance` = allowed)
    """)
    return


@app.cell
def imports():
    import os
    import random
    from collections import Counter
    from pathlib import Path

    import marimo as mo
    import pyarrow.parquet as pq
    from datasets import load_dataset

    return Counter, Path, load_dataset, mo, os, pq, random


@app.cell
def config(Path):
    DATASET_ID = "allenai/wildguardmix"
    DATASET_CONFIG = "wildguardtrain"
    DATASET_SPLIT = "train"
    SAMPLE_SIZE = 9_000
    SEED = 42
    ENV_PATH = Path("scripts/.env")
    OUTPUT_PATH = Path("wildguardtrain_9000_seed42.parquet")
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
    RESPONSE_ACTION = {"refusal": "blocked", "compliance": "allowed"}
    return (
        DATASET_CONFIG,
        DATASET_ID,
        DATASET_SPLIT,
        ENV_PATH,
        OUTPUT_PATH,
        RESPONSE_ACTION,
        SAMPLE_SIZE,
        SEED,
        STRATUM_COLUMNS,
        VALID_LABELS,
    )


@app.cell
def load_data(
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_SPLIT,
    ENV_PATH,
    load_dataset,
    os,
):
    def _read_env_value(path, key):
        if not path.is_file():
            return None
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            candidate, separator, value = line.partition("=")
            if separator and candidate.strip() == key:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
                    value = value[1:-1]
                return value or None
        return None


    _hf_token = os.environ.get("HF_TOKEN") or _read_env_value(ENV_PATH, "HF_TOKEN")
    if not _hf_token:
        raise RuntimeError(
            f"HF_TOKEN is required in the environment or {ENV_PATH}."
        )

    wildguardtrain = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        token=_hf_token,
    )
    wildguardtrain = wildguardtrain.add_column(
        "source_index", list(range(len(wildguardtrain)))
    )
    return (wildguardtrain,)


@app.cell
def eligibility(
    SEED,
    STRATUM_COLUMNS,
    VALID_LABELS,
    mo,
    random,
    wildguardtrain,
):
    def _has_non_empty_response(row):
        response = row.get("response")
        return isinstance(response, str) and bool(response.strip())

    def _is_eligible(row):
        prompt = row.get("prompt")
        return (
            isinstance(prompt, str)
            and bool(prompt.strip())
            and _has_non_empty_response(row)
            and all(
                row.get(column) in VALID_LABELS[column]
                for column in STRATUM_COLUMNS
            )
        )

    response_row_count = sum(
        _has_non_empty_response(_row) for _row in wildguardtrain
    )
    _complete_cases = wildguardtrain.filter(
        _is_eligible,
        desc="Keeping response-bearing rows with complete binary labels",
    )
    _prompts = _complete_cases["prompt"]
    _candidate_indices = list(range(len(_complete_cases)))
    random.Random(f"{SEED}:dedupe").shuffle(_candidate_indices)
    _seen_prompts = set()
    _unique_indices = []
    for _index in _candidate_indices:
        _prompt = _prompts[_index]
        if _prompt not in _seen_prompts:
            _seen_prompts.add(_prompt)
            _unique_indices.append(_index)

    eligible_cases = _complete_cases.select(sorted(_unique_indices))
    eligibility_summary = [
        {"population": "All WildGuardTrain rows", "rows": len(wildguardtrain)},
        {"population": "Non-empty responses", "rows": response_row_count},
        {
            "population": "Eligible complete labels",
            "rows": len(_complete_cases),
        },
        {
            "population": "Duplicate-prompt rows removed",
            "rows": len(_complete_cases) - len(eligible_cases),
        },
        {
            "population": "Unique-prompt eligible pool",
            "rows": len(eligible_cases),
        },
    ]
    eligibility_view = mo.vstack(
        [mo.md("## Eligibility"), mo.ui.table(eligibility_summary)]
    )
    eligibility_view
    return (eligible_cases,)


@app.cell
def quotas(SAMPLE_SIZE, STRATUM_COLUMNS, eligible_cases):
    _raw_strata = {}
    for _eligible_index, _labels in enumerate(
        zip(*(eligible_cases[_column] for _column in STRATUM_COLUMNS))
    ):
        _raw_strata.setdefault(tuple(_labels), []).append(_eligible_index)

    source_strata = {
        _key: tuple(_raw_strata[_key]) for _key in sorted(_raw_strata)
    }
    source_counts = {
        _key: len(_indices) for _key, _indices in source_strata.items()
    }
    if SAMPLE_SIZE > len(eligible_cases):
        raise ValueError(
            f"Requested {SAMPLE_SIZE:,} rows but only {len(eligible_cases):,} are eligible"
        )

    _ideal_counts = {
        _key: SAMPLE_SIZE * _count / len(eligible_cases)
        for _key, _count in source_counts.items()
    }
    joint_quotas = {
        _key: int(_ideal_counts[_key]) for _key in source_counts
    }
    _remaining = SAMPLE_SIZE - sum(joint_quotas.values())
    for _key in sorted(
        source_counts,
        key=lambda _item: (-(_ideal_counts[_item] % 1), _item),
    )[:_remaining]:
        joint_quotas[_key] += 1

    assert sum(joint_quotas.values()) == SAMPLE_SIZE
    assert all(
        joint_quotas[_key] <= source_counts[_key] for _key in source_counts
    )
    return joint_quotas, source_counts, source_strata


@app.cell
def selection(SEED, eligible_cases, joint_quotas, random, source_strata):
    def _select_indices():
        rng = random.Random(SEED)
        chosen = []
        for key in sorted(source_strata):
            pool = list(source_strata[key])
            rng.shuffle(pool)
            chosen.extend(pool[: joint_quotas[key]])
        rng.shuffle(chosen)
        return chosen


    selected_indices = _select_indices()
    _repeat_indices = _select_indices()
    assert selected_indices == _repeat_indices
    selected_cases = eligible_cases.select(selected_indices)
    return (selected_cases,)


@app.cell
def validation(
    Counter,
    SAMPLE_SIZE,
    STRATUM_COLUMNS,
    VALID_LABELS,
    joint_quotas,
    mo,
    selected_cases,
):
    selected_counts = Counter(
        zip(*(selected_cases[_column] for _column in STRATUM_COLUMNS))
    )

    assert len(selected_cases) == SAMPLE_SIZE
    assert len(set(selected_cases["source_index"])) == SAMPLE_SIZE
    assert len(set(selected_cases["prompt"])) == SAMPLE_SIZE
    assert all(
        isinstance(_response, str) and bool(_response.strip())
        for _response in selected_cases["response"]
    )
    assert all(
        _value in VALID_LABELS[_column]
        for _column in STRATUM_COLUMNS
        for _value in selected_cases[_column]
    )
    assert dict(selected_counts) == joint_quotas

    validation_summary = [
        {"check": "Selected rows", "result": f"{len(selected_cases):,}"},
        {
            "check": "Distinct source rows",
            "result": f"{len(set(selected_cases['source_index'])):,}",
        },
        {
            "check": "Distinct prompts",
            "result": f"{len(set(selected_cases['prompt'])):,}",
        },
        {"check": "Non-empty responses", "result": "all"},
        {"check": "Complete valid labels", "result": "all"},
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
    Counter,
    RESPONSE_ACTION,
    SAMPLE_SIZE,
    eligible_cases,
    mo,
    selected_cases,
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
                "source_pct": round(100 * _source_count / len(eligible_cases), 4),
                "selected_rows": _selected_count,
                "selected_pct": round(100 * _selected_count / SAMPLE_SIZE, 4),
            }
        )

    marginal_distribution = []
    _marginal_specs = (
        ("Prompt harmfulness", "prompt_harm_label", ("harmful", "unharmful")),
        ("Response harmfulness", "response_harm_label", ("harmful", "unharmful")),
        ("Response action", "response_refusal_label", ("refusal", "compliance")),
    )
    for _dimension, _column, _values in _marginal_specs:
        _source_marginal = Counter(eligible_cases[_column])
        _selected_marginal = Counter(selected_cases[_column])
        for _value in _values:
            marginal_distribution.append(
                {
                    "dimension": _dimension,
                    "category": RESPONSE_ACTION.get(_value, _value),
                    "source_rows": _source_marginal[_value],
                    "source_pct": round(
                        100 * _source_marginal[_value] / len(eligible_cases), 4
                    ),
                    "selected_rows": _selected_marginal[_value],
                    "selected_pct": round(
                        100 * _selected_marginal[_value] / SAMPLE_SIZE, 4
                    ),
                }
            )

    distribution_view = mo.vstack(
        [
            mo.md("## Joint distribution"),
            mo.ui.table(joint_distribution),
            mo.md("## Marginal distributions"),
            mo.ui.table(marginal_distribution),
        ]
    )
    distribution_view
    return


@app.cell
def save_parquet(OUTPUT_PATH, SAMPLE_SIZE, mo, pq, selected_cases):
    _bytes_written = selected_cases.to_parquet(str(OUTPUT_PATH))
    parquet_row_count = pq.ParquetFile(str(OUTPUT_PATH)).metadata.num_rows
    assert parquet_row_count == SAMPLE_SIZE

    save_summary = [
        {"output": str(OUTPUT_PATH.resolve())},
        {"rows": f"{parquet_row_count:,}"},
        {"bytes": f"{_bytes_written:,}"},
    ]
    save_view = mo.vstack(
        [mo.md("## Saved Parquet"), mo.ui.table(save_summary)]
    )
    save_view
    return


@app.cell
def preview(mo, selected_cases):
    _preview_columns = (
        "source_index",
        "prompt",
        "response",
        "prompt_harm_label",
        "response_harm_label",
        "response_refusal_label",
        "adversarial",
        "subcategory",
    )
    preview_rows = [
        {_column: _row[_column] for _column in _preview_columns}
        for _row in selected_cases.select(range(min(25, len(selected_cases))))
    ]
    preview_view = mo.vstack(
        [mo.md("## Selected-row preview"), mo.ui.table(preview_rows)]
    )
    preview_view
    return


if __name__ == "__main__":
    app.run()
