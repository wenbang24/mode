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
    # Aegis 2.0 train 10,000-case selector

    Selects 10,000 prompt-safety cases from the pinned official
    [`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0)
    training Parquet after excluding blank/`REDACTED` prompts and removing normalized
    prompt keys with conflicting labels.

    The saved Parquet uses the same prompt-level contract as the WildGuardTrain
    selector for PIGuard, AdaSteer, and GuardAgent.
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
        read_cases,
        stratified_sample,
        write_cases,
    )

    return (
        CASE_COLUMNS,
        Counter,
        Path,
        deduplicate_consistent,
        hf_hub_download,
        mo,
        pq,
        read_cases,
        stratified_sample,
        write_cases,
    )


@app.cell
def configuration(Path):
    WORKSPACE = Path(__file__).resolve().parent
    SOURCE_REPO = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
    SOURCE_DATASET = f"{SOURCE_REPO}/train"
    SOURCE_REVISION = "cd1abe041ba6f595fea47a67f650bcc0a809ea81"
    SOURCE_FILE = "aegis_v2_train.parquet"
    OUTPUT_PATH = WORKSPACE / "aegis2_train_10000_seed42.parquet"
    SEED = 42
    SAMPLE_SIZE = 10_000
    EXPECTED_LABEL_QUOTAS = {"harmful": 5_057, "unharmful": 4_943}
    return (
        EXPECTED_LABEL_QUOTAS,
        OUTPUT_PATH,
        SAMPLE_SIZE,
        SEED,
        SOURCE_DATASET,
        SOURCE_FILE,
        SOURCE_REPO,
        SOURCE_REVISION,
    )


@app.cell
def load_source(
    Path,
    SOURCE_FILE,
    SOURCE_REPO,
    SOURCE_REVISION,
    hf_hub_download,
    pq,
):
    source_file = Path(
        hf_hub_download(
            repo_id=SOURCE_REPO,
            repo_type="dataset",
            filename=SOURCE_FILE,
            revision=SOURCE_REVISION,
        )
    )
    source_table = pq.read_table(
        source_file,
        columns=["id", "prompt", "prompt_label", "violated_categories"],
    )
    source_rows = source_table.to_pylist()
    return (source_rows,)


@app.cell
def eligibility(SOURCE_DATASET, source_rows):
    candidates = []
    _excluded_prompt_rows = 0
    _ignored_label_rows = 0
    for _source_index, _row in enumerate(source_rows):
        _prompt = (_row["prompt"] or "").strip()
        _source_label = (_row["prompt_label"] or "").strip().casefold()
        if not _prompt or _prompt.casefold() == "redacted":
            _excluded_prompt_rows += 1
            continue
        if _source_label not in {"safe", "unsafe"}:
            _ignored_label_rows += 1
            continue
        _subcategory = (_row["violated_categories"] or "").strip() or None
        candidates.append(
            {
                "case_id": f"aegis2:{_row['id']}",
                "source_dataset": SOURCE_DATASET,
                "source_index": _source_index,
                "prompt": _prompt,
                "prompt_harm_label": {
                    "unsafe": "harmful",
                    "safe": "unharmful",
                }[_source_label],
                "adversarial": None,
                "subcategory": _subcategory,
            }
        )
    excluded_prompt_rows = _excluded_prompt_rows
    ignored_label_rows = _ignored_label_rows
    return candidates, excluded_prompt_rows, ignored_label_rows


@app.cell
def deduplicate(SEED, candidates, deduplicate_consistent):
    eligible_rows, conflicting_prompt_keys = deduplicate_consistent(
        candidates, seed=SEED
    )
    assert conflicting_prompt_keys == 167
    assert len(eligible_rows) == 22_850
    return conflicting_prompt_keys, eligible_rows


@app.cell
def sample(
    EXPECTED_LABEL_QUOTAS,
    SAMPLE_SIZE,
    SEED,
    eligible_rows,
    stratified_sample,
):
    selected_rows, label_quotas, eligible_label_counts = stratified_sample(
        eligible_rows,
        SAMPLE_SIZE,
        SEED,
        key=lambda row: row["prompt_harm_label"],
    )
    assert label_quotas == EXPECTED_LABEL_QUOTAS
    _repeat_rows, _repeat_quotas, _ = stratified_sample(
        eligible_rows,
        SAMPLE_SIZE,
        SEED,
        key=lambda row: row["prompt_harm_label"],
    )
    assert _repeat_quotas == label_quotas
    assert [row["case_id"] for row in _repeat_rows] == [
        row["case_id"] for row in selected_rows
    ]
    return (selected_rows,)


@app.cell
def save_and_validate(
    CASE_COLUMNS,
    Counter,
    EXPECTED_LABEL_QUOTAS,
    OUTPUT_PATH,
    read_cases,
    selected_rows,
    write_cases,
):
    selected_label_counts = Counter(
        row["prompt_harm_label"] for row in selected_rows
    )
    assert dict(selected_label_counts) == EXPECTED_LABEL_QUOTAS
    output_bytes = write_cases(OUTPUT_PATH, selected_rows)
    validated_rows = read_cases(OUTPUT_PATH)
    assert tuple(validated_rows[0]) == CASE_COLUMNS
    assert [row["case_id"] for row in validated_rows] == [
        row["case_id"] for row in selected_rows
    ]
    return output_bytes, selected_label_counts, validated_rows


@app.cell
def summary(
    OUTPUT_PATH,
    candidates,
    conflicting_prompt_keys,
    eligible_rows,
    excluded_prompt_rows,
    ignored_label_rows,
    mo,
    output_bytes,
    selected_label_counts,
    source_rows,
    validated_rows,
):
    mo.vstack(
        [
            mo.md(
                f"""
    ## Selection complete

    - Source rows: **{len(source_rows):,}**
    - Blank/REDACTED prompts excluded: **{excluded_prompt_rows:,}**
    - Unknown labels ignored: **{ignored_label_rows:,}**
    - Candidate rows: **{len(candidates):,}**
    - Conflicting normalized prompt keys removed: **{conflicting_prompt_keys:,}**
    - Eligible normalized-unique rows: **{len(eligible_rows):,}**
    - Selected labels: **{dict(selected_label_counts)}**
    - Output: `{OUTPUT_PATH.name}` ({output_bytes / 1_000_000:.2f} MB)
    """
            ),
            mo.ui.table(validated_rows[:20], selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
