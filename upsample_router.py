# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "marimo==0.24.0",
#     "accelerate>=1.2,<2",
#     "fsspec>=2024.6",
#     "pyarrow>=18",
#     "torch>=2.7",
#     "peft>=0.17,<0.19",
#     "transformers==4.57.6",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    # MoDE router up-sampling

    Prepare a resumable, route-balanced training set and compare B0, B1, and G1.
    Validation, calibration, and Aegis test membership are frozen. The generated
    cases are exported for a **separate** `benchmark_guardrails.py` run across all
    three experts, then imported here using `most_confident_successful` (all-failed
    cases become human review).

    Every expensive action has its own button. Outputs live in
    `artifacts/qwen_router/upsampling/<prepared-dir>/<step>/`; completed steps
    have `done.json`. Drive uploads omit checkpoints and `epochs/` folders.
    """)
    return


@app.cell
def imports():
    import os

    os.environ["USE_TF"] = "0"
    os.environ["USE_TORCH"] = "1"

    import json
    import shutil
    from pathlib import Path

    import marimo as mo
    from scripts import judge_confidence, laya_router, qwen_router, training_cases, upsampling
    from scripts.local_judge import (
        LocalJudgeServer,
        judge_key,
        normalize_endpoint,
        server_panel,
    )

    WORKSPACE = Path(__file__).resolve().parent
    DEFAULT_SETTINGS = qwen_router.DEFAULT_SETTINGS
    local_judge_server = LocalJudgeServer()
    return (
        DEFAULT_SETTINGS,
        WORKSPACE,
        judge_confidence,
        judge_key,
        json,
        laya_router,
        local_judge_server,
        mo,
        normalize_endpoint,
        os,
        Path,
        qwen_router,
        server_panel,
        shutil,
        training_cases,
        upsampling,
    )


@app.cell(hide_code=True)
def drive_filesystem(mo):
    # Paste the molab gdrive_fsspec cell here. Paths are relative to My Drive root;
    # use fs.put_file, fs.get_file, fs.ls/fs.exists, and fs.makedirs.
    fs = None
    mo.callout(
        "Drive is not connected. Paste the molab gdrive_fsspec cell here to enable restore and uploads.",
        kind="warn",
    )
    return (fs,)


@app.cell
def controls(DEFAULT_SETTINGS, WORKSPACE, mo):
    training_benchmark_control = mo.ui.text(
        value=str(WORKSPACE / "benchmarks/unseen"),
        label="Confidence-labelled Aegis unseen benchmark run directory",
        full_width=True,
    )
    train_parquet_control = mo.ui.text(
        value=str(WORKSPACE / "aegis2_unseen_10000_seed42.parquet"),
        label="Aegis unseen train-source Parquet",
        full_width=True,
    )
    test_parquet_control = mo.ui.text(
        value=str(WORKSPACE / "aegis2_test.parquet"),
        label="Frozen Aegis test Parquet",
        full_width=True,
    )
    output_control = mo.ui.text(
        value=str(WORKSPACE / "artifacts/qwen_router"),
        label="Artifact output root",
        full_width=True,
    )
    prepared_path_control = mo.ui.text(value="", label="Saved prepared directory (optional)", full_width=True)
    base_path_control = mo.ui.text(value="", label="Saved base checkpoint directory (optional)", full_width=True)
    device_control = mo.ui.dropdown(options=["auto", "cuda", "cpu"], value="auto", label="Training device")
    training_controls = mo.ui.dictionary({
        "micro_batch": mo.ui.number(start=1, value=DEFAULT_SETTINGS["micro_batch"], step=1, label="Micro-batch"),
        "grad_accum": mo.ui.number(start=1, value=DEFAULT_SETTINGS["grad_accum"], step=1, label="Gradient accumulation"),
        "learning_rate": mo.ui.number(start=0.000001, value=DEFAULT_SETTINGS["learning_rate"], step=0.000001, label="Learning rate"),
        "max_len": mo.ui.number(start=128, value=DEFAULT_SETTINGS["max_len"], step=128, label="Maximum sequence length"),
        "weight_decay": mo.ui.number(start=0, value=DEFAULT_SETTINGS["weight_decay"], step=0.001, label="Weight decay"),
        "warmup_ratio": mo.ui.number(start=0, stop=1, value=DEFAULT_SETTINGS["warmup_ratio"], step=0.01, label="Warmup ratio"),
        "gradient_checkpointing": mo.ui.checkbox(value=DEFAULT_SETTINGS["gradient_checkpointing"], label="Gradient checkpointing"),
        "amp": mo.ui.checkbox(value=DEFAULT_SETTINGS["amp"], label="Mixed precision"),
        "lora_rank": mo.ui.number(start=1, value=DEFAULT_SETTINGS["lora_rank"], step=1, label="LoRA rank"),
        "lora_alpha": mo.ui.number(start=1, value=DEFAULT_SETTINGS["lora_alpha"], step=1, label="LoRA alpha"),
        "brier_weight": mo.ui.number(start=0, value=DEFAULT_SETTINGS["brier_weight"], step=0.05, label="Brier loss weight"),
    })
    mo.vstack([
        mo.md("## Inputs and training controls"),
        training_benchmark_control,
        train_parquet_control,
        test_parquet_control,
        output_control,
        prepared_path_control,
        base_path_control,
        device_control,
        training_controls,
    ])
    return (
        base_path_control,
        device_control,
        output_control,
        prepared_path_control,
        test_parquet_control,
        train_parquet_control,
        training_benchmark_control,
        training_controls,
    )


@app.cell
def settings(training_controls, qwen_router):
    training_settings = qwen_router.validate_settings({
        **qwen_router.DEFAULT_SETTINGS,
        **training_controls.value,
        "epochs": 1,
        "seed": 42,
    })
    return (training_settings,)


@app.cell
def preparation_controls(mo):
    prepare_button = mo.ui.run_button(label="1 · Prepare frozen splits")
    download_base_button = mo.ui.run_button(label="2 · Download base checkpoint")
    folds_button = mo.ui.run_button(label="3 · Train OOF folds and label difficulty")
    mo.vstack([
        mo.md("## Prepare and tag original training cases"),
        prepare_button,
        download_base_button,
        folds_button,
    ])
    return download_base_button, folds_button, prepare_button


@app.cell
def prepare_action(
    fs,
    mo,
    output_control,
    prepare_button,
    qwen_router,
    shutil,
    test_parquet_control,
    train_parquet_control,
    training_benchmark_control,
    upsampling,
    json,
    Path,
):
    prepared_data = None
    if prepare_button.value:
        try:
            if not training_benchmark_control.value.strip():
                raise ValueError("Enter the confidence-labelled unseen benchmark run directory")
            prepared_data = qwen_router.prepare_data(
                training_benchmark_control.value,
                train_parquet_control.value,
                test_parquet_control.value,
                output_control.value,
            )
            _prep_name = Path(prepared_data["directory"]).name
            _step_dir = Path(output_control.value) / "upsampling" / _prep_name / "prepare"
            _drive_dir = f"MoDE/upsampling/{_prep_name}/prepare"
            _prepared_metadata = _step_dir / "prepared.json"
            if (_step_dir / "done.json").is_file() and _prepared_metadata.is_file():
                _stored = json.loads(_prepared_metadata.read_text())
                _prepared_dir = Path(_stored["directory"])
                if not _prepared_dir.is_dir():
                    _prepared_dir = _step_dir / "prepared_data"
                prepared_data = {"directory": str(_prepared_dir), "manifest": _stored["manifest"]}
            else:
                shutil.copytree(prepared_data["directory"], _step_dir / "prepared_data", dirs_exist_ok=True)
                upsampling.write_json(_prepared_metadata, {
                    "directory": prepared_data["directory"],
                    "manifest": prepared_data["manifest"],
                })
                upsampling.finish_step(
                    _step_dir, {"prepared_directory": str(prepared_data["directory"])}, fs, _drive_dir)
            _view = mo.callout(f"Prepared frozen splits: {prepared_data['directory']}", kind="success")
        except Exception as exc:
            _view = mo.callout(f"Preparation failed: {exc}", kind="danger")
    else:
        _view = mo.md("Preparation is idle.")
    _view
    return (prepared_data,)


@app.cell
def active_prepared(prepared_data, output_control, prepared_path_control, upsampling, restored_files, mo, Path):
    current_prepared = None
    try:
        if prepared_path_control.value.strip():
            current_prepared = upsampling.load_prepared_directory(
                output_control.value, prepared_path_control.value.strip())
        elif restored_files:
            current_prepared = upsampling.load_prepared_directory(output_control.value)
        elif prepared_data is not None:
            current_prepared = prepared_data
        else:
            current_prepared = upsampling.load_prepared_directory(output_control.value)
    except Exception as exc:
        _view = mo.callout(f"Prepared data is not available yet: {exc}", kind="warn")
    else:
        _view = mo.callout(f"Using prepared dataset `{Path(current_prepared['directory']).name}`", kind="info")
    _view
    return (current_prepared,)


@app.cell
def base_checkpoint_action(
    base_path_control,
    current_prepared,
    download_base_button,
    fs,
    json,
    mo,
    output_control,
    qwen_router,
    upsampling,
    Path,
):
    downloaded_base = None
    if download_base_button.value:
        try:
            if current_prepared is None:
                raise ValueError("Prepare data first")
            _prep_name = Path(current_prepared["directory"]).name
            _step_dir = Path(output_control.value) / "upsampling" / _prep_name / "base"
            _drive_dir = f"MoDE/upsampling/{_prep_name}/base"
            _metadata_path = _step_dir / "base.json"
            if (_step_dir / "done.json").is_file() and _metadata_path.is_file() and Path(
                    json.loads(_metadata_path.read_text())["directory"]).is_dir():
                downloaded_base = json.loads(_metadata_path.read_text())["directory"]
            else:
                downloaded_base = qwen_router.download_base(Path(output_control.value) / "cache")
                upsampling.write_json(_metadata_path, {"directory": str(downloaded_base)})
                upsampling.finish_step(_step_dir, {"downloaded": True}, fs, _drive_dir)
            _base_view = mo.callout(f"Base checkpoint ready: {downloaded_base}", kind="success")
        except Exception as exc:
            _base_view = mo.callout(f"Base checkpoint download failed: {exc}", kind="danger")
    else:
        _base_view = mo.md("Base checkpoint download is idle.")
    _base_view
    return (downloaded_base,)


@app.cell
def folds_action(
    base_path_control,
    current_prepared,
    device_control,
    folds_button,
    fs,
    json,
    mo,
    output_control,
    training_settings,
    upsampling,
    Path,
):
    if folds_button.value:
        try:
            if current_prepared is None:
                raise ValueError("Prepare data first")
            _base_directory = base_path_control.value.strip()
            if not _base_directory:
                _base_file = Path(output_control.value) / "upsampling" / Path(current_prepared["directory"]).name / "base/base.json"
                if _base_file.is_file():
                    _base_directory = json.loads(_base_file.read_text())["directory"]
            if not _base_directory:
                raise ValueError("Download or enter the base checkpoint directory")
            _prep_name = Path(current_prepared["directory"]).name
            _fold_dir = Path(output_control.value) / "upsampling" / _prep_name / "folds"

            def _upload_oof(path):
                upsampling.upload_progress_file(
                    fs, path, _fold_dir, f"MoDE/upsampling/{_prep_name}/folds")

            _report = upsampling.run_folds(
                current_prepared, _base_directory, output_control.value, training_settings,
                device_control.value, fs=fs, drive_root=f"MoDE/upsampling/{_prep_name}/folds",
                on_progress=_upload_oof,
            )
            _fold_view = mo.callout(f"OOF folds complete: {_report['directory']}", kind="success")
        except Exception as exc:
            _fold_view = mo.callout(f"Fold training failed: {exc}", kind="danger")
    else:
        _fold_view = mo.md("OOF fold training is idle.")
    _fold_view
    return


@app.cell
def judge_controls(local_judge_server, mo, server_panel):
    local_settings, local_server_view, get_local_status = server_panel(
        mo, local_judge_server, model="Qwen/Qwen3.8-27B", gpu_memory=0.8)
    endpoint_mode = mo.ui.dropdown(
        options=["Local vLLM", "Hosted OpenAI-compatible"],
        value="Local vLLM",
        label="LLM endpoint",
    )
    hosted_endpoint = mo.ui.text(value="https://api.openai.com/v1", label="Hosted endpoint URL")
    hosted_model = mo.ui.text(value="Qwen/Qwen3.8-27B", label="Hosted model ID")
    judge_timeout = mo.ui.number(start=1, value=180, step=1, label="Judge request timeout (seconds)")
    mo.vstack([
        mo.md("## LLM endpoint"),
        endpoint_mode,
        local_server_view,
        hosted_endpoint,
        hosted_model,
        judge_timeout,
    ])
    return endpoint_mode, get_local_status, hosted_endpoint, hosted_model, judge_timeout, local_settings


@app.cell
def judge_configuration(
    endpoint_mode,
    get_local_status,
    hosted_endpoint,
    hosted_model,
    judge_key,
    judge_timeout,
    local_settings,
    normalize_endpoint,
):
    try:
        if endpoint_mode.value == "Local vLLM":
            if get_local_status()["state"] != "ready":
                raise RuntimeError("Start the local vLLM server first")
            judge_endpoint = normalize_endpoint(
                f"http://127.0.0.1:{int(local_settings.value['port'])}/v1")
            judge_model = local_settings.value["model"]
        else:
            judge_endpoint = normalize_endpoint(hosted_endpoint.value)
            judge_model = hosted_model.value.strip()
            if not judge_model:
                raise ValueError("Enter a hosted model ID")
        judge_api_key = judge_key(judge_endpoint, "JUDGE_API_KEY")
        judge_config = {
            "endpoint": judge_endpoint,
            "model": judge_model,
            "api_key": judge_api_key,
            "timeout": float(judge_timeout.value),
            "error": None,
        }
    except Exception as exc:
        judge_config = {"error": str(exc)}
    return (judge_config,)


@app.cell
def category_controls(mo):
    categories_button = mo.ui.run_button(label="4 · Categorize train, validation, and test")
    category_view = mo.vstack([
        mo.md("## Category labels"),
        mo.md("The category inventory comes from the train-source Parquet. Existing single categories and benign cases need no LLM call; ambiguous harmful prompts use up to three strict-JSON attempts."),
        categories_button,
    ])
    category_view
    return (categories_button,)


@app.cell
def categories_action(
    categories_button,
    current_prepared,
    fs,
    judge_config,
    mo,
    output_control,
    test_parquet_control,
    train_parquet_control,
    upsampling,
    training_cases,
    json,
    Path,
):
    category_report = None
    if categories_button.value:
        try:
            if current_prepared is None:
                raise ValueError("Prepare data first")
            _prep_name = Path(current_prepared["directory"]).name
            _step_dir = Path(output_control.value) / "upsampling" / _prep_name / "categories"
            _drive_dir = f"MoDE/upsampling/{_prep_name}/categories"
            if (_step_dir / "done.json").is_file():
                category_report = json.loads((_step_dir / "done.json").read_text())
            else:
                if judge_config.get("error"):
                    raise ValueError(judge_config["error"])
                _train_source = training_cases.read_cases(train_parquet_control.value, expected_rows=None)
                _categories = upsampling.build_category_list(_train_source)
                upsampling.write_json(_step_dir / "categories.json", _categories)

                def _call_category(case, choices):
                    for _ in range(3):
                        response = upsampling.request_json(
                            judge_config["endpoint"], judge_config["api_key"], judge_config["model"],
                            upsampling.category_prompt(case, choices), 0,
                            timeout=judge_config["timeout"], attempts=1,
                            system_prompt=upsampling.CATEGORY_SYSTEM_PROMPT,
                        )
                        if set(response) == {"category"} and response["category"] in choices:
                            return response["category"]
                    return None

                _train_cases = json.loads((Path(current_prepared["directory"]) / "train_cases.json").read_text())
                _validation_entries = json.loads((Path(current_prepared["directory"]) / "validation_cases.json").read_text())
                _validation_cases = [entry["case"] for entry in _validation_entries]
                _test_cases = training_cases.read_cases(test_parquet_control.value, expected_rows=None)

                def _upload_category_log(path):
                    upsampling.upload_progress_file(fs, path, _step_dir, _drive_dir)

                upsampling.label_categories(
                    _train_cases, _categories, _step_dir / "train_categories.jsonl",
                    _call_category, workers=4, on_progress=_upload_category_log)
                upsampling.label_categories(
                    _validation_cases, _categories, _step_dir / "validation_categories.jsonl",
                    _call_category, workers=4, on_progress=_upload_category_log)
                upsampling.label_categories(
                    _test_cases, _categories, _step_dir / "test_categories.jsonl",
                    _call_category, workers=4, on_progress=_upload_category_log)
                category_report = {
                    "categories": len(_categories),
                    "train": len(_train_cases),
                    "validation": len(_validation_cases),
                    "test": len(_test_cases),
                }
                upsampling.finish_step(_step_dir, category_report, fs, _drive_dir)
            _view = mo.callout(json.dumps(category_report, indent=2), kind="success")
        except Exception as exc:
            _view = mo.callout(f"Category labeling failed: {exc}", kind="danger")
    else:
        _view = mo.md("Category labeling is idle.")
    _view
    return (category_report,)


@app.cell
def balance_controls(mo):
    balance_n = mo.ui.number(start=1, value=600, step=1, label="Target N per route")
    balance_button = mo.ui.run_button(label="5 · Balance original train set")
    mo.vstack([
        mo.md("## Route and category balance"),
        balance_n,
        balance_button,
    ])
    return balance_button, balance_n


@app.cell
def balance_action(
    balance_button,
    balance_n,
    current_prepared,
    mo,
    output_control,
    upsampling,
    fs,
    json,
    Path,
):
    balance_report = None
    if balance_button.value:
        try:
            if current_prepared is None:
                raise ValueError("Prepare data first")
            _prep_name = Path(current_prepared["directory"]).name
            _step_dir = Path(output_control.value) / "upsampling" / _prep_name / "balance"
            _balance_path = _step_dir / "balance.json"
            if (_step_dir / "done.json").is_file() and _balance_path.is_file():
                balance_report = json.loads(_balance_path.read_text())
            else:
                _categories_dir = Path(output_control.value) / "upsampling" / _prep_name / "categories"
                _difficulty_rows = upsampling.read_jsonl(
                    Path(output_control.value) / "upsampling" / _prep_name / "folds/difficulty.jsonl")
                _difficulty = {row["case_id"]: row["difficulty"] for row in _difficulty_rows}
                _categories = {
                    row["case_id"]: row["category"]
                    for row in upsampling.read_jsonl(_categories_dir / "train_categories.jsonl")
                }
                _cases = json.loads((Path(current_prepared["directory"]) / "train_cases.json").read_text())
                _records = [{
                    **case,
                    "target": current_prepared["manifest"]["targets"][case["case_id"]],
                    "difficulty": _difficulty[case["case_id"]],
                    "category": _categories[case["case_id"]],
                } for case in _cases]
                _balance = upsampling.balance_cases(_records, int(balance_n.value))
                balance_report = {
                    "routes": _balance["routes"],
                    "kept_train_case_ids": _balance["kept_train_case_ids"],
                    "below_n": {
                        route: counts["target"] < int(balance_n.value)
                        for route, counts in _balance["routes"].items()
                    },
                }
                upsampling.write_json(_balance_path, balance_report)
                upsampling.finish_step(
                    _step_dir, {"routes": _balance["routes"]}, fs,
                    f"MoDE/upsampling/{_prep_name}/balance")
            _view = mo.vstack([
                mo.callout(json.dumps(balance_report["routes"], indent=2), kind="success"),
                mo.md("Routes marked `true` below have targets below N, as required by the 50% generated-case cap."),
                mo.plain_text(json.dumps(balance_report["below_n"], indent=2)),
            ])
        except Exception as exc:
            _view = mo.callout(f"Balancing failed: {exc}", kind="danger")
    else:
        _view = mo.md("Balancing is idle.")
    _view
    return (balance_report,)


@app.cell
def generation_controls(mo):
    generate_button = mo.ui.run_button(label="6 · Generate route-targeted cases")
    check_button = mo.ui.run_button(label="7 · Judge harm labels and export generated.parquet")
    mo.vstack([
        mo.md("## Generate and judge"),
        generate_button,
        check_button,
    ])
    return check_button, generate_button


@app.cell
def generate_action(
    current_prepared,
    fs,
    generate_button,
    judge_config,
    mo,
    output_control,
    test_parquet_control,
    upsampling,
    training_cases,
    json,
    Path,
):
    generated_rows = None
    if generate_button.value:
        try:
            if current_prepared is None:
                raise ValueError("Prepare data first")
            if judge_config.get("error"):
                raise ValueError(judge_config["error"])
            _prep_name = Path(current_prepared["directory"]).name
            _root = Path(output_control.value) / "upsampling" / _prep_name
            _step_dir = _root / "generate"
            _drive_dir = f"MoDE/upsampling/{_prep_name}/generate"
            if (_step_dir / "done.json").is_file():
                generated_rows = upsampling.read_jsonl(_step_dir / "deduped.jsonl")
            else:
                _category_rows = upsampling.read_jsonl(_root / "categories/train_categories.jsonl")
                _difficulty_rows = upsampling.read_jsonl(_root / "folds/difficulty.jsonl")
                _difficulty = {row["case_id"]: row["difficulty"] for row in _difficulty_rows}
                _category = {row["case_id"]: row["category"] for row in _category_rows}
                _train_cases = json.loads((Path(current_prepared["directory"]) / "train_cases.json").read_text())
                _rows = [{
                    **case,
                    "target": current_prepared["manifest"]["targets"][case["case_id"]],
                    "difficulty": _difficulty[case["case_id"]],
                    "category": _category[case["case_id"]],
                } for case in _train_cases]
                _balance_data = json.loads((_root / "balance/balance.json").read_text())
                _balance = {
                    **_balance_data,
                    "kept_cases": [row for row in _rows
                                   if row["case_id"] in set(_balance_data["kept_train_case_ids"])],
                }
                _requests = upsampling.allocate_requests(_rows, _balance)
                _frozen_rows = list(_train_cases)
                _validation_entries = json.loads((Path(current_prepared["directory"]) / "validation_cases.json").read_text())
                _frozen_rows.extend(entry["case"] for entry in _validation_entries)
                _frozen_rows.extend(training_cases.read_cases(test_parquet_control.value, expected_rows=None))
                _calibration_entries = json.loads((Path(current_prepared["directory"]) / "calibration_cases.json").read_text())
                _frozen_rows.extend(entry["case"] for entry in _calibration_entries)

                def call_generation(prompt):
                    for _ in range(3):
                        response = upsampling.request_json(
                            judge_config["endpoint"], judge_config["api_key"], judge_config["model"],
                            prompt, 0.9, timeout=judge_config["timeout"], attempts=1,
                            system_prompt=upsampling.GENERATION_SYSTEM_PROMPT)
                        if set(response) == {"prompt"} and isinstance(response["prompt"], str):
                            return response
                    return {"prompt": ""}

                def _upload_generation_log(path):
                    upsampling.upload_progress_file(fs, path, _step_dir, _drive_dir)

                generated_rows = upsampling.run_generation(
                    _requests, _rows, _step_dir / "generated_raw.jsonl", _step_dir / "deduped.jsonl",
                    _frozen_rows, call_generation, workers=4, on_progress=_upload_generation_log)
                upsampling.write_json(_step_dir / "requests.json", _requests)
                upsampling.finish_step(
                    _step_dir, {"requested": len(_requests), "deduped": len(generated_rows)}, fs, _drive_dir)
            _view = mo.callout(f"Generated unique prompts: {len(generated_rows)}", kind="success")
        except Exception as exc:
            _view = mo.callout(f"Generation failed: {exc}", kind="danger")
    else:
        _view = mo.md("Generation is idle.")
    _view
    return (generated_rows,)


@app.cell
def check_action(
    check_button,
    current_prepared,
    fs,
    judge_config,
    json,
    mo,
    output_control,
    upsampling,
    Path,
):
    checked_parquet = None
    if check_button.value:
        try:
            if current_prepared is None:
                raise ValueError("Prepare data first")
            if judge_config.get("error"):
                raise ValueError(judge_config["error"])
            _prep_name = Path(current_prepared["directory"]).name
            _root = Path(output_control.value) / "upsampling" / _prep_name
            _step_dir = _root / "check"
            _drive_dir = f"MoDE/upsampling/{_prep_name}/check"
            checked_parquet = _step_dir / "generated.parquet"
            if not (_step_dir / "done.json").is_file():
                _candidates = upsampling.read_jsonl(_root / "generate/deduped.jsonl")
                if not _candidates:
                    raise ValueError("Generate and deduplicate prompts first")

                def _judge_prompt(prompt):
                    for _ in range(3):
                        response = upsampling.request_json(
                            judge_config["endpoint"], judge_config["api_key"], judge_config["model"],
                            json.dumps({"prompt": prompt[:1500]}, ensure_ascii=False),
                            0, timeout=judge_config["timeout"], attempts=1,
                            system_prompt=upsampling.HARM_SYSTEM_PROMPT)
                        if set(response) == {"harmful"} and type(response["harmful"]) is bool:
                            return response
                    raise ValueError("Harm judge returned invalid JSON three times")

                def _upload_check_log(path):
                    upsampling.upload_progress_file(fs, path, _step_dir, _drive_dir)

                _checked = upsampling.run_check(
                    _candidates, _step_dir / "checked.jsonl", _judge_prompt, workers=4,
                    on_progress=_upload_check_log)
                _cases = upsampling.write_generated_parquet(_checked, checked_parquet)
                upsampling.finish_step(_step_dir, {
                    "cases": len(_cases),
                    "judge_passed": sum(row["passed"] for row in _checked),
                    "generated_parquet": str(checked_parquet),
                }, fs, _drive_dir)
            _view = mo.vstack([
                mo.callout(f"Generated benchmark input: {checked_parquet}", kind="success"),
                mo.md("Run `benchmark_guardrails.py` manually on this parquet with **all three experts** and `target_rule=most_confident_successful`. Then paste the completed run directory in step 8."),
            ])
        except Exception as exc:
            _view = mo.callout(f"Generated-case check failed: {exc}", kind="danger")
    else:
        _view = mo.md("Generated-case judging is idle.")
    _view
    return (checked_parquet,)


@app.cell
def confidence_controls(mo):
    benchmark_run_control = mo.ui.text(value="", label="Generated benchmark run directory", full_width=True)
    score_confidence_button = mo.ui.run_button(label="Score AdaSteer judge confidence")
    mo.vstack([
        mo.md("## Optional AdaSteer judge confidence"),
        benchmark_run_control,
        score_confidence_button,
        mo.md("This calls `score_adasteer` and `apply_confidence` on the completed generated run. It is required for the `most_confident_successful` route labels."),
    ])
    return benchmark_run_control, score_confidence_button


@app.cell
def confidence_action(
    benchmark_run_control,
    current_prepared,
    fs,
    judge_confidence,
    judge_config,
    mo,
    score_confidence_button,
    json,
    upsampling,
    Path,
):
    if score_confidence_button.value:
        try:
            if not benchmark_run_control.value.strip():
                raise ValueError("Enter the completed generated benchmark run directory")
            if judge_config.get("error"):
                raise ValueError(judge_config["error"])
            if current_prepared is None:
                raise ValueError("Prepare data first")
            _run_dir = benchmark_run_control.value.strip()
            _prep_name = Path(current_prepared["directory"]).name
            _remote_dir = f"MoDE/upsampling/{_prep_name}/benchmark/{Path(_run_dir).name}"

            def upload_confidence_progress(_counts):
                upsampling.upload_progress_file(
                    fs, Path(_run_dir) / judge_confidence.CONFIDENCE_FILE,
                    Path(_run_dir), _remote_dir)

            _counts = judge_confidence.score_adasteer(
                _run_dir, judge_config["endpoint"], judge_config["model"],
                timeout=judge_config["timeout"], workers=4, api_key=judge_config["api_key"],
                progress=upload_confidence_progress)
            _labels = judge_confidence.apply_confidence(_run_dir, judge_config["model"], lambda row: row)
            if fs is not None:
                upsampling.upload_json_reports(fs, _run_dir, _remote_dir)
            _view = mo.callout(json.dumps({
                "scoring": _counts,
                "cases_relabelled": len(_labels),
                "next": "Continue to import only after all three expert results and confidence scoring are complete.",
                "drive_upload": fs is not None,
            }, indent=2), kind="success")
        except Exception as exc:
            _view = mo.callout(f"AdaSteer confidence scoring failed: {exc}", kind="danger")
    else:
        _view = mo.md("AdaSteer confidence scoring is idle.")
    _view
    return


@app.cell
def import_controls(mo):
    import_button = mo.ui.run_button(label="8 · Import benchmark and build G1")
    mo.vstack([
        mo.md("## Import measured routes"),
        import_button,
    ])
    return (import_button,)


@app.cell
def import_action(
    benchmark_run_control,
    current_prepared,
    fs,
    import_button,
    laya_router,
    mo,
    output_control,
    test_parquet_control,
    training_cases,
    upsampling,
    json,
    Path,
):
    import_report = None
    if import_button.value:
        try:
            if current_prepared is None:
                raise ValueError("Prepare data first")
            if not benchmark_run_control.value.strip():
                raise ValueError("Enter the generated-case benchmark run directory")
            _prep_name = Path(current_prepared["directory"]).name
            _root = Path(output_control.value) / "upsampling" / _prep_name
            _step_dir = _root / "import"
            _drive_dir = f"MoDE/upsampling/{_prep_name}/import"
            _generated_parquet = _root / "check/generated.parquet"
            if (_step_dir / "done.json").is_file():
                import_report = json.loads((_step_dir / "done.json").read_text())
            else:
                _run_dir = Path(benchmark_run_control.value).expanduser().resolve()
                _manifest = json.loads((_run_dir / "run_config.json").read_text())
                if _manifest.get("target_rule") != "most_confident_successful":
                    raise ValueError("Generated benchmark must use target_rule=most_confident_successful")
                if set(_manifest.get("selected_guardrails", ())) != set(laya_router.EXPERTS):
                    raise ValueError("Generated benchmark must include all three experts")
                _benchmark = laya_router.load_benchmark(
                    _run_dir, _generated_parquet, expected_rows=None, split="generated")
                laya_router.check_matching_experts(
                    current_prepared["manifest"]["benchmark"], _benchmark.manifest)
                if fs is not None:
                    _remote = f"MoDE/upsampling/{_prep_name}/benchmark/{_run_dir.name}"
                    upsampling.upload_json_reports(fs, _run_dir, _remote)

                _train_cases = json.loads((Path(current_prepared["directory"]) / "train_cases.json").read_text())
                _difficulty = {row["case_id"]: row["difficulty"]
                               for row in upsampling.read_jsonl(_root / "folds/difficulty.jsonl")}
                _categories = {row["case_id"]: row["category"]
                               for row in upsampling.read_jsonl(_root / "categories/train_categories.jsonl")}
                _train_rows = [{
                    **case,
                    "target": current_prepared["manifest"]["targets"][case["case_id"]],
                    "difficulty": _difficulty[case["case_id"]],
                    "category": _categories[case["case_id"]],
                } for case in _train_cases]
                _balance_data = json.loads((_root / "balance/balance.json").read_text())
                _balance = {
                    **_balance_data,
                    "kept_cases": [row for row in _train_rows
                                   if row["case_id"] in set(_balance_data["kept_train_case_ids"])],
                }
                _originals, _added, _g1_targets, _leftovers = upsampling.build_g1_cases(
                    _balance, _benchmark.cases, _benchmark.targets, seed=42)
                _original_targets = {
                    case["case_id"]: current_prepared["manifest"]["targets"][case["case_id"]]
                    for case in _originals
                }
                _g1_targets.update(_original_targets)
                _g1_cases = _originals + _added
                _g1_rows = laya_router.labeled_rows(_g1_cases, _g1_targets)

                _source_train = upsampling.read_jsonl(Path(current_prepared["directory"]) / "train.jsonl")
                _b0_cases = json.loads((Path(current_prepared["directory"]) / "train_cases.json").read_text())
                _b0_targets = {case["case_id"]: current_prepared["manifest"]["targets"][case["case_id"]]
                               for case in _b0_cases}
                _b1_rows = upsampling.weighted_training_rows(_source_train)
                _variants = {
                    "B0": (_source_train, _b0_cases, _b0_targets),
                    "B1": (_b1_rows, _b0_cases, _b0_targets),
                    "G1": (_g1_rows, _g1_cases, _g1_targets),
                }
                _prepared_variants = {}
                for _name, (_rows, _cases, _targets) in _variants.items():
                    _prepared_variants[_name] = upsampling.prepare_variant(
                        current_prepared, _step_dir / "variants" / _name,
                        _rows, _cases, _targets, _name)
                _raw_rows = upsampling.read_jsonl(_root / "generate/generated_raw.jsonl")
                _deduped = upsampling.read_jsonl(_root / "generate/deduped.jsonl")
                _checked = upsampling.read_jsonl(_root / "check/checked.jsonl")
                _requests = json.loads((_root / "generate/requests.json").read_text())
                _losses = upsampling.count_stages(
                    _requests, _raw_rows, _deduped, _checked,
                    _benchmark.cases, _benchmark.targets, _added)
                upsampling.write_json(_step_dir / "losses.json", _losses)
                upsampling.write_json(_step_dir / "leftovers.json", _leftovers)
                import_report = {
                    "benchmark_cases": len(_benchmark.cases),
                    "target_counts": dict(__import__("collections").Counter(_benchmark.targets.values())),
                    "g1_cases": len(_g1_cases),
                    "added_cases": len(_added),
                    "leftover_case_ids": len(_leftovers),
                    "prepared_variants": {
                        _name: info["directory"] for _name, info in _prepared_variants.items()
                    },
                }
                upsampling.finish_step(_step_dir, import_report, fs, _drive_dir)
            _view = mo.vstack([
                mo.callout(json.dumps(import_report, indent=2), kind="success"),
                mo.plain_text((_step_dir / "losses.json").read_text()),
            ])
        except Exception as exc:
            _view = mo.callout(f"Benchmark import failed: {exc}", kind="danger")
    else:
        _view = mo.md("Benchmark import is idle.")
    _view
    return (import_report,)


@app.cell
def experiment_controls(mo):
    test_benchmark_control = mo.ui.text(value="", label="Aegis test benchmark run directory (optional)", full_width=True)
    train_button = mo.ui.run_button(label="9 · Train all remaining B0/B1/G1 seed runs")
    mo.vstack([
        mo.md("## Experiments"),
        test_benchmark_control,
        train_button,
    ])
    return test_benchmark_control, train_button


@app.cell
def experiment_action(
    current_prepared,
    device_control,
    fs,
    mo,
    output_control,
    base_path_control,
    test_benchmark_control,
    test_parquet_control,
    train_button,
    training_settings,
    upsampling,
    json,
    Path,
):
    if train_button.value:
        try:
            if current_prepared is None:
                raise ValueError("Prepare data first")
            _prep_name = Path(current_prepared["directory"]).name
            _root = Path(output_control.value) / "upsampling" / _prep_name
            _base_file = _root / "base/base.json"
            _base_directory = base_path_control.value.strip()
            if not _base_directory and _base_file.is_file():
                _base_directory = json.loads(_base_file.read_text())["directory"]
            if not _base_directory:
                raise ValueError("Download the base checkpoint first")
            _results = []
            for _variant in ("B0", "B1", "G1"):
                _variant_dir = _root / "import/variants" / _variant
                _prepared = {
                    "directory": str(_variant_dir),
                    "manifest": json.loads((_variant_dir / "manifest.json").read_text()),
                }
                for _seed in (42, 43, 44):
                    _name = f"{_variant}-seed{_seed}"
                    _result = upsampling.run_experiment(
                        _prepared, _base_directory, _root, _name,
                        {**training_settings, "epochs": 1, "seed": _seed},
                        device_control.value,
                        fs=fs,
                        drive_folder=f"MoDE/upsampling/{_prep_name}/runs/{_name}",
                        test_run_dir=test_benchmark_control.value.strip() or None,
                        test_parquet=test_parquet_control.value,
                    )
                    _results.append({"run": _name, **_result})
            _view = mo.callout(json.dumps(_results, indent=2), kind="success")
        except Exception as exc:
            _view = mo.callout(f"Experiment training failed: {exc}", kind="danger")
    else:
        _view = mo.md("Experiment training is idle.")
    _view
    return


@app.cell
def summary_controls(mo):
    summary_button = mo.ui.run_button(label="10 · Aggregate validation and optional test metrics")
    mo.vstack([
        mo.md("## Bootstrap summary"),
        summary_button,
        mo.md("Each metric uses 1,000 case-level bootstrap resamples with seed 0, aggregating the three saved training seeds."),
    ])
    return (summary_button,)


@app.cell
def summary_action(
    current_prepared,
    mo,
    output_control,
    summary_button,
    test_benchmark_control,
    test_parquet_control,
    upsampling,
    laya_router,
    fs,
    json,
    Path,
):
    summary_report = None
    if summary_button.value:
        try:
            if current_prepared is None:
                raise ValueError("Prepare data first")
            _prep_name = Path(current_prepared["directory"]).name
            _root = Path(output_control.value) / "upsampling" / _prep_name
            _summary_dir = _root / "summary"
            _summary_path = _summary_dir / "summary.json"
            if (_summary_dir / "done.json").is_file() and _summary_path.is_file():
                summary_report = json.loads(_summary_path.read_text())
            else:
                _validation_entries = json.loads(
                    (Path(current_prepared["directory"]) / "validation_cases.json").read_text())
                _validation_cases = [entry["case"] for entry in _validation_entries]
                _validation_targets = {
                    entry["case"]["case_id"]: entry["target"] for entry in _validation_entries
                }
                _validation_outcomes = {
                    entry["case"]["case_id"]: entry["outcomes"] for entry in _validation_entries
                }
                _validation_categories = upsampling.read_jsonl(_root / "categories/validation_categories.jsonl")
                summary_report = {}
                _test_data = None
                if test_benchmark_control.value.strip():
                    _test_data = laya_router.load_benchmark(
                        test_benchmark_control.value.strip(), test_parquet_control.value,
                        expected_rows=1914, split="test")
                for _variant in ("B0", "B1", "G1"):
                    _validation_predictions = {}
                    _test_predictions = {}
                    for _seed in (42, 43, 44):
                        _run_dir = _root / "runs" / f"{_variant}-seed{_seed}"
                        _validation_report = json.loads((_run_dir / "validation_report.json").read_text())
                        _validation_predictions[_seed] = {
                            row["case_id"]: row
                            for row in _validation_report["predictions"]["finetuned_qwen"]
                        }
                        _test_path = _run_dir / "test_report.json"
                        if _test_data is not None:
                            if not _test_path.is_file():
                                raise FileNotFoundError(
                                    f"Missing test report for {_variant}-seed{_seed}; rerun with the test benchmark directory")
                            _test_report = json.loads(_test_path.read_text())
                            _test_predictions[_seed] = {
                                row["case_id"]: row for row in _test_report["predictions"]["finetuned_qwen"]
                            }
                    _variant_result = {
                        "validation": upsampling.summarize_predictions(
                            _validation_cases, _validation_targets, _validation_outcomes,
                            _validation_predictions,
                            {row["case_id"]: row["category"] for row in _validation_categories},
                            resamples=1000, seed=0),
                    }
                    if _test_data is not None:
                        _test_categories = upsampling.read_jsonl(_root / "categories/test_categories.jsonl")
                        _variant_result["test"] = upsampling.summarize_predictions(
                            _test_data.cases, _test_data.targets, _test_data.outcomes, _test_predictions,
                            {row["case_id"]: row["category"] for row in _test_categories},
                            resamples=1000, seed=0)
                    summary_report[_variant] = _variant_result
                upsampling.write_json(_summary_path, summary_report)
                upsampling.finish_step(
                    _summary_dir, {"variants": list(summary_report), "bootstrap_resamples": 1000},
                    fs, f"MoDE/upsampling/{_prep_name}/summary")
            _view = mo.vstack([
                mo.callout("Summary saved and uploaded.", kind="success"),
                mo.plain_text(json.dumps(summary_report, indent=2)),
            ])
        except Exception as exc:
            _view = mo.callout(f"Summary failed: {exc}", kind="danger")
    else:
        _view = mo.md("Summary aggregation is idle.")
    _view
    return (summary_report,)


@app.cell
def restore_controls(mo):
    restore_button = mo.ui.run_button(label="Restore completed steps and partial JSONL logs from Drive")
    restore_button
    return (restore_button,)


@app.cell
def restore_action(
    fs,
    mo,
    output_control,
    restore_button,
    upsampling,
    Path,
):
    restored_files = []
    if restore_button.value:
        try:
            if fs is None:
                _view = mo.callout("Drive restore skipped because fs is None. Paste the molab Drive filesystem cell into drive_filesystem.", kind="warn")
            else:
                local_root = Path(output_control.value) / "upsampling"
                restored_files = upsampling.restore_from_drive(
                    fs, "MoDE/upsampling", local_root)
                _view = mo.callout(f"Restored {len(restored_files)} files into {local_root}", kind="success")
        except Exception as exc:
            _view = mo.callout(f"Drive restore failed: {exc}", kind="danger")
    else:
        _view = mo.md("Drive restore is idle.")
    _view
    return (restored_files,)


if __name__ == "__main__":
    app.run()
