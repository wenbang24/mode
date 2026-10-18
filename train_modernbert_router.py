# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "marimo==0.24.0",
#     "accelerate>=1.2,<2",
#     "pyarrow>=18",
#     "torch>=2.7",
#     "transformers==4.57.6",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    # Train the MoDE router with ModernBERT

    Learn the **fastest successful expert** for a prompt: AdaSteer, PIGuard, or
    GuardAgent. Benchmark all three on **Aegis unseen 1,000** in
    `benchmark_guardrails.py`, then provide that run's directory below.
    The original prompt is the only model input; benchmark success and timings
    supply its training label.

    The unseen pool is split 80/20 for training and validation with seed 42.
    All labeled training examples are used; confidence is uncalibrated softmax probability.
    **Aegis test (1,914 prompts)** stays separate for final evaluation.
    You can fine-tune before running its benchmark: leave the test benchmark
    directory blank until step 5. Only the reserved test Parquet is needed
    during preparation, to check for overlap with training prompts.

    Run on Molab with the RTX 6000 Pro (96 GB), from this repository with its
    `scripts/` directory and Parquets available. Locally, open with
    `uvx marimo@0.24.0 edit train_modernbert_router.py --sandbox`.
    Every expensive or writing action requires its own button.
    """)
    return


@app.cell
def imports():
    import json
    from pathlib import Path

    import marimo as mo
    from scripts.modernbert_router import (
        BASE_MODEL,
        BASE_REVISION,
        DEFAULT_SETTINGS,
        OUTCOME_NOTE,
        ModernBERTRouter,
        cuda_smoke_test,
        download_base,
        evaluate_test,
        export_router,
        inspect_training,
        prepare_data,
        train_router,
    )

    WORKSPACE = Path(__file__).resolve().parent
    return (
        BASE_MODEL, BASE_REVISION, DEFAULT_SETTINGS, OUTCOME_NOTE, ModernBERTRouter,
        WORKSPACE, Path, cuda_smoke_test, download_base, evaluate_test, export_router,
        inspect_training, json, mo, prepare_data, train_router,
    )


@app.cell(hide_code=True)
def assumptions(BASE_MODEL, BASE_REVISION, OUTCOME_NOTE, mo):
    mo.vstack([
        mo.callout(OUTCOME_NOTE, kind="info"),
        mo.md(f"Base: `{BASE_MODEL}@{BASE_REVISION}`. "
              "Exact latency ties use AdaSteer → PIGuard → GuardAgent. "
              "All-failed cases have no training label and remain in evaluation success rates. "
              "Runs with only one target expert are rejected; inspect the target distribution first."),
    ])
    return


@app.cell
def input_controls(WORKSPACE, mo):
    training_benchmark_control = mo.ui.text(
        value=str(WORKSPACE / "benchmarks/unseen") if (WORKSPACE / "benchmarks/unseen/run_config.json").is_file() else "",
        label="Unseen Aegis benchmark run directory (contains run_config.json)", full_width=True)
    train_parquet_control = mo.ui.text(
        value=str(WORKSPACE / "aegis2_unseen_1000_seed42.parquet"), label="Original unseen prompts", full_width=True)
    test_parquet_control = mo.ui.text(
        value=str(WORKSPACE / "aegis2_test.parquet"), label="Reserved test prompts (Parquet only, for overlap checks)", full_width=True)
    output_control = mo.ui.text(
        value=str(WORKSPACE / "artifacts/modernbert_router"), label="Artifact output root", full_width=True)
    prepare_button = mo.ui.run_button(label="1 · Validate benchmark and prepare training data")
    mo.vstack([
        mo.md("## 1 · Benchmark supervision"),
        mo.md("Use a complete run for all three experts, with no outstanding errors. "
              "The run folder is printed by the benchmark. JSONL takes precedence over grouped JSON "
              "so interrupted/resumed work cannot silently use an older grouped snapshot. "
              "An Aegis test benchmark run is not required to prepare data or fine-tune."),
        training_benchmark_control, train_parquet_control,
        test_parquet_control, output_control, prepare_button,
    ])
    return (
        output_control, prepare_button, test_parquet_control,
        train_parquet_control, training_benchmark_control,
    )


@app.cell
def prepare_action(
    json, mo, output_control, prepare_button, prepare_data, test_parquet_control,
    train_parquet_control, training_benchmark_control,
):
    prepared_data = None
    if prepare_button.value:
        try:
            if not training_benchmark_control.value.strip():
                raise ValueError("Enter the unseen Aegis benchmark run directory")
            prepared_data = prepare_data(
                training_benchmark_control.value, train_parquet_control.value,
                test_parquet_control.value, output_control.value)
            _view = mo.vstack([
                mo.callout(f"Prepared data: {prepared_data['directory']}", kind="success"),
                mo.plain_text(json.dumps(prepared_data["manifest"]["splits"], indent=2)),
            ])
        except Exception as _exc:
            _view = mo.callout(f"Data preparation failed: {_exc}", kind="danger")
    else:
        _view = mo.md("Data preparation is idle; no benchmark files have been read.")
    _view
    return (prepared_data,)


@app.cell
def model_controls(WORKSPACE, mo):
    base_cache_control = mo.ui.text(
        value=str(WORKSPACE / ".cache/modernbert_router"), label="Pinned base checkpoint cache", full_width=True)
    download_button = mo.ui.run_button(label="2 · Download / reuse pinned ModernBERT checkpoint")
    mo.vstack([mo.md("## 2 · Base checkpoint"), base_cache_control, download_button])
    return base_cache_control, download_button


@app.cell
def download_action(base_cache_control, download_base, download_button, mo):
    base_directory = None
    if download_button.value:
        try:
            with mo.status.spinner(title="Preparing the pinned ModernBERT-large checkpoint"):
                base_directory = download_base(base_cache_control.value)
            _view = mo.callout(f"Base checkpoint ready: {base_directory}", kind="success")
        except Exception as _exc:
            _view = mo.callout(f"Checkpoint download failed: {_exc}", kind="danger")
    else:
        _view = mo.md("No checkpoint download requested.")
    _view
    return (base_directory,)


@app.cell
def training_controls(mo):
    epochs_control = mo.ui.number(start=1, value=8, step=1, label="Epochs")
    micro_batch_control = mo.ui.number(start=1, value=8, step=1, label="Micro-batch")
    accumulation_control = mo.ui.number(start=1, value=2, step=1, label="Gradient accumulation")
    learning_rate_control = mo.ui.number(start=0.0000001, value=0.00002, step=0.000001, label="Learning rate")
    max_len_control = mo.ui.number(start=128, stop=8192, value=1024, step=128, label="Sequence tokens")
    device_control = mo.ui.dropdown(options=["auto", "cuda", "mps", "cpu"], value="auto", label="Device")
    amp_control = mo.ui.switch(value=True, label="CUDA mixed precision")
    inspect_button = mo.ui.run_button(label="3 · Check tokens and optimizer budget")
    smoke_button = mo.ui.run_button(label="Run CUDA smoke test (20 examples)")
    train_button = mo.ui.run_button(label="4 · Fine-tune and validate", kind="success")
    mo.vstack([
        mo.md("## 3–4 · Inspect and fine-tune"),
        mo.md("Fully train ModernBERT-large and a new three-class head using cross-entropy. "
              "Defaults use AdamW, 0.01 weight decay, 10% linear warmup, dynamic padding, "
              "gradient checkpointing, and seed 42. CUDA uses BF16 when supported, otherwise FP16; "
              "MPS and CPU use FP32. Auto selects CUDA, then MPS, then CPU. "
              "Each training click starts a new run. The lowest validation-loss checkpoint is saved "
              "for evaluation and export. No additional calibration split is taken."),
        mo.hstack([epochs_control, micro_batch_control, accumulation_control]),
        mo.hstack([learning_rate_control, max_len_control, device_control]), amp_control,
        mo.hstack([inspect_button, smoke_button, train_button]),
    ])
    return (
        accumulation_control, amp_control, device_control, epochs_control,
        inspect_button, learning_rate_control, max_len_control,
        micro_batch_control, smoke_button, train_button,
    )


@app.cell
def training_settings(
    DEFAULT_SETTINGS, accumulation_control, amp_control, epochs_control,
    learning_rate_control, max_len_control, micro_batch_control,
):
    settings = {
        **DEFAULT_SETTINGS,
        "epochs": int(epochs_control.value), "micro_batch": int(micro_batch_control.value),
        "grad_accum": int(accumulation_control.value), "learning_rate": learning_rate_control.value,
        "max_len": int(max_len_control.value), "amp": amp_control.value,
    }
    return (settings,)


@app.cell
def inspect_action(base_directory, inspect_button, inspect_training, json, mo, prepared_data, settings):
    if inspect_button.value:
        try:
            if prepared_data is None or base_directory is None:
                raise ValueError("Prepare data and download the checkpoint first")
            _report = inspect_training(prepared_data["directory"], base_directory, settings)
            _view = mo.plain_text(json.dumps(_report, indent=2))
        except Exception as _exc:
            _view = mo.callout(f"Training inspection failed: {_exc}", kind="danger")
    else:
        _view = mo.md("Token inspection is idle. Training also repeats this check before loading weights.")
    _view
    return


@app.cell
def train_action(
    base_directory, device_control, json, mo, output_control, prepared_data,
    settings, train_button, train_router,
):
    trained_run = None
    if train_button.value:
        try:
            if prepared_data is None or base_directory is None:
                raise ValueError("Prepare data and download the checkpoint first")
            with mo.status.spinner(title="Fine-tuning ModernBERT and evaluating the validation prompts"):
                trained_run = train_router(prepared_data, base_directory, output_control.value,
                                           settings, device_control.value)
            _view = mo.vstack([
                mo.callout(f"Run saved: {trained_run['directory']}", kind="success"),
                mo.plain_text(json.dumps(trained_run["inspection"], indent=2)),
                mo.ui.table(trained_run["validation"]["summary"], selection=None),
                mo.md("Validation results above include the all-failed prompts. "
                      "Training loss uses only examples with a routing target. "
                      "Confidence is uncalibrated softmax probability. Inspect selection counts for collapse."),
            ])
        except Exception as _exc:
            _view = mo.callout(f"Training failed: {_exc}. Check the console and the saved run directory.", kind="danger")
    else:
        _view = mo.md("Training is idle.")
    _view
    return (trained_run,)


@app.cell
def smoke_action(base_directory, cuda_smoke_test, json, mo, output_control, prepared_data, settings, smoke_button):
    if smoke_button.value:
        try:
            if prepared_data is None or base_directory is None:
                raise ValueError("Prepare data and download the checkpoint first")
            with mo.status.spinner(title="Checking CUDA training, checkpoint save, and reload on 20 examples"):
                _report = cuda_smoke_test(prepared_data, base_directory, output_control.value, settings)
            _view = mo.plain_text(json.dumps(_report, indent=2))
        except Exception as _exc:
            _view = mo.callout(f"CUDA smoke test failed: {_exc}", kind="danger")
    else:
        _view = mo.md("CUDA smoke test is idle. Run it on Molab before the full fine-tune.")
    _view
    return


@app.cell
def saved_run_controls(mo):
    test_benchmark_control = mo.ui.text(
        value="", label="Aegis test benchmark run directory (required only for step 5)", full_width=True)
    saved_run_control = mo.ui.text(
        value="", label="Saved run directory (optional: evaluate/export an earlier completed run)", full_width=True)
    test_button = mo.ui.run_button(label="5 · Evaluate on the reserved Aegis test set")
    export_button = mo.ui.run_button(label="6 · Export and verify checkpoint reload")
    mo.vstack([
        mo.md("## 5–6 · Final evaluation and export"),
        mo.md("Benchmark **Aegis 2.0** with the exact same three expert artifacts for the test action. "
              "Test results never enter fine-tuning or checkpoint selection. A saved run path takes precedence "
              "over this session's trained run. Export does not require a test run; absent final-test "
              "results must not be treated as evidence of test performance."),
        test_benchmark_control, saved_run_control, mo.hstack([test_button, export_button]),
    ])
    return export_button, saved_run_control, test_benchmark_control, test_button


@app.cell
def active_run(saved_run_control, trained_run):
    active_run_directory = saved_run_control.value.strip() or (trained_run["directory"] if trained_run else None)
    return (active_run_directory,)


@app.cell
def test_action(
    active_run_directory, device_control, evaluate_test, mo, test_benchmark_control,
    test_button, test_parquet_control,
):
    if test_button.value:
        try:
            if not active_run_directory or not test_benchmark_control.value.strip():
                raise ValueError("Choose a trained run and an Aegis test benchmark directory")
            with mo.status.spinner(title="Comparing trained, fixed-expert, and oracle policies on Aegis test"):
                _report = evaluate_test(active_run_directory, test_benchmark_control.value,
                                        test_parquet_control.value, device_control.value)
            _view = mo.ui.table(_report["summary"], selection=None)
        except Exception as _exc:
            _view = mo.callout(f"Test evaluation failed: {_exc}", kind="danger")
    else:
        _view = mo.md("Final test evaluation is idle.")
    _view
    return


@app.cell
def export_action(active_run_directory, device_control, export_button, export_router, mo, output_control):
    exported_directory = None
    if export_button.value:
        try:
            if not active_run_directory:
                raise ValueError("Train a router or provide a completed saved run")
            with mo.status.spinner(title="Exporting and verifying a fresh checkpoint reload"):
                exported_directory = export_router(active_run_directory, output_control.value, device_control.value)
            _view = mo.callout(f"Reload verified. Exported router: {exported_directory}", kind="success")
        except Exception as _exc:
            _view = mo.callout(f"Export failed: {_exc}", kind="danger")
    else:
        _view = mo.md("Export is idle.")
    _view
    return (exported_directory,)


@app.cell
def demo_controls(mo):
    demo_prompt = mo.ui.text_area(value="Explain how rainbows form.", label="Prompt to route", full_width=True)
    demo_button = mo.ui.run_button(label="Select an expert from the exported checkpoint")
    mo.vstack([
        mo.md("## Prompt-only routing demo"), demo_prompt, demo_button,
        mo.md("Reuse in MoDE: `from scripts.modernbert_router import ModernBERTRouter`, then "
              "`with ModernBERTRouter(export_path) as router: result = router.predict(prompt)`. "
              "The result contains `expert`, `probabilities`, `confidence`, and `truncated`. "
              "This selects an expert; it does not execute it."),
    ])
    return demo_button, demo_prompt


@app.cell
def demo_action(ModernBERTRouter, demo_button, demo_prompt, device_control, exported_directory, json, mo):
    if demo_button.value:
        try:
            if not exported_directory:
                raise ValueError("Export a checkpoint first")
            with ModernBERTRouter(exported_directory, device_control.value) as _router:
                _prediction = _router.predict(demo_prompt.value)
            _view = mo.plain_text(json.dumps(_prediction, indent=2))
        except Exception as _exc:
            _view = mo.callout(f"Routing demo failed: {_exc}", kind="danger")
    else:
        _view = mo.md("Routing demo is idle.")
    _view
    return


if __name__ == "__main__":
    app.run()
