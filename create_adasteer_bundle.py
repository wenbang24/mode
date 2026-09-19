# /// script
# dependencies = [
#     "accelerate==1.2.1",
#     "marimo==0.24.0",
#     "numpy==2.1.3",
#     "pyarrow==18.1.0",
#     "torch",
#     "tqdm==4.67.1",
#     "transformers==4.46.3",
# ]
# requires-python = ">=3.10,<3.13"
#
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
#
# [tool.uv.sources]
# torch = [
#     { index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def title(mo):
    mo.md("""
    # AdaSteer decoder-model bundle notebook

    This marimo notebook uses the official AdaSteer `Probe` with a standard
    text-only decoder model to train on all 10,000 WildGuard or Aegis cases,
    tune on validation, and report metrics on the untouched test split.
    """)
    return


@app.cell
def imports():
    import os
    from pathlib import Path
    from types import SimpleNamespace

    import marimo as mo
    from scripts.local_judge import LocalJudgeServer, server_panel, normalize_endpoint, judge_key, positive_timeout
    local_server = LocalJudgeServer()

    from scripts.experts.adasteer import AdaSteer
    from scripts.experts.adasteer_bundle import (
        DEFAULT_MODEL_ID,
        DEFAULT_OUTPUT_ROOT,
        build_bundle,
        model_slug,
        preflight_bundle_inputs,
        verify_bundle,
    )

    return (
        local_server, server_panel, normalize_endpoint, judge_key, positive_timeout,
        AdaSteer,
        DEFAULT_MODEL_ID,
        DEFAULT_OUTPUT_ROOT,
        Path,
        SimpleNamespace,
        build_bundle,
        mo,
        model_slug,
        os,
        preflight_bundle_inputs,
        verify_bundle,
    )


@app.cell
def local_server_controls(mo, local_server, server_panel):
    server_settings, server_view, get_server_status = server_panel(mo, local_server)
    endpoint_mode = mo.ui.dropdown(options=["Existing endpoint", "Managed local"], value="Existing endpoint", label="Judge connection")
    judge_timeout_control = mo.ui.number(start=1, value=180, label="Judge request timeout (seconds)")
    check_judge_button = mo.ui.run_button(label="Check judge")
    mo.vstack([endpoint_mode, server_view, judge_timeout_control, check_judge_button,
               mo.md("Keep the local judge running throughout the AdaSteer build.")])
    return server_settings, get_server_status, endpoint_mode, judge_timeout_control, check_judge_button


@app.cell
def local_server_status(mo, get_server_status):
    mo.ui.table([get_server_status()])
    return


@app.cell
def setup_adasteer(Path, mo):
    import subprocess

    adasteer_checkout = Path("AdaSteer").resolve()
    if not (adasteer_checkout / ".git").is_dir():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/MuyuenLP/AdaSteer.git",
                str(adasteer_checkout),
            ],
            check=True,
        )
    adasteer_commit = subprocess.run(
        ["git", "-C", str(adasteer_checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    mo.callout(
        f"Official AdaSteer checkout ready at `{adasteer_checkout}` "
        f"(commit `{adasteer_commit}`).",
        kind="success",
    )
    return adasteer_checkout, adasteer_commit


@app.cell
def controls(DEFAULT_MODEL_ID, DEFAULT_OUTPUT_ROOT, adasteer_checkout, mo):
    official_root_control = mo.ui.text(
        value=str(adasteer_checkout),
        label="Official AdaSteer checkout",
        full_width=True,
    )
    model_id_control = mo.ui.text(
        value=DEFAULT_MODEL_ID,
        label="Decoder model ID or path",
        full_width=True,
    )
    revision_control = mo.ui.text(
        value="", label="Model revision", full_width=True
    )
    dataset_preset_control = mo.ui.dropdown(
        options=["WildGuard", "Aegis 2.0"],
        value="WildGuard",
        label="Dataset",
    )
    rd_probe_layer_control = mo.ui.number(
        start=0, stop=255, value=5, label="RD probe layer"
    )
    hd_probe_layer_control = mo.ui.number(
        start=0, stop=255, value=13, label="HD probe layer"
    )
    output_control = mo.ui.text(
        value=str(DEFAULT_OUTPUT_ROOT),
        label="Bundle output root",
        full_width=True,
    )
    api_base_control = mo.ui.text(
        value="https://ai.hackclub.com/proxy/v1",
        label="Judge API base",
        full_width=True,
    )
    judge_model_control = mo.ui.text(
        value="openai/gpt-4o", label="Paper judge model", full_width=True
    )
    overwrite_control = mo.ui.checkbox(
        value=False, label="Replace an existing bundle"
    )
    preflight_button = mo.ui.run_button(label="1. Preflight")
    build_button = mo.ui.run_button(label="2. Train, validate, and test")
    demo_prompt_control = mo.ui.text_area(
        value="Explain how rainbows form.",
        label="Demonstration prompt",
        full_width=True,
    )
    demo_button = mo.ui.run_button(label="3. Run demonstration")

    controls_view = mo.vstack(
        [
            official_root_control,
            model_id_control,
            revision_control,
            dataset_preset_control,
            mo.hstack([rd_probe_layer_control, hd_probe_layer_control]),
            output_control,
            api_base_control,
            judge_model_control,
            overwrite_control,
            mo.hstack([preflight_button, build_button, demo_button]),
            demo_prompt_control,
            mo.callout(
                "Requires an fp16 CUDA environment compatible with Transformers 4.46.3. "
                "The full held-out test split runs after validation. Set HACKCLUB_API_KEY "
                "or JUDGE_API_KEY for hosted judging; local endpoints can run without a key. "
                "Set HF_TOKEN when the model requires it.",
                kind="info",
            ),
        ]
    )
    controls_view
    return (
        api_base_control,
        build_button,
        dataset_preset_control,
        demo_button,
        demo_prompt_control,
        judge_model_control,
        hd_probe_layer_control,
        model_id_control,
        official_root_control,
        output_control,
        overwrite_control,
        preflight_button,
        rd_probe_layer_control,
        revision_control,
    )


@app.cell
def configuration(
    Path,
    dataset_preset_control,
    hd_probe_layer_control,
    model_id_control,
    model_slug,
    official_root_control,
    output_control,
    rd_probe_layer_control,
    revision_control,
):
    workspace = Path(__file__).resolve().parent
    dataset_paths = {
        "WildGuard": {
            "train": workspace / "wildguardtrain_10000_seed42.parquet",
            "validation": workspace
            / "wildguardtrain_validation_1000_seed42.parquet",
            "test": workspace / "wildguardtest.parquet",
        },
        "Aegis 2.0": {
            "train": workspace / "aegis2_train_10000_seed42.parquet",
            "validation": workspace / "aegis2_validation.parquet",
            "test": workspace / "aegis2_test.parquet",
        },
    }[dataset_preset_control.value]
    official_root = Path(official_root_control.value).expanduser()
    train_path = dataset_paths["train"]
    validation_path = dataset_paths["validation"]
    test_path = dataset_paths["test"]
    output_root = Path(output_control.value).expanduser()
    model_id = model_id_control.value.strip()
    revision = revision_control.value.strip() or None
    rd_probe_layer = int(rd_probe_layer_control.value)
    hd_probe_layer = int(hd_probe_layer_control.value)
    target_bundle = (
        output_root / model_slug(train_path.stem) / model_slug(model_id)
    )
    return (
        hd_probe_layer,
        model_id,
        official_root,
        output_root,
        rd_probe_layer,
        revision,
        target_bundle,
        test_path,
        train_path,
        validation_path,
    )


@app.cell
def credentials(os):
    hf_token = os.environ.get("HF_TOKEN") or None
    return (hf_token,)


@app.cell
def preflight(
    hf_token,
    hd_probe_layer,
    mo,
    model_id,
    official_root,
    preflight_bundle_inputs,
    preflight_button,
    rd_probe_layer,
    revision,
    test_path,
    train_path,
    validation_path,
):
    if preflight_button.value:
        preflight_result = preflight_bundle_inputs(
            official_root=official_root,
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            model_id=model_id,
            revision=revision,
            token=hf_token,
            rd_probe_layer=rd_probe_layer,
            hd_probe_layer=hd_probe_layer,
        )
        preflight_view = mo.vstack(
            [mo.md("## Preflight passed"), mo.ui.table([preflight_result])]
        )
    else:
        preflight_result = None
        preflight_view = mo.callout("Press **1. Preflight** before starting the GPU build.")
    preflight_view
    return


@app.cell
def judge_context(
    AdaSteer, SimpleNamespace, api_base_control, judge_model_control,
    endpoint_mode, server_settings, judge_timeout_control,
    local_server, normalize_endpoint, judge_key, positive_timeout,
):
    managed_judge = endpoint_mode.value == "Managed local"
    local_settings = server_settings.value
    judge_endpoint = normalize_endpoint(
        f"http://127.0.0.1:{local_settings['port']}/v1" if managed_judge else api_base_control.value
    )
    selected_judge_model = (local_settings["model"] if managed_judge else judge_model_control.value).strip()
    judge_timeout = positive_timeout(judge_timeout_control.value)

    def paper_context():
        if not selected_judge_model:
            raise ValueError("Enter a judge model ID")
        if managed_judge:
            local_server.require_ready(local_settings)
        return SimpleNamespace(
            api_key="" if managed_judge else judge_key(judge_endpoint, "HACKCLUB_API_KEY"),
            api_base=judge_endpoint, judge_model=selected_judge_model,
            judge_timeout=judge_timeout, seed=42, constrain_judgments=managed_judge,
        )

    def judge_refusal(prompt, response):
        return AdaSteer._paper_judge(paper_context(), prompt, response, "refusal")["result"]

    def judge_compliance(prompt, response):
        return AdaSteer._paper_judge(paper_context(), prompt, response, "compliance")["result"]

    return judge_compliance, judge_refusal, paper_context, judge_endpoint, selected_judge_model, judge_timeout, managed_judge


@app.cell
def check_judge(mo, check_judge_button, judge_refusal, judge_compliance):
    if check_judge_button.value:
        try:
            result = {
                "refusal": judge_refusal("Explain rainbows.", "I cannot help with that request."),
                "compliance": judge_compliance("Say hello.", "Hello!"),
            }
            mo.output.replace(mo.ui.table([result]))
        except Exception as exc:
            mo.output.replace(mo.callout(str(exc), kind="danger"))
    return


@app.cell
def build(
    managed_judge,
    build_bundle,
    build_button,
    hf_token,
    hd_probe_layer,
    judge_compliance,
    selected_judge_model,
    judge_endpoint,
    paper_context,
    judge_refusal,
    mo,
    model_id,
    official_root,
    output_root,
    overwrite_control,
    rd_probe_layer,
    revision,
    test_path,
    train_path,
    validation_path,
):
    if build_button.value:
        paper_context()
        built_bundle = build_bundle(
            official_root=official_root,
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            output_root=output_root,
            judge_refusal=judge_refusal,
            judge_compliance=judge_compliance,
            model_id=model_id,
            revision=revision,
            token=hf_token,
            judge_model=selected_judge_model,
            judge_endpoint=judge_endpoint,
            judge_constrained=managed_judge,
            rd_probe_layer=rd_probe_layer,
            hd_probe_layer=hd_probe_layer,
            overwrite=overwrite_control.value,
        )
        build_view = mo.callout(f"Saved and verified: `{built_bundle}`", kind="success")
    else:
        built_bundle = None
        build_view = mo.callout(
            "Press **2. Train, validate, and test** after preflight passes."
        )
    build_view
    return


@app.cell
def bundle_status(mo, target_bundle, verify_bundle):
    if target_bundle.is_dir():
        bundle_metadata = verify_bundle(target_bundle)
        status_view = mo.vstack(
            [
                mo.md("## Current bundle"),
                mo.ui.table(
                    [
                        {
                            "path": str(target_bundle.resolve()),
                            "model": bundle_metadata["model_id"],
                            "model type": bundle_metadata["model_type"],
                            "layers × width": (
                                f"{bundle_metadata['num_hidden_layers']} × "
                                f"{bundle_metadata['hidden_size']}"
                            ),
                            "RD / HD probe layers": (
                                f"{bundle_metadata['rd_probe_layer']} / "
                                f"{bundle_metadata['hd_probe_layer']}"
                            ),
                            "train / validation / test": " / ".join(
                                f"{bundle_metadata['datasets'][split]['rows']:,}"
                                for split in ("train", "validation", "test")
                            ),
                            "behavior groups": bundle_metadata[
                                "behavior_group_counts"
                            ],
                            "RD / HD calibration": (
                                f"{len(bundle_metadata['calibration_records']['rd'])} / "
                                f"{len(bundle_metadata['calibration_records']['hd'])}"
                            ),
                            "validation harmful refusal": bundle_metadata[
                                "paper_metrics"
                            ]["final_harmful_refusal_rate"],
                            "validation benign compliance": bundle_metadata[
                                "paper_metrics"
                            ]["final_benign_full_compliance_rate"],
                            "test harmful refusal": bundle_metadata[
                                "test_metrics"
                            ]["harmful_refusal_rate"],
                            "test benign compliance": bundle_metadata[
                                "test_metrics"
                            ]["benign_full_compliance_rate"],
                            "test overall / balanced": (
                                f"{bundle_metadata['test_metrics']['overall_success_rate']:.1%} / "
                                f"{bundle_metadata['test_metrics']['balanced_success_rate']:.1%}"
                            ),
                            "official_commit": bundle_metadata[
                                "official_commit"
                            ],
                        }
                    ]
                ),
            ]
        )
    else:
        bundle_metadata = None
        work = target_bundle.parent / f".{target_bundle.name}.work"
        evaluations = work / "evaluations.jsonl"
        positions = work / "positions.jsonl"
        status_view = mo.callout(
            f"No completed bundle at `{target_bundle}`. Resume state: "
            f"{len(evaluations.read_text().splitlines()) if evaluations.exists() else 0} "
            f"judged generations and "
            f"{len(positions.read_text().splitlines()) if positions.exists() else 0} positions."
        )
    status_view
    return


@app.cell
def demonstration(
    AdaSteer,
    paper_context,
    judge_timeout,
    demo_button,
    demo_prompt_control,
    hf_token,
    mo,
    model_id,
    official_root,
    revision,
    target_bundle,
):
    if demo_button.value:
        context = paper_context()
        expert = AdaSteer(
            official_root,
            target_bundle,
            context.api_key,
            context.api_base,
            context.judge_model,
            model_id,
            hf_token,
            revision,
            judge_timeout=judge_timeout,
        )
        try:
            demo_outcome = expert.run(demo_prompt_control.value)
        finally:
            expert.close()
        demo_view = mo.vstack(
            [
                mo.md("## AdaSteer demonstration"),
                mo.ui.table(
                    [
                        {
                            "prompt": demo_prompt_control.value,
                            "response": demo_outcome.response,
                            "block": demo_outcome.block,
                            **{
                                key: demo_outcome.metadata[key]
                                for key in (
                                    "rd_position",
                                    "hd_position",
                                    "rd_coefficient",
                                    "hd_coefficient",
                                )
                            },
                        }
                    ]
                ),
            ]
        )
    else:
        demo_outcome = None
        demo_view = mo.callout("Build a bundle, then press **3. Run demonstration**.")
    demo_view
    return


@app.cell(hide_code=True)
def notes(mo):
    mo.md("""
    ## What this workflow owns

    - The official repository supplies `Probe`; removable decoder-layer hooks apply steering.
    - The model's selected-judge-labeled behavior—not stored source response labels—forms
      the RD/HD groups and calibrates the two bounded affine laws.
    - The generated bundle is separate from the checkout, so upstream vectors and
      source files are never overwritten.
    """)
    return


if __name__ == "__main__":
    app.run()
