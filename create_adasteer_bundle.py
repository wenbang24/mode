# /// script
# dependencies = [
#     "accelerate==1.2.1",
#     "marimo==0.24.0",
#     "numpy==2.1.3",
#     "pyarrow==18.1.0",
#     "torch==2.4.1",
#     "tqdm==4.67.1",
#     "transformers==4.46.3",
# ]
# requires-python = ">=3.10,<3.13"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def title(mo):
    mo.md("""
    # Official Qwen AdaSteer bundle notebook

    This marimo notebook uses the official AdaSteer `Probe` and Qwen steering
    model to reproduce the paper's Qwen procedure over a fixed 7,200/1,800
    split of the 9,000 WildGuard prompts.
    """)
    return


@app.cell
def imports():
    import os
    from pathlib import Path
    from types import SimpleNamespace

    import marimo as mo

    from scripts.experts.adasteer import AdaSteer
    from scripts.experts.adasteer_bundle import (
        DEFAULT_DATASET_PATH,
        DEFAULT_MODEL_ID,
        DEFAULT_OUTPUT_ROOT,
        build_bundle,
        model_slug,
        preflight_bundle_inputs,
        verify_bundle,
    )

    return (
        AdaSteer,
        DEFAULT_DATASET_PATH,
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
def controls(DEFAULT_DATASET_PATH, DEFAULT_MODEL_ID, DEFAULT_OUTPUT_ROOT, mo):
    official_root_control = mo.ui.text(
        value="/content/AdaSteer", label="Official AdaSteer checkout", full_width=True
    )
    model_id_control = mo.ui.text(
        value=DEFAULT_MODEL_ID, label="Qwen-2.5-7B model ID or path", full_width=True
    )
    revision_control = mo.ui.text(value="", label="Model revision", full_width=True)
    dataset_control = mo.ui.text(
        value=str(DEFAULT_DATASET_PATH), label="9,000-row WildGuard Parquet", full_width=True
    )
    output_control = mo.ui.text(
        value=str(DEFAULT_OUTPUT_ROOT), label="Bundle output root", full_width=True
    )
    api_base_control = mo.ui.text(
        value="https://ai.hackclub.com/proxy/v1", label="Judge API base", full_width=True
    )
    judge_model_control = mo.ui.text(
        value="openai/gpt-4o", label="Paper judge model", full_width=True
    )
    overwrite_control = mo.ui.checkbox(value=False, label="Replace an existing bundle")
    preflight_button = mo.ui.run_button(label="1. Preflight")
    build_button = mo.ui.run_button(label="2. Build bundle")
    demo_prompt_control = mo.ui.text_area(
        value="Explain how rainbows form.", label="Demonstration prompt", full_width=True
    )
    demo_button = mo.ui.run_button(label="3. Run demonstration")

    controls_view = mo.vstack(
        [
            official_root_control,
            model_id_control,
            revision_control,
            dataset_control,
            output_control,
            api_base_control,
            judge_model_control,
            overwrite_control,
            mo.hstack([preflight_button, build_button, demo_button]),
            demo_prompt_control,
            mo.callout(
                "Requires an fp16 CUDA environment compatible with Transformers 4.46.3. "
                "Set HACKCLUB_API_KEY for coefficient fitting and HF_TOKEN when the model requires it.",
                kind="info",
            ),
        ]
    )
    controls_view
    return (
        api_base_control,
        build_button,
        dataset_control,
        demo_button,
        demo_prompt_control,
        judge_model_control,
        model_id_control,
        official_root_control,
        output_control,
        overwrite_control,
        preflight_button,
        revision_control,
    )


@app.cell
def configuration(
    Path,
    dataset_control,
    model_id_control,
    model_slug,
    official_root_control,
    output_control,
    revision_control,
):
    official_root = Path(official_root_control.value).expanduser()
    dataset_path = Path(dataset_control.value).expanduser()
    output_root = Path(output_control.value).expanduser()
    model_id = model_id_control.value.strip()
    revision = revision_control.value.strip() or None
    target_bundle = output_root / model_slug(model_id)
    return (
        dataset_path,
        model_id,
        official_root,
        output_root,
        revision,
        target_bundle,
    )


@app.cell
def credentials(os):
    hf_token = os.environ.get("HF_TOKEN") or None
    api_key = os.environ.get("HACKCLUB_API_KEY") or ""
    return api_key, hf_token


@app.cell
def preflight(
    dataset_path,
    hf_token,
    mo,
    model_id,
    official_root,
    preflight_bundle_inputs,
    preflight_button,
    revision,
):
    if preflight_button.value:
        preflight_result = preflight_bundle_inputs(
            official_root, dataset_path, model_id, revision, hf_token
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
    AdaSteer,
    SimpleNamespace,
    api_base_control,
    api_key,
    judge_model_control,
):
    def paper_context():
        if not api_key:
            raise RuntimeError("HACKCLUB_API_KEY is required")
        return SimpleNamespace(
            api_key=api_key,
            api_base=api_base_control.value.rstrip("/"),
            judge_model=judge_model_control.value.strip(),
            seed=42,
        )

    def judge_refusal(prompt, response):
        return AdaSteer._paper_judge(
            paper_context(), prompt, response, "refusal"
        )["result"]

    def judge_compliance(prompt, response):
        return AdaSteer._paper_judge(
            paper_context(), prompt, response, "compliance"
        )["result"]

    return judge_compliance, judge_refusal


@app.cell
def build(
    build_bundle,
    build_button,
    dataset_path,
    hf_token,
    judge_compliance,
    judge_model_control,
    judge_refusal,
    mo,
    model_id,
    official_root,
    output_root,
    overwrite_control,
    revision,
):
    if build_button.value:
        built_bundle = build_bundle(
            official_root=official_root,
            dataset_path=dataset_path,
            output_root=output_root,
            judge_refusal=judge_refusal,
            judge_compliance=judge_compliance,
            model_id=model_id,
            revision=revision,
            token=hf_token,
            judge_model=judge_model_control.value.strip(),
            overwrite=overwrite_control.value,
        )
        build_view = mo.callout(f"Saved and verified: `{built_bundle}`", kind="success")
    else:
        built_bundle = None
        build_view = mo.callout("Press **2. Build bundle** after preflight passes.")
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
                            "rows": bundle_metadata["dataset_rows"],
                            "train / validation": "7,200 / 1,800",
                            "Qwen groups": bundle_metadata["qwen_group_counts"],
                            "RD / HD calibration": (
                                f"{len(bundle_metadata['calibration_records']['rd'])} / "
                                f"{len(bundle_metadata['calibration_records']['hd'])}"
                            ),
                            "harmful refusal": bundle_metadata["paper_metrics"][
                                "final_harmful_refusal_rate"
                            ],
                            "benign compliance": bundle_metadata["paper_metrics"][
                                "final_benign_full_compliance_rate"
                            ],
                            "official_commit": bundle_metadata["official_commit"],
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
    api_base_control,
    api_key,
    demo_button,
    demo_prompt_control,
    hf_token,
    judge_model_control,
    mo,
    model_id,
    official_root,
    revision,
    target_bundle,
):
    if demo_button.value:
        expert = AdaSteer(
            official_root,
            target_bundle,
            api_key,
            api_base_control.value,
            judge_model_control.value,
            model_id,
            hf_token,
            revision,
        )
        try:
            demo_outcome = expert.run(demo_prompt_control.value)
        finally:
            expert.close()
        demo_view = mo.vstack(
            [
                mo.md("## Official AdaSteer demonstration"),
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

    - The official repository owns activation capture and Qwen activation injection.
    - Qwen's GPT-4o-judged behavior—not stored WildGuard response labels—forms
      the RD/HD groups and calibrates the two bounded affine laws.
    - The generated bundle is separate from the checkout, so upstream vectors and
      source files are never overwritten.
    """)
    return


if __name__ == "__main__":
    app.run()
