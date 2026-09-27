# /// script
# dependencies = [
#     "httpx==0.27.2",
#     "marimo==0.24.0",
#     "openai==1.7.2",
#     "pyarrow==18.1.0",
#     "pyautogen==0.2.35",
#     "python-levenshtein==0.23.0",
#     "termcolor==2.4.0",
# ]
# requires-python = ">=3.12,<3.13"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def title(mo):
    mo.md("""
    # GuardAgent policy notebook

    Build GuardAgent's online policy memory from all 10,000 WildGuard or Aegis
    training cases, evaluate the frozen policy on validation, and unlock the
    untouched test split only after validation is complete. The notebook uses the
    official two-stage planner/code-generator pinned to commit
    `eb8797f0f3570800c1f596c40418edc929994a24`; generated code is evaluated by a
    restricted assignment-only interpreter.
    """)
    return


@app.cell
def imports():
    import json
    import os
    import subprocess
    import time
    from pathlib import Path

    import marimo as mo

    from scripts.local_judge import (
        LocalJudgeServer,
        judge_key,
        normalize_endpoint,
        positive_timeout,
        server_panel,
    )
    from scripts.experts.guardagent import (
        ALLOW_CODE,
        BLOCK_CODE,
        MEMORY_SHOTS,
        GuardAgent,
        restricted_execute,
    )
    from scripts.experts.guardagent_policy import (
        DATASET_PRESETS,
        DEFAULT_OUTPUT_ROOT,
        UPSTREAM_COMMIT,
        UPSTREAM_REPOSITORY,
        build_identity,
        check_guardagent,
        model_slug,
        policy_status,
        preflight_policy_inputs,
        run_policy_stage,
        verify_policy_bundle,
    )

    local_server = LocalJudgeServer()
    return (
        ALLOW_CODE,
        BLOCK_CODE,
        DATASET_PRESETS,
        DEFAULT_OUTPUT_ROOT,
        GuardAgent,
        MEMORY_SHOTS,
        Path,
        UPSTREAM_COMMIT,
        UPSTREAM_REPOSITORY,
        build_identity,
        check_guardagent,
        json,
        judge_key,
        local_server,
        mo,
        model_slug,
        normalize_endpoint,
        policy_status,
        positive_timeout,
        preflight_policy_inputs,
        restricted_execute,
        run_policy_stage,
        server_panel,
        subprocess,
        time,
        verify_policy_bundle,
    )


@app.cell(hide_code=True)
def connection_controls(local_server, mo, server_panel):
    server_settings, server_view, get_server_status = server_panel(
        mo, local_server, tool_call_parser="hermes"
    )
    connection_mode_control = mo.ui.dropdown(
        options=["Hack Club API", "Managed local"],
        value="Hack Club API",
        label="GuardAgent connection",
    )
    request_timeout_control = mo.ui.number(
        start=1, value=180, step=1, label="Request timeout (seconds)"
    )
    check_guardagent_button = mo.ui.run_button(label="Check GuardAgent")
    mo.vstack(
        [
            connection_mode_control,
            server_view,
            request_timeout_control,
            check_guardagent_button,
            mo.md(
                "Managed local starts vLLM 0.27 with automatic tool choice and the "
                "Hermes parser. Keep it running for the complete build and evaluation."
            ),
        ]
    )
    return (
        check_guardagent_button,
        connection_mode_control,
        get_server_status,
        request_timeout_control,
        server_settings,
    )


@app.cell(hide_code=True)
def local_server_status(get_server_status, mo):
    mo.ui.table([get_server_status()])
    return


@app.cell(hide_code=True)
def official_checkout(
    Path,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    mo,
    subprocess,
):
    guardagent_checkout = Path(".guardagent/GuardAgent").resolve()
    if not (guardagent_checkout / ".git").is_dir():
        guardagent_checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                UPSTREAM_REPOSITORY,
                str(guardagent_checkout),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(guardagent_checkout),
                "checkout",
                "--detach",
                UPSTREAM_COMMIT,
            ],
            check=True,
        )
    guardagent_commit = subprocess.run(
        ["git", "-C", str(guardagent_checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if guardagent_commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"GuardAgent checkout is {guardagent_commit}; expected pinned commit {UPSTREAM_COMMIT}"
        )
    mo.callout(
        f"Official GuardAgent checkout ready at `{guardagent_checkout}` "
        f"(commit `{guardagent_commit}`).",
        kind="success",
    )
    return (guardagent_checkout,)


@app.cell(hide_code=True)
def workflow_controls(DATASET_PRESETS, DEFAULT_OUTPUT_ROOT, mo):
    dataset_preset_control = mo.ui.dropdown(
        options=list(DATASET_PRESETS), value="WildGuard", label="Dataset"
    )
    hosted_api_base_control = mo.ui.text(
        value="https://ai.hackclub.com/proxy/v1",
        label="Hack Club API base",
        full_width=True,
    )
    hosted_model_control = mo.ui.text(
        value="openai/gpt-5.6-luna",
        label="Hack Club model",
        full_width=True,
    )
    output_root_control = mo.ui.text(
        value=str(DEFAULT_OUTPUT_ROOT),
        label="Artifact output root",
        full_width=True,
    )
    case_budget_control = mo.ui.number(
        start=1, value=100, step=1, label="Cases per action"
    )
    overwrite_control = mo.ui.checkbox(
        value=False, label="Replace existing build on the next Train action"
    )
    preflight_button = mo.ui.run_button(label="1. Preflight")
    train_button = mo.ui.run_button(label="2. Train / resume")
    validation_button = mo.ui.run_button(label="3. Validate / resume")
    test_button = mo.ui.run_button(label="4. Test / resume")
    refresh_status_button = mo.ui.run_button(label="Refresh status")
    demo_prompt_control = mo.ui.text_area(
        value="Explain how rainbows form.",
        label="Demonstration prompt",
        full_width=True,
    )
    demo_button = mo.ui.run_button(label="5. Run demonstration")
    mo.vstack(
        [
            dataset_preset_control,
            hosted_api_base_control,
            hosted_model_control,
            output_root_control,
            mo.hstack([case_budget_control, overwrite_control]),
            mo.hstack(
                [
                    preflight_button,
                    train_button,
                    validation_button,
                    test_button,
                ]
            ),
            refresh_status_button,
            demo_prompt_control,
            demo_button,
            mo.callout(
                "Training appends a memory only when execution succeeds and the verdict "
                "matches the label. Validation and test are frozen. Errors are checkpointed "
                "and retried on the next action; five consecutive connection failures stop "
                "the current action. Excluding preflight and retries, a complete "
                "A clean run makes three model calls per case (planning, code "
                "generation, and termination): about 38,097 for WildGuard and "
                "39,309 for Aegis, before retries.",
                kind="info",
            ),
        ]
    )
    return (
        case_budget_control,
        dataset_preset_control,
        demo_button,
        demo_prompt_control,
        hosted_api_base_control,
        hosted_model_control,
        output_root_control,
        overwrite_control,
        preflight_button,
        refresh_status_button,
        test_button,
        train_button,
        validation_button,
    )


@app.cell
def configuration(
    DATASET_PRESETS,
    Path,
    case_budget_control,
    connection_mode_control,
    dataset_preset_control,
    guardagent_checkout,
    hosted_api_base_control,
    hosted_model_control,
    model_slug,
    normalize_endpoint,
    output_root_control,
    positive_timeout,
    request_timeout_control,
    server_settings,
):
    workspace = Path(__file__).resolve().parent
    dataset_preset = DATASET_PRESETS[dataset_preset_control.value]
    dataset_slug = dataset_preset["slug"]
    dataset_paths = {
        split_name: workspace / dataset_preset[split_name][0]
        for split_name in ("train", "validation", "test")
    }
    expected_counts = {
        split_name: dataset_preset[split_name][1]
        for split_name in ("train", "validation", "test")
    }
    official_root = guardagent_checkout
    train_path = dataset_paths["train"]
    validation_path = dataset_paths["validation"]
    test_path = dataset_paths["test"]
    output_root = Path(output_root_control.value).expanduser()
    case_budget = int(case_budget_control.value)
    request_timeout = positive_timeout(request_timeout_control.value)

    if connection_mode_control.value == "Managed local":
        provider = "Managed local"
        api_endpoint = normalize_endpoint(
            f"http://127.0.0.1:{int(server_settings.value['port'])}/v1"
        )
        core_model = server_settings.value["model"].strip()
        tool_call_parser = server_settings.value["tool_call_parser"].strip()
    else:
        provider = "Hack Club API"
        api_endpoint = normalize_endpoint(hosted_api_base_control.value)
        core_model = hosted_model_control.value.strip()
        tool_call_parser = None

    if not core_model:
        raise ValueError("GuardAgent model must not be empty")
    target_bundle = output_root / dataset_slug / model_slug(core_model)
    return (
        api_endpoint,
        case_budget,
        core_model,
        dataset_slug,
        expected_counts,
        official_root,
        provider,
        request_timeout,
        target_bundle,
        test_path,
        tool_call_parser,
        train_path,
        validation_path,
    )


@app.cell
def run_context(
    MEMORY_SHOTS,
    api_endpoint,
    build_identity,
    core_model,
    dataset_slug,
    expected_counts,
    judge_key,
    local_server,
    official_root,
    preflight_policy_inputs,
    provider,
    server_settings,
    test_path,
    tool_call_parser,
    train_path,
    validation_path,
):
    def resolve_guardagent_run():
        preflight_value = preflight_policy_inputs(
            official_root=official_root,
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            expected_counts=expected_counts,
        )
        if provider == "Managed local":
            local_server.require_ready(server_settings.value)
            key_value = ""
        else:
            key_value = judge_key(api_endpoint, "HACKCLUB_API_KEY")
        identity_value = build_identity(
            dataset=dataset_slug,
            model=core_model,
            provider=provider,
            endpoint=api_endpoint,
            tool_call_parser=tool_call_parser,
            source_hashes=preflight_value["source_hashes"],
            upstream_source_hashes=preflight_value["upstream_source_hashes"],
            seed=42,
            num_shots=MEMORY_SHOTS,
            upstream_commit=preflight_value["official_commit"],
        )
        return preflight_value, identity_value, key_value

    return (resolve_guardagent_run,)


@app.cell(hide_code=True)
def preflight_action(
    expected_counts,
    json,
    mo,
    official_root,
    preflight_button,
    preflight_policy_inputs,
    test_path,
    train_path,
    validation_path,
):
    preflight_report = None
    if preflight_button.value:
        preflight_report = preflight_policy_inputs(
            official_root=official_root,
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            expected_counts=expected_counts,
        )
    mo.md(
        "Preflight has not run."
        if preflight_report is None
        else "```json\n" + json.dumps(preflight_report, indent=2) + "\n```"
    )
    return


@app.cell(hide_code=True)
def connection_check(
    api_endpoint,
    check_guardagent,
    check_guardagent_button,
    core_model,
    json,
    judge_key,
    local_server,
    mo,
    official_root,
    provider,
    request_timeout,
    server_settings,
):
    connection_check_report = None
    if check_guardagent_button.value:
        if provider == "Managed local":
            local_server.require_ready(server_settings.value)
            connection_check_key = ""
        else:
            connection_check_key = judge_key(api_endpoint, "HACKCLUB_API_KEY")
        connection_check_report = check_guardagent(
            official_root=official_root,
            api_key=connection_check_key,
            endpoint=api_endpoint,
            model=core_model,
            seed=42,
            timeout=request_timeout,
        )
    mo.md(
        "GuardAgent connection has not been checked."
        if connection_check_report is None
        else "```json\n"
        + json.dumps(connection_check_report, indent=2)
        + "\n```"
    )
    return


@app.cell(hide_code=True)
def train_action(
    case_budget,
    json,
    mo,
    official_root,
    overwrite_control,
    request_timeout,
    resolve_guardagent_run,
    run_policy_stage,
    target_bundle,
    test_path,
    train_button,
    train_path,
    validation_path,
):
    train_report = None
    if train_button.value:
        train_preflight, train_identity, train_key = resolve_guardagent_run()
        train_report = run_policy_stage(
            split="train",
            official_root=official_root,
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            target=target_bundle,
            identity=train_identity,
            api_key=train_key,
            case_budget=case_budget,
            timeout=request_timeout,
            overwrite=overwrite_control.value,
        )
    mo.md(
        "Train action is idle."
        if train_report is None
        else "```json\n" + json.dumps(train_report, indent=2) + "\n```"
    )
    return (train_report,)


@app.cell(hide_code=True)
def validation_action(
    case_budget,
    json,
    mo,
    official_root,
    request_timeout,
    resolve_guardagent_run,
    run_policy_stage,
    target_bundle,
    test_path,
    train_path,
    validation_button,
    validation_path,
):
    validation_report = None
    if validation_button.value:
        validation_preflight, validation_identity, validation_key = (
            resolve_guardagent_run()
        )
        validation_report = run_policy_stage(
            split="validation",
            official_root=official_root,
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            target=target_bundle,
            identity=validation_identity,
            api_key=validation_key,
            case_budget=case_budget,
            timeout=request_timeout,
        )
    mo.md(
        "Validation action is idle."
        if validation_report is None
        else "```json\n" + json.dumps(validation_report, indent=2) + "\n```"
    )
    return (validation_report,)


@app.cell(hide_code=True)
def test_action(
    case_budget,
    json,
    mo,
    official_root,
    request_timeout,
    resolve_guardagent_run,
    run_policy_stage,
    target_bundle,
    test_button,
    test_path,
    train_path,
    validation_path,
):
    test_report = None
    if test_button.value:
        test_preflight, test_identity, test_key = resolve_guardagent_run()
        test_report = run_policy_stage(
            split="test",
            official_root=official_root,
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            target=target_bundle,
            identity=test_identity,
            api_key=test_key,
            case_budget=case_budget,
            timeout=request_timeout,
        )
    mo.md(
        "Test action is idle. It remains locked until validation is complete."
        if test_report is None
        else "```json\n" + json.dumps(test_report, indent=2) + "\n```"
    )
    return (test_report,)


@app.cell(hide_code=True)
def artifact_status(
    expected_counts,
    mo,
    policy_status,
    refresh_status_button,
    target_bundle,
    test_report,
    train_report,
    validation_report,
):
    _ = (
        refresh_status_button.value,
        train_report,
        validation_report,
        test_report,
    )
    status_shapes = {
        name: range(count) for name, count in expected_counts.items()
    }
    artifact_status_report = policy_status(target_bundle, status_shapes)
    status_rows = []
    for status_split, status_value in artifact_status_report.get(
        "splits", {}
    ).items():
        status_rows.append(
            {
                "split": status_split,
                "complete": status_value["complete_cases"],
                "expected": status_value["expected_cases"],
                "attempts": status_value["attempts"],
                "end-to-end accuracy": status_value["metrics"][
                    "end_to_end_accuracy"
                ],
                "valid-only accuracy": status_value["metrics"][
                    "valid_only_accuracy"
                ],
                "coverage": status_value["metrics"]["executable_coverage"],
            }
        )
    mo.vstack(
        [
            mo.md(
                f"### Artifact status: **{artifact_status_report['state']}**  \n"
                f"`{artifact_status_report['path']}`  \n"
                f"Learned memory size: `{artifact_status_report.get('learned_memory_size')}`"
            ),
            mo.ui.table(status_rows),
        ]
    )
    return


@app.cell(hide_code=True)
def demonstration(
    GuardAgent,
    MEMORY_SHOTS,
    api_endpoint,
    core_model,
    demo_button,
    demo_prompt_control,
    json,
    judge_key,
    local_server,
    mo,
    official_root,
    provider,
    request_timeout,
    server_settings,
    target_bundle,
    time,
    verify_policy_bundle,
):
    demo_report = None
    if demo_button.value:
        verified_metadata = verify_policy_bundle(target_bundle)
        if provider == "Managed local":
            local_server.require_ready(server_settings.value)
            demo_key = ""
        else:
            demo_key = judge_key(api_endpoint, "HACKCLUB_API_KEY")
        demo_agent = GuardAgent(
            official_root,
            demo_key,
            api_endpoint,
            core_model,
            seed=42,
            policy_bundle=target_bundle,
            num_shots=MEMORY_SHOTS,
            timeout=request_timeout,
        )
        demo_started = time.perf_counter()
        try:
            demo_outcome = demo_agent.run(demo_prompt_control.value)
        finally:
            demo_agent.close()
        demo_report = {
            "verdict": "block" if demo_outcome.block else "allow",
            "provider": provider,
            "endpoint": api_endpoint,
            "model": core_model,
            "latency_seconds": time.perf_counter() - demo_started,
            "retrieved_examples": demo_outcome.metadata["retrieved_examples"],
            "task_plan": demo_outcome.metadata["task_plan"],
            "restricted_code": demo_outcome.metadata["generated_code"],
            "bundle_fingerprint": verified_metadata["fingerprint"],
        }
    mo.md(
        "A verified, completed bundle is required for the demonstration."
        if demo_report is None
        else "```json\n" + json.dumps(demo_report, indent=2) + "\n```"
    )
    return


@app.cell(hide_code=True)
def built_in_checks(
    ALLOW_CODE,
    BLOCK_CODE,
    MEMORY_SHOTS,
    expected_counts,
    json,
    mo,
    restricted_execute,
):
    allow_check = restricted_execute(ALLOW_CODE, "hello")
    block_check = restricted_execute(BLOCK_CODE, "unsafe")
    if allow_check["access_denied"] or not block_check["access_denied"]:
        raise AssertionError("restricted executor allow/block check failed")
    if MEMORY_SHOTS != 3:
        raise AssertionError("GuardAgent must use three nearest examples")
    if expected_counts["train"] != 10_000:
        raise AssertionError(
            "policy generation must use all 10,000 training cases"
        )
    built_in_check_report = {
        "restricted_allow": "passed",
        "restricted_block": "passed",
        "nearest_examples": MEMORY_SHOTS,
        "training_cases": expected_counts["train"],
        "test_locked_until_validation": True,
    }
    mo.callout(json.dumps(built_in_check_report, indent=2), kind="success")
    return


if __name__ == "__main__":
    app.run()
