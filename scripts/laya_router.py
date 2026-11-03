"""Benchmark supervision, evaluation, and prompt-only inference for the MoDE router.

Heavy dependencies are imported only by model actions, so data checks and tests
can run without downloading Laya or loading any expert weights.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

from scripts.training_cases import normalize_prompt, read_cases, stratified_sample


EXPERTS = ("adasteer", "piguard", "guardagent")
# Router targets: an expert, or human review when every expert fails.
HUMAN = "human"
ROUTES = (*EXPERTS, HUMAN)
LAYA_COMMIT = "d74e8be46ef5241e4ba2b16cd25e896f24f71373"
BASE_MODEL = "convaiinnovations/laya"
BASE_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
QUESTIONS = {
    "expert": {
        "type": "choice",
        "instructions": "Choose the defensive expert that will successfully handle this prompt, or human review if none will.",
        "criteria": {
            "adasteer": "AdaSteer: steer the responding model toward safe, useful responses.",
            "piguard": "PIGuard: use a trained prompt-safety classifier to block or allow the prompt.",
            "guardagent": "GuardAgent: use a Python safety policy to block or allow the prompt.",
            "human": "Human review: no automated expert will handle this prompt successfully.",
        },
    }
}
ARTIFACT_KEYS = (
    "adasteer_bundle_sha256", "piguard_artifact_sha256", "guardagent_policy_sha256",
)
BEHAVIOR_KEYS = (
    "adasteer_bundle_fingerprint", "responding_model", "responding_max_new_tokens",
    "judge_model", "judge_temperature", "judge_max_tokens", "gate_outcome_assumption",
    "piguard_block_threshold", "guardagent_block_threshold", "refusal_text",
)
OUTCOME_NOTE = (
    "AdaSteer success uses local response judgments. PIGuard and GuardAgent success "
    "uses prompt-label gate assumptions; allowing a benign prompt does not prove helpfulness. "
    "Expert cost is the recorded guardrail_seconds, excluding judge/setup time. "
    "Gate costs omit response generation. Cost plus measured router latency is an estimate, "
    "not measured end-to-end production latency. Confidence is a choice probability, "
    "not the selected expert's probability of success."
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def assert_disjoint(left, right):
    if {row["case_id"] for row in left} & {row["case_id"] for row in right}:
        raise ValueError("Training/validation and test case IDs overlap")
    if {normalize_prompt(row["prompt"]) for row in left} & {normalize_prompt(row["prompt"]) for row in right}:
        raise ValueError("Training/validation and test normalized prompts overlap")


def fastest_successful(outcomes):
    """None means all experts failed; tuple order provides deterministic ties."""
    successful = [expert for expert in EXPERTS if outcomes[expert]["succeeded"]]
    return min(successful, key=lambda expert: outcomes[expert]["guardrail_seconds"]) if successful else None


def decision_confidence(outcome):
    """Gate confidence in its own block/allow decision, else the judge's P(success); None if neither."""
    metadata = outcome.get("guard_metadata")
    score = metadata.get("harmful_score", metadata.get("policy_score")) if isinstance(metadata, dict) else None
    blocked = outcome.get("guard_blocked")
    if type(score) in (int, float) and 0 <= score <= 1 and isinstance(blocked, bool):
        return float(score) if blocked else 1.0 - float(score)
    judged = outcome.get("judge_confidence")
    return float(judged) if type(judged) in (int, float) and 0 <= judged <= 1 else None


def most_confident_successful(outcomes):
    """Experts without any confidence rank below scored ones; ties go to the faster expert."""
    successful = [expert for expert in EXPERTS if outcomes[expert]["succeeded"]]
    def rank(expert):
        confidence = decision_confidence(outcomes[expert])
        return (confidence is None, -(confidence or 0.0), outcomes[expert]["guardrail_seconds"])
    return min(successful, key=rank) if successful else None


TARGET_RULES = {
    "fastest_successful": fastest_successful,
    "most_confident_successful": most_confident_successful,
}


@dataclass
class BenchmarkData:
    manifest: dict
    cases: list[dict]
    outcomes: dict[str, dict[str, dict]]
    targets: dict[str, str | None]
    results_path: Path


def load_benchmark(run_dir, parquet_path, *, expected_rows, split):
    """Validate an entire benchmark, using JSONL as the authoritative resume log."""
    run_dir = Path(run_dir).expanduser().resolve()
    parquet_path = Path(parquet_path).expanduser().resolve()
    cases = read_cases(parquet_path, expected_rows=expected_rows)
    manifest = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    config = {key: value for key, value in manifest.items() if key != "run_fingerprint"}
    run_fingerprint = manifest.get("run_fingerprint")
    if run_fingerprint != fingerprint(config):
        raise ValueError("Benchmark run_config.json fingerprint is invalid")
    version = manifest.get("schema_version")
    if version not in (3, 4) or (version == 3 and split != "test"):
        raise ValueError("Unseen supervision requires benchmark schema 4; test results support schemas 3 and 4")
    if manifest.get("dataset") != "aegis" or manifest.get("data_split", "test") != split:
        raise ValueError(f"Expected an Aegis {split} benchmark")
    if manifest.get("test_sha256") != sha256_file(parquet_path):
        raise ValueError("Benchmark parquet hash differs from the original prompts")
    if manifest.get("case_count") != len(cases):
        raise ValueError("Benchmark case_count differs from the original prompts")
    if sorted(manifest.get("selected_guardrails", [])) != sorted(EXPERTS):
        raise ValueError("Benchmark must include exactly AdaSteer, PIGuard, and GuardAgent")
    for key in ARTIFACT_KEYS:
        value = manifest.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing or invalid expert provenance: {key}")
    target_rule = manifest.get("target_rule", "fastest_successful")
    if target_rule not in TARGET_RULES:
        raise ValueError(f"Unknown benchmark target_rule: {target_rule!r}")

    results_path = run_dir / "results.jsonl"
    if results_path.is_file():
        records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").split("\n") if line.strip()]
    else:
        results_path = run_dir / "prompt_results.json"
        grouped = json.loads(results_path.read_text(encoding="utf-8"))
        records = []
        for group in grouped:
            for result in group["results"]:
                if "prompt" in result and result["prompt"] != group["prompt"]:
                    raise ValueError("Grouped benchmark contains conflicting prompt text")
                records.append({**result, "prompt": group["prompt"]})
    originals = {row["case_id"]: row for row in cases}
    latest = {}
    for record in records:
        case_id, expert = record.get("case_id"), record.get("expert")
        if case_id not in originals or expert not in EXPERTS:
            raise ValueError(f"Unexpected benchmark case/expert: {case_id!r}, {expert!r}")
        if record.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Result run fingerprint differs from its manifest")
        for key in ("prompt", "prompt_harm_label"):
            if record.get(key) != originals[case_id][key]:
                raise ValueError(f"Result {key} differs from original case {case_id}")
        if record.get("dataset") != "aegis":
            raise ValueError("Result dataset differs from its manifest")
        latest[case_id, expert] = record
    missing = len(cases) * len(EXPERTS) - len(latest)
    if missing:
        raise ValueError(f"Benchmark incomplete: {missing} case/expert results missing; run/resume all three experts")
    for (case_id, expert), record in latest.items():
        seconds = record.get("guardrail_seconds")
        if record.get("error") or type(record.get("succeeded")) is not bool:
            raise ValueError(f"Benchmark has an errored or unknown outcome: {case_id}/{expert}; resume it first")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"Invalid guardrail_seconds for {case_id}/{expert}")
    outcomes = {case_id: {expert: latest[case_id, expert] for expert in EXPERTS} for case_id in originals}
    targets = {case_id: TARGET_RULES[target_rule](values) or HUMAN for case_id, values in outcomes.items()}
    return BenchmarkData(manifest, cases, outcomes, targets, results_path)


def check_matching_experts(train_manifest, test_manifest):
    for key in ARTIFACT_KEYS + BEHAVIOR_KEYS:
        if train_manifest.get(key) != test_manifest.get(key):
            raise ValueError(f"Training and test expert configuration differs: {key}")


def split_cases(cases, seed=42):
    ordered = sorted(cases, key=lambda row: row["case_id"])
    train, _, _ = stratified_sample(ordered, int(len(ordered) * 0.8), seed,
                                    key=lambda row: row["prompt_harm_label"])
    selected = {row["case_id"] for row in train}
    valid = [row for row in ordered if row["case_id"] not in selected]
    assert_disjoint(train, valid)
    return {"train": train, "validation": valid}


def labeled_rows(cases, targets):
    return [{"state": row["prompt"], "questions": QUESTIONS, "expected": {"expert": targets[row["case_id"]] or HUMAN}}
            for row in cases]


def prepare_data(run_dir, train_parquet, test_parquet, output_root):
    data = load_benchmark(run_dir, train_parquet, expected_rows=10000, split="unseen")
    test_cases = read_cases(test_parquet, expected_rows=1914)
    assert_disjoint(data.cases, test_cases)
    splits = split_cases(data.cases)
    manifest = {
        "schema_version": 1,
        "benchmark": data.manifest,
        "results_path": str(data.results_path),
        "results_sha256": sha256_file(data.results_path),
        "train_parquet": str(Path(train_parquet).resolve()),
        "test_parquet": str(Path(test_parquet).resolve()),
        "test_sha256": sha256_file(test_parquet),
        "questions": QUESTIONS,
        "seed": 42,
        "split_case_ids": {name: [row["case_id"] for row in cases] for name, cases in splits.items()},
        "targets": data.targets,
        "splits": {},
        "outcome_note": OUTCOME_NOTE,
    }
    for name, cases in splits.items():
        rows = labeled_rows(cases, data.targets)
        if not rows:
            raise ValueError(f"No targets in the {name} split")
        counts = dict(Counter(row["expected"]["expert"] for row in rows))
        if name == "train" and len(counts) < 2:
            raise ValueError(f"Training targets contain only one expert: {counts}; inspect benchmark separation before fine-tuning")
        manifest["splits"][name] = {"cases": len(cases), "labeled": len(rows),
                                    "all_failed": counts.get(HUMAN, 0), "target_counts": counts}
    directory = Path(output_root).expanduser().resolve() / "datasets" / fingerprint(manifest)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    for name, cases in splits.items():
        rows = labeled_rows(cases, data.targets)
        (directory / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    # Save all cases/outcomes, including all-failed prompts, for honest validation.
    write_json(directory / "validation_cases.json", [
        {"case": row, "outcomes": data.outcomes[row["case_id"]], "target": data.targets[row["case_id"]]}
        for row in splits["validation"]
    ])
    write_json(directory / "train_cases.json", splits["train"])
    manifest["file_sha256"] = {name: sha256_file(directory / name)
                               for name in ("train.jsonl", "validation.jsonl", "validation_cases.json", "train_cases.json")}
    write_json(directory / "manifest.json", manifest)
    return {"directory": str(directory), "manifest": manifest}


def download_base(cache_dir):
    from huggingface_hub import snapshot_download

    return snapshot_download(BASE_MODEL, revision=BASE_REVISION, cache_dir=str(cache_dir))


def make_train_config(settings, *, eval_data=None):
    from laya.train import TrainConfig

    return TrainConfig(**settings, eval_data=eval_data, calib_frac=0.1, calib_max=400,
                       seed=42, calib_seed=42, freeze_encoder=False, log_every=10)


def inspect_training(data_dir, base_dir, settings):
    """Use Laya's own tokenizer, item builder, split and budget calculation."""
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_sequence
    from laya.train import dry_run, to_internal

    data_dir = Path(data_dir)
    config = make_train_config(settings, eval_data=str(data_dir / "validation.jsonl"))
    report = dry_run(str(data_dir / "train.jsonl"), str(base_dir), config)
    if report["skipped"] or report.get("eval_skipped"):
        raise ValueError(f"Laya preprocessing skipped labeled examples: {report}")
    _fix_tokenizer_config(str(base_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(Path(base_dir) / "tokenizer"))
    question = to_internal("expert", QUESTIONS["expert"])
    truncation = {}
    for name in ("train", "validation"):
        rows = [json.loads(line) for line in (data_dir / f"{name}.jsonl").read_text().split("\n") if line.strip()]
        stats = [build_sequence(tokenizer, row["state"], question, settings["max_len"],
                                settings["head_max_len"], return_truncation_stats=True)[2] for row in rows]
        truncation[name] = {"cases": len(stats), "truncated": sum(row["truncated"] for row in stats),
                            "state_tokens_dropped": sum(row["state_tokens_dropped"] for row in stats)}
    return {**report, "truncation": truncation}


def cuda_smoke_test(prepared, base_dir, output_root, settings):
    """Exercise real CUDA forward/backward, save, and reload before a full run."""
    import torch
    from laya.train import finetune

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke test requires the Molab NVIDIA GPU; no CUDA device is available here")
    rows = [json.loads(line) for line in (Path(prepared["directory"]) / "train.jsonl").read_text().split("\n") if line.strip()][:20]
    if len(rows) < 20:
        raise ValueError("CUDA smoke test needs at least 20 labeled training examples")
    root = Path(output_root).expanduser().resolve() / "smoke"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="cuda-", dir=root))
    data_path = directory / "train.jsonl"
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    smoke_settings = {**settings, "epochs": 1, "micro_batch": 2, "grad_accum": 1, "amp": True}
    # Calibration on two held-out examples is intentionally small: this checks execution, not quality.
    report = finetune(str(data_path), str(base_dir), str(directory / "checkpoint"),
                       config=make_train_config(smoke_settings), device="cuda")
    if not report["epoch_loss"] or not all(math.isfinite(loss) for loss in report["epoch_loss"]):
        raise RuntimeError("CUDA smoke test produced non-finite training loss")
    with MoDERouter(directory / "checkpoint", "cuda") as router:
        prediction = router.predict("Explain how rainbows form.")
    result = {"directory": str(directory), "gpu": torch.cuda.get_device_name(),
              "summary": report, "reload_prediction": prediction,
              "note": "Execution check only; tiny calibration slice is not quality evidence."}
    write_json(directory / "smoke_report.json", result)
    return result


def resolve_device(requested="auto"):
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class MoDERouter:
    """Reload a trained Laya checkpoint and select an expert from prompt text."""

    def __init__(self, checkpoint_dir, device="auto", *, max_len=None, head_max_len=None, questions=None):
        import laya

        checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.questions = questions if questions is not None else json.loads(
            (checkpoint_dir / "questions.json").read_text(encoding="utf-8"))
        if self.questions != QUESTIONS:
            raise ValueError("Checkpoint question schema differs from the MoDE router schema")
        self.device = resolve_device(device)
        self.agent = laya.load(str(checkpoint_dir), device=self.device)
        self.max_len = max_len
        self.head_max_len = head_max_len

    def predict(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        result = self.agent.predict(prompt, self.questions, max_len=self.max_len,
                                    head_max_len=self.head_max_len)
        answer = result["answers"]["expert"]
        expert = answer["choice"]
        probabilities = answer["probabilities"]
        if expert not in ROUTES or set(probabilities) != set(ROUTES):
            raise ValueError("Laya returned an invalid expert choice")
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
               for value in probabilities.values()):
            raise ValueError("Laya returned invalid choice probabilities")
        if not math.isclose(sum(probabilities.values()), 1, abs_tol=0.001):
            raise ValueError("Laya choice probabilities do not sum to one")
        return {"expert": expert, "probabilities": probabilities, "confidence": probabilities[expert],
                "truncated": bool(result.get("usage", {}).get("truncated", False))}

    def close(self):
        self.agent.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _synchronize(device):
    import torch

    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elif str(device) == "mps":
        torch.mps.synchronize()


def predict_cases(router, cases):
    # One warm-up, excluded from per-prompt steady-state latency.
    router.predict(cases[0]["prompt"])
    predictions = []
    for case in cases:
        _synchronize(router.device)
        started = time.perf_counter()
        result = router.predict(case["prompt"])
        _synchronize(router.device)
        predictions.append({"case_id": case["case_id"], **result,
                            "router_seconds": time.perf_counter() - started})
    return predictions


def _percentile(values, percentile):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * percentile) - 1)] if values else None


def selection_outcome(outcomes, target, choice):
    """(succeeded, expert seconds); human review succeeds only when every expert fails."""
    if choice == HUMAN:
        return target == HUMAN, 0.0
    return outcomes[choice]["succeeded"], outcomes[choice]["guardrail_seconds"]


def summarize_policy(name, cases, outcomes, targets, predictions):
    """Every case has a target; all-failed cases target human review."""
    by_id = {row["case_id"]: row for row in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {row["case_id"] for row in cases}:
        raise ValueError("Predictions must cover every evaluation case exactly once")
    summaries = []
    for subset in ("all", "harmful", "unharmful"):
        selected = [case for case in cases if subset == "all" or case["prompt_harm_label"] == subset]
        if not selected:
            continue
        chosen = [by_id[case["case_id"]] for case in selected]
        results = [selection_outcome(outcomes[case["case_id"]], targets[case["case_id"]] or HUMAN, p["expert"])
                   for case, p in zip(selected, chosen)]
        costs = [seconds for _, seconds in results]
        seconds = [p.get("router_seconds", 0.0) for p in chosen]
        summaries.append({
            "policy": name, "subset": subset, "cases": len(selected),
            "target_cases": len(selected),
            "all_failed": sum((targets[case["case_id"]] or HUMAN) == HUMAN for case in selected),
            "routing_accuracy": sum(p["expert"] == (targets[case["case_id"]] or HUMAN)
                                    for case, p in zip(selected, chosen)) / len(selected),
            "selected_expert_success": sum(succeeded for succeeded, _ in results) / len(selected),
            "human_escalations": sum(p["expert"] == HUMAN for p in chosen),
            "selection_counts": dict(Counter(p["expert"] for p in chosen)),
            "mean_expert_cost_seconds": sum(costs) / len(costs),
            "median_router_seconds": _percentile(seconds, 0.5),
            "p95_router_seconds": _percentile(seconds, 0.95),
            "mean_estimated_cost_seconds": sum(a + b for a, b in zip(costs, seconds)) / len(costs),
            "truncated": sum(p.get("truncated", False) for p in chosen),
        })
    return summaries


def evaluate_checkpoints(cases, outcomes, targets, base_dir, checkpoint_dir, settings, device):
    reports, predictions = [], {}
    for name, directory in (("base_laya", base_dir), ("finetuned_laya", checkpoint_dir)):
        # Same question and token budgets for both checkpoints.
        with MoDERouter(directory, device, max_len=settings["max_len"],
                        head_max_len=settings["head_max_len"],
                        questions=QUESTIONS if name == "base_laya" else None) as router:
            predictions[name] = predict_cases(router, cases)
        reports.extend(summarize_policy(name, cases, outcomes, targets, predictions[name]))
    for name in (*EXPERTS, "fastest_successful_oracle"):
        choices = [{"case_id": case["case_id"], "expert": name if name in EXPERTS else targets[case["case_id"]] or HUMAN}
                   for case in cases]
        reports.extend(summarize_policy(name, cases, outcomes, targets, choices))
    return {"summary": reports, "predictions": predictions, "outcome_note": OUTCOME_NOTE}


def train_router(prepared, base_dir, output_root, settings, device="auto"):
    import torch
    from laya.train import finetune

    data_dir = Path(prepared["directory"])
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest != prepared["manifest"]:
        raise ValueError("Prepared manifest changed; prepare the data again")
    for filename, expected in manifest["file_sha256"].items():
        if sha256_file(data_dir / filename) != expected:
            raise ValueError(f"Prepared data changed: {filename}; prepare the data again")
    device = resolve_device(device)
    settings = {**settings, "amp": settings.get("amp", True) and device == "cuda"}
    inspection = inspect_training(data_dir, base_dir, settings)
    root = Path(output_root).expanduser().resolve() / "runs"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    shutil.copytree(data_dir, run_dir / "training_data")
    run = {
        "directory": str(run_dir), "base_dir": str(base_dir), "settings": settings,
        "device": device, "laya_commit": LAYA_COMMIT, "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION, "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "data_manifest": manifest, "inspection": inspection,
    }
    write_json(run_dir / "run.json", run)
    config = make_train_config(settings, eval_data=str(run_dir / "training_data/validation.jsonl"))
    summary = finetune(str(run_dir / "training_data/train.jsonl"), str(base_dir),
                       str(run_dir / "checkpoint"), config=config, device=device)
    write_json(run_dir / "training_summary.json", summary)
    entries = json.loads((run_dir / "training_data/validation_cases.json").read_text())
    validation = evaluate_checkpoints(
        [entry["case"] for entry in entries],
        {entry["case"]["case_id"]: entry["outcomes"] for entry in entries},
        {entry["case"]["case_id"]: entry["target"] for entry in entries},
        base_dir, run_dir / "checkpoint", settings, device,
    )
    write_json(run_dir / "validation_report.json", validation)
    return {**run, "summary": summary, "validation": validation}


def evaluate_test(run_dir, benchmark_dir, test_parquet, device="auto"):
    run_dir = Path(run_dir).expanduser().resolve()
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    data = load_benchmark(benchmark_dir, test_parquet, expected_rows=1914, split="test")
    manifest = run["data_manifest"]
    if sha256_file(test_parquet) != manifest["test_sha256"]:
        raise ValueError("Test parquet differs from the test set reserved during preparation")
    check_matching_experts(manifest["benchmark"], data.manifest)
    training = json.loads((run_dir / "training_data/train_cases.json").read_text())
    validation = json.loads((run_dir / "training_data/validation_cases.json").read_text())
    assert_disjoint(training, data.cases)
    assert_disjoint([entry["case"] for entry in validation], data.cases)
    report = evaluate_checkpoints(data.cases, data.outcomes, data.targets, run["base_dir"],
                                  run_dir / "checkpoint", run["settings"], resolve_device(device))
    report["benchmark"] = data.manifest
    report["results_sha256"] = sha256_file(data.results_path)
    write_json(run_dir / "test_report.json", report)
    return report


def export_router(run_dir, output_root, device="auto"):
    run_dir = Path(run_dir).expanduser().resolve()
    if not (run_dir / "validation_report.json").is_file():
        raise ValueError("Training and validation must complete before export")
    root = Path(output_root).expanduser().resolve() / "exports"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_dir.name
    if destination.exists():
        raise FileExistsError(f"Export already exists: {destination}; choose a different output root")
    # Build and verify in a staging directory so an interrupted copy is never a finished export.
    staging = Path(tempfile.mkdtemp(prefix=".export-", dir=root))
    try:
        shutil.copytree(run_dir / "checkpoint", staging, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("checkpoint_latest"))
        shutil.copytree(run_dir / "training_data", staging / "training_data")
        for path in run_dir.glob("*.json"):
            shutil.copy2(path, staging / path.name)
        with MoDERouter(staging, device) as router:
            prediction = router.predict("Explain how rainbows form.")
        write_json(staging / "reload_check.json", prediction)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return str(destination)
