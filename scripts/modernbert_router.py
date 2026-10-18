"""ModernBERT routing with the same supervision and metrics as the Laya trainer.

Model dependencies are imported only by explicit model actions.
"""
from __future__ import annotations

from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import tempfile

from scripts.laya_router import (
    EXPERTS, OUTCOME_NOTE, assert_disjoint, check_matching_experts, load_benchmark,
    predict_cases, prepare_data, resolve_device, sha256_file, summarize_policy, write_json,
)


BASE_MODEL = "answerdotai/ModernBERT-large"
BASE_REVISION = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
ID2LABEL = dict(enumerate(EXPERTS))
LABEL2ID = {expert: index for index, expert in ID2LABEL.items()}
DEFAULT_SETTINGS = {
    "epochs": 8, "micro_batch": 8, "grad_accum": 2, "learning_rate": 2e-5,
    "max_len": 1024, "weight_decay": 0.01, "warmup_ratio": 0.1,
    "gradient_checkpointing": True, "amp": True,
}


def download_base(cache_dir):
    from huggingface_hub import snapshot_download

    return snapshot_download(
        BASE_MODEL, revision=BASE_REVISION, cache_dir=str(Path(cache_dir).expanduser()),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
    )


def validate_settings(settings):
    settings = {**DEFAULT_SETTINGS, **settings}
    for name in ("epochs", "micro_batch", "grad_accum", "max_len"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("learning_rate", "weight_decay", "warmup_ratio"):
        value = settings[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if settings["learning_rate"] == 0 or settings["warmup_ratio"] > 1:
        raise ValueError("learning_rate must be positive and warmup_ratio must be at most one")
    for name in ("amp", "gradient_checkpointing"):
        if type(settings[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    return settings


def read_labeled(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]
    if not rows:
        raise ValueError(f"No labeled examples: {path}")
    for row in rows:
        if not isinstance(row.get("state"), str) or not row["state"].strip():
            raise ValueError("Training prompts must be nonempty strings")
        if row.get("expected", {}).get("expert") not in LABEL2ID:
            raise ValueError("Unknown expert training label")
    return rows


def encode_rows(rows, tokenizer, max_len):
    # No question, benchmark outcome, or latency text enters the encoder.
    encoded = tokenizer([row["state"] for row in rows], truncation=True,
                        max_length=max_len, return_token_type_ids=False)
    return [{**{key: values[index] for key, values in encoded.items()},
             "labels": LABEL2ID[row["expected"]["expert"]]} for index, row in enumerate(rows)]


def inspect_training(data_dir, base_dir, settings):
    from transformers import AutoConfig, AutoTokenizer

    settings = validate_settings(settings)
    config = AutoConfig.from_pretrained(base_dir, local_files_only=True)
    if config.model_type != "modernbert":
        raise ValueError("Expected a ModernBERT checkpoint")
    if settings["max_len"] > config.max_position_embeddings:
        raise ValueError(f"max_len exceeds model context ({config.max_position_embeddings})")
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    truncation = {}
    for name in ("train", "validation"):
        rows = read_labeled(Path(data_dir) / f"{name}.jsonl")
        lengths = [len(tokenizer(row["state"], truncation=False,
                                return_token_type_ids=False)["input_ids"]) for row in rows]
        truncation[name] = {
            "cases": len(rows), "truncated": sum(length > settings["max_len"] for length in lengths),
            "tokens_dropped": sum(max(0, length - settings["max_len"]) for length in lengths),
            "target_counts": {expert: sum(row["expected"]["expert"] == expert for row in rows) for expert in EXPERTS},
        }
    updates = math.ceil(math.ceil(truncation["train"]["cases"] / settings["micro_batch"]) / settings["grad_accum"])
    return {"truncation": truncation, "effective_batch": settings["micro_batch"] * settings["grad_accum"],
            "updates_per_epoch": updates, "total_updates": updates * settings["epochs"]}


def _precision(device, amp):
    import torch

    enabled = device == "cuda" and amp
    bf16 = enabled and torch.cuda.is_bf16_supported()
    return {"bf16": bf16, "fp16": enabled and not bf16}


def _fit(data_dir, base_dir, directory, settings, device):
    import torch
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        Trainer, TrainingArguments, set_seed,
    )

    settings = validate_settings(settings)
    set_seed(42)
    args = TrainingArguments(
        output_dir=str(directory / "epochs"), num_train_epochs=settings["epochs"],
        per_device_train_batch_size=settings["micro_batch"],
        per_device_eval_batch_size=settings["micro_batch"],
        gradient_accumulation_steps=settings["grad_accum"], learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"], warmup_ratio=settings["warmup_ratio"],
        lr_scheduler_type="linear", optim="adamw_torch", seed=42, data_seed=42,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2,
        gradient_checkpointing=settings["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cpu=device == "cpu", **_precision(device, settings["amp"]),
        dataloader_pin_memory=device == "cuda", report_to="none",
        logging_strategy="epoch", logging_nan_inf_filter=False,
    )
    if args.device.type != device:
        raise RuntimeError(f"Requested {device}, but Trainer selected {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_dir, local_files_only=True, num_labels=len(EXPERTS),
        id2label=ID2LABEL, label2id=LABEL2ID, problem_type="single_label_classification",
        reference_compile=False, attn_implementation="sdpa",
    )
    model.config.modernbert_router = {
        "schema_version": 1, "max_len": settings["max_len"], "amp": settings["amp"],
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
    }
    trainer = Trainer(
        model=model, args=args,
        train_dataset=encode_rows(read_labeled(data_dir / "train.jsonl"), tokenizer, settings["max_len"]),
        eval_dataset=encode_rows(read_labeled(data_dir / "validation.jsonl"), tokenizer, settings["max_len"]),
        data_collator=DataCollatorWithPadding(tokenizer), processing_class=tokenizer,
    )
    result = trainer.train()
    losses = [result.training_loss, trainer.state.best_metric]
    losses.extend(row[key] for row in trainer.state.log_history for key in ("loss", "eval_loss") if key in row)
    if any(value is None or not math.isfinite(value) for value in losses):
        raise RuntimeError("Training produced non-finite loss; checkpoint is not ready for export")
    checkpoint = directory / "checkpoint"
    trainer.save_model(str(checkpoint))
    tokenizer.save_pretrained(checkpoint)
    summary = {"metrics": result.metrics, "best_validation_loss": trainer.state.best_metric,
               "best_checkpoint": trainer.state.best_model_checkpoint, "history": trainer.state.log_history}
    write_json(directory / "training_summary.json", summary)
    del trainer, model
    if device == "cuda":
        torch.cuda.empty_cache()
    return summary


class ModernBERTRouter:
    """Load an exported router locally and select an expert using prompt text only."""

    def __init__(self, checkpoint_dir, device="auto"):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self.device = resolve_device(device)
        if self.device not in ("cpu", "mps", "cuda"):
            raise ValueError("device must be auto, cuda, mps, or cpu")
        checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        config = AutoConfig.from_pretrained(checkpoint_dir, local_files_only=True)
        metadata = getattr(config, "modernbert_router", {})
        if (config.model_type != "modernbert" or config.id2label != ID2LABEL
                or config.label2id != LABEL2ID or metadata.get("schema_version") != 1):
            raise ValueError("Checkpoint does not contain the ModernBERT MoDE router schema")
        self.max_len = metadata.get("max_len")
        if type(self.max_len) is not int or not 1 <= self.max_len <= config.max_position_embeddings:
            raise ValueError("Checkpoint has an invalid max_len")
        self.precision = _precision(self.device, metadata.get("amp", False))
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir, local_files_only=True, reference_compile=False, attn_implementation="sdpa",
        ).to(self.device).eval()
        self._torch = torch

    def predict(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if self.model is None:
            raise RuntimeError("Router is closed")
        torch = self._torch
        # Measure the full input length for the truncation flag, then use the same
        # tokenizer truncation path as training so special tokens are preserved.
        length = len(self.tokenizer(prompt, truncation=False, return_token_type_ids=False)["input_ids"])
        inputs = self.tokenizer(prompt, truncation=True, max_length=self.max_len,
                                return_token_type_ids=False, return_tensors="pt").to(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device, enabled=any(self.precision.values()),
            dtype=torch.bfloat16 if self.precision["bf16"] else torch.float16,
        ):
            probabilities = self.model(**inputs).logits[0].float().softmax(-1).tolist()
        if (len(probabilities) != len(EXPERTS) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
                or not math.isclose(sum(probabilities), 1, abs_tol=0.001)):
            raise RuntimeError("Model returned invalid expert probabilities")
        selected = max(range(len(EXPERTS)), key=probabilities.__getitem__)
        return {"expert": EXPERTS[selected], "probabilities": dict(zip(EXPERTS, probabilities)),
                "confidence": probabilities[selected], "truncated": length > self.max_len}

    def close(self):
        self.model = None
        if self.device == "cuda":
            self._torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def evaluate_checkpoint(cases, outcomes, targets, checkpoint_dir, device):
    with ModernBERTRouter(checkpoint_dir, device) as router:
        predictions = predict_cases(router, cases)
    reports = summarize_policy("finetuned_modernbert", cases, outcomes, targets, predictions)
    for name in (*EXPERTS, "fastest_successful_oracle"):
        choices = [{"case_id": case["case_id"], "expert": name if name in EXPERTS else (
            targets[case["case_id"]] or min(EXPERTS, key=lambda expert: outcomes[case["case_id"]][expert]["guardrail_seconds"]))}
                   for case in cases]
        reports.extend(summarize_policy(name, cases, outcomes, targets, choices))
    return {"summary": reports, "predictions": {"finetuned_modernbert": predictions}, "outcome_note": OUTCOME_NOTE}


def train_router(prepared, base_dir, output_root, settings, device="auto"):
    import torch

    settings = validate_settings(settings)
    data_dir = Path(prepared["directory"])
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest != prepared["manifest"]:
        raise ValueError("Prepared manifest changed; prepare the data again")
    for filename, expected in manifest["file_sha256"].items():
        if sha256_file(data_dir / filename) != expected:
            raise ValueError(f"Prepared data changed: {filename}; prepare the data again")
    device = resolve_device(device)
    inspection = inspect_training(data_dir, base_dir, settings)
    root = Path(output_root).expanduser().resolve() / "runs"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    shutil.copytree(data_dir, directory / "training_data")
    run = {
        "directory": str(directory), "base_dir": str(base_dir), "settings": settings,
        "device": device, "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "versions": {name: version(name) for name in ("torch", "transformers", "accelerate", "marimo", "pyarrow")},
        "precision": _precision(device, settings["amp"]),
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "data_manifest": manifest, "inspection": inspection,
    }
    write_json(directory / "run.json", run)
    summary = _fit(directory / "training_data", base_dir, directory, settings, device)
    entries = json.loads((directory / "training_data/validation_cases.json").read_text())
    validation = evaluate_checkpoint(
        [entry["case"] for entry in entries],
        {entry["case"]["case_id"]: entry["outcomes"] for entry in entries},
        {entry["case"]["case_id"]: entry["target"] for entry in entries}, directory / "checkpoint", device,
    )
    write_json(directory / "validation_report.json", validation)
    return {**run, "summary": summary, "validation": validation}


def cuda_smoke_test(prepared, base_dir, output_root, settings):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke test requires an NVIDIA GPU")
    settings = validate_settings({**settings, "epochs": 1, "micro_batch": 2, "grad_accum": 1, "amp": True})
    rows = read_labeled(Path(prepared["directory"]) / "train.jsonl")[:20]
    if len(rows) < 20:
        raise ValueError("CUDA smoke test needs at least 20 labeled training examples")
    root = Path(output_root).expanduser().resolve() / "smoke"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="cuda-", dir=root))
    data_dir = directory / "training_data"
    data_dir.mkdir()
    for name, selected in (("train", rows[:16]), ("validation", rows[16:])):
        (data_dir / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
    summary = _fit(data_dir, base_dir, directory, settings, "cuda")
    with ModernBERTRouter(directory / "checkpoint", "cuda") as router:
        prediction = router.predict("Explain how rainbows form.")
    report = {"directory": str(directory), "gpu": torch.cuda.get_device_name(),
              "summary": summary, "reload_prediction": prediction,
              "note": "Execution check on 20 training examples only; not quality evidence."}
    write_json(directory / "smoke_report.json", report)
    return report


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
    report = evaluate_checkpoint(data.cases, data.outcomes, data.targets, run_dir / "checkpoint", resolve_device(device))
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
    staging = Path(tempfile.mkdtemp(prefix=".export-", dir=root))
    try:
        shutil.copytree(run_dir / "checkpoint", staging, dirs_exist_ok=True)
        shutil.copytree(run_dir / "training_data", staging / "training_data")
        for path in run_dir.glob("*.json"):
            shutil.copy2(path, staging / path.name)
        prompt = "Explain how rainbows form."
        with ModernBERTRouter(run_dir / "checkpoint", device) as router:
            original = router.predict(prompt)
        with ModernBERTRouter(staging, device) as router:
            prediction = router.predict(prompt)
        if (original["expert"] != prediction["expert"] or original["truncated"] != prediction["truncated"]
                or any(not math.isclose(original["probabilities"][expert], prediction["probabilities"][expert],
                                        rel_tol=1e-5, abs_tol=1e-6) for expert in EXPERTS)):
            raise RuntimeError("Exported checkpoint predictions differ after reload")
        write_json(staging / "reload_check.json", prediction)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return str(destination)
