# /// script
# dependencies = [
#     "accelerate>=0.33",
#     "bitsandbytes>=0.45",
#     "datasets>=2.21",
#     "marimo",
#     "numpy>=1.26",
#     "peft>=0.12",
#     "pyarrow>=17",
#     "safetensors>=0.4",
#     "sentencepiece>=0.2",
#     "torch>=2.4",
#     "transformers>=4.48,<6",
# ]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from pathlib import Path
    from collections import Counter
    import datetime as dt
    import difflib
    import hashlib
    import importlib
    import json
    import math
    import os
    import platform
    import random
    import shutil
    import subprocess
    import sys
    import tempfile
    import textwrap
    import time
    import urllib.error
    import urllib.parse
    import urllib.request

    import numpy as np
    import pyarrow.parquet as pq

    WORKSPACE = Path(__file__).resolve().parent
    DATASET_PATH = WORKSPACE / "wildguardtrain_9000_seed42.parquet"
    RUNTIME_ROOT = WORKSPACE / ".piguard"
    UPSTREAM_DIR = RUNTIME_ROOT / "PIGuard"
    ARTIFACT_ROOT = WORKSPACE / "artifacts" / "piguard_custom"
    CACHE_VERSION = 1
    STEP_VERSIONS = {
        "prepare": 1,
        "stage1": 1,
        "scan": 1,
        "generate": 1,
        "stage2": 1,
        "evaluate": 1,
    }
    STEP_LABELS = {
        "prepare": "1 · Prepare pinned source + data",
        "stage1": "2 · Train stage one",
        "scan": "3 · Scan biased tokens",
        "generate": "4 · Generate + refine benign prompts",
        "stage2": "5 · Retrain from untouched base",
        "evaluate": "6 · Evaluate final checkpoint",
    }
    UPSTREAM_REPO = "https://github.com/leolee99/PIGuard.git"
    UPSTREAM_COMMIT = "1b5751e88bf7475acbedfc8eda795ce060307c84"
    SEED = 42
    LABELS = {"unharmful": 0, "harmful": 1}
    PRESETS = {
        "DeBERTa V3 Base": "microsoft/deberta-v3-base",
        "Qwen 2.5 3B": "Qwen/Qwen2.5-3B",
    }
    KNOWN_PARAMETER_COUNTS = {
        "microsoft/deberta-v3-base": 184_000_000,
        "Qwen/Qwen2.5-3B": 3_090_000_000,
    }
    ENCODER_MODEL_TYPES = {
        "albert",
        "bert",
        "big_bird",
        "camembert",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "flaubert",
        "funnel",
        "layoutlm",
        "longformer",
        "mpnet",
        "rembert",
        "roberta",
        "xlm",
        "xlm-roberta",
        "xlnet",
    }

    MODE_PIGUARD_SOURCE = textwrap.dedent(r"""from __future__ import annotations

    import json
    import os
    from collections import OrderedDict
    from pathlib import Path

    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from PIGuard import PIGuard as OfficialPIGuard


    ENCODER_MODEL_TYPES = {
        "albert", "bert", "big_bird", "camembert", "deberta", "deberta-v2",
        "distilbert", "electra", "flaubert", "funnel", "layoutlm", "longformer",
        "mpnet", "rembert", "roberta", "xlm", "xlm-roberta", "xlnet",
    }


    class PIGuard(OfficialPIGuard):
        # Small compatibility layer over the pinned official PIGuard class.

        def __init__(self, model_name, num_labels, device):
            self.training_mode = os.environ.get("PIGUARD_TRAINING_MODE", "full")
            self.revision = os.environ.get("PIGUARD_MODEL_REVISION") or None
            self.trust_remote_code = os.environ.get("PIGUARD_TRUST_REMOTE_CODE") == "1"
            self.hf_token = os.environ.get("HF_TOKEN") or None
            self.device = device

            if (
                self.training_mode == "full"
                and model_name == "microsoft/deberta-v3-base"
                and self.revision in (None, "main")
                and not self.trust_remote_code
            ):
                super().__init__(model_name, num_labels, device)
                self.pooling = "first"
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                return

            nn.Module.__init__(self)
            common = {
                "revision": self.revision,
                "token": self.hf_token,
                "trust_remote_code": self.trust_remote_code,
            }
            common = {key: value for key, value in common.items() if value is not None}
            self.config = AutoConfig.from_pretrained(model_name, **common)
            self.pooling = "first" if self.config.model_type in ENCODER_MODEL_TYPES else "last"

            if self.training_mode == "qlora":
                if not torch.cuda.is_available():
                    raise RuntimeError("QLoRA requires a CUDA GPU and bitsandbytes.")
                from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
                from transformers import BitsAndBytesConfig

                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                quantization = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
                backbone = AutoModel.from_pretrained(
                    model_name,
                    config=self.config,
                    quantization_config=quantization,
                    device_map={"": 0},
                    **common,
                )
                backbone = prepare_model_for_kbit_training(backbone)
                if hasattr(backbone.config, "use_cache"):
                    backbone.config.use_cache = False
                lora = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=int(os.environ.get("PIGUARD_LORA_R", "16")),
                    lora_alpha=int(os.environ.get("PIGUARD_LORA_ALPHA", "32")),
                    lora_dropout=float(os.environ.get("PIGUARD_LORA_DROPOUT", "0.05")),
                    target_modules=os.environ.get("PIGUARD_LORA_TARGETS", "all-linear"),
                    bias="none",
                )
                self.deberta = get_peft_model(backbone, lora)
            elif self.training_mode == "full":
                self.deberta = AutoModel.from_pretrained(model_name, config=self.config, **common).to(device)
            else:
                raise ValueError(f"Unsupported training mode: {self.training_mode}")

            self.classifier = nn.Linear(self.deberta.config.hidden_size, num_labels).to(device)
            self.loss_fct = nn.CrossEntropyLoss().to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **common)
            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token_id is None:
                    raise ValueError("Tokenizer has neither a pad token nor an EOS token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token

        def _pool(self, hidden, attention_mask):
            if self.pooling == "first":
                return hidden[:, 0, :]
            reverse_offset = attention_mask.flip(1).long().argmax(1)
            last_index = attention_mask.shape[1] - 1 - reverse_offset
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), last_index]

        def forward(self, input_ids, attention_mask, labels=None, mode="train"):
            outputs = self.deberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            pooled = self._pool(outputs["last_hidden_state"], attention_mask)
            logits = self.classifier(pooled.to(self.classifier.weight.dtype))
            loss = None if labels is None else self.loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
            return logits, loss

        def classify(self, input_text):
            encoded = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )
            logits, _ = self(
                encoded["input_ids"].to(self.device),
                encoded["attention_mask"].to(self.device),
            )
            return logits

        def state_dict(self, *args, **kwargs):
            if self.training_mode != "qlora":
                return super().state_dict(*args, **kwargs)
            from peft import get_peft_model_state_dict

            state = OrderedDict()
            for key, value in get_peft_model_state_dict(self.deberta).items():
                state[f"__piguard_adapter__.{key}"] = value
            for key, value in self.classifier.state_dict().items():
                state[f"classifier.{key}"] = value
            return state

        def load_state_dict(self, state_dict, strict=True):
            if self.training_mode != "qlora":
                return super().load_state_dict(state_dict, strict=strict)
            from peft import set_peft_model_state_dict

            adapter = {
                key.removeprefix("__piguard_adapter__."): value
                for key, value in state_dict.items()
                if key.startswith("__piguard_adapter__.")
            }
            head = {
                key.removeprefix("classifier."): value
                for key, value in state_dict.items()
                if key.startswith("classifier.")
            }
            if not adapter or not head:
                raise RuntimeError("Checkpoint is missing the PEFT adapter or classifier head.")
            adapter_result = set_peft_model_state_dict(self.deberta, adapter)
            self.classifier.load_state_dict(head, strict=True)
            return adapter_result

        def export_bundle(self, output_dir):
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            if self.training_mode == "qlora":
                self.deberta.save_pretrained(output / "adapter")
            else:
                self.deberta.save_pretrained(output / "backbone")
            torch.save(self.classifier.state_dict(), output / "classifier_head.pth")
            self.tokenizer.save_pretrained(output / "tokenizer")
            (output / "piguard_bundle.json").write_text(
                json.dumps(
                    {"training_mode": self.training_mode, "pooling": self.pooling},
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
    """)

    def file_sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    DATASET_SHA256 = (
        file_sha256(DATASET_PATH) if DATASET_PATH.exists() else "missing"
    )

    def atomic_text(path, text):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(text)
            temporary = Path(handle.name)
        os.replace(temporary, path)

    def write_json(path, value):
        atomic_text(
            path, json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )

    def read_json(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def read_jsonl(path):
        path = Path(path)
        if not path.exists():
            return []
        rows = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {path}:{line_number}"
                    ) from exc
        return rows

    def append_jsonl(path, row):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def command(args, cwd=None, env=None, check=True, stdout=None):
        return subprocess.run(
            [str(arg) for arg in args],
            cwd=cwd,
            env=env,
            check=check,
            text=True,
            stdout=stdout if stdout is not None else subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def patch_upstream():
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        if not (UPSTREAM_DIR / ".git").exists():
            command(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    UPSTREAM_REPO,
                    UPSTREAM_DIR,
                ]
            )
            command(
                [
                    "git",
                    "-C",
                    UPSTREAM_DIR,
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    UPSTREAM_COMMIT,
                ]
            )
            command(
                [
                    "git",
                    "-C",
                    UPSTREAM_DIR,
                    "checkout",
                    "--detach",
                    UPSTREAM_COMMIT,
                ]
            )
        head = command(
            ["git", "-C", UPSTREAM_DIR, "rev-parse", "HEAD"]
        ).stdout.strip()
        if head != UPSTREAM_COMMIT:
            raise RuntimeError(
                f"Refusing to patch unexpected PIGuard commit {head}."
            )

        pristine = command(
            ["git", "-C", UPSTREAM_DIR, "show", "HEAD:train.py"]
        ).stdout
        replacements = [
            (
                "from PIGuard import PIGuard",
                "from mode_piguard import PIGuard",
            ),
            ("from eval import evaluate\n", ""),
            (
                "    # tokenizer initial\n    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')",
                "    # tokenizer initial (model ID supplied by the notebook)\n"
                "    model_name = os.environ['PIGUARD_MODEL_NAME']\n"
                "    revision = os.environ.get('PIGUARD_MODEL_REVISION') or None\n"
                "    trust_remote_code = os.environ.get('PIGUARD_TRUST_REMOTE_CODE') == '1'\n"
                "    hf_token = os.environ.get('HF_TOKEN') or None\n"
                "    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, token=hf_token, trust_remote_code=trust_remote_code)\n"
                "    if tokenizer.pad_token_id is None:\n"
                "        tokenizer.pad_token = tokenizer.eos_token",
            ),
            (
                "    model = PIGuard('microsoft/deberta-v3-base', num_labels=2, device=device) ",
                "    model = PIGuard(model_name, num_labels=2, device=device)",
            ),
            ("    best_accuracy = 0", "    best_accuracy = -1"),
            (
                "                            torch.save(model.state_dict(), best_model_path)\n"
                '                            print(f"Saved to {best_model_path}.")',
                "                            torch.save(model.state_dict(), best_model_path)\n"
                "                            model.export_bundle(os.path.join(save_path, 'best_bundle'))\n"
                '                            print(f"Saved to {best_model_path}.")',
            ),
            (
                "    # evaluate on overall test set\n"
                '    logger.info("Evaluate Best Model on Test Sets.")\n'
                "    model.load_state_dict(torch.load(best_model_path, map_location=device))\n"
                '    logger.info(f"Loaded model from {best_model_path}.")\n'
                "    evaluate(model, args.dataset_root)",
                "    # The notebook evaluates its fixed WildGuard test split.\n"
                '    logger.info(f"Best validation checkpoint: {best_model_path}")',
            ),
        ]
        patched = pristine
        for old, new in replacements:
            if patched.count(old) != 1:
                raise RuntimeError(
                    f"Pinned train.py did not match the expected patch hunk: {old[:70]!r}"
                )
            patched = patched.replace(old, new)
        atomic_text(UPSTREAM_DIR / "train.py", patched)
        atomic_text(UPSTREAM_DIR / "mode_piguard.py", MODE_PIGUARD_SOURCE)

        train_diff = command(
            ["git", "-C", UPSTREAM_DIR, "diff", "--", "train.py"]
        ).stdout
        shim_diff = "".join(
            difflib.unified_diff(
                [],
                MODE_PIGUARD_SOURCE.splitlines(keepends=True),
                fromfile="/dev/null",
                tofile="b/mode_piguard.py",
            )
        )
        return train_diff + shim_diff

    def _apportion(count, ratios=(0.8, 0.1, 0.1)):
        raw = [count * ratio for ratio in ratios]
        result = [math.floor(value) for value in raw]
        for index in sorted(
            range(len(raw)),
            key=lambda i: (raw[i] - result[i], -i),
            reverse=True,
        )[: count - sum(result)]:
            result[index] += 1
        return result

    def prepare_data(run_dir, smoke=False):
        if not DATASET_PATH.exists():
            raise FileNotFoundError(DATASET_PATH)
        table = pq.read_table(
            DATASET_PATH,
            columns=["prompt", "prompt_harm_label", "source_index"],
        )
        source = table.to_pylist()
        if len(source) != 9000:
            raise ValueError(
                f"Expected 9,000 WildGuard rows, found {len(source):,}."
            )
        prompts = [row["prompt"] for row in source]
        if len(set(prompts)) != 9000:
            raise ValueError(
                "WildGuard selection must contain 9,000 unique prompts."
            )
        counts = Counter(row["prompt_harm_label"] for row in source)
        if counts != Counter({"harmful": 5148, "unharmful": 3852}):
            raise ValueError(f"Unexpected label counts: {dict(counts)}")

        split_rows = {"train": [], "valid": [], "test": []}
        manifest_rows = []
        for source_label, numeric_label in LABELS.items():
            label_rows = [
                row
                for row in source
                if row["prompt_harm_label"] == source_label
            ]
            label_rows.sort(
                key=lambda row: hashlib.sha256(
                    f"{SEED}\0{row['prompt']}".encode()
                ).hexdigest()
            )
            train_n, valid_n, test_n = _apportion(len(label_rows))
            boundaries = (train_n, train_n + valid_n)
            groups = {
                "train": label_rows[: boundaries[0]],
                "valid": label_rows[boundaries[0] : boundaries[1]],
                "test": label_rows[boundaries[1] :],
            }
            if len(groups["test"]) != test_n:
                raise AssertionError("Stratified split allocation failed.")
            for split, rows in groups.items():
                split_rows[split].extend(
                    {"prompt": row["prompt"], "label": numeric_label}
                    for row in rows
                )
                manifest_rows.extend(
                    {
                        "prompt_sha256": hashlib.sha256(
                            row["prompt"].encode()
                        ).hexdigest(),
                        "source_index": row["source_index"],
                        "source_label": source_label,
                        "label": numeric_label,
                        "split": split,
                    }
                    for row in rows
                )

        for split, rows in split_rows.items():
            rows.sort(
                key=lambda row: hashlib.sha256(
                    f"order\0{SEED}\0{row['prompt']}".encode()
                ).hexdigest()
            )
            write_json(Path(run_dir) / "data" / f"{split}.json", rows)
        split_sizes = {key: len(value) for key, value in split_rows.items()}
        if split_sizes != {"train": 7200, "valid": 900, "test": 900}:
            raise AssertionError(split_sizes)

        manifest_path = Path(run_dir) / "manifests" / "split_manifest.jsonl"
        atomic_text(
            manifest_path,
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in manifest_rows
            ),
        )
        data_manifest = {
            "source": str(DATASET_PATH),
            "source_sha256": DATASET_SHA256,
            "seed": SEED,
            "label_mapping": LABELS,
            "source_label_counts": dict(counts),
            "split_sizes": split_sizes,
            "split_label_counts": {
                split: dict(Counter(row["label"] for row in rows))
                for split, rows in split_rows.items()
            },
            "unique_prompts": len(set(prompts)),
        }
        write_json(
            Path(run_dir) / "manifests" / "data_manifest.json", data_manifest
        )

        if smoke:
            for split in ("train", "valid", "test"):
                rows = split_rows[split]
                tiny = [
                    next(row for row in rows if row["label"] == label)
                    for label in (0, 1)
                ]
                tiny += [
                    next(
                        row
                        for row in rows
                        if row["label"] == label and row not in tiny
                    )
                    for label in (0, 1)
                ]
                write_json(
                    Path(run_dir) / "data" / f"smoke_{split}.json", tiny
                )
        return data_manifest

    def model_parameter_count(model_id, revision=None):
        if model_id in KNOWN_PARAMETER_COUNTS:
            return KNOWN_PARAMETER_COUNTS[model_id]
        from huggingface_hub import HfApi

        info = HfApi(token=os.environ.get("HF_TOKEN")).model_info(
            model_id,
            revision=revision or None,
            files_metadata=True,
        )
        safetensors = getattr(info, "safetensors", None)
        total = (
            safetensors.get("total")
            if isinstance(safetensors, dict)
            else getattr(safetensors, "total", None)
        )
        if total is None:
            raise ValueError(
                "Could not determine the custom model's parameter count; choose Full or QLoRA manually."
            )
        return int(total)

    def resolve_training_mode(config):
        requested = config["training_mode"]
        if requested == "auto":
            count = model_parameter_count(
                config["model_id"], config.get("revision")
            )
            return ("full" if count <= 500_000_000 else "qlora"), count
        try:
            count = model_parameter_count(
                config["model_id"], config.get("revision")
            )
        except Exception:
            count = None
        return requested, count

    def _fingerprint(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True).encode()
        ).hexdigest()[:12]

    def pipeline_artifacts(config, root=ARTIFACT_ROOT):
        config = {
            "model_id": config["model_id"].strip(),
            "revision": config.get("revision", "").strip(),
            "training_mode": config["training_mode"],
            "trust_remote_code": bool(config["trust_remote_code"]),
            "smoke": bool(config["smoke"]),
            "batch_size": int(config["batch_size"]),
            "max_length": int(config["max_length"]),
            "ai_base_url": config["ai_base_url"].strip().rstrip("/"),
            "ai_model": config["ai_model"].strip(),
        }
        common = {"cache_version": CACHE_VERSION}
        prepare = {
            **common,
            "step_version": STEP_VERSIONS["prepare"],
            "upstream_commit": UPSTREAM_COMMIT,
            "dataset_sha256": DATASET_SHA256,
            "seed": SEED,
            "smoke": config["smoke"],
        }
        prepare_key = _fingerprint(prepare)
        training = {
            "model_id": config["model_id"],
            "revision": config["revision"],
            "training_mode": config["training_mode"],
            "trust_remote_code": config["trust_remote_code"],
            "smoke": config["smoke"],
            "batch_size": config["batch_size"],
            "max_length": config["max_length"],
            "epochs": 1 if config["smoke"] else 3,
            "save_step": 1 if config["smoke"] else 200,
            "warmup": 0 if config["smoke"] else 100,
            "seed": SEED,
        }
        stage1 = {
            **common,
            "step_version": STEP_VERSIONS["stage1"],
            "prepare": prepare_key,
            "stage": "stage1",
            **training,
        }
        stage1_key = _fingerprint(stage1)
        scan = {
            **common,
            "step_version": STEP_VERSIONS["scan"],
            "stage1": stage1_key,
            "vocabulary_limit": 512 if config["smoke"] else None,
            "saved_tokens": 300,
        }
        scan_key = _fingerprint(scan)
        generate = {
            **common,
            "step_version": STEP_VERSIONS["generate"],
            "prepare": prepare_key,
            "scan": scan_key,
            "ai_base_url": config["ai_base_url"],
            "ai_model": config["ai_model"],
            "target": 4 if config["smoke"] else 1000,
            "request_budget": 400,
        }
        generate_key = _fingerprint(generate)
        stage2 = {
            **common,
            "step_version": STEP_VERSIONS["stage2"],
            "prepare": prepare_key,
            "generate": generate_key,
            "stage": "stage2",
            **training,
        }
        stage2_key = _fingerprint(stage2)
        evaluate = {
            **common,
            "step_version": STEP_VERSIONS["evaluate"],
            "prepare": prepare_key,
            "stage2": stage2_key,
            "batch_size": config["batch_size"],
            "max_length": config["max_length"],
            "threshold": 0.5,
        }
        inputs = {
            "prepare": prepare,
            "stage1": stage1,
            "scan": scan,
            "generate": generate,
            "stage2": stage2,
            "evaluate": evaluate,
        }
        keys = {step: _fingerprint(value) for step, value in inputs.items()}
        root = Path(root)
        return {
            "root": root,
            "config": config,
            "steps": {
                step: {
                    "step": step,
                    "key": keys[step],
                    "inputs": value,
                    "path": root / "cache" / step / keys[step],
                }
                for step, value in inputs.items()
            },
        }

    _REQUIRED_OUTPUTS = {
        "prepare": (
            "data/train.json",
            "data/valid.json",
            "data/test.json",
            "manifests/data_manifest.json",
            "patched-source.diff",
        ),
        "stage1": ("checkpoints/best_model.pth",),
        "scan": ("biased_tokens.json",),
        "generate": ("generated/benign_prompts.jsonl",),
        "stage2": ("checkpoints/best_model.pth",),
        "evaluate": ("metrics.json", "test_predictions.jsonl"),
    }

    def _required_outputs(artifact):
        outputs = _REQUIRED_OUTPUTS[artifact["step"]]
        if artifact["step"] == "prepare" and artifact["inputs"]["smoke"]:
            outputs += (
                "data/smoke_train.json",
                "data/smoke_valid.json",
                "data/smoke_test.json",
            )
        return outputs

    def _cache_manifest(artifact):
        path = artifact["path"] / "manifest.json"
        if not path.exists():
            return None
        try:
            manifest = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if (
            manifest.get("schema_version") != CACHE_VERSION
            or manifest.get("step") != artifact["step"]
            or manifest.get("key") != artifact["key"]
            or manifest.get("inputs") != artifact["inputs"]
            or not all(
                (artifact["path"] / relative).exists()
                for relative in _required_outputs(artifact)
            )
        ):
            return None
        return manifest

    def _complete_artifact(artifact, result, config=None):
        # ponytail: one writer per cache key; add file locks if concurrent sessions need the same key.
        manifest = {
            "schema_version": CACHE_VERSION,
            "step": artifact["step"],
            "key": artifact["key"],
            "inputs": artifact["inputs"],
            "result": result,
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        if config is not None:
            manifest["config"] = config
        write_json(artifact["path"] / "manifest.json", manifest)
        return {
            "status": "completed",
            "step": artifact["step"],
            "key": artifact["key"],
            "artifact_dir": str(artifact["path"]),
            **result,
        }

    def _training_matches(saved, current):
        return (
            saved.get("model_id") == current["model_id"]
            and (saved.get("revision") or "") == current["revision"]
            and saved.get("training_mode") == current["training_mode"]
            and bool(saved.get("trust_remote_code"))
            == current["trust_remote_code"]
            and bool(saved.get("smoke")) == current["smoke"]
            and saved.get("batch_size") == current["batch_size"]
            and saved.get("max_length") == current["max_length"]
        )

    def _legacy_generation_matches(run_dir, config):
        manifest_path = run_dir / "manifests" / "ai_generation.json"
        output_path = run_dir / "generated" / "benign_prompts.jsonl"
        if not manifest_path.exists() or not output_path.exists():
            return False
        try:
            manifest = read_json(manifest_path)
            rows = read_jsonl(output_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        target = 4 if config["smoke"] else 1000
        return (
            manifest.get("model") == config["ai_model"]
            and str(manifest.get("base_url", "")).rstrip("/")
            == config["ai_base_url"]
            and manifest.get("accepted") == target
            and len(rows) == target
        )

    def _legacy_artifact(pipeline, step):
        root = pipeline["root"]
        config = pipeline["config"]
        if not root.exists():
            return None
        for run_dir in sorted(root.iterdir()):
            config_path = run_dir / "run_config.json"
            if run_dir.name == "cache" or not config_path.exists():
                continue
            try:
                saved = read_json(config_path)
                data_manifest = read_json(
                    run_dir / "manifests" / "data_manifest.json"
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            base_matches = (
                saved.get("upstream_commit") == UPSTREAM_COMMIT
                and saved.get("dataset_sha256") == DATASET_SHA256
                and data_manifest.get("seed") == SEED
                and bool(saved.get("smoke")) == config["smoke"]
            )
            if not base_matches:
                continue
            training_matches = _training_matches(saved, config)
            generation_matches = training_matches and _legacy_generation_matches(
                run_dir, config
            )
            paths = {
                "prepare": run_dir,
                "stage1": run_dir / "stage1",
                "scan": run_dir,
                "generate": run_dir,
                "stage2": run_dir / "stage2",
                "evaluate": run_dir / "stage2" / "evaluation",
            }
            complete = {
                "prepare": all(
                    (paths["prepare"] / relative).exists()
                    for relative in _required_outputs(
                        pipeline["steps"]["prepare"]
                    )
                ),
                "stage1": training_matches
                and (paths["stage1"] / "stage_manifest.json").exists()
                and (paths["stage1"] / "checkpoints/best_model.pth").exists(),
                "scan": training_matches
                and (paths["scan"] / "biased_tokens.json").exists(),
                "generate": generation_matches,
                "stage2": generation_matches
                and (paths["stage2"] / "stage_manifest.json").exists()
                and (paths["stage2"] / "checkpoints/best_model.pth").exists(),
                "evaluate": generation_matches
                and (paths["evaluate"] / "metrics.json").exists()
                and (paths["evaluate"] / "test_predictions.jsonl").exists(),
            }
            if complete[step]:
                return {
                    **pipeline["steps"][step],
                    "path": paths[step],
                    "legacy": True,
                    "legacy_run": run_dir,
                }
        return None

    def _resolved_artifact(pipeline, step):
        artifact = pipeline["steps"][step]
        if _cache_manifest(artifact) is not None:
            return artifact
        return _legacy_artifact(pipeline, step)

    def _cached_result(artifact):
        result = {
            "status": "cached",
            "step": artifact["step"],
            "key": artifact["key"],
            "artifact_dir": str(artifact["path"]),
            "legacy": bool(artifact.get("legacy")),
        }
        manifest = None if artifact.get("legacy") else _cache_manifest(artifact)
        if manifest is not None and isinstance(manifest.get("result"), dict):
            result.update(manifest["result"])
        return result

    def _require_artifact(pipeline, step):
        artifact = _resolved_artifact(pipeline, step)
        if artifact is None:
            raise RuntimeError(f"Complete {STEP_LABELS[step]} first.")
        return artifact

    def _effective_training_config(config):
        if not config["model_id"]:
            raise ValueError(
                "Choose a preset or enter a Hugging Face model ID."
            )
        resolved_mode, parameter_count = resolve_training_mode(config)
        return {
            key: config[key]
            for key in (
                "model_id",
                "revision",
                "training_mode",
                "trust_remote_code",
                "smoke",
                "batch_size",
                "max_length",
            )
        } | {
            "resolved_training_mode": resolved_mode,
            "parameter_count": parameter_count,
            "upstream_commit": UPSTREAM_COMMIT,
            "dataset_sha256": DATASET_SHA256,
        }

    def prepare_run(pipeline):
        cached = _resolved_artifact(pipeline, "prepare")
        if cached is not None:
            return _cached_result(cached)
        artifact = pipeline["steps"]["prepare"]
        artifact["path"].mkdir(parents=True, exist_ok=True)
        source_diff = patch_upstream()
        data_manifest = prepare_data(
            artifact["path"], smoke=pipeline["config"]["smoke"]
        )
        atomic_text(artifact["path"] / "patched-source.diff", source_diff)
        return _complete_artifact(
            artifact, {"splits": data_manifest["split_sizes"]}
        )

    def _require_training_platform(mode):
        if mode != "qlora":
            return
        import torch

        if platform.system() != "Linux" or not torch.cuda.is_available():
            raise RuntimeError(
                "QLoRA/4-bit needs Linux, a CUDA GPU, and bitsandbytes. Use a GPU host or select Full manually."
            )

    def run_official_stage(pipeline, stage):
        if stage not in {"stage1", "stage2"}:
            raise ValueError(stage)
        cached = _resolved_artifact(pipeline, stage)
        if cached is not None:
            return _cached_result(cached)
        prepared = _require_artifact(pipeline, "prepare")
        generated = (
            _require_artifact(pipeline, "generate")
            if stage == "stage2"
            else None
        )
        artifact = pipeline["steps"][stage]
        config = _effective_training_config(pipeline["config"])
        mode = config["resolved_training_mode"]
        _require_training_platform(mode)
        patch_upstream()
        output_dir = artifact["path"]
        output_dir.mkdir(parents=True, exist_ok=True)
        if stage == "stage2":
            build_stage2_data(
                prepared["path"],
                generated["path"],
                output_dir,
                smoke=config["smoke"],
            )
        prefix = "smoke_" if config["smoke"] else ""
        train_path = (
            prepared["path"] / "data" / f"{prefix}train.json"
            if stage == "stage1"
            else output_dir / "data" / f"{prefix}stage2_train.json"
        )
        valid_path = prepared["path"] / "data" / f"{prefix}valid.json"
        checkpoint_dir = output_dir / "checkpoints"
        logs_dir = output_dir / "logs"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(
            {
                "PIGUARD_MODEL_NAME": config["model_id"],
                "PIGUARD_MODEL_REVISION": config.get("revision") or "",
                "PIGUARD_TRUST_REMOTE_CODE": "1"
                if config["trust_remote_code"]
                else "0",
                "PIGUARD_TRAINING_MODE": mode,
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        stage_batch_size = (
            (2 if stage == "stage1" else 4)
            if config["smoke"]
            else config["batch_size"]
        )
        args = [
            sys.executable,
            "train.py",
            "--name",
            stage,
            "--train_set",
            train_path,
            "--valid_set",
            valid_path,
            "--checkpoint_path",
            checkpoint_dir,
            "--logs",
            logs_dir,
            "--batch_size",
            stage_batch_size,
            "--eval_batch_size",
            stage_batch_size,
            "--max_length",
            config["max_length"],
            "--epochs",
            1 if config["smoke"] else 3,
            "--save_step",
            1 if config["smoke"] else 200,
            "--warmup",
            0 if config["smoke"] else 100,
            "--save_thres",
            -1,
            "--seed",
            SEED,
        ]
        log_path = output_dir / "training.log"
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        with log_path.open("w", encoding="utf-8") as log_handle:
            result = command(
                args,
                cwd=UPSTREAM_DIR,
                env=environment,
                check=False,
                stdout=log_handle,
            )
        if result.returncode:
            tail = "\n".join(
                log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[-30:]
            )
            raise RuntimeError(
                f"Official PIGuard training failed (exit {result.returncode}).\n{tail}"
            )
        checkpoint = checkpoint_dir / "best_model.pth"
        if not checkpoint.exists():
            raise RuntimeError(
                "Official training completed without best_model.pth."
            )
        record = {
            "stage": stage,
            "started_at": started,
            "fresh_base_model": True,
            "checkpoint": str(checkpoint),
            "bundle": str(checkpoint_dir / "best_bundle"),
            "official_train_py": str(UPSTREAM_DIR / "train.py"),
            "log": str(log_path),
        }
        return _complete_artifact(artifact, record, config=config)

    def _artifact_training_config(artifact):
        if artifact.get("legacy"):
            config = read_json(artifact["legacy_run"] / "run_config.json")
        else:
            manifest = _cache_manifest(artifact)
            config = None if manifest is None else manifest.get("config")
        if not isinstance(config, dict) or not config.get(
            "resolved_training_mode"
        ):
            raise RuntimeError(
                f"Stored {artifact['step']} artifact has no training configuration."
            )
        return config

    def load_piguard_model(artifact):
        import torch

        config = _artifact_training_config(artifact)
        _require_training_platform(config["resolved_training_mode"])
        patch_upstream()
        if str(UPSTREAM_DIR) not in sys.path:
            sys.path.insert(0, str(UPSTREAM_DIR))
        module = importlib.import_module("mode_piguard")
        module = importlib.reload(module)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        old_environment = {
            key: os.environ.get(key)
            for key in (
                "PIGUARD_TRAINING_MODE",
                "PIGUARD_MODEL_REVISION",
                "PIGUARD_TRUST_REMOTE_CODE",
            )
        }
        try:
            os.environ["PIGUARD_TRAINING_MODE"] = config[
                "resolved_training_mode"
            ]
            os.environ["PIGUARD_MODEL_REVISION"] = config.get("revision") or ""
            os.environ["PIGUARD_TRUST_REMOTE_CODE"] = (
                "1" if config["trust_remote_code"] else "0"
            )
            model = module.PIGuard(
                config["model_id"], num_labels=2, device=device
            )
        finally:
            for key, value in old_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        checkpoint = artifact["path"] / "checkpoints" / "best_model.pth"
        if not checkpoint.exists():
            raise RuntimeError(
                f"Run {artifact['step']} training before loading its checkpoint."
            )
        model.load_state_dict(
            torch.load(checkpoint, map_location=device), strict=True
        )
        model.eval()
        return model, device, config

    def scan_biased_tokens(pipeline):
        cached = _resolved_artifact(pipeline, "scan")
        if cached is not None:
            return _cached_result(cached)
        import torch

        stage1 = _require_artifact(pipeline, "stage1")
        artifact = pipeline["steps"]["scan"]
        model, device, config = load_piguard_model(stage1)
        tokenizer = model.tokenizer
        vocabulary = sorted(
            tokenizer.get_vocab().items(), key=lambda item: item[1]
        )
        special_ids = set(tokenizer.all_special_ids)
        if config["smoke"]:
            vocabulary = vocabulary[:512]
        batch_size = 128 if config["resolved_training_mode"] == "full" else 32
        scores = []
        with torch.no_grad():
            for start in range(0, len(vocabulary), batch_size):
                items = vocabulary[start : start + batch_size]
                candidates = [
                    (
                        token,
                        token_id,
                        tokenizer.decode(
                            [token_id], skip_special_tokens=True
                        ).strip(),
                    )
                    for token, token_id in items
                    if token_id not in special_ids
                ]
                candidates = [
                    item
                    for item in candidates
                    if item[2] and len(item[2]) <= 80 and item[2].isprintable()
                ]
                if not candidates:
                    continue
                batch = tokenizer(
                    [decoded for _, _, decoded in candidates],
                    padding=True,
                    truncation=True,
                    max_length=min(config["max_length"], 64),
                    return_tensors="pt",
                )
                logits, _ = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                harmful = (
                    torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
                )
                for (token, token_id, decoded), score in zip(
                    candidates, harmful
                ):
                    scores.append(
                        {
                            "token_id": token_id,
                            "token": token,
                            "decoded": decoded,
                            "harmful_probability": score,
                        }
                    )
        scores.sort(key=lambda row: row["harmful_probability"], reverse=True)
        output = {
            "pooling": model.pooling,
            "vocabulary_size": len(tokenizer),
            "tokens_scanned": len(vocabulary),
            "smoke_limited": config["smoke"],
            "top_biased_tokens": scores[:300],
        }
        write_json(artifact["path"] / "biased_tokens.json", output)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = {
            key: value
            for key, value in output.items()
            if key != "top_biased_tokens"
        } | {"saved": len(output["top_biased_tokens"])}
        return _complete_artifact(artifact, result)

    class AIAPIError(RuntimeError):
        def __init__(self, message, status=None, retry_after=None):
            super().__init__(message)
            self.status = status
            self.retry_after = retry_after

    def _normalize_ai_base_url(base_url):
        value = base_url.strip().rstrip("/")
        parsed = urllib.parse.urlparse(value)
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if (
            not parsed.hostname
            or parsed.scheme not in {"http", "https"}
            or (parsed.scheme == "http" and not local)
            or parsed.username
            or parsed.password
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "AI API base URL must use HTTPS (or HTTP on localhost) and contain no credentials, query, or fragment."
            )
        return value

    def _ai_transport(base_url, path, api_key, payload=None):
        url = f"{_normalize_ai_base_url(base_url)}/{path.lstrip('/')}"
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            url,
            data=body,
            method="GET" if payload is None else "POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            retry_after = exc.headers.get("Retry-After")
            detail = exc.read().decode(errors="replace")[:500]
            raise AIAPIError(
                detail or str(exc),
                exc.code,
                float(retry_after) if retry_after else None,
            ) from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            raise AIAPIError(str(exc)) from exc

    def _spend_budget(budget):
        if budget["used"] >= budget["limit"]:
            raise RuntimeError(
                f"AI API request budget exhausted ({budget['limit']}). Progress is saved; resume later."
            )
        budget["used"] += 1

    def ai_chat(
        model_id,
        messages,
        api_key,
        budget,
        transport=None,
        sleep_fn=time.sleep,
    ):
        if transport is None:
            raise ValueError("AI transport is required.")
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4000,
        }
        for attempt in range(5):
            _spend_budget(budget)
            try:
                response = transport("chat/completions", api_key, payload)
                return response["choices"][0]["message"]["content"]
            except AIAPIError as exc:
                if exc.status != 429 or attempt == 4:
                    raise
                sleep_fn(
                    exc.retry_after if exc.retry_after is not None else 60
                )
        raise AssertionError("unreachable")

    def validate_ai_model(model_id, api_key, budget, transport=None):
        if transport is None:
            raise ValueError("AI transport is required.")
        _spend_budget(budget)
        response = transport("models", api_key, None)
        available = {
            row["id"]
            for row in response.get("data", [])
            if isinstance(row, dict) and row.get("id")
        }
        if model_id not in available:
            sample = ", ".join(sorted(available)[:8]) or "no models returned"
            raise ValueError(
                f"AI model {model_id!r} is unavailable. API returned: {sample}"
            )

    def parse_prompt_response(content):
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            left, right = text.find("["), text.rfind("]")
            if left < 0 or right <= left:
                raise ValueError(
                    "AI API response did not contain a JSON array."
                )
            payload = json.loads(text[left : right + 1])
        if isinstance(payload, dict):
            payload = payload.get("prompts", [])
        if not isinstance(payload, list):
            raise ValueError("AI API response must be a JSON list.")
        prompts = []
        for item in payload:
            prompt = (
                item
                if isinstance(item, str)
                else item.get("prompt")
                if isinstance(item, dict)
                else None
            )
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt.strip())
        return prompts

    def accept_candidates(candidates, required_groups, seen):
        accepted = []
        for prompt, required in zip(candidates, required_groups):
            normalized = " ".join(prompt.casefold().split())
            if normalized in seen or not all(
                token.casefold() in prompt.casefold() for token in required
            ):
                continue
            seen.add(normalized)
            accepted.append((prompt, required))
        return accepted

    def generate_benign_prompts(pipeline, transport=None):
        cached = _resolved_artifact(pipeline, "generate")
        if cached is not None:
            return _cached_result(cached)
        prepared = _require_artifact(pipeline, "prepare")
        scanned = _require_artifact(pipeline, "scan")
        artifact = pipeline["steps"]["generate"]
        config = pipeline["config"]
        ai_base_url = _normalize_ai_base_url(config["ai_base_url"])
        ai_model = config["ai_model"].strip()
        if not ai_model:
            raise ValueError("Enter an AI model ID.")
        biased_path = scanned["path"] / "biased_tokens.json"
        tokens = [
            row["decoded"]
            for row in read_json(biased_path)["top_biased_tokens"]
        ]
        tokens = list(
            dict.fromkeys(token for token in tokens if token.strip())
        )
        target = artifact["inputs"]["target"]
        output_path = artifact["path"] / "generated" / "benign_prompts.jsonl"
        existing = read_jsonl(output_path)
        if len(existing) > target:
            raise RuntimeError(
                f"Resume file already has {len(existing)} rows, above target {target}."
            )
        wildguard = (
            read_json(prepared["path"] / "data" / "train.json")
            + read_json(prepared["path"] / "data" / "valid.json")
            + read_json(prepared["path"] / "data" / "test.json")
        )
        source_seen = {
            " ".join(row["prompt"].casefold().split()) for row in wildguard
        }
        resume_prompts = [
            " ".join(row["prompt"].casefold().split()) for row in existing
        ]
        if len(set(resume_prompts)) != len(
            resume_prompts
        ) or source_seen.intersection(resume_prompts):
            raise RuntimeError(
                "Resume JSONL contains duplicate prompts or a WildGuard prompt."
            )
        seen = source_seen.union(resume_prompts)
        if len(existing) == target:
            return _complete_artifact(
                artifact,
                {
                    "target": target,
                    "accepted": target,
                    "requests_this_session": 0,
                    "request_budget": 400,
                    "base_url": ai_base_url,
                    "model": ai_model,
                },
            )
        if len(tokens) < 6:
            raise RuntimeError(
                "The token scan returned too few usable tokens."
            )
        api_key = os.environ.get("AI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set AI_API_KEY in the environment; the notebook never stores it."
            )
        if transport is None:

            def configured_transport(path, key, payload=None):
                return _ai_transport(ai_base_url, path, key, payload)

            transport = configured_transport
        budget = {"used": 0, "limit": 400}
        validate_ai_model(ai_model, api_key, budget, transport=transport)

        accepted_count = len(existing)
        round_number = 0
        while accepted_count < target:
            batch_n = min(10, target - accepted_count)
            groups = []
            for offset in range(batch_n):
                group_size = 1 + ((accepted_count + offset) % 3)
                start = ((accepted_count + offset) * 3 + round_number) % len(
                    tokens
                )
                groups.append(
                    [
                        tokens[(start + index) % len(tokens)]
                        for index in range(group_size)
                    ]
                )
            generation_messages = [
                {
                    "role": "system",
                    "content": "Create ordinary, clearly harmless user prompts for classifier training. Return only a JSON array of objects with a prompt field. Do not generate harmful instructions.",
                },
                {
                    "role": "user",
                    "content": "Write one benign prompt for each required-token group. Include every required token naturally and verbatim. Groups: "
                    + json.dumps(groups, ensure_ascii=False),
                },
            ]
            generated = parse_prompt_response(
                ai_chat(
                    ai_model,
                    generation_messages,
                    api_key,
                    budget,
                    transport=transport,
                )
            )
            generated = generated[:batch_n]
            refinement_messages = [
                {
                    "role": "system",
                    "content": "Review prompts for benign intent. Rewrite ambiguous items into clearly harmless requests while preserving every required token verbatim. Return only a JSON array of objects with a prompt field.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        [
                            {"prompt": prompt, "required_tokens": group}
                            for prompt, group in zip(generated, groups)
                        ],
                        ensure_ascii=False,
                    ),
                },
            ]
            refined = parse_prompt_response(
                ai_chat(
                    ai_model,
                    refinement_messages,
                    api_key,
                    budget,
                    transport=transport,
                )
            )
            for prompt, required in accept_candidates(refined, groups, seen):
                row = {
                    "prompt": prompt,
                    "label": 0,
                    "required_tokens": required,
                    "source": "ai_generated_benign",
                    "model": ai_model,
                }
                append_jsonl(output_path, row)
                accepted_count += 1
                if accepted_count == target:
                    break
            round_number += 1
            if round_number > 200:
                raise RuntimeError(
                    "Too many rejected generations; partial progress is saved."
                )
        result = {
            "target": target,
            "accepted": accepted_count,
            "requests_this_session": budget["used"],
            "request_budget": budget["limit"],
            "base_url": ai_base_url,
            "model": ai_model,
        }
        return _complete_artifact(artifact, result)

    def build_stage2_data(prepare_dir, generation_dir, output_dir, smoke=False):
        generated = read_jsonl(
            generation_dir / "generated" / "benign_prompts.jsonl"
        )
        expected = 4 if smoke else 1000
        if len(generated) != expected:
            raise RuntimeError(
                f"Need exactly {expected} generated benign prompts; found {len(generated)}."
            )
        train = read_json(prepare_dir / "data" / "train.json")
        added = [{"prompt": row["prompt"], "label": 0} for row in generated]
        write_json(output_dir / "data" / "stage2_train.json", train + added)
        write_json(
            output_dir / "stage2_data.json",
            {
                "base_train_rows": len(train),
                "generated_benign_rows": len(added),
                "total_rows": len(train) + len(added),
                "validation_unchanged": True,
            },
        )
        if smoke:
            smoke_train = read_json(prepare_dir / "data" / "smoke_train.json")
            write_json(
                output_dir / "data" / "smoke_stage2_train.json",
                smoke_train + added,
            )
        return len(train) + len(added)

    def evaluate_stage(pipeline):
        cached = _resolved_artifact(pipeline, "evaluate")
        if cached is not None:
            return _cached_result(cached)
        import torch

        prepared = _require_artifact(pipeline, "prepare")
        stage2 = _require_artifact(pipeline, "stage2")
        artifact = pipeline["steps"]["evaluate"]
        model, device, config = load_piguard_model(stage2)
        test_name = "smoke_test.json" if config["smoke"] else "test.json"
        rows = read_json(prepared["path"] / "data" / test_name)
        predictions = []
        batch_size = config["batch_size"]
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                chunk = rows[start : start + batch_size]
                encoded = model.tokenizer(
                    [row["prompt"] for row in chunk],
                    padding=True,
                    truncation=True,
                    max_length=config["max_length"],
                    return_tensors="pt",
                )
                logits, _ = model(
                    encoded["input_ids"].to(device),
                    encoded["attention_mask"].to(device),
                )
                probabilities = (
                    torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
                )
                for row, probability in zip(chunk, probabilities):
                    predictions.append(
                        {
                            "prompt": row["prompt"],
                            "label": row["label"],
                            "prediction": int(probability >= 0.5),
                            "harmful_probability": probability,
                        }
                    )
        labels = np.array([row["label"] for row in predictions])
        predicted = np.array([row["prediction"] for row in predictions])
        tn = int(((labels == 0) & (predicted == 0)).sum())
        fp = int(((labels == 0) & (predicted == 1)).sum())
        fn = int(((labels == 1) & (predicted == 0)).sum())
        tp = int(((labels == 1) & (predicted == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics = {
            "stage": "stage2",
            "rows": len(rows),
            "accuracy": float((labels == predicted).mean()),
            "benign_accuracy": tn / (tn + fp) if tn + fp else 0.0,
            "harmful_accuracy": tp / (tp + fn) if tp + fn else 0.0,
            "precision_harmful": precision,
            "recall_harmful": recall,
            "f1_harmful": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "pooling": model.pooling,
            "training_mode": config["resolved_training_mode"],
        }
        output_dir = artifact["path"]
        atomic_text(
            output_dir / "test_predictions.jsonl",
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in predictions
            ),
        )
        write_json(output_dir / "metrics.json", metrics)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _complete_artifact(artifact, metrics)

    def classify_prompt(pipeline, prompt):
        import torch

        if not prompt.strip():
            raise ValueError("Enter a prompt.")
        artifact = _resolved_artifact(pipeline, "stage2")
        if artifact is None:
            artifact = _resolved_artifact(pipeline, "stage1")
        if artifact is None:
            raise RuntimeError("Complete stage-one or stage-two training first.")
        model, _, _ = load_piguard_model(artifact)
        with torch.no_grad():
            probability = torch.softmax(
                model.classify(prompt).float(), dim=-1
            )[0, 1].item()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "stage": artifact["step"],
            "label": "harmful" if probability >= 0.5 else "benign",
            "harmful_probability": probability,
        }

    def artifact_status(pipeline):
        status = {}
        for step, artifact in pipeline["steps"].items():
            resolved = _resolved_artifact(pipeline, step)
            if resolved is not None:
                state = "cached"
                path = resolved["path"]
                legacy = bool(resolved.get("legacy"))
            else:
                path = artifact["path"]
                state = (
                    "partial"
                    if path.exists() and any(path.iterdir())
                    else "missing"
                )
                legacy = False
            status[step] = {
                "state": state,
                "key": artifact["key"],
                "path": path,
                "legacy": legacy,
            }
        return status

    def self_check():
        assert (
            _normalize_ai_base_url("https://openrouter.ai/api/v1/")
            == "https://openrouter.ai/api/v1"
        )
        assert (
            _normalize_ai_base_url("http://localhost:8000/v1/")
            == "http://localhost:8000/v1"
        )
        try:
            _normalize_ai_base_url("http://example.com/v1")
        except ValueError:
            pass
        else:
            raise AssertionError("insecure remote AI URL was accepted")
        assert _apportion(5148) == [4118, 515, 515]
        assert _apportion(3852) == [3082, 385, 385]
        parsed = parse_prompt_response(
            '```json\n[{"prompt":"A safe test"}]\n```'
        )
        assert parsed == ["A safe test"]
        seen = {"duplicate"}
        accepted = accept_candidates(
            ["Duplicate", "Please explain reset safely"],
            [["Duplicate"], ["reset"]],
            seen,
        )
        assert accepted == [("Please explain reset safely", ["reset"])]

        attempts = {"count": 0}

        def fake_transport(path, api_key, payload=None):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise AIAPIError("retry", status=429, retry_after=0)
            return {
                "choices": [{"message": {"content": '[{"prompt":"safe"}]'}}]
            }

        budget = {"used": 0, "limit": 2}
        assert parse_prompt_response(
            ai_chat(
                "fake",
                [],
                "fake",
                budget,
                transport=fake_transport,
                sleep_fn=lambda _: None,
            )
        ) == ["safe"]
        assert budget["used"] == 2
        try:
            _spend_budget(budget)
        except RuntimeError:
            pass
        else:
            raise AssertionError("request budget did not stop")

        with tempfile.TemporaryDirectory() as temporary:
            resume = Path(temporary) / "resume.jsonl"
            append_jsonl(resume, {"prompt": "one", "label": 0})
            append_jsonl(resume, {"prompt": "two", "label": 0})
            assert [row["prompt"] for row in read_jsonl(resume)] == [
                "one",
                "two",
            ]

        base_config = {
            "model_id": "example/model",
            "revision": "main",
            "training_mode": "full",
            "trust_remote_code": False,
            "smoke": True,
            "batch_size": 2,
            "max_length": 512,
            "ai_base_url": "https://example.test/v1",
            "ai_model": "example/generator",
        }

        def keys(config):
            return {
                step: artifact["key"]
                for step, artifact in pipeline_artifacts(config)[
                    "steps"
                ].items()
            }

        original = keys(base_config)
        assert keys(dict(base_config)) == original
        ai_changed = keys(
            base_config
            | {
                "ai_base_url": "https://other.example/v1/",
                "ai_model": "example/other-generator",
            }
        )
        assert all(
            original[step] == ai_changed[step]
            for step in ("prepare", "stage1", "scan")
        )
        assert all(
            original[step] != ai_changed[step]
            for step in ("generate", "stage2", "evaluate")
        )
        for setting, value in (("batch_size", 3), ("max_length", 256)):
            changed = keys(base_config | {setting: value})
            assert changed["prepare"] == original["prepare"]
            assert all(
                changed[step] != original[step]
                for step in (
                    "stage1",
                    "scan",
                    "generate",
                    "stage2",
                    "evaluate",
                )
            )
        smoke_changed = keys(base_config | {"smoke": False})
        assert all(
            original[step] != smoke_changed[step] for step in original
        )

        with tempfile.TemporaryDirectory() as temporary:
            pipeline = pipeline_artifacts(base_config, root=temporary)
            prepared = pipeline["steps"]["prepare"]
            for relative in _required_outputs(prepared):
                path = prepared["path"] / relative
                if path.suffix == ".json":
                    write_json(path, {})
                else:
                    atomic_text(path, "test\n")
            _complete_artifact(prepared, {"splits": {}})
            assert prepare_run(pipeline)["status"] == "cached"

            generated = pipeline["steps"]["generate"]
            append_jsonl(
                generated["path"] / "generated" / "benign_prompts.jsonl",
                {"prompt": "partial", "label": 0},
            )
            assert artifact_status(pipeline)["generate"]["state"] == "partial"

            scanned = pipeline["steps"]["scan"]
            write_json(scanned["path"] / "biased_tokens.json", {})
            write_json(
                scanned["path"] / "manifest.json",
                {
                    "schema_version": CACHE_VERSION,
                    "step": "scan",
                    "key": "wrong",
                    "inputs": scanned["inputs"],
                },
            )
            assert _cache_manifest(scanned) is None
            _complete_artifact(scanned, {"saved": 0})
            assert _cache_manifest(scanned) is not None
            assert scan_biased_tokens(pipeline)["status"] == "cached"

            for number in range(1, 4):
                append_jsonl(
                    generated["path"]
                    / "generated"
                    / "benign_prompts.jsonl",
                    {"prompt": f"cached {number}", "label": 0},
                )
            _complete_artifact(generated, {"target": 4, "accepted": 4})
            assert generate_benign_prompts(pipeline)["status"] == "cached"

            stage2 = pipeline["steps"]["stage2"]
            atomic_text(
                stage2["path"] / "checkpoints" / "best_model.pth", "test\n"
            )
            _complete_artifact(stage2, {"stage": "stage2"}, config={})
            assert run_official_stage(pipeline, "stage2")["status"] == "cached"

            evaluated = pipeline["steps"]["evaluate"]
            write_json(evaluated["path"] / "metrics.json", {})
            atomic_text(evaluated["path"] / "test_predictions.jsonl", "")
            _complete_artifact(evaluated, {"rows": 0})
            assert evaluate_stage(pipeline)["status"] == "cached"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "legacy-run"
            saved = {
                key: base_config[key]
                for key in (
                    "model_id",
                    "revision",
                    "training_mode",
                    "trust_remote_code",
                    "smoke",
                    "batch_size",
                    "max_length",
                )
            } | {
                "resolved_training_mode": "full",
                "upstream_commit": UPSTREAM_COMMIT,
                "dataset_sha256": DATASET_SHA256,
            }
            write_json(legacy / "run_config.json", saved)
            write_json(
                legacy / "manifests" / "data_manifest.json", {"seed": SEED}
            )
            for name in ("train", "valid", "test"):
                write_json(legacy / "data" / f"{name}.json", [])
                write_json(legacy / "data" / f"smoke_{name}.json", [])
            atomic_text(legacy / "patched-source.diff", "test\n")
            atomic_text(
                legacy / "stage1" / "checkpoints" / "best_model.pth",
                "test\n",
            )
            write_json(legacy / "stage1" / "stage_manifest.json", {})
            legacy_pipeline = pipeline_artifacts(base_config, root=root)
            assert _legacy_artifact(legacy_pipeline, "prepare") is not None
            assert _legacy_artifact(legacy_pipeline, "stage1") is not None
            assert not (root / "cache").exists()
        return {
            "ai_url_validation": "ok",
            "split_math": "ok",
            "ai_response_parsing": "ok",
            "deduplication": "ok",
            "429_retry_and_budget": "ok",
            "jsonl_resume": "ok",
            "step_cache_keys": "ok",
            "cache_completion_and_partial_resume": "ok",
            "legacy_cache_discovery": "ok",
        }

    return (
        ARTIFACT_ROOT,
        PRESETS,
        STEP_LABELS,
        UPSTREAM_COMMIT,
        UPSTREAM_REPO,
        artifact_status,
        classify_prompt,
        evaluate_stage,
        generate_benign_prompts,
        json,
        mo,
        pipeline_artifacts,
        prepare_run,
        run_official_stage,
        scan_biased_tokens,
        self_check,
    )


@app.cell
def title(UPSTREAM_COMMIT, UPSTREAM_REPO, mo):
    mo.md(f"""
    # Custom-backbone PIGuard

    This notebook is a thin wrapper around the official [PIGuard repository]({UPSTREAM_REPO.removesuffix(".git")}) at
    `{UPSTREAM_COMMIT}`. It patches only model selection, causal-model pooling, PEFT persistence, and the final
    dataset-specific evaluation hand-off. The official loss, optimizer loop, scheduler, validation, checkpoint
    selection, and three-epoch production default remain in `train.py`.

    **Task semantics:** WildGuard prompt harmfulness (`unharmful → 0`, `harmful → 1`), not prompt-injection labeling.
    """)
    return


@app.cell(hide_code=True)
def controls(mo):
    backbone_preset = mo.ui.dropdown(
        options=["DeBERTa V3 Base", "Qwen 2.5 3B", "Custom"],
        value="DeBERTa V3 Base",
        label="Backbone preset",
    )
    custom_model_id = mo.ui.text(
        label="Custom Hugging Face model ID", placeholder="org/model"
    )
    model_revision = mo.ui.text(
        label="Model revision (optional)", placeholder="main or commit SHA"
    )
    training_mode = mo.ui.dropdown(
        options=["auto", "full", "qlora"], value="auto", label="Training mode"
    )
    trust_remote_code = mo.ui.checkbox(
        label="Allow trust_remote_code for this model", value=False
    )
    smoke_mode = mo.ui.checkbox(
        label="Smoke workflow (2 steps, 4 generated prompts, 512-token scan)",
        value=True,
    )
    batch_size = mo.ui.number(
        start=1, stop=64, step=1, value=2, label="Micro-batch size"
    )
    max_length = mo.ui.number(
        start=64, stop=4096, step=64, value=512, label="Maximum token length"
    )
    ai_base_url = mo.ui.text(
        label="AI API base URL",
        value="https://openrouter.ai/api/v1",
        placeholder="https://provider.example/v1",
    )
    ai_model_id = mo.ui.text(
        label="AI model ID", value="z-ai/glm-5.2", placeholder="provider/model"
    )

    prepare_button = mo.ui.run_button(label="1 · Prepare pinned source + data")
    stage1_button = mo.ui.run_button(label="2 · Train stage one")
    scan_button = mo.ui.run_button(label="3 · Scan biased tokens")
    generate_button = mo.ui.run_button(
        label="4 · Generate + refine benign prompts"
    )
    stage2_button = mo.ui.run_button(label="5 · Retrain from untouched base")
    evaluate_button = mo.ui.run_button(label="6 · Evaluate final checkpoint")
    return (
        ai_base_url,
        ai_model_id,
        backbone_preset,
        batch_size,
        custom_model_id,
        evaluate_button,
        generate_button,
        max_length,
        model_revision,
        prepare_button,
        scan_button,
        smoke_mode,
        stage1_button,
        stage2_button,
        training_mode,
        trust_remote_code,
    )


@app.cell(hide_code=True)
def configuration(
    PRESETS,
    ai_base_url,
    ai_model_id,
    backbone_preset,
    batch_size,
    custom_model_id,
    max_length,
    model_revision,
    pipeline_artifacts,
    smoke_mode,
    training_mode,
    trust_remote_code,
):
    selected_model_id = (
        custom_model_id.value.strip()
        if backbone_preset.value == "Custom"
        else PRESETS[backbone_preset.value]
    )
    run_config = {
        "model_id": selected_model_id,
        "revision": model_revision.value.strip(),
        "training_mode": training_mode.value,
        "trust_remote_code": trust_remote_code.value,
        "smoke": smoke_mode.value,
        "batch_size": int(batch_size.value),
        "max_length": int(max_length.value),
        "ai_base_url": ai_base_url.value.strip().rstrip("/"),
        "ai_model": ai_model_id.value.strip(),
    }
    pipeline = pipeline_artifacts(run_config)
    return (pipeline,)


@app.cell
def control_panel(
    ai_base_url,
    ai_model_id,
    backbone_preset,
    batch_size,
    custom_model_id,
    evaluate_button,
    generate_button,
    max_length,
    mo,
    model_revision,
    prepare_button,
    pipeline,
    scan_button,
    smoke_mode,
    stage1_button,
    stage2_button,
    training_mode,
    trust_remote_code,
):
    mo.vstack(
        [
            mo.md("## Configuration"),
            mo.hstack([backbone_preset, custom_model_id]),
            mo.hstack([model_revision, training_mode, trust_remote_code]),
            mo.hstack([smoke_mode, batch_size, max_length]),
            mo.md(
                "Auto uses full fine-tuning at ≤500M parameters and QLoRA above that. "
                "QLoRA requires Linux + CUDA + bitsandbytes. `HF_TOKEN` is used for gated Hugging Face models."
            ),
            mo.md("## Benign-prompt API"),
            mo.hstack([ai_base_url, ai_model_id]),
            mo.md(
                "The endpoint must expose OpenAI-compatible `/models` and `/chat/completions` routes. "
                "The key is read only from `AI_API_KEY`; it is never stored. Generation is resumable and capped at 400 requests per session."
            ),
            mo.md("## Pipeline"),
            mo.hstack([prepare_button, stage1_button, scan_button]),
            mo.hstack([generate_button, stage2_button, evaluate_button]),
            mo.md(f"**Cache directory:** `{pipeline['root'] / 'cache'}`"),
        ]
    )
    return


@app.cell
def actions(
    evaluate_button,
    evaluate_stage,
    generate_benign_prompts,
    generate_button,
    json,
    mo,
    prepare_button,
    prepare_run,
    pipeline,
    run_official_stage,
    scan_biased_tokens,
    scan_button,
    stage1_button,
    stage2_button,
):
    _result = {"status": "ready"}
    try:
        if prepare_button.value:
            _result = prepare_run(pipeline)
        elif stage1_button.value:
            _result = run_official_stage(pipeline, "stage1")
        elif scan_button.value:
            _result = scan_biased_tokens(pipeline)
        elif generate_button.value:
            _result = generate_benign_prompts(pipeline)
        elif stage2_button.value:
            _result = run_official_stage(pipeline, "stage2")
        elif evaluate_button.value:
            _result = evaluate_stage(pipeline)
    except Exception as _exc:
        _result = {
            "status": "error",
            "type": type(_exc).__name__,
            "message": str(_exc),
        }
    action_result = _result
    mo.md(
        "## Last action\n```json\n"
        + json.dumps(action_result, indent=2)
        + "\n```"
    )
    return (action_result,)


@app.cell
def status(STEP_LABELS, action_result, artifact_status, mo, pipeline):
    _refresh = action_result
    _status = artifact_status(pipeline)
    _icons = {"cached": "✅", "partial": "🟨", "missing": "⬜"}
    _rows = "\n".join(
        f"- {_icons[item['state']]} {STEP_LABELS[step]} — "
        f"**{item['state']}** · `{item['key']}`"
        + (" · legacy" if item["legacy"] else "")
        for step, item in _status.items()
    )
    _diff_path = _status["prepare"]["path"] / "patched-source.diff"
    _diff_preview = (
        _diff_path.read_text(encoding="utf-8")
        if _diff_path.exists()
        else "Prepare the run to materialize the verified patch."
    )
    mo.md(
        f"## Artifacts\n{_rows}\n\n<details><summary>Runtime patch</summary>\n\n```diff\n{_diff_preview}\n```\n</details>"
    )
    return


@app.cell(hide_code=True)
def inference_controls(mo):
    inference_prompt = mo.ui.text_area(
        label="Prompt",
        placeholder="Classify a prompt with the latest available checkpoint",
    )
    inference_button = mo.ui.run_button(label="Classify")
    return inference_button, inference_prompt


@app.cell
def inference(
    classify_prompt,
    inference_button,
    inference_prompt,
    json,
    mo,
    pipeline,
):
    _inference_result = {"status": "waiting"}
    if inference_button.value:
        try:
            _inference_result = classify_prompt(
                pipeline, inference_prompt.value
            )
        except Exception as _infer_exc:
            _inference_result = {
                "status": "error",
                "type": type(_infer_exc).__name__,
                "message": str(_infer_exc),
            }
    mo.vstack(
        [
            mo.md("## Inference"),
            inference_prompt,
            inference_button,
            mo.md(
                "```json\n" + json.dumps(_inference_result, indent=2) + "\n```"
            ),
        ]
    )
    return


@app.cell
def checks(json, mo, self_check):
    self_check_report = self_check()
    mo.md(
        "## Built-in checks\n```json\n"
        + json.dumps(self_check_report, indent=2)
        + "\n```"
    )
    return


if __name__ == "__main__":
    app.run()
