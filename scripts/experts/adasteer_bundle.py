"""Build and run WildGuard-derived bundles with official AdaSteer Qwen code."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATASET_PATH = Path("wildguardtrain_9000_seed42.parquet")
DEFAULT_OUTPUT_ROOT = Path("artifacts/adasteer")
EXPECTED_ROWS = 9_000
QWEN_LAYERS = 28
QWEN_HIDDEN_SIZE = 3_584
RD_PROBE_LAYER = 5
HD_PROBE_LAYER = 13
MAX_LENGTH = 512
CALIBRATION_SIZE = 15
CANDIDATE_SCAN_LIMIT = 100
SEED = 42
STRENGTH_GRID = tuple(
    value
    for magnitude in (0.05, 0.10, 0.20, 0.30, 0.50, 0.75)
    for value in (magnitude, -magnitude)
)
GROUPS = (
    "harmful_refusal",
    "harmful_compliance",
    "benign_compliance",
    "benign_refusal",
)
ARTIFACT_SHAPES = {
    "RD/mean_diff.pkl": (QWEN_LAYERS, QWEN_HIDDEN_SIZE),
    "RD/class_a.pkl": (QWEN_LAYERS, 1, QWEN_HIDDEN_SIZE),
    "RD/class_b.pkl": (QWEN_LAYERS, 1, QWEN_HIDDEN_SIZE),
    "HD/mean_diff.pkl": (QWEN_LAYERS, QWEN_HIDDEN_SIZE),
    "HD/proj.pkl": (QWEN_LAYERS, QWEN_HIDDEN_SIZE),
    "HD/class_a.pkl": (QWEN_LAYERS, 1, QWEN_HIDDEN_SIZE),
    "HD/class_b.pkl": (QWEN_LAYERS, 1, QWEN_HIDDEN_SIZE),
}
REQUIRED_CHECKOUT_FILES = (
    "adasteer/extract/Probing/probe.py",
    "adasteer/models/For_Steering_QwenModel_adasteer.py",
    "adasteer/lib/_pickle.py",
)


def model_slug(model_id: str) -> str:
    slug = "".join(
        character if character.isalnum() or character in ".-_" else "-"
        for character in model_id.lower()
    ).strip("-.")
    if not slug:
        raise ValueError("model ID does not produce a usable artifact directory name")
    return slug


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_key(row: dict[str, Any]) -> str:
    prompt_label = row.get("prompt_harm_label")
    action_label = row.get("response_refusal_label")
    if prompt_label == "harmful":
        return "harmful_refusal" if action_label == "refusal" else "harmful_compliance"
    if prompt_label == "unharmful":
        return "benign_refusal" if action_label == "refusal" else "benign_compliance"
    raise ValueError(f"unsupported WildGuard labels: {prompt_label!r}, {action_label!r}")


def validate_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = list(rows)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS:,} rows, found {len(rows):,}")
    required = {
        "source_index",
        "prompt",
        "response",
        "prompt_harm_label",
        "response_harm_label",
        "response_refusal_label",
    }
    source_indices = set()
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"row {index} is missing columns: {sorted(missing)}")
        if not isinstance(row["source_index"], int):
            raise ValueError(f"row {index} has an invalid source_index")
        if row["source_index"] in source_indices:
            raise ValueError(f"duplicate source_index: {row['source_index']}")
        source_indices.add(row["source_index"])
        for field in ("prompt", "response"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"row {index} has an empty {field}")
        if row["prompt_harm_label"] not in {"harmful", "unharmful"}:
            raise ValueError(f"row {index} has an invalid prompt_harm_label")
        if row["response_harm_label"] not in {"harmful", "unharmful"}:
            raise ValueError(f"row {index} has an invalid response_harm_label")
        if row["response_refusal_label"] not in {"refusal", "compliance"}:
            raise ValueError(f"row {index} has an invalid response_refusal_label")
        counts[group_key(row)] += 1
    missing_groups = set(GROUPS) - set(counts)
    if missing_groups:
        raise ValueError(f"WildGuard data has empty groups: {sorted(missing_groups)}")
    return rows, {name: counts[name] for name in GROUPS}


def load_wildguard_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    import pyarrow.parquet as pq

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pq.read_table(path)
    required = {
        "source_index",
        "prompt",
        "response",
        "prompt_harm_label",
        "response_harm_label",
        "response_refusal_label",
    }
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"missing Parquet columns: {sorted(missing)}")
    return validate_rows(table.select(sorted(required)).to_pylist())


def validate_qwen_config(config: Any) -> None:
    values = config if isinstance(config, dict) else vars(config)
    if values.get("model_type") != "qwen2":
        raise ValueError("AdaSteer bundle building supports only Qwen2 models")
    if values.get("num_hidden_layers") != QWEN_LAYERS:
        raise ValueError(f"expected {QWEN_LAYERS} decoder layers")
    if values.get("hidden_size") != QWEN_HIDDEN_SIZE:
        raise ValueError(f"expected hidden size {QWEN_HIDDEN_SIZE}")


def validate_checkout(root: Path) -> tuple[Path, str]:
    root = Path(root).expanduser().resolve()
    missing = [str(root / relative) for relative in REQUIRED_CHECKOUT_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"invalid official AdaSteer checkout; missing {missing}")
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("official AdaSteer root must be a Git checkout") from exc
    return root, commit


def _runtime_preflight(
    official_root: Path,
    dataset_path: Path,
    model_id: str,
    revision: str | None,
    token: str | None,
) -> tuple[Path, str, list[dict[str, Any]], dict[str, int], Any, Any]:
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    if transformers.__version__ != "4.46.3":
        raise RuntimeError(
            f"official AdaSteer requires transformers 4.46.3, found {transformers.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    root, commit = validate_checkout(official_root)
    rows, group_counts = load_wildguard_rows(dataset_path)
    config = AutoConfig.from_pretrained(model_id, revision=revision, token=token)
    validate_qwen_config(config)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
    if not tokenizer.chat_template:
        raise ValueError("the selected tokenizer has no chat template")
    return root, commit, rows, group_counts, config, tokenizer


def preflight_bundle_inputs(
    official_root: Path,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    root, commit, rows, group_counts, config, _tokenizer = _runtime_preflight(
        official_root, dataset_path, model_id, revision, token
    )
    return {
        "official_root": str(root),
        "official_commit": commit,
        "dataset": str(Path(dataset_path).expanduser().resolve()),
        "dataset_rows": len(rows),
        "group_counts": group_counts,
        "model_id": model_id,
        "revision": revision,
        "layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
    }


def finalize_directions(group_activations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    means: dict[str, np.ndarray] = {}
    for name in GROUPS:
        activations = np.asarray(group_activations[name])
        if activations.ndim != 3 or activations.shape[0] != QWEN_LAYERS or activations.shape[2] != QWEN_HIDDEN_SIZE:
            raise ValueError(f"{name} activations have unexpected shape {activations.shape}")
        means[name] = activations.mean(axis=1, dtype=np.float64).astype(np.float32)

    rd = means["harmful_refusal"] - means["harmful_compliance"]
    hd = means["benign_compliance"] - means["harmful_compliance"]
    denominator = np.square(rd).sum(axis=1, keepdims=True)
    if np.any(denominator <= 1e-12):
        raise ValueError("at least one RD layer has zero norm")
    projection = rd * ((rd * hd).sum(axis=1, keepdims=True) / denominator)
    artifacts = {
        "RD/mean_diff.pkl": rd,
        "RD/class_a.pkl": means["harmful_refusal"][:, None, :],
        "RD/class_b.pkl": means["harmful_compliance"][:, None, :],
        "HD/mean_diff.pkl": hd,
        "HD/proj.pkl": projection,
        "HD/class_a.pkl": means["benign_compliance"][:, None, :],
        "HD/class_b.pkl": means["harmful_compliance"][:, None, :],
    }
    for name, value in artifacts.items():
        if value.shape != ARTIFACT_SHAPES[name] or not np.isfinite(value).all():
            raise ValueError(f"invalid {name}: shape={value.shape}")
    return {name: value.astype(np.float16) for name, value in artifacts.items()}


def prompt_positions(
    rd_hidden: np.ndarray,
    hd_hidden: np.ndarray,
    artifacts: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rd = np.asarray(artifacts["RD/mean_diff.pkl"], dtype=np.float32)
    hd = np.asarray(artifacts["HD/mean_diff.pkl"], dtype=np.float32)
    rd_center = np.asarray(artifacts["RD/class_b.pkl"], dtype=np.float32)[:, 0, :]
    hd_center = np.asarray(artifacts["HD/class_b.pkl"], dtype=np.float32)[:, 0, :]
    rd_positions = (np.asarray(rd_hidden, dtype=np.float32) - rd_center[RD_PROBE_LAYER]) @ rd[RD_PROBE_LAYER]
    hd_positions = (np.asarray(hd_hidden, dtype=np.float32) - hd_center[HD_PROBE_LAYER]) @ hd[HD_PROBE_LAYER]
    return rd_positions, hd_positions


def fit_law(records: list[dict[str, Any]], position_key: str) -> dict[str, Any]:
    positions = np.asarray([record[position_key] for record in records], dtype=np.float64)
    strengths = np.asarray([record["strength"] for record in records], dtype=np.float64)
    if len(records) != CALIBRATION_SIZE or not np.isfinite(positions).all() or not np.isfinite(strengths).all():
        raise ValueError(f"expected {CALIBRATION_SIZE} finite calibration records")
    design = np.column_stack([positions, np.ones_like(positions)])
    slope, intercept = np.linalg.lstsq(design, strengths, rcond=None)[0]
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "minimum": float(strengths.min()),
        "maximum": float(strengths.max()),
        "samples": len(records),
    }


def predict_law(law: dict[str, Any], position: float) -> float:
    return float(np.clip(law["slope"] * position + law["intercept"], law["minimum"], law["maximum"]))


def set_model_coefficients(model: Any, torch_module: Any, rd_coefficient: float, hd_coefficient: float) -> None:
    device = model.model.steer_vector.device
    model.model.alpha_list = torch_module.tensor([rd_coefficient], dtype=torch_module.float16, device=device)
    model.model.beta_list = torch_module.tensor([hd_coefficient], dtype=torch_module.float16, device=device)


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_pickle(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        return np.asarray(pickle.load(handle))


class QwenSteeringRuntime:
    """Thin runtime around the official Qwen activation-injection model."""

    def __init__(
        self,
        official_root: Path,
        bundle: Path,
        model_id: str = DEFAULT_MODEL_ID,
        token: str | None = None,
        revision: str | None = None,
        max_new_tokens: int = 128,
        laws: dict[str, dict[str, Any]] | None = None,
    ):
        import torch
        from transformers import AutoConfig, AutoTokenizer

        if not 1 <= max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be between 1 and 512")
        self.official_root, self.official_commit = validate_checkout(official_root)
        self.bundle = Path(bundle).expanduser().resolve()
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.torch = torch
        config = AutoConfig.from_pretrained(model_id, revision=revision, token=token)
        validate_qwen_config(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.artifacts = {name: _load_pickle(self.bundle / name) for name in ARTIFACT_SHAPES}
        if laws is None:
            metadata = verify_bundle(self.bundle)
            if metadata["model_id"] != model_id or metadata.get("revision") != revision:
                raise ValueError("bundle model or revision does not match the requested model")
            laws = metadata["coefficient_laws"]
        self.laws = laws

        if str(self.official_root) not in sys.path:
            sys.path.insert(0, str(self.official_root))
        from adasteer.models.For_Steering_QwenModel_adasteer import Qwen_for_Steering_dynamic

        self._workspace = tempfile.TemporaryDirectory(prefix="mode-adasteer-")
        vector_parent = Path(self._workspace.name) / "vectors"
        vector_parent.mkdir()
        (vector_parent / "qwen25-7b-instruct").symlink_to(self.bundle, target_is_directory=True)
        with _working_directory(Path(self._workspace.name)):
            self.model = Qwen_for_Steering_dynamic.from_pretrained(
                model_id,
                token=token,
                revision=revision,
                torch_dtype=torch.float16,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            self.model.get_steer(str(self.bundle / "RD/mean_diff.pkl"), alpha=0)
        self.model.eval()

    def _encoded(self, prompt: str) -> Any:
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer(
            formatted,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to("cuda")

    def positions(self, prompt: str) -> tuple[float, float]:
        encoded = self._encoded(prompt)
        self.model.reset_alpha()
        with self.torch.inference_mode():
            outputs = self.model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        rd_hidden = outputs.hidden_states[RD_PROBE_LAYER + 1][0, -1].float().cpu().numpy()
        hd_hidden = outputs.hidden_states[HD_PROBE_LAYER + 1][0, -1].float().cpu().numpy()
        rd_positions, hd_positions = prompt_positions(rd_hidden[None, :], hd_hidden[None, :], self.artifacts)
        return float(rd_positions[0]), float(hd_positions[0])

    def generate(self, prompt: str, rd_coefficient: float, hd_coefficient: float) -> str:
        encoded = self._encoded(prompt)
        set_model_coefficients(self.model, self.torch, rd_coefficient, hd_coefficient)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

    def adaptive_generate(self, prompt: str) -> tuple[str, dict[str, float]]:
        rd_position, hd_position = self.positions(prompt)
        rd_coefficient = predict_law(self.laws["rd"], rd_position)
        hd_coefficient = predict_law(self.laws["hd"], hd_position)
        response = self.generate(prompt, rd_coefficient, hd_coefficient)
        return response, {
            "rd_position": rd_position,
            "hd_position": hd_position,
            "rd_coefficient": rd_coefficient,
            "hd_coefficient": hd_coefficient,
        }

    def close(self) -> None:
        model, tokenizer = self.model, self.tokenizer
        self.model = self.tokenizer = None
        del model, tokenizer
        self._workspace.cleanup()
        gc.collect()
        self.torch.cuda.empty_cache()


def _format_prompts(tokenizer: Any, prompts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def _extract_activations(
    official_root: Path,
    rows: list[dict[str, Any]],
    model_id: str,
    revision: str | None,
    token: str | None,
    tokenizer: Any,
    progress: Callable[[str], None],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[list[dict[str, Any]], np.ndarray, np.ndarray]]]:
    import torch
    from transformers import AutoModelForCausalLM

    if str(official_root) not in sys.path:
        sys.path.insert(0, str(official_root))
    from adasteer.extract.Probing.probe import Probe

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        revision=revision,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    probe = object.__new__(Probe)
    probe.model = model
    grouped_rows = {name: [row for row in rows if group_key(row) == name] for name in GROUPS}
    group_activations: dict[str, np.ndarray] = {}
    calibration_hidden: dict[str, tuple[list[dict[str, Any]], np.ndarray, np.ndarray]] = {}
    try:
        for name in GROUPS:
            selected = grouped_rows[name]
            progress(f"Extracting official Qwen activations: {name} ({len(selected):,})")
            encoded = tokenizer(
                _format_prompts(tokenizer, [row["prompt"] for row in selected]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
                max_length=MAX_LENGTH,
            )
            tensor = Probe._get_hidden_sentence_embeddings(probe, encoded)
            activations = tensor.numpy()
            group_activations[name] = activations
            if name in {"harmful_compliance", "benign_refusal"}:
                calibration_hidden[name] = (
                    selected,
                    activations[RD_PROBE_LAYER].astype(np.float32),
                    activations[HD_PROBE_LAYER].astype(np.float32),
                )
            del tensor, encoded
    finally:
        del probe, model
        gc.collect()
        torch.cuda.empty_cache()
    return group_activations, calibration_hidden


def _candidate_records(
    calibration_hidden: dict[str, tuple[list[dict[str, Any]], np.ndarray, np.ndarray]],
    artifacts: dict[str, np.ndarray],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for name, (rows, rd_hidden, hd_hidden) in calibration_hidden.items():
        rd_positions, hd_positions = prompt_positions(rd_hidden, hd_hidden, artifacts)
        result[name] = [
            {
                "source_index": row["source_index"],
                "prompt": row["prompt"],
                "rd_position": float(rd_position),
                "hd_position": float(hd_position),
            }
            for row, rd_position, hd_position in zip(rows, rd_positions, hd_positions)
        ]
    return result


def _judgment_fields(judgment: dict[str, Any]) -> dict[str, bool]:
    return {name: bool(judgment[name]) for name in ("safe", "attack_followed", "refusal", "useful")}


def calibrate_laws(
    runtime: QwenSteeringRuntime,
    candidates: dict[str, list[dict[str, Any]]],
    judge: Callable[[str, str], dict[str, Any]],
    progress: Callable[[str], None] = print,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rng = random.Random(SEED)
    harmful = list(candidates["harmful_compliance"])
    benign = list(candidates["benign_refusal"])
    rng.shuffle(harmful)
    rng.shuffle(benign)

    rd_records: list[dict[str, Any]] = []
    for candidate in harmful[:CANDIDATE_SCAN_LIMIT]:
        baseline = runtime.generate(candidate["prompt"], 0.0, 0.0)
        baseline_judgment = judge(candidate["prompt"], baseline)
        if baseline_judgment["refusal"]:
            continue
        for strength in STRENGTH_GRID:
            response = runtime.generate(candidate["prompt"], strength, 0.0)
            judgment = judge(candidate["prompt"], response)
            if judgment["refusal"]:
                rd_records.append(
                    {
                        **{key: candidate[key] for key in ("source_index", "rd_position", "hd_position")},
                        "strength": strength,
                        "judgment": _judgment_fields(judgment),
                        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                    }
                )
                progress(f"RD calibration {len(rd_records)}/{CALIBRATION_SIZE}")
                break
        if len(rd_records) == CALIBRATION_SIZE:
            break
    if len(rd_records) != CALIBRATION_SIZE:
        raise RuntimeError(
            f"found {len(rd_records)} successful RD cases; expected {CALIBRATION_SIZE} within {CANDIDATE_SCAN_LIMIT} candidates"
        )
    rd_law = fit_law(rd_records, "rd_position")

    hd_records: list[dict[str, Any]] = []
    for candidate in benign[:CANDIDATE_SCAN_LIMIT]:
        baseline = runtime.generate(candidate["prompt"], 0.0, 0.0)
        baseline_judgment = judge(candidate["prompt"], baseline)
        if not baseline_judgment["refusal"]:
            continue
        rd_coefficient = predict_law(rd_law, candidate["rd_position"])
        for strength in STRENGTH_GRID:
            response = runtime.generate(candidate["prompt"], rd_coefficient, strength)
            judgment = judge(candidate["prompt"], response)
            if judgment["safe"] and not judgment["refusal"]:
                hd_records.append(
                    {
                        **{key: candidate[key] for key in ("source_index", "rd_position", "hd_position")},
                        "rd_coefficient": rd_coefficient,
                        "strength": strength,
                        "judgment": _judgment_fields(judgment),
                        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                    }
                )
                progress(f"HD calibration {len(hd_records)}/{CALIBRATION_SIZE}")
                break
        if len(hd_records) == CALIBRATION_SIZE:
            break
    if len(hd_records) != CALIBRATION_SIZE:
        raise RuntimeError(
            f"found {len(hd_records)} successful HD cases; expected {CALIBRATION_SIZE} within {CANDIDATE_SCAN_LIMIT} candidates"
        )
    return (
        {"rd": {**rd_law, "probe_layer": RD_PROBE_LAYER}, "hd": {**fit_law(hd_records, "hd_position"), "probe_layer": HD_PROBE_LAYER}},
        {"rd": rd_records, "hd": hd_records},
    )


def _write_artifacts(directory: Path, artifacts: dict[str, np.ndarray]) -> None:
    for relative, value in artifacts.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(np.asarray(value, dtype=np.float16), handle)


def _install_bundle(temporary: Path, target: Path, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists; enable replacement to overwrite it")
    backup = target.with_name(f".{target.name}.backup")
    if backup.exists():
        raise FileExistsError(f"stale bundle backup exists: {backup}")
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(temporary, target)
    except Exception:
        if backup.exists():
            os.replace(backup, target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def verify_bundle(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    metadata_path = path / "bundle.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2:
        raise ValueError("unsupported AdaSteer bundle schema")
    if not isinstance(metadata.get("model_id"), str) or not metadata["model_id"]:
        raise ValueError("bundle has no model ID")
    if metadata.get("revision") is not None and not isinstance(metadata["revision"], str):
        raise ValueError("bundle has an invalid model revision")
    dataset_hash = metadata.get("dataset_sha256", "")
    if len(dataset_hash) != 64 or any(character not in "0123456789abcdef" for character in dataset_hash):
        raise ValueError("bundle has an invalid dataset hash")
    group_counts = metadata.get("group_counts", {})
    if (
        metadata.get("dataset_rows") != EXPECTED_ROWS
        or set(group_counts) != set(GROUPS)
        or not all(isinstance(count, int) and count > 0 for count in group_counts.values())
        or sum(group_counts.values()) != EXPECTED_ROWS
    ):
        raise ValueError("bundle does not describe exactly 9,000 WildGuard rows")
    if metadata.get("vector_shape") != [QWEN_LAYERS, QWEN_HIDDEN_SIZE] or metadata.get("vector_dtype") != "float16":
        raise ValueError("bundle has an invalid vector contract")
    if metadata.get("rd_probe_layer") != RD_PROBE_LAYER or metadata.get("hd_probe_layer") != HD_PROBE_LAYER:
        raise ValueError("bundle has invalid probe layers")
    if not isinstance(metadata.get("official_commit"), str) or not metadata["official_commit"]:
        raise ValueError("bundle has no official checkout commit")
    dependencies = metadata.get("dependencies", {})
    if not all(isinstance(dependencies.get(name), str) and dependencies[name] for name in ("numpy", "pyarrow", "torch", "transformers")):
        raise ValueError("bundle has incomplete dependency versions")
    for relative, shape in ARTIFACT_SHAPES.items():
        artifact_path = path / relative
        value = _load_pickle(artifact_path)
        if value.shape != shape or value.dtype != np.float16 or not np.isfinite(value).all():
            raise ValueError(f"invalid bundle artifact {relative}")
        if metadata.get("artifact_sha256", {}).get(relative) != sha256_file(artifact_path):
            raise ValueError(f"bundle artifact hash differs: {relative}")
    for name in ("rd", "hd"):
        law = metadata.get("coefficient_laws", {}).get(name, {})
        values = [law.get(key) for key in ("slope", "intercept", "minimum", "maximum")]
        if law.get("samples") != CALIBRATION_SIZE or not all(isinstance(value, (int, float)) and np.isfinite(value) for value in values):
            raise ValueError(f"invalid {name} coefficient law")
        if law["minimum"] > law["maximum"]:
            raise ValueError(f"invalid {name} coefficient bounds")
        expected_layer = RD_PROBE_LAYER if name == "rd" else HD_PROBE_LAYER
        if law.get("probe_layer") != expected_layer:
            raise ValueError(f"invalid {name} coefficient probe layer")
        records = metadata.get("calibration_records", {}).get(name, [])
        if len(records) != CALIBRATION_SIZE or not all(isinstance(record.get("source_index"), int) for record in records):
            raise ValueError(f"invalid {name} calibration records")
    return metadata


def build_bundle(
    official_root: Path,
    dataset_path: Path,
    output_root: Path,
    judge: Callable[[str, str], dict[str, Any]],
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    token: str | None = None,
    overwrite: bool = False,
    progress: Callable[[str], None] = print,
) -> Path:
    root, commit, rows, group_counts, _config, tokenizer = _runtime_preflight(
        official_root, dataset_path, model_id, revision, token
    )
    dataset_path = Path(dataset_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / model_slug(model_id)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists; enable replacement to overwrite it")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=output_root))
    try:
        group_activations, calibration_hidden = _extract_activations(
            root, rows, model_id, revision, token, tokenizer, progress
        )
        artifacts = finalize_directions(group_activations)
        candidates = _candidate_records(calibration_hidden, artifacts)
        del group_activations, calibration_hidden
        gc.collect()
        _write_artifacts(temporary, artifacts)
        runtime = QwenSteeringRuntime(root, temporary, model_id, token, revision, laws={})
        try:
            laws, calibration_records = calibrate_laws(runtime, candidates, judge, progress)
        finally:
            runtime.close()
        artifact_hashes = {name: sha256_file(temporary / name) for name in ARTIFACT_SHAPES}
        metadata = {
            "schema_version": 2,
            "model_id": model_id,
            "revision": revision,
            "dataset_path": str(dataset_path),
            "dataset_rows": len(rows),
            "dataset_sha256": sha256_file(dataset_path),
            "group_counts": group_counts,
            "vector_shape": [QWEN_LAYERS, QWEN_HIDDEN_SIZE],
            "vector_dtype": "float16",
            "rd_probe_layer": RD_PROBE_LAYER,
            "hd_probe_layer": HD_PROBE_LAYER,
            "coefficient_laws": laws,
            "calibration_records": calibration_records,
            "strength_grid": list(STRENGTH_GRID),
            "candidate_scan_limit": CANDIDATE_SCAN_LIMIT,
            "official_root": str(root),
            "official_commit": commit,
            "official_source_sha256": {
                relative: sha256_file(root / relative) for relative in REQUIRED_CHECKOUT_FILES
            },
            "artifact_sha256": artifact_hashes,
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pyarrow", "torch", "transformers")
            },
        }
        (temporary / "bundle.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        verify_bundle(temporary)
        _install_bundle(temporary, target, overwrite)
        progress(f"Saved AdaSteer bundle: {target}")
        return target
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
