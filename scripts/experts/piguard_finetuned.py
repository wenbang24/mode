"""Adapter for a locally fine-tuned official PIGuard checkpoint."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .base import ExpertOutcome, release_cuda, require_prompt


# These are the generated archives present in the project workspace. The
# WildGuard export predates embedded dataset metadata, so its archive digest is
# the provenance marker. The two Aegis archives have the same model artifact
# but different ZIP timestamps.
_KNOWN_ARTIFACT_DATASETS = {
    "43142ac801042af56d47f9c0fe597a95d79e8f7a87c75876a3215b25dbf29f7e": "wildguard",
    "d9e0bed6d8b10306caf9643189f10e3078890366a8d08a0b1810539c91c25c0a": "aegis",
    "0084b73d184b83ffad07710d4f07883622331270069fef7bfc21023b9ef51c81": "aegis",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe path in PIGuard archive: {name!r}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"unsafe path in PIGuard archive: {name!r}")
    return target


class PIGuardArtifact:
    """Load a generated PIGuard export and expose its harmful-prompt gate."""

    name = "piguard"

    @classmethod
    def inspect_artifact(cls, artifact: Path) -> dict[str, Any]:
        """Read export metadata without importing torch or loading model weights."""

        path = Path(artifact).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".zip":
            raise ValueError("PIGuard artifact must be a generated .zip bundle")
        digest = _sha256_file(path)
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            bundle_files = [
                name for name in names
                if name.endswith("piguard_bundle.json")
            ]
            if len(bundle_files) != 1:
                raise ValueError("PIGuard archive must contain exactly one piguard_bundle.json")
            bundle_member = bundle_files[0]
            bundle_prefix = bundle_member.removesuffix("piguard_bundle.json")
            bundle_metadata = json.loads(archive.read(bundle_member))
            manifest = {}
            manifest_members = [
                name for name in names if name.endswith("manifest.json")
            ]
            if len(manifest_members) > 1:
                raise ValueError("PIGuard archive contains multiple manifests")
            if manifest_members:
                manifest = json.loads(archive.read(manifest_members[0]))
            adapter_members = [
                name for name in names if name.endswith("adapter_config.json")
            ]
            adapter_config = (
                json.loads(archive.read(adapter_members[0]))
                if adapter_members else {}
            )
            backbone_config_members = [
                name for name in names if name.endswith("backbone/config.json")
            ]
            backbone_config = (
                json.loads(archive.read(backbone_config_members[0]))
                if backbone_config_members else {}
            )
        config = manifest.get("config", {})
        training_mode = bundle_metadata.get("training_mode")
        pooling = bundle_metadata.get("pooling")
        if training_mode not in {"full", "qlora"} or pooling not in {"first", "last"}:
            raise ValueError("PIGuard bundle has invalid training mode or pooling metadata")
        required_members = {
            f"{bundle_prefix}piguard_bundle.json",
            f"{bundle_prefix}classifier_head.pth",
            f"{bundle_prefix}tokenizer/tokenizer_config.json",
        }
        if training_mode == "full":
            required_members.add(f"{bundle_prefix}backbone/config.json")
        else:
            required_members.add(f"{bundle_prefix}adapter/adapter_config.json")
        missing = required_members.difference(names)
        if missing:
            raise ValueError(f"PIGuard bundle is incomplete; missing {sorted(missing)}")
        if int(config.get("max_length", 512)) < 1:
            raise ValueError("PIGuard archive has an invalid maximum sequence length")
        model_id = (
            config.get("model_id")
            or adapter_config.get("base_model_name_or_path")
            or ("microsoft/deberta-v3-base" if backbone_config.get("model_type") == "deberta-v2" else None)
        )
        dataset = _KNOWN_ARTIFACT_DATASETS.get(digest)
        if dataset is None:
            dataset = cls._manifest_dataset(manifest)
        if not model_id:
            raise ValueError("PIGuard archive does not identify its base model")
        return {
            "path": str(path),
            "sha256": digest,
            "dataset": dataset,
            "model_id": model_id,
            "training_mode": training_mode,
            "pooling": pooling,
            "max_length": int(config.get("max_length", 512)),
            "bundle_prefix": bundle_prefix,
        }

    @staticmethod
    def _manifest_dataset(value: Any) -> str | None:
        """Accept explicit dataset provenance from newer exported manifests."""

        if isinstance(value, dict):
            for key in ("dataset", "dataset_slug", "training_dataset"):
                item = value.get(key)
                if isinstance(item, str):
                    normalized = item.casefold().replace(" ", "")
                    if normalized in {"wildguard", "wildguardtrain", "wildguardtest"}:
                        return "wildguard"
                    if normalized in {"aegis", "aegis2", "aegis2.0", "aegis2test"}:
                        return "aegis"
            for item in value.values():
                result = PIGuardArtifact._manifest_dataset(item)
                if result:
                    return result
        elif isinstance(value, list):
            for item in value:
                result = PIGuardArtifact._manifest_dataset(item)
                if result:
                    return result
        return None

    def __init__(
        self,
        artifact: Path,
        expected_dataset: str,
        cache_root: Path,
        token: str | None = None,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.artifact = Path(artifact).expanduser().resolve()
        self.metadata = self.inspect_artifact(self.artifact)
        if self.metadata["dataset"] != expected_dataset:
            raise ValueError(
                "PIGuard artifact provenance does not match the selected dataset: "
                f"expected {expected_dataset!r}, found {self.metadata['dataset']!r}"
            )
        if self.metadata["training_mode"] not in {"full", "qlora"}:
            raise ValueError("PIGuard export must use full or qlora training mode")
        if self.metadata["pooling"] not in {"first", "last"}:
            raise ValueError("PIGuard export has an unsupported pooling rule")

        self.bundle, self.manifest = self._extract_bundle(Path(cache_root))
        self.model_id = self.metadata["model_id"]
        self.pooling = self.metadata["pooling"]
        self.max_length = self.metadata["max_length"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.bundle / "tokenizer", local_files_only=True
        )

        model_kwargs: dict[str, Any] = {
            "token": token,
            "revision": self.manifest.get("config", {}).get("revision") or None,
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
        }
        model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}
        if self.device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        elif self.device.type == "mps":
            model_kwargs["torch_dtype"] = torch.float16

        if self.metadata["training_mode"] == "qlora":
            from peft import PeftModel

            backbone = AutoModel.from_pretrained(self.model_id, **model_kwargs)
            adapter_path = self.bundle / "adapter"
            self.model = PeftModel.from_pretrained(
                backbone, adapter_path, is_trainable=False
            )
        else:
            self.model = AutoModel.from_pretrained(
                self.bundle / "backbone", local_files_only=True
            )
        self.model.to(self.device).eval()
        hidden_size = int(self.model.config.hidden_size)
        self.classifier = torch.nn.Linear(hidden_size, 2)
        state = torch.load(
            self.bundle / "classifier_head.pth",
            map_location="cpu",
            weights_only=True,
        )
        self.classifier.load_state_dict(state, strict=True)
        self.classifier.to(self.device).eval()

    def _extract_bundle(self, cache_root: Path) -> tuple[Path, dict[str, Any]]:
        cache_root = Path(cache_root).expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        target = cache_root / self.metadata["sha256"]
        bundle_dir = target / "bundle"
        manifest_path = target / "manifest.json"
        if bundle_dir.is_dir() and (bundle_dir / "piguard_bundle.json").is_file():
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file() else {}
            )
            return bundle_dir, manifest

        staging = Path(tempfile.mkdtemp(prefix=self.metadata["sha256"][:12] + "-", dir=cache_root))
        try:
            with zipfile.ZipFile(self.artifact) as archive:
                prefix = self.metadata["bundle_prefix"]
                members = [
                    name for name in archive.namelist()
                    if name.startswith(prefix) and not name.endswith("/")
                ]
                for name in members:
                    relative = name.removeprefix(prefix)
                    destination = _safe_member_path(staging / "bundle", relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(name) as source, destination.open("wb") as target_file:
                        shutil.copyfileobj(source, target_file)
                manifest_members = [
                    name for name in archive.namelist()
                    if name.endswith("manifest.json")
                ]
                if manifest_members:
                    (staging / "manifest.json").write_bytes(
                        archive.read(manifest_members[0])
                    )
            if not (staging / "bundle" / "piguard_bundle.json").is_file():
                raise ValueError("could not extract a complete PIGuard bundle")
            if target.exists():
                shutil.rmtree(target)
            staging.rename(target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file() else {}
        )
        return bundle_dir, manifest

    def run(self, prompt: str) -> ExpertOutcome:
        prompt = require_prompt(prompt)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            if self.pooling == "first":
                pooled = hidden[:, 0, :]
            else:
                last_index = (
                    attention_mask.shape[1]
                    - 1
                    - attention_mask.flip(dims=(1,)).long().argmax(dim=1)
                )
                pooled = hidden[
                    self.torch.arange(hidden.shape[0], device=self.device), last_index
                ]
            logits = self.classifier(pooled.to(self.classifier.weight.dtype))
            probabilities = self.torch.softmax(logits.float(), dim=-1)[0]
        harmful_score = float(probabilities[1].item())
        block = harmful_score >= 0.5
        return ExpertOutcome(
            block=block,
            metadata={
                "model": self.model_id,
                "artifact": str(self.artifact),
                "artifact_sha256": self.metadata["sha256"],
                "dataset": self.metadata["dataset"],
                "training_mode": self.metadata["training_mode"],
                "pooling": self.pooling,
                "label": "harmful" if block else "benign",
                "harmful_score": harmful_score,
            },
        )

    def close(self) -> None:
        model, classifier, tokenizer = self.model, self.classifier, self.tokenizer
        self.model = self.classifier = self.tokenizer = None
        del model, classifier, tokenizer
        release_cuda()


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


class PIGuardFineTuned:
    name = "piguard_finetuned"
    model_id = "microsoft/deberta-v3-base"

    def __init__(
        self,
        root: Path,
        checkpoint: Path | None = None,
        token: str | None = None,
        revision: str | None = None,
    ):
        import torch
        from huggingface_hub import snapshot_download

        self.root = Path(root).resolve()
        self.checkpoint = Path(
            checkpoint or self.root / "logs/best_model.pth"
        ).resolve()
        required = [
            self.root / "PIGuard.py",
            self.root / "datasets/train.json",
            self.root / "datasets/valid.json",
            self.checkpoint,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"invalid PIGuard training output; missing {missing}")

        self.torch = torch
        self.revision = revision
        base_model = snapshot_download(self.model_id, revision=revision, token=token)
        module_name = "_mode_piguard_" + hashlib.sha256(
            str(self.root).encode()
        ).hexdigest()[:12]
        official = _load_module(module_name, self.root / "PIGuard.py")
        self.model = official.PIGuard(
            base_model, num_labels=2, device=torch.device("cuda")
        )
        state = torch.load(self.checkpoint, map_location="cuda", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("PIGuard checkpoint must contain a state dictionary")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def run(self, prompt: str) -> ExpertOutcome:
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(
                self.model.classify([require_prompt(prompt)]).float(), dim=-1
            )[0]
        prediction_id = int(probabilities.argmax().item())
        return ExpertOutcome(
            block=prediction_id == 1,
            metadata={
                "model": self.model_id,
                "revision": self.revision,
                "checkpoint": str(self.checkpoint),
                "label": "injection" if prediction_id == 1 else "benign",
                "injection_score": float(probabilities[1].item()),
            },
        )

    def close(self) -> None:
        model = self.model
        self.model = None
        del model
        release_cuda()
