"""Adapter for a locally fine-tuned official PIGuard checkpoint."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from .base import ExpertOutcome, release_cuda, require_prompt


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
