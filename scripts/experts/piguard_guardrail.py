"""Adapter for the released PIGuard checkpoint."""

from __future__ import annotations

from typing import Any

from .base import ExpertOutcome, release_cuda, require_prompt


class PIGuardGuardrail:
    name = "piguard_guardrail"
    model_id = "leolee99/PIGuard"

    def __init__(self, token: str | None = None, revision: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=token, revision=revision
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            token=token,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to("cuda").eval()
        self.id2label = getattr(self.model.config, "id2label", {})
        label2id = {
            str(key).strip().lower(): int(value)
            for key, value in getattr(self.model.config, "label2id", {}).items()
        }
        self.injection_id = label2id.get("injection", 1)

    def run(self, prompt: str) -> ExpertOutcome:
        encoded = self.tokenizer(
            require_prompt(prompt),
            truncation=True,
            max_length=int(
                getattr(self.model.config, "max_position_embeddings", 512) or 512
            ),
            return_tensors="pt",
        ).to("cuda")
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model(**encoded).logits.float(), dim=-1)[0]
        prediction_id = int(probabilities.argmax().item())
        label: Any = self.id2label.get(
            prediction_id, self.id2label.get(str(prediction_id))
        )
        label = str(label).strip().lower() if label is not None else ""
        if label not in {"benign", "injection"}:
            raise ValueError(f"unexpected PIGuard label: {label!r}")
        return ExpertOutcome(
            block=label == "injection",
            metadata={
                "model": self.model_id,
                "revision": self.revision,
                "label": label,
                "injection_score": float(probabilities[self.injection_id].item()),
            },
        )

    def close(self) -> None:
        model, tokenizer = self.model, self.tokenizer
        self.model = self.tokenizer = None
        del model, tokenizer
        release_cuda()
