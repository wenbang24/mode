"""Shared contract for defensive experts."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Protocol


def require_prompt(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    return prompt


@dataclass(frozen=True)
class ExpertOutcome:
    """A route decision. ``block`` always means the evaluated artifact is unsafe."""

    block: bool
    response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.block, bool):
            raise ValueError("block must be a boolean")
        if self.response is not None and not isinstance(self.response, str):
            raise ValueError("response must be a string or null")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")


class Expert(Protocol):
    name: str
    model_id: str

    def run(self, prompt: str) -> ExpertOutcome: ...

    def close(self) -> None: ...


def release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:1_000]
