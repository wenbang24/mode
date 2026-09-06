"""Defensive expert adapters."""

from importlib import import_module


_EXPORTS = {
    "AdaSteer": ".adasteer",
    "Expert": ".base",
    "ExpertOutcome": ".base",
    "GuardAgent": ".guardagent",
    "PIGuardFineTuned": ".piguard_finetuned",
    "PIGuardGuardrail": ".piguard_guardrail",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
