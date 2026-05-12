"""Agent modules for prompt optimization."""

__all__ = ["run_optimization_pipeline"]


def run_optimization_pipeline(*args, **kwargs):
    from ledger_ai.agents.pipeline import run_optimization_pipeline as _run

    return _run(*args, **kwargs)
