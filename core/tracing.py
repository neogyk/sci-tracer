"""MLflow tracing and experiment tracking for the SciTracer agent pipeline.

Two complementary layers
------------------------
1. **MLflow Tracing** (LLM observability)
   ``mlflow.dspy.autolog()`` automatically instruments every DSPy
   ``Predict`` / ``ChainOfThought`` call, capturing inputs, outputs, token
   counts, and latency as structured spans visible in the MLflow UI Traces tab.
   Custom ``@mlflow.trace`` decorators add higher-level WORKFLOW / AGENT spans
   around the researcher → formalizer → verifier → executor pipeline.

2. **MLflow Runs** (experiment tracking)
   ``mlflow.start_run()`` groups each benchmark execution.  Per-problem and
   aggregate metrics (accuracy, n_attempts, refinement_rate, elapsed_s …) are
   logged with ``mlflow.log_metric`` / ``mlflow.log_metrics``.  The full
   EpisodicMemory JSONL is saved as a run artifact.

Quick start
-----------
::

    from core.tracing import setup_tracing, start_benchmark_run

    # Call once at process start — enables DSPy autolog + sets experiment
    setup_tracing(experiment_name="aime2026", tracking_uri="mlruns")

    with start_benchmark_run(run_name="gpt-4o-train-30") as run:
        # ... benchmark loop (metrics logged inside) ...
        pass

    # Open the UI
    #   mlflow ui --backend-store-uri mlruns

Decorators
----------
Use ``trace_agent``, ``trace_chain``, ``trace_tool``, ``trace_workflow``
to annotate any function that should appear as a named span in the trace tree:

::

    from core.tracing import trace_agent

    class Researcher(dspy.Module):
        @trace_agent("Researcher")
        def forward(self, prompt: str) -> ResearchPlan:
            ...
"""
from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Optional

log = logging.getLogger("scitracer.tracing")

try:
    import mlflow
    import mlflow.dspy
    from mlflow.entities import SpanType
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    log.warning("mlflow not installed — tracing disabled.  pip install mlflow")


# ══════════════════════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════════════════════

def setup_tracing(
    experiment_name: str  = "scitracer",
    tracking_uri:    str  = "mlruns",
    dspy_autolog:    bool = True,
) -> None:
    """Configure MLflow and enable DSPy autologging.

    Call once at process start before any agents are instantiated.

    Parameters
    ----------
    experiment_name : MLflow experiment name (created if it doesn't exist).
    tracking_uri    : local directory or remote URI, e.g. ``"mlruns"`` or
                      ``"http://localhost:5000"``.
    dspy_autolog    : whether to enable ``mlflow.dspy.autolog()``  which
                      automatically traces every DSPy Predict/ChainOfThought
                      call including token counts and latency.
    """
    if not _MLFLOW_AVAILABLE:
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if dspy_autolog:
        mlflow.dspy.autolog(
            log_compiles    = True,
            log_traces      = True,
            disable         = False,
            silent          = False,
        )
        log.info("[tracing] mlflow.dspy.autolog enabled")

    log.info("[tracing] MLflow experiment=%r  uri=%r", experiment_name, tracking_uri)


# ══════════════════════════════════════════════════════════════════════════════
# Run context manager
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def start_benchmark_run(
    run_name:  str,
    params:    dict[str, Any] | None = None,
    tags:      dict[str, str]  | None = None,
) -> Generator[Any, None, None]:
    """Context manager that wraps a benchmark execution in an MLflow run.

    Parameters
    ----------
    run_name : human-readable label shown in the MLflow UI.
    params   : hyperparameters logged at run start (split, limit, model, …).
    tags     : arbitrary string tags attached to the run.

    Yields
    ------
    The active ``mlflow.ActiveRun`` object (rarely needed directly).

    Example
    -------
    ::

        with start_benchmark_run("gpt4o-aime-train-30",
                                 params={"split": "train", "limit": 30}) as run:
            for i, example in enumerate(examples):
                ...
                log_episode_metrics(episode, step=i)
            log_benchmark_summary(summary)
    """
    if not _MLFLOW_AVAILABLE:
        yield None
        return

    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        if params:
            mlflow.log_params(params)
        log.info("[tracing] MLflow run started: %s  (id=%s)", run_name, run.info.run_id)
        try:
            yield run
        finally:
            log.info("[tracing] MLflow run ended: %s", run_name)


# ══════════════════════════════════════════════════════════════════════════════
# Metric logging helpers
# ══════════════════════════════════════════════════════════════════════════════

def log_episode_metrics(episode: Any, step: int) -> None:
    """Log per-episode metrics for one problem run.

    Parameters
    ----------
    episode : a ``ReasoningEpisode`` instance from ``memory.episodic_memory``.
    step    : problem index (used as the x-axis step in MLflow charts).
    """
    if not _MLFLOW_AVAILABLE:
        return
    if mlflow.active_run() is None:
        return

    metrics: dict[str, float] = {
        "episode/correct":          float(episode.is_correct),
        "episode/n_attempts":       float(episode.n_attempts),
        "episode/had_refinement":   float(episode.had_refinement),
        "episode/first_failed":     float(episode.first_attempt_failed),
        "episode/elapsed_s":        float(episode.metadata.get("elapsed_s", 0.0)),
    }
    # Running accuracy (fraction correct up to this step)
    mlflow.log_metrics(metrics, step=step)


def log_benchmark_summary(
    summary:       dict[str, Any],
    episodes_path: Optional[Path] = None,
) -> None:
    """Log aggregate benchmark metrics and optional JSONL artifact.

    Parameters
    ----------
    summary       : the dict returned by ``run_benchmark()``.
    episodes_path : path to the ``episodes.jsonl`` file to attach as an artifact.
    """
    if not _MLFLOW_AVAILABLE:
        return
    if mlflow.active_run() is None:
        return

    mlflow.log_metrics({
        "benchmark/accuracy":        summary.get("accuracy", 0.0),
        "benchmark/n_correct":       float(summary.get("n_correct", 0)),
        "benchmark/n_total":         float(summary.get("n_total", 0)),
        "benchmark/run_executor":    float(summary.get("run_executor", True)),
    })

    # Per-subject breakdown (Frontier Science benchmark)
    for subj, stats in summary.get("subject_breakdown", {}).items():
        mlflow.log_metric(f"subject/{subj}/accuracy", stats.get("accuracy", 0.0))

    # Episodic memory artifact
    if episodes_path and Path(episodes_path).exists():
        mlflow.log_artifact(str(episodes_path), artifact_path="episodes")
        log.info("[tracing] episodes artifact logged: %s", episodes_path)


def log_verification_attempt(attempt: Any, episode_step: int) -> None:
    """Log one VerificationAttempt's outcome to the active MLflow run.

    Parameters
    ----------
    attempt      : a ``VerificationAttempt`` instance.
    episode_step : problem index for the x-axis.
    """
    if not _MLFLOW_AVAILABLE:
        return
    if mlflow.active_run() is None:
        return

    mlflow.log_metrics({
        "verifier/is_sat":           float(attempt.is_sat),
        "verifier/is_unsat":         float(attempt.is_unsat),
        "verifier/success":          float(attempt.success),
        "verifier/had_refinement":   float(attempt.had_refinement),
        "verifier/elapsed_s":        attempt.elapsed_s,
    }, step=episode_step * 100 + attempt.attempt_number)


# ══════════════════════════════════════════════════════════════════════════════
# Span decorators
# ══════════════════════════════════════════════════════════════════════════════

def _make_span_decorator(span_type: str) -> Callable:
    """Factory: return a decorator that wraps a function in an MLflow span."""
    def decorator(name: str) -> Callable:
        def wrapper(fn: Callable) -> Callable:
            if not _MLFLOW_AVAILABLE:
                return fn

            @functools.wraps(fn)
            def inner(*args, **kwargs):
                with mlflow.start_span(name=name, span_type=span_type):
                    return fn(*args, **kwargs)

            return inner
        return wrapper
    return decorator


# Convenient named decorators matching MLflow SpanType vocabulary
trace_workflow = _make_span_decorator("WORKFLOW")   # full pipeline run
trace_agent    = _make_span_decorator("AGENT")      # researcher, verifier …
trace_chain    = _make_span_decorator("CHAIN")      # formalizer
trace_tool     = _make_span_decorator("TOOL")       # executor, z3 runner
trace_retriever= _make_span_decorator("RETRIEVER")  # RAG step
trace_memory   = _make_span_decorator("MEMORY")     # episodic memory writes
