"""Evolution judge — evaluates execution results and produces refinement hints.

Used by the self-evolution loop in main.py:
    after each round the judge decides whether the goal was met and, if not,
    produces a critique + refinement_hint that the Researcher uses in the next round.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import dspy


# ── DSPy signature ────────────────────────────────────────────────────────────

class EvolutionJudgeSignature(dspy.Signature):
    """Evaluate whether the execution results satisfy the original research goal.

    Read the goal, the stdout produced by the orchestrator, and the per-step
    logs.  Decide whether the goal has been met (satisfied=True) or not.
    If not satisfied, explain exactly what is missing or broken (critique) and
    suggest concrete changes the planner should make in the next round
    (refinement_hint).
    """
    question: str = dspy.InputField(
        desc="The original research question / goal that must be satisfied"
    )
    execution_success: bool = dspy.InputField(
        desc="Whether the orchestrator run.py exited with returncode 0"
    )
    stdout: str = dspy.InputField(
        desc="Full stdout from the orchestrator run (may be empty)"
    )
    stderr: str = dspy.InputField(
        desc="Full stderr from the orchestrator run (may be empty)"
    )
    step_summary: str = dspy.InputField(
        desc=(
            "One line per plan step: 'step N  <tool>  <success/FAILED>  "
            "<first 120 chars of stdout>'"
        )
    )
    satisfied: bool = dspy.OutputField(
        desc="True if the execution results fully satisfy the original goal"
    )
    critique: str = dspy.OutputField(
        desc=(
            "If not satisfied: bullet-point list of specific deficiencies "
            "(wrong output, missing steps, runtime errors, unmet postconditions). "
            "Empty string if satisfied."
        )
    )
    refinement_hint: str = dspy.OutputField(
        desc=(
            "If not satisfied: actionable instructions for the planner in the next "
            "round (e.g. 'add a data-normalisation step before curve fitting', "
            "'use scipy.optimize instead of manual gradient descent', "
            "'the Z3 model must include the transition invariant X'). "
            "Be specific — the planner will use this verbatim as additional context. "
            "Empty string if satisfied."
        )
    )


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvolutionVerdict:
    round_index: int
    satisfied: bool
    critique: str
    refinement_hint: str

    def as_context(self) -> str:
        """Format as a string to inject into the next Researcher call."""
        if self.satisfied:
            return ""
        return (
            f"[Evolution round {self.round_index} feedback]\n"
            f"The previous attempt did NOT satisfy the goal.\n\n"
            f"Critique:\n{self.critique}\n\n"
            f"What to fix in this round:\n{self.refinement_hint}"
        )


# ── Module ────────────────────────────────────────────────────────────────────

class EvolutionJudge(dspy.Module):
    """Wraps EvolutionJudgeSignature with a ChainOfThought predictor."""

    def __init__(self) -> None:
        self._predict = dspy.ChainOfThought(EvolutionJudgeSignature)

    def forward(
        self,
        *,
        question: str,
        round_index: int,
        execution_success: bool,
        stdout: str,
        stderr: str,
        step_logs,          # list[StepLog] from executor
    ) -> EvolutionVerdict:
        step_summary = "\n".join(
            f"step {lg.index}  {lg.tool}  "
            f"{'OK' if lg.success else 'FAILED'}  "
            f"{(lg.stdout or '')[:120].replace(chr(10), ' ')}"
            for lg in step_logs
        ) if step_logs else "(no steps)"

        pred = self._predict(
            question=question,
            execution_success=execution_success,
            stdout=(stdout or "")[:4000],
            stderr=(stderr or "")[:2000],
            step_summary=step_summary,
        )
        return EvolutionVerdict(
            round_index=round_index,
            satisfied=bool(getattr(pred, "satisfied", False)),
            critique=getattr(pred, "critique", ""),
            refinement_hint=getattr(pred, "refinement_hint", ""),
        )
