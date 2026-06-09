"""Synthesis Agent — final stage of the SciTracer pipeline.

Collects outputs from every upstream agent and produces a single, coherent
final answer.  It acts as the "closing argument": it knows what the plan said,
what each step actually produced, which formal properties hold, and what code
was generated — and it synthesises all of that into a definitive response.

Inputs (from the pipeline)
--------------------------
question            Original research question
research_plan       Verified plan as formatted text (Hoare triples)
execution_trace     Accumulated context from PlanExecutor or step-by-step runner
verification_summary  CTL properties that hold / fail, plus refiner suggestion
code_artifacts      Paths / content of generated Python files

Outputs
-------
final_answer        Direct, complete answer to the original question
executive_summary   1–3 sentence abstract of what was done and found
confidence          HIGH / MEDIUM / LOW with justification
gaps                Any open questions or unresolved steps
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import dspy


# ── DSPy Signature ─────────────────────────────────────────────────────────────

class SynthesisSignature(dspy.Signature):
    """Synthesise a final solution from the outputs of all upstream agents.

    You are the last agent in the SciTracer pipeline.  You have access to:
      • The original research question
      • The verified research plan — a sequence of Hoare triples {P} C {Q}
      • The execution trace — what each step actually produced
      • The verification summary — which CTL properties hold and which failed
      • Code artifacts — paths to generated Python files and their key outputs

    Your task
    ---------
    1. Read the execution trace carefully.  The answer is likely already there
       as the result of the final step.  Extract and present it clearly.
    2. If the execution trace is incomplete or contradictory, synthesise the
       best possible answer from the partial evidence.
    3. Fill gaps using the research plan — if a step was skipped or errored,
       reason from its {P} and {Q} to infer the expected result.
    4. Assess confidence honestly.  If critical steps failed or verification
       properties did not hold, say so.
    5. Keep the final_answer focused and concrete — it must directly answer
       the original question.

    Rules
    -----
    - Do NOT invent results that are not supported by the execution trace or
      the research plan.
    - Do NOT repeat the plan verbatim — synthesise, do not summarise.
    - If a numerical answer is expected, provide the number with units.
    - If code is the deliverable, reference the artifact path and describe
      what the code does.
    """

    question: str = dspy.InputField(
        desc="The original research question that initiated the pipeline"
    )
    research_plan: str = dspy.InputField(
        desc=(
            "The verified research plan as formatted Hoare-triple steps. "
            "Each step has {P} precondition, C command, {Q} postcondition."
        )
    )
    execution_trace: str = dspy.InputField(
        desc=(
            "Accumulated context from all executed plan steps. "
            "Each entry shows: step description, concrete result produced, "
            "and whether the postcondition {Q} was satisfied (PASS/FAIL). "
            "The final entry's result is the primary answer candidate."
        )
    )
    verification_summary: str = dspy.InputField(
        desc=(
            "Results of formal verification: which CTL properties hold, "
            "which failed, and the LLM refiner's suggestion if any failed. "
            "Use this to assess the reliability of the plan."
        )
    )
    code_artifacts: str = dspy.InputField(
        desc=(
            "Paths and descriptions of generated Python files, one per plan step. "
            "Each entry lists the step, file path, and whether generation succeeded. "
            "Empty string if no code was generated."
        )
    )

    final_answer: str = dspy.OutputField(
        desc=(
            "The complete, direct answer to the original question. "
            "Must be self-contained — a reader who has not seen the plan or trace "
            "should be able to understand the answer. "
            "If the answer is numerical, include the value, units, and derivation. "
            "If the answer is code, include the key logic or the file path."
        )
    )
    executive_summary: str = dspy.OutputField(
        desc=(
            "1–3 sentences summarising: what was done, how, and what was found. "
            "Written for a reader who wants the key insight without the details."
        )
    )
    confidence: str = dspy.OutputField(
        desc=(
            "Assessment of answer reliability. "
            "Format: 'HIGH | MEDIUM | LOW : <one-sentence justification>'. "
            "HIGH = all steps passed and all CTL properties hold. "
            "MEDIUM = most steps passed or minor verification failures. "
            "LOW = critical steps failed or the execution trace is incomplete."
        )
    )
    gaps: str = dspy.OutputField(
        desc=(
            "Bullet list of open questions, unresolved steps, or missing evidence. "
            "Write 'None' if the pipeline completed without gaps."
        )
    )


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class SynthesisResult:
    """Structured output of the SynthesisAgent."""

    question:            str
    final_answer:        str
    executive_summary:   str
    confidence:          str
    gaps:                str
    reasoning:           str = ""

    @property
    def confidence_level(self) -> str:
        """Returns 'HIGH', 'MEDIUM', or 'LOW' extracted from confidence field."""
        upper = self.confidence.upper()
        for level in ("HIGH", "MEDIUM", "LOW"):
            if upper.startswith(level):
                return level
        return "UNKNOWN"

    def to_str(self) -> str:
        sep  = "═" * 70
        thin = "─" * 70
        lines = [
            "",
            sep,
            f"  SYNTHESIS  [{self.confidence_level}]",
            sep,
            "",
            "  FINAL ANSWER",
            thin,
        ]
        for line in self.final_answer.splitlines():
            lines.append(f"  {line}")
        lines += [
            "",
            "  EXECUTIVE SUMMARY",
            thin,
        ]
        for line in self.executive_summary.splitlines():
            lines.append(f"  {line}")
        lines += [
            "",
            "  CONFIDENCE",
            thin,
            f"  {self.confidence}",
            "",
            "  GAPS",
            thin,
        ]
        for line in self.gaps.splitlines():
            lines.append(f"  {line}")
        lines.append(sep)
        return "\n".join(lines)


# ── Agent ──────────────────────────────────────────────────────────────────────

class SynthesisAgent(dspy.Module):
    """Collect all pipeline agent outputs and produce a final synthesised answer.

    The agent is the last stage in the SciTracer pipeline.  It receives
    structured summaries from every upstream component and reasons over them
    with ChainOfThought to emit a final_answer, executive_summary, confidence
    level, and a list of any remaining gaps.

    Usage
    -----
    ::

        agent = SynthesisAgent()
        result = agent.synthesize(
            question          = "...",
            research_plan     = plan_str,
            step_results      = step_results,      # list[dict] from executor
            verification      = verification,      # list[VerificationResult]
            suggestion        = suggestion,        # str from verifier
            code_artifacts    = exec_result,       # dict from step2code_executor
            execution_context = final_context,     # str from PlanExecutor
        )
        print(result.to_str())
    """

    def __init__(self) -> None:
        super().__init__()
        self._synthesise = dspy.ChainOfThought(SynthesisSignature)

    # ── Public API ─────────────────────────────────────────────────────────────

    def synthesize(
        self,
        question:          str,
        research_plan:     str,
        step_results:      list[dict[str, Any]],
        verification:      list[Any],          # list[VerificationResult]
        suggestion:        str,
        code_artifacts:    dict[str, Any] | None = None,
        execution_context: str = "",
    ) -> SynthesisResult:
        """Synthesise a final answer from all pipeline agent outputs.

        Parameters
        ----------
        question:
            The original research question.
        research_plan:
            Plan serialised as text (from ``_plan_to_str`` or ``Plan.to_str()``).
        step_results:
            Per-step execution records — list of dicts with keys:
            ``step``, ``command``, ``conditions_check``, ``output_assessment``,
            ``accumulated_context``, ``attempts``.
        verification:
            ``list[VerificationResult]`` from ``verify_kripke``.
        suggestion:
            LLM-generated refinement suggestion from the verifier.
        code_artifacts:
            Dict returned by ``step2code_executor``:
            ``{n_steps, n_errors, artifacts: [{step, task, path, error}]}``.
        execution_context:
            Accumulated context string from ``PlanExecutor`` (optional;
            used when the plan executor ran instead of step2code_executor).
        """
        execution_trace   = self._format_trace(step_results, execution_context)
        verification_sum  = self._format_verification(verification, suggestion)
        code_art_str      = self._format_code_artifacts(code_artifacts)

        pred = self._synthesise(
            question             = question,
            research_plan        = research_plan,
            execution_trace      = execution_trace,
            verification_summary = verification_sum,
            code_artifacts       = code_art_str,
        )

        return SynthesisResult(
            question          = question,
            final_answer      = pred.final_answer,
            executive_summary = pred.executive_summary,
            confidence        = pred.confidence,
            gaps              = pred.gaps,
            reasoning         = getattr(pred, "reasoning", ""),
        )

    def forward(self, **kwargs: Any) -> SynthesisResult:
        """DSPy-compatible entry point — delegates to ``synthesize``."""
        return self.synthesize(**kwargs)

    # ── Formatters ─────────────────────────────────────────────────────────────

    @staticmethod
    def _format_trace(
        step_results: list[dict[str, Any]],
        execution_context: str,
    ) -> str:
        """Build the execution_trace string from step_results + accumulated context."""
        if not step_results and not execution_context:
            return "(no execution trace available)"

        parts: list[str] = []

        if step_results:
            for sr in step_results:
                passed  = str(sr.get("conditions_check", "")).upper().startswith("PASS")
                matched = str(sr.get("output_assessment", "")).upper().startswith("MATCH")
                status  = "✓" if (passed and matched) else "✗"
                context = sr.get("accumulated_context", "") or ""
                parts.append(
                    f"[{status}] Step {sr.get('step', '?')}: {sr.get('command', '')[:80]}\n"
                    f"    P-check : {sr.get('conditions_check', '')}\n"
                    f"    Q-check : {sr.get('output_assessment', '')}\n"
                    f"    result  : {context[:300]}"
                )

        if execution_context:
            parts.append(
                "ACCUMULATED EXECUTION CONTEXT (from PlanExecutor):\n"
                + execution_context[-2000:]   # last 2000 chars — most recent results
            )

        return "\n\n".join(parts)

    @staticmethod
    def _format_verification(
        verification: list[Any],
        suggestion: str,
    ) -> str:
        """Summarise VerificationResult list + refiner suggestion."""
        if not verification:
            return "(verification not run)"

        lines: list[str] = []
        n_pass  = sum(1 for r in verification if getattr(r, "holds", False))
        n_fail  = sum(1 for r in verification if not getattr(r, "holds", True) and not getattr(r, "error", False))
        n_error = sum(1 for r in verification if getattr(r, "error", False))

        lines.append(
            f"CTL verification: {n_pass} pass  {n_fail} fail  {n_error} error"
            f"  (total={len(verification)})"
        )
        for r in verification:
            holds = getattr(r, "holds", None)
            err   = getattr(r, "error", None)
            prop  = getattr(r, "property", "?")
            wit   = getattr(r, "witness", "") or ""
            tag   = "PASS" if holds else ("ERROR" if err else "FAIL")
            witness_note = f"  witness={wit}" if wit else ""
            lines.append(f"  [{tag}] {prop}{witness_note}")

        if suggestion:
            lines.append(f"\nRefiner suggestion:\n  {suggestion[:400]}")

        return "\n".join(lines)

    @staticmethod
    def _format_code_artifacts(artifacts: dict[str, Any] | None) -> str:
        """Summarise code generation results."""
        if not artifacts:
            return "(no code generated)"

        n_steps  = artifacts.get("n_steps", 0)
        n_errors = artifacts.get("n_errors", 0)
        items    = artifacts.get("artifacts", [])

        lines = [
            f"Code generation: {n_steps} steps  {n_errors} errors"
        ]
        for a in items:
            tag  = "✗" if a.get("error") else "✓"
            path = a.get("path", "") or "(no file)"
            task = (a.get("task") or "")[:60]
            err  = f"  error={a['error']}" if a.get("error") else ""
            lines.append(f"  [{tag}] step={a.get('step', '?')}  {task}\n"
                         f"          → {path}{err}")
        return "\n".join(lines)
