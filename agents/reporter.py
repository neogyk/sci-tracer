"""Reporter agent — writes REPORT.md from the full pipeline outputs.

Sections
--------
1. Hypothesis
2. Methodology
3. Plan  (Hoare-triple steps with tool used)
4. Execution Steps  (per-step: reasoning, produced code, stdout, PASS/FAIL)
5. Results
6. Conclusion
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dspy

from agents.researcher import ResearchPlan, Plan
from agents.synthesis import SynthesisResult


# ── DSPy signature ────────────────────────────────────────────────────────────

class ReportSignature(dspy.Signature):
    """Summarise a completed SciTracer pipeline run into structured report sections.

    You have the full context of what was planned, executed, verified, and
    concluded.  Write each section clearly and concisely for a scientific
    reader who was not present during the run.

    Rules
    -----
    - Do NOT invent results not present in execution_trace or synthesis.
    - Each section must stand alone — avoid forward references.
    - Use plain English; avoid jargon unless it is domain-standard.
    - Bullet lists are preferred over dense paragraphs for execution sections.
    """

    question:             str = dspy.InputField(desc="The original research question")
    hypothesis:           str = dspy.InputField(desc="Falsifiable hypothesis from the research plan")
    methodology:          str = dspy.InputField(desc="High-level methodology from the research plan")
    plan_text:            str = dspy.InputField(desc="Ordered plan steps with tool used per step")
    execution_trace:      str = dspy.InputField(desc="Per-step: reasoning, stdout output, PASS/FAIL, repair count")
    tool_use_summary:     str = dspy.InputField(desc="Which tool was used at each step and why")
    verification_summary: str = dspy.InputField(desc="Formal verification outcomes (CTL properties, Z3 results)")
    synthesis:            str = dspy.InputField(desc="Final answer and executive summary from SynthesisAgent")

    hypothesis_section:   str = dspy.OutputField(desc="2-4 sentences restating the hypothesis in light of the results")
    methodology_section:  str = dspy.OutputField(desc="3-6 sentences describing the methodology used")
    plan_section:         str = dspy.OutputField(desc="Bullet list: each plan step, its goal, and the tool assigned")
    execution_section:    str = dspy.OutputField(desc="Bullet list: per step — what reasoning led to the code, what the code produced, PASS/FAIL")
    results_section:      str = dspy.OutputField(desc="Narrative of the key quantitative and qualitative results")
    conclusion_section:   str = dspy.OutputField(desc="2-5 sentences: what was concluded, confidence level, open questions")


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ReportResult:
    report_path:         str
    hypothesis_section:  str = ""
    methodology_section: str = ""
    plan_section:        str = ""
    execution_section:   str = ""
    results_section:     str = ""
    conclusion_section:  str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _plan_to_text(plan: Any) -> str:
    if plan is None:
        return ""
    if isinstance(plan, ResearchPlan):
        steps = plan.steps
    else:
        steps = plan
    if isinstance(steps, Plan):
        steps = steps.instructions
    if not steps:
        return str(plan)[:2000]
    lines = []
    for i, step in enumerate(steps, 1):
        desc = getattr(step, "description", None) or getattr(step, "state", str(step))
        tool = getattr(step, "tool", "") or ""
        lines.append(f"{i}. {desc}" + (f"  [tool: {tool}]" if tool else ""))
    return "\n".join(lines)


def _step_logs_to_text(step_logs: list) -> tuple[str, str]:
    """Return (execution_trace, tool_use_summary) from a list of StepLog objects."""
    trace_lines: list[str] = []
    tool_lines:  list[str] = []

    for log in step_logs:
        status = "PASS" if log.success else f"FAIL (repairs: {log.repairs})"
        reasoning = log.reasoning.strip()[:300] if log.reasoning else "—"
        stdout    = log.stdout.strip()[:400]    if log.stdout    else "—"

        trace_lines.append(
            f"Step {log.index:02d} [{status}]  task: {log.task_name[:80]}\n"
            f"  Reasoning : {reasoning}\n"
            f"  Output    : {stdout}"
        )
        tool_lines.append(
            f"Step {log.index:02d}: tool={log.tool or 'python_exec'}  file={log.file_path}"
        )

    return "\n\n".join(trace_lines), "\n".join(tool_lines)


def _synthesis_to_text(synthesis: SynthesisResult | None) -> str:
    if synthesis is None:
        return ""
    parts = []
    if synthesis.executive_summary:
        parts.append(f"Summary: {synthesis.executive_summary}")
    if synthesis.final_answer:
        parts.append(f"Final answer: {synthesis.final_answer}")
    if synthesis.confidence:
        parts.append(f"Confidence: {synthesis.confidence}")
    if synthesis.gaps and synthesis.gaps.lower() != "none":
        parts.append(f"Gaps: {synthesis.gaps}")
    return "\n".join(parts)


def _code_appendix(step_logs: list) -> str:
    """Render all generated step files as fenced code blocks."""
    if not step_logs:
        return ""
    blocks = ["## Appendix — Generated Code\n"]
    for log in step_logs:
        status = "PASS" if log.success else "FAIL"
        blocks.append(
            f"### Step {log.index:02d} — `{log.task_name[:60]}` [{status}]\n"
            f"**File:** `{log.file_path}`  |  **Tool:** `{log.tool or 'python_exec'}`\n\n"
            f"```python\n{log.code.strip()}\n```\n"
        )
    return "\n".join(blocks)


def _render_markdown(
    question:            str,
    hypothesis_section:  str,
    methodology_section: str,
    plan_section:        str,
    execution_section:   str,
    results_section:     str,
    conclusion_section:  str,
    code_appendix:       str,
    timestamp:           str,
) -> str:
    def section(title: str, body: str) -> str:
        return f"## {title}\n\n{body.strip()}\n"

    parts = [
        "# SciTracer Research Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Question:** {question.strip()}",
        "",
        "---",
        "",
        section("Hypothesis",      hypothesis_section),
        "---",
        "",
        section("Methodology",     methodology_section),
        "---",
        "",
        section("Plan",            plan_section),
        "---",
        "",
        section("Execution Steps", execution_section),
        "---",
        "",
        section("Results",         results_section),
        "---",
        "",
        section("Conclusion",      conclusion_section),
    ]
    if code_appendix:
        parts += ["---", "", code_appendix]

    return "\n".join(parts)


# ── Agent ─────────────────────────────────────────────────────────────────────

class Reporter(dspy.Module):
    """Generate REPORT.md from the full pipeline outputs.

    Parameters
    ----------
    output_dir:
        Directory where REPORT.md is written.  Defaults to the current
        working directory.
    """

    def __init__(self, output_dir: str | Path = ".") -> None:
        super().__init__()
        self.writer     = dspy.ChainOfThought(ReportSignature)
        self.output_dir = Path(output_dir)

    def forward(
        self,
        question:             str,
        research_plan:        ResearchPlan | Plan | None = None,
        execution_trace:      str  = "",
        verification_summary: str  = "",
        synthesis:            SynthesisResult | None = None,
        step_logs:            list | None = None,
    ) -> ReportResult:
        """Write REPORT.md and return a ReportResult.

        Parameters
        ----------
        question:
            The original research question / task prompt.
        research_plan:
            ResearchPlan from the Researcher agent.
        execution_trace:
            Fallback execution text (used when step_logs is not available).
        verification_summary:
            Text summary from the Verifier.
        synthesis:
            SynthesisResult from SynthesisAgent, if available.
        step_logs:
            List of StepLog objects from ExecutionResult.step_logs.
            When provided, supersedes execution_trace and adds the code appendix.
        """
        hypothesis  = getattr(research_plan, "hypothesis",  "") if research_plan else ""
        methodology = getattr(research_plan, "methodology", "") if research_plan else ""
        plan_text   = _plan_to_text(research_plan)
        synth_text  = _synthesis_to_text(synthesis)

        if step_logs:
            trace_text, tool_text = _step_logs_to_text(step_logs)
            appendix = _code_appendix(step_logs)
        else:
            trace_text = execution_trace
            tool_text  = ""
            appendix   = ""

        pred = self.writer(
            question             = question,
            hypothesis           = hypothesis or "Not specified.",
            methodology          = methodology or "Not specified.",
            plan_text            = plan_text[:3000],
            execution_trace      = trace_text[:4000],
            tool_use_summary     = tool_text[:1000],
            verification_summary = verification_summary[:1000],
            synthesis            = synth_text[:1500],
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        md = _render_markdown(
            question            = question,
            hypothesis_section  = pred.hypothesis_section,
            methodology_section = pred.methodology_section,
            plan_section        = pred.plan_section,
            execution_section   = pred.execution_section,
            results_section     = pred.results_section,
            conclusion_section  = pred.conclusion_section,
            code_appendix       = appendix,
            timestamp           = timestamp,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "REPORT.md"
        report_path.write_text(md, encoding="utf-8")
        print(f"[reporter] REPORT.md written → {report_path}")

        return ReportResult(
            report_path         = str(report_path),
            hypothesis_section  = pred.hypothesis_section,
            methodology_section = pred.methodology_section,
            plan_section        = pred.plan_section,
            execution_section   = pred.execution_section,
            results_section     = pred.results_section,
            conclusion_section  = pred.conclusion_section,
        )
