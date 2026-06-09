from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import dspy
from pydantic import BaseModel, Field

from agents.researcher import InstructionStep  # noqa: F401  (re-export convenience)
from tools.kripke import VerificationResult  # noqa: F401


_LEVEL_INDENT = {1: "", 2: "  ", 3: "    "}
_LEVEL_LABEL  = {1: "PHASE", 2: "TASK", 3: "OP"}


# ── Unified plan schema ────────────────────────────────────────────────────────

class Step(BaseModel):
    """One node in the research plan — a Hoare triple with recursive sub-steps.

    Structure
    ---------
    A Step is simultaneously:
      • A Hoare triple  {P} C {Q}  (precondition / command / postcondition)
      • A task item  (description, tool_name)
      • A tree node  (sub_steps carries any finer-grained decomposition)

    Leaves (sub_steps=[]) are atomic operations ready for code generation.
    Internal nodes group related leaves under a shared pre/post boundary.
    """

    id: int = Field(description="Sequential step identifier, unique within the plan")
    description: str = Field(
        description="Imperative sentence: what this step does and why it is necessary"
    )

    # ── Hoare triple ──────────────────────────────────────────────────────────
    precondition: str = Field(
        description="{P} — typed logical state that must hold before this step executes"
    )
    command: str = Field(
        description="C — exact operation or algorithm with all parameters named explicitly"
    )
    postcondition: str = Field(
        description="{Q} — typed logical state guaranteed after this step completes"
    )

    # ── Tool ──────────────────────────────────────────────────────────────────
    tool_name: str = Field(
        default="",
        description="Registry tool used by this step (e.g. 'arxiv_search', 'python_exec')"
    )

    # ── Recursive decomposition ───────────────────────────────────────────────
    sub_steps: list["Step"] = Field(
        default_factory=list,
        description=(
            "Ordered list of finer-grained sub-steps that together implement this step. "
            "Empty for leaf (atomic) steps. "
            "The sequential composition of sub_steps must faithfully implement command C: "
            "  sub_steps[0].precondition == self.precondition  (P preserved), "
            "  sub_steps[-1].postcondition == self.postcondition  (Q preserved), "
            "  sub_steps[i].postcondition == sub_steps[i+1].precondition  (R consistent)."
        )
    )

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def is_leaf(self) -> bool:
        return len(self.sub_steps) == 0

    def leaves(self) -> list["Step"]:
        """Return all leaf steps in execution order."""
        if self.is_leaf:
            return [self]
        return [leaf for s in self.sub_steps for leaf in s.leaves()]

    def all_steps(self) -> list["Step"]:
        """Pre-order traversal of this step and all descendants."""
        result = [self]
        for s in self.sub_steps:
            result.extend(s.all_steps())
        return result

    def depth(self) -> int:
        """Maximum nesting depth (0 = leaf)."""
        if self.is_leaf:
            return 0
        return 1 + max(s.depth() for s in self.sub_steps)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_str(self, indent: int = 0) -> str:
        pad   = "  " * indent
        tool  = f"  [{self.tool_name}]" if self.tool_name else ""
        lines = [
            f"{pad}Step {self.id}: {self.description}{tool}",
            f"{pad}  {{P}} {self.precondition}",
            f"{pad}   C  {self.command}",
            f"{pad}  {{Q}} {self.postcondition}",
        ]
        for sub in self.sub_steps:
            lines.append(sub.to_str(indent + 2))
        return "\n".join(lines)

    # ── Converters ────────────────────────────────────────────────────────────

    @classmethod
    def from_instruction_step(
        cls,
        s: "InstructionStep",
        all_steps: list["InstructionStep"] | None = None,
    ) -> "Step":
        """Convert an ``InstructionStep`` (flat Pydantic model) to a ``Step``.

        When *all_steps* is provided, direct children (steps whose
        ``parent_step`` equals ``s.step``) are embedded as ``sub_steps``
        recursively.
        """
        children: list[Step] = []
        if all_steps:
            direct = [c for c in all_steps if getattr(c, "parent_step", None) == s.step]
            children = [cls.from_instruction_step(c, all_steps) for c in direct]

        return cls(
            id           = s.step,
            description  = s.task or "",
            precondition = s.precondition or "",
            command      = s.command or "",
            postcondition= s.postcondition or "",
            tool_name    = s.tool or "",
            sub_steps    = children,
        )

    def to_instruction_step(self) -> "InstructionStep":
        """Convert back to a flat ``InstructionStep`` (level inferred from depth)."""
        level_map = {0: 3, 1: 2}          # leaf=3 (atomic op), depth-1=2 (task)
        level     = level_map.get(self.depth(), 1)
        parent    = self.sub_steps[0].id if self.sub_steps else None
        return InstructionStep(
            step          = self.id,
            task          = self.description,
            precondition  = self.precondition,
            command       = self.command,
            postcondition = self.postcondition,
            tool          = self.tool_name,
            level         = level,
            parent_step   = None,
            dependencies  = [],
            input_type    = "",
            reasoning     = "",
        )


# allow the recursive self-reference to resolve
Step.model_rebuild()


class Plan(BaseModel):
    """Universal research plan schema.

    A Plan is a typed program:
      INPUT      — described in ``goal`` and the precondition of ``steps[0]``
      BODY       — the ordered list of ``steps`` (each a recursive ``Step``)
      OUTPUT     — the postcondition of ``steps[-1]``

    The ``steps`` list holds top-level steps only.  Sub-steps live inside
    ``Step.sub_steps`` recursively, forming a tree rather than a flat list.
    """

    goal: str = Field(
        description="The concrete objective this plan aims to achieve"
    )
    theory: str = Field(
        default="",
        description="Theoretical background or formal framework the plan operates within"
    )
    hypothesis: str = Field(
        default="",
        description="The central claim or conjecture the plan will verify or falsify"
    )
    methodology: str = Field(
        default="",
        description="High-level description of the approach and algorithmic strategy"
    )
    tool_list: list[str] = Field(
        default_factory=list,
        description="Names of all external tools required across all steps"
    )
    steps: list[Step] = Field(
        default_factory=list,
        description="Top-level ordered steps; each may contain recursive sub_steps"
    )

    # ── Convenience ───────────────────────────────────────────────────────────

    def all_steps(self) -> list[Step]:
        """Flat pre-order list of every step and sub-step in the plan."""
        return [node for s in self.steps for node in s.all_steps()]

    def leaves(self) -> list[Step]:
        """All leaf (atomic) steps in execution order."""
        return [node for s in self.steps for node in s.leaves()]

    def to_str(self) -> str:
        sep = "─" * 70
        lines = [
            sep,
            f"  GOAL        : {self.goal}",
            f"  THEORY      : {self.theory}",
            f"  HYPOTHESIS  : {self.hypothesis}",
            f"  METHODOLOGY : {self.methodology}",
        ]
        if self.tool_list:
            lines.append(f"  TOOLS       : {', '.join(self.tool_list)}")
        lines.append(sep)
        for s in self.steps:
            lines.append(s.to_str(indent=0))
        lines.append(sep)
        return "\n".join(lines)

    # ── Converters ────────────────────────────────────────────────────────────

    @classmethod
    def from_research_plan(cls, rp: Any) -> "Plan":
        """Build a ``Plan`` from a ``ResearchPlan`` (dataclass from agents/rag.py).

        Top-level steps (those with ``parent_step=None`` or level==1) become
        the plan's ``steps``; their children are nested recursively via
        ``Step.from_instruction_step``.
        """
        all_flat: list[InstructionStep] = list(rp.steps or [])
        roots = [
            s for s in all_flat
            if getattr(s, "parent_step", None) is None
        ]
        if not roots:
            roots = all_flat   # flat plan — no hierarchy info

        tool_list = sorted({s.tool for s in all_flat if s.tool})

        return cls(
            goal        = getattr(rp, "research_goal", "") or "",
            theory      = getattr(rp, "theory", "") or "",
            hypothesis  = getattr(rp, "hypothesis", "") or "",
            methodology = getattr(rp, "methodology", "") or "",
            tool_list   = tool_list,
            steps       = [Step.from_instruction_step(r, all_flat) for r in roots],
        )

    def to_instruction_steps(self) -> list[InstructionStep]:
        """Flatten the tree back to a list of ``InstructionStep`` objects."""
        return [node.to_instruction_step() for node in self.all_steps()]


def _plan_to_str(steps: "list[InstructionStep] | Plan") -> str:
    """Serialize a research plan to a human-readable string for LLM prompts.

    Accepts either:
      • a ``Plan`` instance  — uses ``Plan.to_str()`` directly
      • a ``list[InstructionStep]``  — renders with level-based indentation
    """
    if isinstance(steps, Plan):
        return steps.to_str()

    if not steps:
        return ""

    lines: list[str] = []
    for s in steps:
        lvl    = getattr(s, "level", 2)
        indent = _LEVEL_INDENT.get(lvl, "  ")
        label  = _LEVEL_LABEL.get(lvl, "STEP")
        tool_line = f"ACTION: {s.tool}" if s.tool else "ACTION: (none — pure reasoning)"
        deps = ", ".join(str(d) for d in s.dependencies) if s.dependencies else "none"

        if lvl == 1:
            lines += [
                f"",
                f"{indent}{'━' * 60}",
                f"{indent}[{label} {s.step}] {s.task}",
                f"{indent}  {{P}} {s.precondition}",
                f"{indent}  {{Q}} {s.postcondition}",
            ]
        else:
            lines += [
                f"{indent}[{label} {s.step}] {s.task}",
                f"{indent}  Dependencies:      {deps}",
                f"{indent}  {{P}} Precondition: {s.precondition}",
                f"{indent}  Input type:        {s.input_type}",
                f"{indent}  {tool_line}",
                f"{indent}   C  Command:       {s.command}",
                f"{indent}  Tool rationale:    {s.tool_rationale}",
                f"{indent}  {{Q}} Postcondition: {s.postcondition}",
                f"{indent}  Reasoning:         {s.reasoning}",
            ]
    return "\n".join(lines)



# ── Extra DSPy signatures ─────────────────────────────────────────────────────

class PlanRefinement(dspy.Signature):
    """Improve a research plan based on rich Z3 diagnostic feedback from formal verification.

    You are given the current plan, the full Kripke structure that was extracted from it,
    and per-property diagnostic reports.  Use this information to produce a revised plan
    that fixes every reported failure.

    READING THE DIAGNOSTICS
    ═══════════════════════
    Each property diagnostic tells you:
      • Which CTL property failed (e.g. AG(safe), EF(goal), AF(done)).
      • The annotated counterexample path: each world with the propositions that hold
        and fail at that world.  The violation world is marked explicitly.
      • For AG failures  — which worlds are MISSING the proposition.
        Fix: add a step that establishes the proposition before reaching those worlds.
      • For AF failures  — which worlds can AVOID the proposition forever (blocking worlds).
        Fix: add a convergence or termination condition that forces eventual progress.
      • For EF failures  — worlds where the proposition HOLDS but are unreachable.
        Fix: add a transition from a reachable state to one of those goal worlds.
      • For EG failures  — worlds forming an infinite cycle inside the proposition.
        Fix: break the cycle with a state change that exits the looping condition.

    REFINEMENT RULES
    ════════════════
    1. Map each failure world back to the plan step it represents (use the world name
       as a clue — e.g. 's2' likely corresponds to Step 2).
    2. For AG(safe) failures: insert a safety-preserving action before the violating step.
    3. For AF(done)/EF(goal) failures: add a new step that creates the missing transition.
    4. For EG(loop) failures: add a loop-exit condition to the cycling step.
    5. Preserve all steps that are not implicated in any failure.
    6. Keep the three-level hierarchy (Phase / Task / AtomicOp).
    """
    plan: str = dspy.InputField(
        desc="Current research plan as a formatted string of InstructionSteps"
    )
    failed_properties: str = dspy.InputField(
        desc=(
            "Semicolon-separated list of CTL properties that failed, each with the "
            "violation world and shortest counterexample path from the initial state. "
            "Example: \"AG(safe) fails at world 's2' (path: s0 → s1 → s2)\""
        )
    )
    kripke_structure: str = dspy.InputField(
        desc=(
            "Full Kripke structure M = (S, R, V) extracted from the plan: "
            "world names, transition edges, and the complete proposition valuation. "
            "Use this to understand what each world represents and which propositions "
            "are missing, misplaced, or unreachable."
        )
    )
    annotated_paths: str = dspy.InputField(
        desc=(
            "Per-failure annotated counterexample paths.  Each world on the path is "
            "labelled with the propositions that hold and fail at that world. "
            "The violation world is marked.  Use this to pinpoint EXACTLY which step "
            "in the plan breaks the invariant or fails to establish the goal."
        )
    )
    diagnostic_report: str = dspy.InputField(
        desc=(
            "Structured per-property diagnostic report from Z3, including: "
            "missing proposition worlds (AG), blocking worlds (AF), unreachable goal "
            "worlds (EF), and cycle worlds (EG).  Each entry includes a suggested fix "
            "direction.  Translate each fix direction into a concrete plan step change."
        )
    )
    refined_plan: str = dspy.OutputField(
        desc=(
            "Revised research plan that fixes every reported CTL failure. "
            "For each failure, state which step was changed and why, then emit the "
            "full updated plan.  The revised plan must pass AG, AF, EF, and EG checks "
            "for all reported properties."
        )
    )


# ── Verification report ───────────────────────────────────────────────────────

@dataclass
class VerificationReport:
    """Unified result from running all verifiers against a research plan.

    Aggregates Z3, Kripke, Lean, step-chain, and type-consistency checks into
    a single object with convenience properties for pass/fail status and a
    one-line summary.
    """
    z3: dict[str, Any]              # verify_z3_pipeline result
    kripke: list[VerificationResult]  # CTL property results
    lean: list[dict[str, Any]]      # per-step Lean autoformalization results
    chain: list[dict[str, Any]]     # postcondition_N ⊨ precondition_{N+1} checks
    type_checks: list[dict[str, Any]]  # input/output type compatibility results
    suggestion: str                 # LLM-generated refinement suggestion

    @property
    def passed(self) -> bool:
        """True when all verifiers agree the plan is sound."""
        z3_ok     = bool(self.z3.get("surviving"))
        kripke_ok = all(r.holds for r in self.kripke if not r.error)
        chain_ok  = all(r.get("entailment_holds") == "yes" for r in self.chain)
        type_ok   = all(r.get("types_compatible") == "yes" for r in self.type_checks)
        return z3_ok and kripke_ok and chain_ok and type_ok

    @property
    def summary(self) -> str:
        """One-line status string with per-verifier counts."""
        status          = "PASSED" if self.passed else "FAILED"
        z3_surviving    = len(self.z3.get("surviving", []))
        z3_total        = len(self.z3.get("paths", []))
        chain_gaps      = sum(1 for r in self.chain if r.get("entailment_holds") != "yes")
        type_mismatches = sum(1 for r in self.type_checks if r.get("types_compatible") != "yes")
        lean_verified   = sum(1 for r in self.lean if r.get("verified"))
        return (
            f"[{status}] z3={z3_surviving}/{z3_total} "
            f"chain_gaps={chain_gaps} "
            f"type_mismatches={type_mismatches} "
            f"lean_verified={lean_verified}"
        )
