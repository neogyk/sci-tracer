"""PlanOrchestrator — two-phase hierarchical planner.

Phase 1: HighLevelPlanner (DSPy ChainOfThought) produces a numbered list of
         high-level milestones from the user goal and background context.
         The LLM decides how many steps are needed — no cap is imposed.

Phase 2: LATS acts as the Worker that executes each milestone.  It receives
         the current milestone and the full macro context and returns a
         detailed micro-solution via Monte Carlo tree search.

The result is compatible with ResearchPlan so the rest of the SciTracer
pipeline (verification, code-generation, synthesis) is unchanged.
"""
import re
import dspy
from dataclasses import dataclass, field
from typing import Any, Optional
from agents.planning.lats import LATS, LATSResult


# ── Signatures ────────────────────────────────────────────────────────────────

class HighLevelPlanner(dspy.Signature):
    """
    Your purpose is to act as a diligent research assistant acting on behalf of the user to conduct research checks into prospective technology research company.
    You are planner agent designed to autonomously discover, formalize, and plan execution strategies for complex, real-world problems at the intersection of empirical science and advanced computation.
    User might be interested in conducting the research or verification some technical task
    in math, physics, chemistry or computer science.

    Critically think about the user question. 
    Suggest a high-level research plan as numbered milestones in a details in a scientific manner.

    Each milestone is a distinct, non-overlapping detailed description of the research or
    solution process.  Use as many milestones as the problem genuinely requires
    — do NOT artificially limit or inflate the count.

    The milestones define the macro-structure.  
    """

    user_goal: str = dspy.InputField(
        desc="The research question or problem to solve"
    )
    context: str = dspy.InputField(
        desc=(
            "Background context: relevant literature summaries, known methods, "
            "prior results, and domain knowledge that should inform the plan."
        )
    )

    macro_plan: str = dspy.OutputField(
        desc=(
            "Numbered list of high-level milestones. "
            "Use exactly as many milestones as the problem requires. "
            "Each milestone must be detailed and concrete and produce a named deliverable."
        )
    )
    methodology: str = dspy.OutputField(
        desc=(
            "2-4 sentence description of the overall scientific methodology: "
            "why this sequence of milestones is appropriate for the problem, "
            "what theoretical framework or algorithmic paradigm underpins it, "
            "and how the milestones chain together causally."
        )
    )


class StepHoareConverter(dspy.Signature):
    """Convert a LATS micro-solution into a precise Hoare triple {P} C {Q}.

    Given the original milestone, the LATS winning answer, and the derivation
    that led to it, produce a complete Hoare triple suitable for formal
    verification and code generation.

    Rules
    -----
    - {P} must describe the concrete input state (data types, shapes, values).
    - C must be a FULL Python expression with inputs, algorithm, parameters,
      and output variable assigned — not just a function name.
    - {Q} must contain machine-verifiable assertions (assert statements).
    - Do NOT invent results not supported by the LATS answer/derivation.
    """

    milestone: str = dspy.InputField(desc="The high-level milestone being decomposed")
    lats_answer: str = dspy.InputField(desc="The LATS winning answer — the micro-solution")
    lats_derivation: str = dspy.InputField(desc="Key steps from the LATS trajectory")
    step_number: str = dspy.InputField(desc="Sequential step number (e.g. '2')")
    prior_postcondition: str = dspy.InputField(
        desc="Postcondition of the preceding step (empty string if first step)"
    )

    precondition: str = dspy.OutputField(
        desc="Hoare precondition {P}: complete logical state before this step"
    )
    command: str = dspy.OutputField(
        desc="Hoare command C: full Python expression with all variables and parameters named"
    )
    postcondition: str = dspy.OutputField(
        desc="Hoare postcondition {Q}: output types, shapes, and assert statements"
    )
    tool: str = dspy.OutputField(
        desc="Tool name (e.g. 'arxiv_search', 'python_exec') or empty string for pure reasoning"
    )
    tool_rationale: str = dspy.OutputField(desc="One sentence: why this tool over alternatives")
    reasoning: str = dspy.OutputField(
        desc="2-3 sentences explaining how the LATS answer informed this Hoare triple"
    )


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class OrchestratorResult:
    """Output of PlanOrchestrator — ResearchPlan-compatible."""

    theory:               str  = ""
    research_goal:        str  = ""
    hypothesis:           str  = ""
    methodology:          str  = ""
    architecture_type:    str  = "orchestrated — macro/micro two-phase planning"
    programming_language: str  = "Python 3"
    dependencies:         list = field(default_factory=list)
    data_sources:         list = field(default_factory=list)
    additional_tools:     list = field(default_factory=list)
    steps:                list = field(default_factory=list)
    decomposition_roots:  list = field(default_factory=list)
    htn_network:          Optional[Any] = None
    macro_plan:           str  = ""
    lats_results:         list = field(default_factory=list)


# ── Module ────────────────────────────────────────────────────────────────────

class PlanOrchestrator(dspy.Module):
    """Two-phase hierarchical planner: CoT macro plan + LATS Worker per milestone.

    Phase 1  HighLevelPlanner (ChainOfThought) — single LLM call that produces
             a numbered milestone list.  The LLM decides the count; no limit.

    Phase 2  LATS Worker — runs independently per milestone.  Rollout budget is
             shared proportionally so total cost stays bounded.

    Each LATS result is converted to a Hoare triple {P} C {Q} via
    StepHoareConverter, giving the verification pipeline a fully typed step.
    """

    def __init__(
        self,
        max_rollouts_total:    int       = 30,
        min_rollouts_per_step: int       = 5,
        retriever:             Any | None = None,
        knowledge_graph:       Any | None = None,
        arxiv_n:               int       = 3,
        verbose:               bool      = True,
    ) -> None:
        super().__init__()

        self.max_rollouts_total    = max_rollouts_total
        self.min_rollouts_per_step = min_rollouts_per_step
        self._retriever            = retriever
        self._knowledge_graph      = knowledge_graph
        self._arxiv_n              = arxiv_n
        self.verbose               = verbose

        self.build_macro_plan       = dspy.ChainOfThought(HighLevelPlanner)
        self.decompose_step         = LATS(max_rollouts=min_rollouts_per_step, use_rag=False, verbose=verbose)
        self._decompose_step_rag: LATS | None = None
        self.hoare_converter        = dspy.ChainOfThought(StepHoareConverter)

    def _get_lats(self, use_rag: bool, rollouts: int) -> LATS:
        if use_rag:
            if self._decompose_step_rag is None:
                self._decompose_step_rag = LATS(
                    max_rollouts    = rollouts,
                    use_rag         = True,
                    retriever       = self._retriever,
                    knowledge_graph = self._knowledge_graph,
                    arxiv_n         = self._arxiv_n,
                    verbose         = self.verbose,
                )
            else:
                self._decompose_step_rag.max_rollouts = rollouts
            return self._decompose_step_rag
        self.decompose_step.max_rollouts = rollouts
        return self.decompose_step

    def forward(
        self,
        user_goal: str,
        context:   str  = "",
        use_rag:   bool = False,
    ) -> OrchestratorResult:
        self._log(f"\n{'='*70}")
        self._log(f"  PlanOrchestrator  goal={user_goal[:80]}")
        self._log(f"{'='*70}")

        # Phase 1: macro plan
        self._log("\n[Phase 1] Building macro plan...")
        #Retrieve the context from RAG:
        macro_out   = self.build_macro_plan(user_goal=user_goal, context=context[:4000] if context else "")
        macro_plan  = macro_out.macro_plan
        methodology = getattr(macro_out, "methodology", "")
        self._log(f"  Macro plan:\n{macro_plan}\n")

        milestones = self._parse_milestones(macro_plan) or [user_goal]
        n          = len(milestones)
        self._log(f"  {n} milestone(s) identified.")

        rollouts_per = max(self.min_rollouts_per_step, self.max_rollouts_total // max(1, n))
        self._log(f"  Rollout budget: {rollouts_per}/milestone x {n} = {rollouts_per * n} total")

        # Phase 2: LATS Worker per milestone
        from agents.researcher import InstructionStep  # avoid circular import

        steps:        list[InstructionStep] = []
        lats_results: list[LATSResult]      = []
        prior_postcondition                 = ""
        global_step                         = 0

        macro_context = (
            f"Overall goal: {user_goal}\n\nFull plan:\n{macro_plan}"
            + (f"\n\nBackground:\n{context[:1500]}" if context else "")
        )

        for idx, milestone in enumerate(milestones, start=1):
            self._log(f"\n[Phase 2 - step {idx}/{n}] {milestone[:100]}")

            lats        = self._get_lats(use_rag=use_rag, rollouts=rollouts_per)
            lats_result = lats.forward(problem=milestone, context=macro_context)
            lats_results.append(lats_result)

            self._log(
                f"  LATS  solved={lats_result.solved}  rollouts={lats_result.rollouts}  "
                f"best_value={lats_result.tree_stats.get('best_value', '?')}"
            )

            try:
                hoare          = self.hoare_converter(
                    milestone           = milestone,
                    lats_answer         = lats_result.answer,
                    lats_derivation     = lats_result.derivation,
                    step_number         = str(idx),
                    prior_postcondition = prior_postcondition,
                )
                precondition   = hoare.precondition
                command        = hoare.command
                postcondition  = hoare.postcondition
                tool           = hoare.tool or ""
                tool_rationale = hoare.tool_rationale or ""
                reasoning      = hoare.reasoning or ""
            except Exception as exc:
                self._log(f"  [!] Hoare converter failed: {exc} — using LATS answer directly")
                precondition   = prior_postcondition or "Problem statement is given."
                command        = lats_result.answer[:400]
                postcondition  = lats_result.derivation[:400] or f"Milestone {idx} complete."
                tool = tool_rationale = reasoning = ""

            global_step += 1
            steps.append(InstructionStep(
                step           = global_step,
                task           = milestone,
                dependencies   = [global_step - 1] if global_step > 1 else [],
                precondition   = precondition,
                input_type     = "str",
                tool           = tool,
                tool_rationale = tool_rationale,
                command        = command,
                postcondition  = postcondition,
                reasoning      = reasoning,
                level          = 2,
            ))
            prior_postcondition = postcondition

        solved_count = sum(1 for r in lats_results if r.solved)
        self._log(f"\n[PlanOrchestrator] Done — {len(steps)} step(s)  LATS solved {solved_count}/{n}")

        return OrchestratorResult(
            theory        = "Two-phase hierarchical planning: CoT macro plan + LATS Workers",
            research_goal = user_goal,
            hypothesis    = lats_results[-1].answer[:200] if lats_results else "",
            methodology   = methodology,
            steps         = steps,
            macro_plan    = macro_plan,
            lats_results  = lats_results,
        )

    @staticmethod
    def _parse_milestones(text: str) -> list[str]:
        """Extract numbered milestones from plan text (handles 1. / 1) / Step 1: formats)."""
        lines      = text.strip().splitlines()
        milestones: list[str] = []
        buffer     = ""

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if buffer:
                    milestones.append(buffer.strip())
                    buffer = ""
                continue
            if re.match(r"^(?:step\s*)?\d+[.):\-]\s+\S", stripped, re.IGNORECASE):
                if buffer:
                    milestones.append(buffer.strip())
                buffer = re.sub(r"^(?:step\s*)?\d+[.):\-]\s+", "", stripped, flags=re.IGNORECASE)
            else:
                buffer = (buffer + " " + stripped).strip() if buffer else stripped

        if buffer:
            milestones.append(buffer.strip())
        return [m for m in milestones if m]

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
