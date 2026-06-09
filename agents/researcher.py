import dspy
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from dataclasses import dataclass, field
from agents.knowledge_constructor import HoareTriplet
from agents.tool_registry import TOOL_REGISTRY, registry_names, registry_descriptions
from agents.experience import with_experience, default_store

class Task(BaseModel):
    state:str = Field(description="General description of the current step in the research plan, what is the goal of this steps what will be delivered")
    action:str = Field(description="What kind of action is necessary to do to finish this task")
    description:str = Field(description="Detailed description of task")
    dependencies: List[str] = Field(
        default_factory=list,
        description=(
            "Exact name of the dependencies of object that is necessary to "
        )
    )
    tool: str = Field(
        description=(
            "Exact tool name from the registry (tool_names input field).  "
            "Example: 'qiskit', 'numpy', 'z3_solver'"
        )
    )
    hoare_triplet:HoareTriplet = Field(
        description=(
            "Hoare triplet that describes the precondition, command and postcondition"
        )
    )

class Plan(BaseModel):
    instructions: List[Task] = Field(
        description=(
            "Ordered list of Task steps to execute. Each step names exactly one "
            "tool from the registry (tool_names input field).  "
            "Example tools: 'python_exec', 'arxiv_search', 'z3_solver'"
        )
    )
    
class ToolSelector(dspy.Signature):
    """Select the single best tool for one plan step from the available registry.

    Called only when the plan step has an empty or unrecognized tool field.
    Return exactly one name from tool_names — no explanation, no punctuation.
    """
    task_description: str = dspy.InputField(desc="What this step needs to do")
    task_action:      str = dspy.InputField(desc="The action type of this step")
    tool_names:       str = dspy.InputField(desc="Comma-separated list of valid tool names")
    tool_descriptions: str = dspy.InputField(desc="name: description pairs for every tool")
    tool: str = dspy.OutputField(desc="Exactly one name from tool_names that best fits this step")


class GoalHypothesisExtractor(dspy.Signature):
    """Derive a concrete research goal and a falsifiable hypothesis from a raw question.

    research_goal must be a measurable objective with a clear success criterion.
    hypothesis must be a falsifiable claim in the form
    'We hypothesize that [claim], verifiable by [criterion].'
    """
    question:      str = dspy.InputField(desc="The raw research question or task description")
    research_goal: str = dspy.OutputField(desc="Concrete, measurable research objective")
    hypothesis:    str = dspy.OutputField(desc="Falsifiable hypothesis with a verifiable criterion")


class InstructionPlan(dspy.Signature):
    """
    """
    question: str = dspy.InputField(
        desc="The research problem or question to solve, usually a scientific problem from physics or mathematics"
    )
    context: str = dspy.InputField(
        desc="Retrieved trajectory context (problem / method / result) to draw from"
    )
    tool_names: str = dspy.InputField(
        desc="Comma-separated list of available tool names. Every Task in the plan MUST use exactly one name from this list."
    )
    tool_descriptions: str = dspy.InputField(
        desc="Newline-separated 'name: description' pairs for each available tool. Use these to choose the right tool for each step."
    )
    theory: str = dspy.OutputField(
        desc=(
            "The established scientific theory, mathematical framework, or prior work "
            "that the plan relies on or will validate. "
            "Name it precisely (e.g. 'Atiyah-Hirzebruch spectral sequence', "
            "'VC dimension theory', 'Riemannian geometry on SPD manifolds'). "
            "List the key axioms, theorems, or invariants the plan depends on. "
            "If the plan validates a theory, state what observable evidence confirms or refutes it."
        )
    )
    research_goal: str = dspy.InputField(
        desc=(
            "A concrete, measurable objective whose achievement can be unambiguously "
            "confirmed or refuted by executing the plan. "
            "Specify: (a) what will be computed, proven, or demonstrated; "
            "(b) the success criterion (numeric threshold, formal proof, benchmark score). "
            "Vague goals such as 'understand X' or 'explore Y' are FORBIDDEN. "
            "Example: 'Prove mutual exclusion satisfies AG(~(c1 & c2)) via Z3 "
            "with <=10 CTL properties and zero counterexamples.'"
        )
    )
    hypothesis: str = dspy.InputField(
        desc=(
            "A falsifiable scientific hypothesis directly testable by executing the plan. "
            "Format: 'We hypothesize that [specific claim], which can be verified by "
            "[concrete observable criterion yielded by the plan].' "
            "The claim must be specific enough that a single experiment can decide it. "
            "Example: 'We hypothesize that UMAP(n_components=2) preserves cluster "
            "structure better than t-SNE, verifiable by silhouette score comparison "
            "UMAP > t-SNE at p < 0.05 (paired t-test, n=10 seeds).'"
        )
    )
    methodology: str = dspy.OutputField(
        desc=(
            "The high-level scientific methodology organising the plan. "
            "Name: (a) the paradigm (e.g. 'deductive algebraic proof', "
            "'empirical benchmarking', 'Bayesian inference', 'simulation study'); "
            "(b) its principal components and how they chain together; "
            "(c) how it connects theory -> hypothesis -> verification. "
            "Example: 'Deductive proof via spectral sequences: (1) construct AHSS, "
            "(2) compute differentials, (3) read off bordism groups, "
            "(4) verify generators satisfy expected torsion relations.'"
        )
    )
    research_plan: Plan = dspy.OutputField(
        desc=(
            "Complete updated plan as an ordered list of InstructionStep objects "
            "organised in a THREE-LEVEL HIERARCHY (see MULTI-LEVEL PLAN STRUCTURE above). "
            "THE PLAN MUST BE DIRECTLY EXECUTABLE: "
            "running `python -c \"import plan_exec; plan_exec.run_plan()\"` must succeed "
            "with no errors, no missing imports, and no placeholder bodies. "
            "Each step is an actionable task card AND a Hoare triple. Required fields: "
            "step (int — sequential across all levels), "
            "level (int — 1=Phase, 2=Task, 3=AtomicOp; MANDATORY), "
            "parent_step (int|None — L1→None, L2→L1 step, L3→L2 step; MANDATORY), "
            "task (str — short imperative sentence: '<verb> <object> [using <tool>]'), "
            "dependencies (list[int] — step numbers that must finish first; [] for step 1), "
            "precondition (str — {P}: input data + type + constraints + causal justification), "
            "command (str — C: exact named algorithm with explicit parameters — NO vague phrases), "
            "tool (str — ACTION: exactly one tool name from tool_names, or '' for L1/pure reasoning), "
            "tool_rationale (str — one sentence justifying the tool choice), "
            "postcondition (str — {Q}: concrete output + new invariants + downstream enablement), "
            "input_type (str — formal type + encoding of primary input), "
            "reasoning (str — numbered deductive chain proving {P} ^ C -> {Q}), "
            "frame (str — comma-separated list of state NOT modified by this step), "
            "command_body (str — MANDATORY for L2/L3: non-empty executable Python function "
            "def step_N(context: dict) -> dict; empty string only for L1 phase headers), "
            "verification_conditions (list — MANDATORY ≥1 Python assertion per L2/L3 step), "
            "completion_metric (str — Python assert expression proving step is done), "
            "precondition_resources (list — one PreconditionResource per {P} variable). "
            "Postcondition of step N must appear in precondition of step N+1. "
            "A step with an empty or placeholder command_body is a planning error — rewrite it."
        )
    )

@dataclass
class ResearchPlan:
    """Full output of the Researcher module — framing fields plus executable steps."""
    theory: str = ""
    research_goal: str = ""
    hypothesis: str = ""
    methodology: str = ""
    # ── inferred plan-level properties ────────────────────────────────────────
    architecture_type: str = ""
    programming_language: str = ""
    dependencies: list = field(default_factory=list)      # list[str]
    data_sources: list = field(default_factory=list)      # list[str]
    additional_tools: list = field(default_factory=list)  # list[str]
    # ── type system ───────────────────────────────────────────────────────────
    plan_types: list = field(default_factory=list)         # list[PlanType]
    type_actions: list = field(default_factory=list)       # list[TypeAction]
    type_tool_bindings: list = field(default_factory=list) # list[TypeToolBinding]
    # ── decomposition outputs ──────────────────────────────────────────────────
    steps: list = field(default_factory=list)
    decomposition_roots: list = field(default_factory=list)  # list[HoareTreeNode]  (tree mode)

class Researcher(dspy.Module):
    """
    Builds a ResearchPlan via a single RAG/CAG call.
    """

    def __init__(self, rag=None):
        self.rag = rag
        self.goal_extractor = dspy.Predict(GoalHypothesisExtractor)
        self.researcher     = dspy.ChainOfThought(InstructionPlan)
        self.tool_selector  = dspy.Predict(ToolSelector)

    def _ensure_tools(self, plan: Plan, tool_names: str, tool_descriptions: str) -> None:
        """Guarantee every Task in the plan has a valid registry tool name.

        Iterates over all tasks and, for any step whose tool field is absent or
        not in the registry, calls ToolSelector to pick the right one in-place.
        """
        valid = set(TOOL_REGISTRY.keys())
        for task in plan.instructions:
            if task.tool and task.tool in valid:
                continue
            try:
                pred = self.tool_selector(
                    task_description  = task.description,
                    task_action       = task.action,
                    tool_names        = tool_names,
                    tool_descriptions = tool_descriptions,
                )
                chosen = pred.tool.strip().strip("'\"")
                task.tool = chosen if chosen in valid else next(iter(valid))
            except Exception:
                task.tool = "python_exec"

    @with_experience(store=default_store, enable=True)
    def forward(
        self,
        prompt: str,
        trajectory: list[str]=None,
        vector_db: Optional[Any] = None,
        hybrid_retriever: Optional["HybridRetriever"] = None,
        initial_hoare_context: str = "",
        tool_descriptions: str = "",
        tool_names: str = "",
    ) -> ResearchPlan:
        context_parts: list[str] = []

        context = ""
        retriever = hybrid_retriever or (self.rag if self.rag is not None else None)
        if retriever is not None:
            try:
                if hasattr(retriever, "retrieve"):
                    # HybridRetriever — returns RetrievedContext with .to_text()
                    context = retriever.retrieve(prompt).to_text()
                elif hasattr(retriever, "sample_trajectory_points"):
                    # RAG module — returns list[{problem, method}]
                    hits = retriever.sample_trajectory_points(prompt)
                    context = "\n\n".join(
                        f"Problem: {h['problem']}\nMethod: {h['method']}"
                        for h in hits
                    )
            except Exception as exc:
                print(f"  [researcher] RAG retrieval failed: {exc}")

        if not tool_names:
            tool_names = registry_names()
        if not tool_descriptions:
            tool_descriptions = registry_descriptions()

        goal_pred = self.goal_extractor(question=prompt)
        research_goal = getattr(goal_pred, "research_goal", prompt)
        hypothesis    = getattr(goal_pred, "hypothesis", "")

        res = self.researcher(
            question=prompt,
            context=context,
            tool_names=tool_names,
            tool_descriptions=tool_descriptions,
            research_goal=research_goal,
            hypothesis=hypothesis,
        )
        def _split_lines(raw: str) -> list[str]:
            sep = "\n" if "\n" in raw else ","
            return [item.strip() for item in raw.split(sep) if item.strip()]

        def _as_list(attr: str) -> list:
            val = getattr(res, attr, None) or []
            return val if isinstance(val, list) else []

        if isinstance(res.research_plan, Plan):
            self._ensure_tools(res.research_plan, tool_names, tool_descriptions)

        return ResearchPlan(
            theory               = getattr(res, "theory", ""),
            research_goal        = getattr(res, "research_goal", ""),
            hypothesis           = getattr(res, "hypothesis", ""),
            methodology          = getattr(res, "methodology", ""),
            architecture_type    = getattr(res, "architecture_type", ""),
            programming_language = getattr(res, "programming_language", ""),
            dependencies         = _split_lines(getattr(res, "dependencies", "")),
            data_sources         = _split_lines(getattr(res, "data_sources", "")),
            additional_tools     = _split_lines(getattr(res, "additional_tools", "")),
            plan_types           = _as_list("plan_types"),
            type_actions         = _as_list("type_actions"),
            type_tool_bindings   = _as_list("type_tool_bindings"),
            steps                = res.research_plan if isinstance(res.research_plan, (Plan, list)) else Plan(instructions=[]),
            #research_plan        =       = res,
        )
