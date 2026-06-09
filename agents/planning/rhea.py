"""RHEA — Rolling Horizon Evolutionary Algorithm for Task Planning.

Formalisation
-------------
An agent must discover a sequence of H high-level actions to achieve a
multi-step scientific objective.  RHEA frames this as evolutionary search:

  Individual (Chromosome)  S = [a_0, a_1, …, a_{H-1}]
    Each a_i is drawn from a dynamic action space grounded by RAG retrieval.

  Population               P competing chromosomes evaluated each generation.

  Forward Model            LLM-guided simulator: given (s_k, a_k) → s_{k+1}.
    When use_rag=True, the RAG memory store is queried with (s_k, a_k) to
    ground the simulator and prevent hallucinated state transitions.

  Fitness Function         LLM evaluator scoring the simulated final state
    against the objective across correctness, safety, cost, and efficiency.

RAG has two distinct roles
--------------------------
1. Action Space Buffer (Dynamic Action Pruning)
   At each mutation locus t:
     q = Embed(s_t ∥ G)
     top-K = VectorDB.query(q)          ← context-aware action candidates
   The top-K slice becomes the mutation buffer for that gene.

2. Agent Memory (Experience Replay & Rollout Calibration)
   At population initialisation:
     seed_plans = MemoryDB.query(objective)   ← shift-buffer init from history

   During fitness evaluation of chromosome S:
     For each (s_k, a_k): context = MemoryDB.query(s_k ∥ a_k)
     This context is fed to the ForwardModel, grounding it in real execution logs.

Rolling Horizon
---------------
After each generation, if the best chromosome's first action exceeds
commit_threshold, that action is committed and the horizon shifts by one:
  S_new = [a_1, …, a_{H-1}, a_new_H]

DSPy wiring
-----------
All LLM calls are DSPy modules (independently optimisable):
  ActionCandidateGen    — propose one action for a gene locus
  ForwardModel          — simulate s_{k+1} from (s_k, a_k, memory_context)
  FitnessEvaluator      — score a full chromosome trajectory
  ChromosomeInitializer — warm-start a chromosome from a memory-retrieved plan
  AnswerExtractor       — synthesise final answer from the winning chromosome
"""
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import dspy


# ── DSPy Signatures ───────────────────────────────────────────────────────────

class ActionCandidateGen(dspy.Signature):
    """Propose the next action for a specific gene locus in a chromosome.

    You are given the current simulated state at locus t, the overall
    objective, and a buffer of top-K semantically relevant candidate actions
    retrieved from the action database.

    Rules
    -----
    - Prefer actions from the action_buffer when they fit the current state.
    - Output exactly one action in one of the standard forms:
        COMPUTE[<formula or algorithm>]
        DERIVE[<algebraic / logical step>]
        SEARCH[<retrieval query>]
        VERIFY[<correctness claim>]
        SYNTHESIZE[<integration of prior results>]
        FINISH[<final answer>]
    - FINISH is only valid at the last locus (horizon_remaining == 0).
    - Make the action as specific and concrete as possible.
    """
    problem:           str = dspy.InputField(desc="The scientific problem to solve")
    current_state:     str = dspy.InputField(desc="Simulated state at the current locus")
    objective:         str = dspy.InputField(desc="Overall goal / success criterion")
    horizon_remaining: str = dspy.InputField(desc="Number of action slots remaining after this one")
    prior_actions:     str = dspy.InputField(desc="Actions already committed in this chromosome")
    action_buffer:     str = dspy.InputField(
        desc="Top-K candidate actions retrieved from the action database; "
             "use these as the primary mutation pool"
    )
    action: str = dspy.OutputField(
        desc="One concrete action in the form COMPUTE[...], DERIVE[...], "
             "SEARCH[...], VERIFY[...], SYNTHESIZE[...], or FINISH[...]"
    )


class ForwardModel(dspy.Signature):
    """Predict the next state after executing an action from the current state.

    When memory_context is non-empty it contains historical execution logs of
    the same (state, action) pair retrieved from the RAG memory store.  Use
    this grounding to make the state transition as accurate as possible and
    to avoid hallucinating plausible-but-incorrect outcomes.
    """
    problem:        str = dspy.InputField(desc="The scientific problem")
    current_state:  str = dspy.InputField(desc="State before the action")
    action:         str = dspy.InputField(desc="Action being executed")
    memory_context: str = dspy.InputField(
        desc="Historical execution logs for (current_state, action) from RAG memory; "
             "empty string if RAG memory is disabled or no match found"
    )
    next_state:  str = dspy.OutputField(
        desc="Simulated state after executing the action; be concrete about what "
             "variables, results, or knowledge are now available"
    )
    state_delta: str = dspy.OutputField(
        desc="One sentence: what specifically changed between current_state and next_state"
    )


class FitnessEvaluator(dspy.Signature):
    """Score a complete chromosome (action sequence + simulated trajectory).

    Evaluate on four dimensions, then combine into a single score:
      Correctness  — does the final state answer the problem correctly?   (0–0.4)
      Safety       — no harmful or irreversible operations?               (0–0.2)
      Efficiency   — does the plan avoid redundant steps?                 (0–0.2)
      Completeness — does the final state fully satisfy the objective?    (0–0.2)

    Return a float in [0.0, 1.0] in the 'score' field.
    """
    problem:               str = dspy.InputField(desc="The scientific problem")
    objective:             str = dspy.InputField(desc="The goal / success criterion")
    chromosome:            str = dspy.InputField(desc="Action sequence [a_0, a_1, ..., a_{H-1}]")
    simulated_trajectory:  str = dspy.InputField(
        desc="Full interleaved sequence: s_0, a_0, s_1, a_1, …, s_H"
    )
    memory_context:        str = dspy.InputField(
        desc="RAG memory context used during simulation; helps calibrate the score"
    )
    score: str = dspy.OutputField(
        desc="Float in [0.0, 1.0]; use the full range — be discriminative"
    )
    reasoning: str = dspy.OutputField(
        desc="2–3 sentences covering correctness, safety, efficiency, completeness"
    )
    completion_assessment: str = dspy.OutputField(
        desc="'solved' | 'partial' | 'failed' — overall quality verdict"
    )


class ChromosomeInitializer(dspy.Signature):
    """Warm-start a chromosome from a historically successful plan.

    Given a retrieved historical plan that solved a similar problem, adapt
    it to the current problem by rewriting actions that are problem-specific
    while keeping the structural skeleton intact.
    """
    problem:          str = dspy.InputField(desc="The current problem to solve")
    historical_plan:  str = dspy.InputField(desc="Retrieved historical plan (trajectory string)")
    horizon:          str = dspy.InputField(desc="Required number of actions H")
    actions: str = dspy.OutputField(
        desc="Comma-separated list of exactly H adapted actions in standard form "
             "COMPUTE[...] | DERIVE[...] | SEARCH[...] | VERIFY[...] | "
             "SYNTHESIZE[...] | FINISH[...].  Last action should be FINISH[...]."
    )


class AnswerExtractor(dspy.Signature):
    """Synthesise the final answer from the best evolutionary chromosome.

    Given the winning chromosome and its full simulated trajectory, extract
    a clean, self-contained answer with supporting derivation.
    """
    problem:              str = dspy.InputField(desc="The original problem")
    best_chromosome:      str = dspy.InputField(desc="Winning action sequence")
    simulated_trajectory: str = dspy.InputField(desc="Full simulated trajectory of the winner")
    answer:     str = dspy.OutputField(desc="Final answer, clearly stated")
    derivation: str = dspy.OutputField(
        desc="Key steps from the trajectory that support and verify the answer"
    )


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RHEAChromosome:
    """One individual in the RHEA population."""
    actions:  list[str]            # gene sequence [a_0, …, a_{H-1}]
    states:   list[str]            # simulated states [s_0, …, s_H]
    fitness:  float = 0.0
    generation: int = 0
    assessment: str = "pending"    # 'solved' | 'partial' | 'failed' | 'pending'
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def horizon(self) -> int:
        return len(self.actions)

    def trajectory_str(self) -> str:
        """Interleaved s_0, a_0, s_1, a_1, … s_H string for LLM prompts."""
        parts: list[str] = []
        for i, (s, a) in enumerate(zip(self.states, self.actions)):
            parts.append(f"State {i}: {s}")
            parts.append(f"Action {i}: {a}")
        if len(self.states) > len(self.actions):
            parts.append(f"State {len(self.actions)}: {self.states[-1]}")
        return "\n".join(parts)

    def chromosome_str(self) -> str:
        return " → ".join(self.actions)

    def is_solved(self) -> bool:
        return self.assessment == "solved"


@dataclass
class RHEAResult:
    """Result returned by RHEA.forward()."""
    answer:           str
    derivation:       str
    best_chromosome:  RHEAChromosome
    generations_run:  int
    population_size:  int
    solved:           bool
    evolution_stats:  dict[str, Any] = field(default_factory=dict)


# ── Retriever (shared with LATS, re-exported here for independence) ───────────

class RHEARetriever:
    """Retrieval backend serving two roles in RHEA:

    Role 1 — Action Space Buffer
      retrieve_actions(state, objective) → top-K candidate action strings
      Used during mutation to restrict the search to semantically relevant moves.

    Role 2 — Memory for Rollout Calibration
      retrieve_memory(state, action) → historical execution context
      Used during fitness evaluation to ground the ForwardModel.
    """

    def __init__(
        self,
        retriever:        Any | None = None,
        knowledge_graph:  Any | None = None,
        trajectory_db:    Any | None = None,
        arxiv_n:          int = 3,
        top_k:            int = 8,
        max_kg_edges:     int = 20,
    ) -> None:
        self.retriever       = retriever
        self.knowledge_graph = knowledge_graph
        self.trajectory_db   = trajectory_db   # TrajectoryVectorDB for memory role
        self.arxiv_n         = arxiv_n
        self.top_k           = top_k
        self.max_kg_edges    = max_kg_edges

    # ── Role 1: Action Space Buffer ───────────────────────────────────────────

    def retrieve_actions(self, state: str, objective: str) -> str:
        """Return a block of candidate actions relevant to (state, objective)."""
        query = f"{state} | goal: {objective}"
        parts: list[str] = []

        if self.retriever is not None:
            try:
                ctx = self.retriever.retrieve(query)
                text = ctx.to_text()
                if text.strip():
                    parts.append(f"[Vector+Graph actions]\n{text}")
            except Exception as exc:
                parts.append(f"[RAG error: {exc}]")

        if self.knowledge_graph is not None:
            try:
                edges = self.knowledge_graph.query(objective)
                if not edges:
                    words = [w for w in objective.lower().split() if len(w) > 3]
                    seen: set[str] = set()
                    for word in words:
                        for e in self.knowledge_graph.query(word):
                            key = f"{e.source}|{e.type}|{e.target}"
                            if key not in seen:
                                seen.add(key)
                                edges.append(e)
                if edges:
                    lines = [
                        f"DERIVE[{e.source} --{e.type}--> {e.target}]"
                        for e in edges[: self.max_kg_edges]
                    ]
                    parts.append(f"[KG-derived actions]\n" + "\n".join(lines))
            except Exception as exc:
                parts.append(f"[KG error: {exc}]")

        if self.arxiv_n > 0:
            try:
                from tools.pub_search import arxiv_search
                results = list(arxiv_search(objective, n=self.arxiv_n))
                if results:
                    lines = [
                        f"SEARCH[{title[:80]}]"
                        for _, title, _ in results[: self.arxiv_n]
                    ]
                    parts.append(f"[ArXiv candidate actions]\n" + "\n".join(lines))
            except Exception as exc:
                parts.append(f"[ArXiv error: {exc}]")

        return "\n\n".join(parts) if parts else "(no actions retrieved)"

    # ── Role 2: Memory / Rollout Calibration ──────────────────────────────────

    def retrieve_memory(self, state: str, action: str) -> str:
        """Return historical execution logs for (state, action) pair."""
        query = f"state: {state[:200]} | action: {action}"
        parts: list[str] = []

        if self.trajectory_db is not None:
            try:
                hits = self.trajectory_db.query_trajectories(query, role="method", n=3)
                if hits:
                    lines = [f"  [{h.get('role','?')}] {h.get('text','')[:300]}"
                             for h in hits]
                    parts.append("[Memory — trajectory DB]\n" + "\n".join(lines))
            except Exception as exc:
                parts.append(f"[Memory error: {exc}]")

        if self.retriever is not None:
            try:
                ctx = self.retriever.retrieve(query)
                text = ctx.to_text()
                if text.strip():
                    parts.append(f"[Memory — vector+graph]\n{text[:600]}")
            except Exception as exc:
                parts.append(f"[Memory retrieval error: {exc}]")

        return "\n\n".join(parts) if parts else ""

    # ── Historical plan initialisation (shift-buffer) ─────────────────────────

    def retrieve_historical_plans(self, objective: str, n: int = 3) -> list[str]:
        """Retrieve historical plans for population seed initialisation."""
        if self.trajectory_db is None and self.retriever is None:
            return []
        plans: list[str] = []
        if self.trajectory_db is not None:
            try:
                hits = self.trajectory_db.query_trajectories(objective, role="method", n=n)
                for h in hits:
                    text = h.get("text", "")
                    if text.strip():
                        plans.append(text[:600])
            except Exception:
                pass
        if self.retriever is not None and len(plans) < n:
            try:
                ctx = self.retriever.retrieve(objective)
                text = ctx.to_text()
                if text.strip():
                    plans.append(text[:600])
            except Exception:
                pass
        return plans[:n]

    @property
    def enabled(self) -> bool:
        return (self.retriever is not None
                or self.knowledge_graph is not None
                or self.trajectory_db is not None)


# ── RHEA Module ───────────────────────────────────────────────────────────────

class RHEA(dspy.Module):
    """Rolling Horizon Evolutionary Algorithm for scientific task planning.

    Parameters
    ----------
    population_size      Number of chromosomes P in the population.
    horizon              Length H of each chromosome (number of actions).
    n_generations        Maximum number of evolutionary generations.
    mutation_rate        Probability of mutating each gene (action) slot.
    crossover_rate       Probability of applying single-point crossover.
    elite_fraction       Fraction of top chromosomes preserved intact each gen.
    commit_threshold     Min fitness for the rolling horizon to commit a_0.
    verbose              Print generation-by-generation progress.

    RAG parameters (all optional)
    --------------------------------
    use_rag          Enable both RAG roles (action buffer + memory).
    retriever        HybridRetriever instance (vector + Neo4j graph).
    knowledge_graph  KnowledgeGraph instance (in-memory edge store).
    trajectory_db    TrajectoryVectorDB for memory / shift-buffer init.
    arxiv_n          ArXiv papers per retrieval call (0 = disabled).
    top_k            Size K of the action buffer slice per mutation locus.
    """

    def __init__(
        self,
        population_size:  int   = 8,
        horizon:          int   = 6,
        n_generations:    int   = 10,
        mutation_rate:    float = 0.3,
        crossover_rate:   float = 0.5,
        elite_fraction:   float = 0.25,
        commit_threshold: float = 0.8,
        verbose:          bool  = True,
        use_rag:          bool  = False,
        retriever:        Any | None = None,
        knowledge_graph:  Any | None = None,
        trajectory_db:    Any | None = None,
        arxiv_n:          int   = 3,
        top_k:            int   = 8,
    ) -> None:
        super().__init__()
        self.population_size  = population_size
        self.horizon          = horizon
        self.n_generations    = n_generations
        self.mutation_rate    = mutation_rate
        self.crossover_rate   = crossover_rate
        self.elite_fraction   = elite_fraction
        self.commit_threshold = commit_threshold
        self.verbose          = verbose
        self.use_rag          = use_rag

        self._retriever = RHEARetriever(
            retriever=retriever       if use_rag else None,
            knowledge_graph=knowledge_graph if use_rag else None,
            trajectory_db=trajectory_db     if use_rag else None,
            arxiv_n=arxiv_n          if use_rag else 0,
            top_k=top_k,
        )

        # DSPy modules
        self._action_gen    = dspy.ChainOfThought(ActionCandidateGen)
        self._forward_model = dspy.ChainOfThought(ForwardModel)
        self._fitness_eval  = dspy.ChainOfThought(FitnessEvaluator)
        self._chr_init      = dspy.ChainOfThought(ChromosomeInitializer)
        self._answer_ext    = dspy.ChainOfThought(AnswerExtractor)

        # Evolution bookkeeping
        self._committed_actions: list[str] = []
        self._best_fitness_per_gen: list[float] = []

    # ── Chromosome initialisation ─────────────────────────────────────────────

    def _init_chromosome_random(self, problem: str, initial_state: str) -> RHEAChromosome:
        """Generate a random chromosome using ActionCandidateGen at each locus."""
        actions: list[str] = []
        states = [initial_state]
        for t in range(self.horizon):
            buf = (self._retriever.retrieve_actions(states[-1], problem)
                   if self.use_rag else "(RAG disabled)")
            is_last = (t == self.horizon - 1)
            resp = self._action_gen(
                problem=problem,
                current_state=states[-1],
                objective=problem,
                horizon_remaining=str(self.horizon - 1 - t),
                prior_actions=" → ".join(actions) if actions else "(none)",
                action_buffer=buf if not is_last else "FINISH[<answer>]",
            )
            action = resp.action.strip()
            if is_last and not action.upper().startswith("FINISH"):
                action = f"FINISH[{action}]"
            actions.append(action)
            # Simulate forward one step (lightweight: no memory context yet)
            fwd = self._forward_model(
                problem=problem,
                current_state=states[-1],
                action=action,
                memory_context="",
            )
            states.append(fwd.next_state.strip())
        return RHEAChromosome(actions=actions, states=states)

    def _init_chromosome_from_memory(
        self, problem: str, initial_state: str, historical_plan: str
    ) -> RHEAChromosome:
        """Warm-start a chromosome by adapting a retrieved historical plan."""
        resp = self._chr_init(
            problem=problem,
            historical_plan=historical_plan,
            horizon=str(self.horizon),
        )
        raw = resp.actions
        # Parse comma-separated actions
        actions = [a.strip() for a in raw.split(",") if a.strip()]
        # Pad or truncate to exactly horizon
        while len(actions) < self.horizon:
            actions.append(f"FINISH[{problem}]")
        actions = actions[: self.horizon]
        # Simulate the chromosome to build state sequence
        states = [initial_state]
        for action in actions:
            fwd = self._forward_model(
                problem=problem,
                current_state=states[-1],
                action=action,
                memory_context="",
            )
            states.append(fwd.next_state.strip())
        return RHEAChromosome(actions=actions, states=states)

    def _initialize_population(
        self, problem: str, initial_state: str
    ) -> list[RHEAChromosome]:
        """Build the initial population, seeding from memory when RAG is active."""
        population: list[RHEAChromosome] = []

        if self.use_rag:
            n_seeded = max(1, self.population_size // 4)
            hist_plans = self._retriever.retrieve_historical_plans(problem, n=n_seeded)
            if self.verbose and hist_plans:
                print(f"  [RHEA] Seeding {len(hist_plans)} chromosome(s) from memory.")
            for plan in hist_plans:
                if len(population) >= n_seeded:
                    break
                try:
                    ch = self._init_chromosome_from_memory(problem, initial_state, plan)
                    population.append(ch)
                except Exception as exc:
                    if self.verbose:
                        print(f"  [RHEA] Memory seed failed: {exc}")

        while len(population) < self.population_size:
            population.append(self._init_chromosome_random(problem, initial_state))

        return population

    # ── Fitness evaluation ────────────────────────────────────────────────────

    def _simulate_and_score(
        self, chromosome: RHEAChromosome, problem: str, initial_state: str
    ) -> RHEAChromosome:
        """Re-simulate a chromosome using the ForwardModel and score it."""
        states  = [initial_state]
        mem_ctx_parts: list[str] = []

        for action in chromosome.actions:
            mem_ctx = ""
            if self.use_rag:
                mem_ctx = self._retriever.retrieve_memory(states[-1], action)
                if mem_ctx:
                    mem_ctx_parts.append(f"({action[:40]}…): {mem_ctx[:200]}")
            fwd = self._forward_model(
                problem=problem,
                current_state=states[-1],
                action=action,
                memory_context=mem_ctx,
            )
            states.append(fwd.next_state.strip())

        # Aggregate memory context for fitness evaluator
        combined_mem = "\n".join(mem_ctx_parts[:5]) if mem_ctx_parts else ""
        chromosome.states = states

        traj_str = chromosome.trajectory_str()
        resp = self._fitness_eval(
            problem=problem,
            objective=problem,
            chromosome=chromosome.chromosome_str(),
            simulated_trajectory=traj_str,
            memory_context=combined_mem,
        )

        try:
            chromosome.fitness = float(resp.score.strip())
        except ValueError:
            import re
            m = re.search(r"[01]?\.\d+|\d+", resp.score)
            chromosome.fitness = float(m.group()) if m else 0.0
        chromosome.fitness = max(0.0, min(1.0, chromosome.fitness))
        chromosome.assessment = resp.completion_assessment.strip().lower()
        return chromosome

    # ── Genetic operators ─────────────────────────────────────────────────────

    def _mutate(
        self,
        chromosome: RHEAChromosome,
        problem: str,
    ) -> RHEAChromosome:
        """Mutate individual gene loci using the action buffer as mutation pool."""
        new_actions = list(chromosome.actions)
        new_states  = list(chromosome.states)

        for t in range(self.horizon):
            if random.random() > self.mutation_rate:
                continue
            # Retrieve context-aware action candidates for this locus
            buf = (self._retriever.retrieve_actions(new_states[t], problem)
                   if self.use_rag else "(RAG disabled)")
            is_last = (t == self.horizon - 1)
            resp = self._action_gen(
                problem=problem,
                current_state=new_states[t],
                objective=problem,
                horizon_remaining=str(self.horizon - 1 - t),
                prior_actions=" → ".join(new_actions[:t]) if t > 0 else "(none)",
                action_buffer=buf if not is_last else "FINISH[<answer>]",
            )
            new_action = resp.action.strip()
            if is_last and not new_action.upper().startswith("FINISH"):
                new_action = f"FINISH[{new_action}]"
            new_actions[t] = new_action
            # Re-simulate from locus t onward
            mem_ctx = (self._retriever.retrieve_memory(new_states[t], new_action)
                       if self.use_rag else "")
            fwd = self._forward_model(
                problem=problem,
                current_state=new_states[t],
                action=new_action,
                memory_context=mem_ctx,
            )
            # Rebuild states from t+1 onward (later states become stale — lazy re-sim)
            if t + 1 < len(new_states):
                new_states[t + 1] = fwd.next_state.strip()

        return RHEAChromosome(
            actions=new_actions,
            states=new_states,
            generation=chromosome.generation,
        )

    @staticmethod
    def _crossover(
        parent_a: RHEAChromosome,
        parent_b: RHEAChromosome,
    ) -> tuple[RHEAChromosome, RHEAChromosome]:
        """Single-point crossover between two chromosomes."""
        H = parent_a.horizon
        if H < 2:
            return parent_a, parent_b
        point = random.randint(1, H - 1)
        child_a_actions = parent_a.actions[:point] + parent_b.actions[point:]
        child_b_actions = parent_b.actions[:point] + parent_a.actions[point:]
        # States are now partially stale — fitness evaluation will re-simulate
        child_a = RHEAChromosome(
            actions=child_a_actions,
            states=parent_a.states[:point + 1] + parent_b.states[point + 1:],
        )
        child_b = RHEAChromosome(
            actions=child_b_actions,
            states=parent_b.states[:point + 1] + parent_a.states[point + 1:],
        )
        return child_a, child_b

    # ── Selection ─────────────────────────────────────────────────────────────

    def _select(self, population: list[RHEAChromosome]) -> list[RHEAChromosome]:
        """Elitism: keep the top elite_fraction; fill the rest with tournament pairs."""
        population = sorted(population, key=lambda c: c.fitness, reverse=True)
        n_elite = max(1, int(self.elite_fraction * self.population_size))
        elite = population[:n_elite]
        # Tournament selection for the rest
        offspring: list[RHEAChromosome] = []
        while len(offspring) < self.population_size - n_elite:
            a, b = random.sample(population[:max(2, n_elite * 2)], 2)
            offspring.append(a if a.fitness >= b.fitness else b)
        return elite + offspring

    # ── Rolling horizon commit ────────────────────────────────────────────────

    def _rolling_commit(
        self,
        population: list[RHEAChromosome],
    ) -> tuple[Optional[str], str]:
        """Commit the first action if the best chromosome exceeds commit_threshold.

        Returns (committed_action, initial_state_for_next_generation).
        """
        best = max(population, key=lambda c: c.fitness)
        if best.fitness >= self.commit_threshold and best.horizon > 0:
            committed = best.actions[0]
            next_state = best.states[1] if len(best.states) > 1 else best.states[0]
            return committed, next_state
        return None, best.states[0]

    # ── Main entry point ──────────────────────────────────────────────────────

    def forward(self, problem: str, context: str = "") -> RHEAResult:
        """Run RHEA to find an action sequence solving the problem.

        Args:
            problem: The scientific problem / research question.
            context: Optional pre-fetched context string (e.g. from upstream RAG).

        Returns:
            RHEAResult with the best answer, derivation, and evolution statistics.
        """
        initial_state = (
            f"Initial state: problem='{problem[:200]}'"
            + (f"\nContext: {context[:300]}" if context else "")
        )

        if self.verbose:
            print(f"\n[RHEA] Population={self.population_size}  "
                  f"Horizon={self.horizon}  Generations={self.n_generations}  "
                  f"RAG={'on' if self.use_rag else 'off'}")

        # ── Initialise population ─────────────────────────────────────────────
        if self.verbose:
            print(f"[RHEA] Initialising population…")
        population = self._initialize_population(problem, initial_state)

        # ── Evaluate initial fitness ──────────────────────────────────────────
        population = [self._simulate_and_score(c, problem, initial_state)
                      for c in population]

        best_overall  = max(population, key=lambda c: c.fitness)
        current_state = initial_state

        # ── Evolutionary loop ─────────────────────────────────────────────────
        for gen in range(1, self.n_generations + 1):
            best_gen = max(population, key=lambda c: c.fitness)
            self._best_fitness_per_gen.append(best_gen.fitness)

            if self.verbose:
                solved_n = sum(1 for c in population if c.is_solved())
                print(f"  Gen {gen:>3}/{self.n_generations}  "
                      f"best={best_gen.fitness:.3f}  "
                      f"mean={sum(c.fitness for c in population)/len(population):.3f}  "
                      f"solved={solved_n}/{len(population)}")

            # Early exit if we found a solved chromosome
            if best_gen.is_solved() and best_gen.fitness >= self.commit_threshold:
                best_overall = best_gen
                if self.verbose:
                    print(f"  [RHEA] Solved at generation {gen}.")
                break

            # ── Genetic operators ─────────────────────────────────────────────
            population = self._select(population)

            new_population: list[RHEAChromosome] = []
            i = 0
            while len(new_population) < self.population_size:
                parent = population[i % len(population)]
                if (random.random() < self.crossover_rate
                        and len(population) >= 2):
                    partner = population[(i + 1) % len(population)]
                    child_a, child_b = self._crossover(parent, partner)
                    new_population.extend([
                        self._mutate(child_a, problem),
                        self._mutate(child_b, problem),
                    ])
                else:
                    new_population.append(self._mutate(parent, problem))
                i += 1
            population = new_population[: self.population_size]

            # ── Re-evaluate fitness ───────────────────────────────────────────
            population = [self._simulate_and_score(c, problem, current_state)
                          for c in population]

            # ── Rolling horizon commit ────────────────────────────────────────
            committed, current_state = self._rolling_commit(population)
            if committed:
                self._committed_actions.append(committed)
                if self.verbose:
                    print(f"  [RHEA] Committed: {committed[:60]}")

            best_gen = max(population, key=lambda c: c.fitness)
            if best_gen.fitness > best_overall.fitness:
                best_overall = best_gen

        # ── Extract final answer ──────────────────────────────────────────────
        if self.verbose:
            print(f"\n[RHEA] Best fitness: {best_overall.fitness:.3f}  "
                  f"assessment={best_overall.assessment}")

        resp = self._answer_ext(
            problem=problem,
            best_chromosome=best_overall.chromosome_str(),
            simulated_trajectory=best_overall.trajectory_str(),
        )

        n_gen_run = len(self._best_fitness_per_gen)
        return RHEAResult(
            answer=resp.answer.strip(),
            derivation=resp.derivation.strip(),
            best_chromosome=best_overall,
            generations_run=n_gen_run,
            population_size=self.population_size,
            solved=best_overall.is_solved(),
            evolution_stats={
                "best_fitness":           best_overall.fitness,
                "best_fitness_per_gen":   self._best_fitness_per_gen,
                "committed_actions":      self._committed_actions,
                "final_assessment":       best_overall.assessment,
                "generations_to_solve":   next(
                    (i + 1 for i, f in enumerate(self._best_fitness_per_gen)
                     if f >= self.commit_threshold),
                    None,
                ),
            },
        )
