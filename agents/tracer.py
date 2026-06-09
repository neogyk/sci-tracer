from __future__ import annotations
import pdb
import hashlib
import logging
import os
import pickle
import uuid
import dspy
import numpy as np
import tqdm
import pdb;

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("scitracer")

from agents.researcher import  (Researcher,
                                InstructionPlan,
                                InstructionStep
)
from agents.rag import HoareDecomposition

from agents.rag import (
    RAG,
    Trajectory,
    KnowledgeGraph,
    ExtractKnowledgeGraph,
    DiscoveredTool,
    ExtractDiscoveredTools,
    HybridRetriever,
    HoareOntologyPipeline,
    GraphStore,
    VectorStore)

from agents.planning.orchestrator import PlanOrchestrator, OrchestratorResult

from tools.file_reader import doc_reader
from tools.kripke import KripkeStructure, VerificationResult, parse_kripke_from_llm
from tools.pub_search import file_reader
from tools.web_search import web_search, github_search, wiki_search, rag_query
from tools.z3_kripke_verifier import Z3KripkeVerifier, Z3DiagnosticReport, verify_kripke_z3
from tools.lean_kripke_verifier import LeanKripkeVerifier, verify_kripke_lean

from core.trajectory_db import TrajectoryVectorDB

from core.tool_registry import _TOOL_REGISTRY, _tool_descriptions_str, _tool_names_str
from core.plan_types import Paper, Plan, Step, PlanRefinement, VerificationReport, _plan_to_str
from core.print_helpers import (
    _parse_structure,
    _print_final_decomposition,
    _print_hoare,
    _print_knowledge_graph,
    _print_kripke,
    _print_verification,
)




class SciTracer(dspy.Module):
    """Main orchestrator for the SciTracer research pipeline.

    Each public method maps to one discrete pipeline step so the caller
    can compose them freely or run the full pipeline via forward().

        tracer = SciTracer()
        literature       = tracer.collect_literature(prompt)

        trajectories     = tracer.make_trajectories(literature)
        
        research_plan    = tracer.make_research_plan(prompt, trajectories)

        formalization    = tracer.formalize_plan(research_plan)
        verification, suggestion = tracer.verify_kripke(research_plan, formalization)
        verification, suggestion = tracer.ast_formalization(research_plan, formalization)
    """

    def __init__(
        self,
        neo4j_uri:      str | None = None,
        neo4j_user:     str | None = None,
        neo4j_password: str | None = None,
    ) -> None:
        super().__init__()
        # Phrases the LLM returns when it cannot extract content from a section.
        _EMPTY_TRAJECTORY_PHRASES = (
            "reference list",
            "list of references",
            "does not provide",
            "does not describe",
            "does not include",
            "does not contain",
            "no clear",
            "not applicable",
        )
    
        # Persistent vector store for prolongation-based RAG
        
        # Retrieval & planning

        
        self.rag = RAG()
        self.vector_db = TrajectoryVectorDB()
        self.knowledge_graph = KnowledgeGraph()
        
        self.kg_extractor  = dspy.ChainOfThought(ExtractKnowledgeGraph)
        self.researcher = Researcher()
        
        #self.executor = dspy.ChainOfThought(Executor)

        # Knowledge graph

        # ── Ontology layer (Neo4j + Qdrant) ───────────────────────────────────
        # GraphStore requires a running Neo4j instance; falls back to None when
        # credentials are absent so the rest of the pipeline continues to work.
        _graph: GraphStore | None = None
        if neo4j_uri and neo4j_user and neo4j_password:
            try:
                _graph = GraphStore(neo4j_uri, neo4j_user, neo4j_password)
            except Exception as exc:
                print(f"  [ontology] GraphStore unavailable: {exc}")
        self._ontology_graph  = _graph
        self._ontology_vector = VectorStore()

        self.ontology_pipeline = HoareOntologyPipeline(
            graph_store=self._ontology_graph,
            vector_store=self._ontology_vector,
        )
        self.hybrid_retriever = HybridRetriever(
            vector_store=self._ontology_vector,
            graph_store=self._ontology_graph,
        )
        

        # Tool discovery from RAG/papers
        self.tool_discoverer = dspy.ChainOfThought(ExtractDiscoveredTools)

        # Agent-based verifier (supplements / replaces formal Kripke when absent)
        self.plan_verifier_agent = PlanVerifierAgent(run_lats=False, verbose=True)

        # Paper relevance filter — runs before downloading full text
        #self._paper_relevance = dspy.Predict(PaperRelevanceCheck)

        # Two-phase orchestrator: CoT macro plan + LATS per milestone
        # use_rag is passed at call time via forward(use_rag=...) — not stored here
        self.plan_orchestrator = PlanOrchestrator(
            max_rollouts_total    = 30,
            min_rollouts_per_step = 5,
            retriever             = self.hybrid_retriever,
            knowledge_graph       = self.knowledge_graph,
            verbose               = True,
        )

        # Synthesis: collects all pipeline outputs → final answer
        self.synthesis_agent = SynthesisAgent()

    # ── Lean formalizer (lazy bootstrap) ─────────────────────────────────────

    @property
    def lean_formalizer(self) -> LeanAutoformalization:
        """Bootstrap the Lean corpus on first access (expensive — seeds + Loogle calls)."""
        if self._lean_formalizer is None:
            print("  [Lean] Bootstrapping corpus (tactics + Loogle queries)…")
            try:
                self._lean_corpus     = bootstrap_corpus(domain="mathematics")
                self._lean_formalizer = LeanAutoformalization(corpus=self._lean_corpus)
                print("  [Lean] Corpus ready.")
            except Exception as exc:
                print(f"  [Lean] Bootstrap failed: {exc} — Lean verification disabled.")
                self._lean_formalizer = None
        return self._lean_formalizer  # type: ignore[return-value]

    # ── Literature API ───────────────────────────────────────────────────────

    _LOCAL_SUFFIXES = {".pdf", ".md", ".txt", ".docx", ".tex"}


    def _collect_from_local(self, prompt: str, folder: str) -> list[Paper]:
        """Read all supported documents from a local folder path."""
        root = Path(folder)
        if not root.exists():
            raise FileNotFoundError(f"Local path does not exist: {folder}")
        if not root.is_dir():
            raise NotADirectoryError(f"Local path is not a directory: {folder}")

        paths = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in self._LOCAL_SUFFIXES
        )
        if not paths:
            print(f"  No supported files found in {folder}")
            return []

        papers: list[Paper] = []
        for path in tqdm.tqdm(paths, desc="Reading local files", unit="file"):
            try:
                if path.suffix.lower() in {".md", ".txt", ".tex"}:
                    text = path.read_text(encoding="utf-8", errors="replace")
                else:
                    # PDF / DOCX — delegate to docling for structured markdown output
                    text = doc_reader(str(path))
                sections = [s.strip() for s in text.split("##") if s.strip()]
                papers.append(Paper(url=str(path), query=prompt, sections=sections))
                tqdm.tqdm.write(f"  Loaded {len(sections)} sections — {path.name}")
            except Exception as exc:
                tqdm.tqdm.write(f"  Skipping {path.name}: {exc}")

        print(f"  Collected {len(papers)} local papers.")
        return papers


    # ── Trajectory API ───────────────────────────────────────────────────────


    def make_trajectories(self, literature: list[Paper]) -> list[Any]:
        """Extract (problem, method, result) trajectories from paper sections.

        Sections are processed concurrently — trajectory extraction and KG extraction
        run in parallel across sections (both are I/O-bound LLM calls).
        Reference-list sections and empty LLM responses are silently skipped.

        Returns a flat list of DSPy Prediction objects with .problem, .method, .result.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Collect candidate (section, query) pairs upfront, skip reference sections
        candidates: list[tuple[str, str]] = []
        n_skipped_ref = 0
        for paper in literature:
            for section in paper.sections:
                if self._is_reference_section(section):
                    n_skipped_ref += 1
                else:
                    candidates.append((section, paper.query))

        total_sections = sum(len(p.sections) for p in literature)
        print(
            f"  Extracting trajectories from {len(candidates)} sections "
            f"({n_skipped_ref} reference sections skipped, "
            f"{total_sections} total, {len(literature)} papers)..."
        )
        self.knowledge_graph.reset()

        # Per-section worker: runs trajectory + KG extraction together
        def _process(section: str, query: str) -> tuple[Any | None, list]:
            traj = None
            rels: list = []
            try:
                t = self.rag.trajectories(paragraph=section, question=query)
                if not self._is_empty_trajectory(t):
                    traj = t
            except Exception:
                pass
            try:
                kg_pred = self.kg_extractor(text=section)
                rels = kg_pred.relationships
            except Exception:
                pass
            return traj, rels

        trajectories: list[Any] = []
        n_empty = 0
        max_workers = min(12, max(1, len(candidates)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_process, sec, qry): (sec, qry)
                for sec, qry in candidates
            }
            for future in tqdm.tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Extracting trajectories",
                unit="section",
            ):
                traj, rels = future.result()
                if traj is None:
                    n_empty += 1
                else:
                    trajectories.append(traj)
                    self.vector_db.store_trajectory(traj)
                if rels:
                    self.knowledge_graph.add(rels)

        print(
            f"  Extracted {len(trajectories)} trajectory points "
            f"({n_empty} empty responses skipped, "
            f"VectorDB total={self.vector_db.trajectory_count()})."
        )
        print(f"  {self.knowledge_graph}")
        _print_knowledge_graph(self.knowledge_graph)
        return trajectories


    def make_research_plan(
        self,
        prompt: str,
        trajectories: list[Any],
        decomposition_mode: str = "tree",
    ) -> ResearchPlan:
        """Build a research plan from trajectory context and VectorDB RAG.

        Trajectory prolongation is intentionally excluded here.  Call
        :meth:`prolong_trajectories` first if you want geodesic-prolonged
        context paragraphs, then pass the result via a custom wrapper.
        By default this method builds context paragraphs directly from the
        raw trajectory text and VectorDB retrieval without prolongation.
        """
        if not trajectories:
            print("  No trajectories — using prompt as the plan.")
            return prompt

        # Convert raw trajectory objects to CAG passages (used as fallback evidence
        # when VectorDB is empty; Researcher will also receive VectorDB RAG hits).
        cag_passages: list[str] = [
            f"Problem: {t.problem}\nMethod: {t.method}\nResult: {t.result}"
            for t in trajectories
            if hasattr(t, "problem")
        ] or [str(t) for t in trajectories]

        hoare_note = (f", {self.vector_db.hoare_count()} Hoare triplets"
                      if self.vector_db.hoare_count() else "")
        print(
            f"  Building research plan  "
            f"trajectories={len(trajectories)}  "
            f"VectorDB={self.vector_db.trajectory_count()}{hoare_note}  "
            f"mode=RAG+CAG (single LLM call)"
        )

        tool_desc = _tool_descriptions_str()
        tool_names = _tool_names_str()
        
        print("Researcher")
        pdb.set_trace()
        _plan = self.researcher.forward(
            prompt, 
            cag_passages,
            vector_db=self.vector_db,
            hybrid_retriever=self.hybrid_retriever,
            initial_hoare_context="",
            tool_descriptions=tool_desc,
            tool_names=tool_names,
        )
        
        steps: list[InstructionStep] = _plan.steps

        # ── Decomposition (mode-dispatched) ───────────────────────────────────
        decomp_roots: list = []
        #TODO having the list of action -> USE the Notion of High-Level Plan and low level plan
        pdb.set_trace()
        htn_network:  HTNNetwork | None = None

        if decomposition_mode == "tree":
            # print(f"\n{sep}")
            # print(f"  TREE DECOMPOSITION  (Hoare sequential composition, branching=2, depth=until-atomic)")
            # print(sep)
            # decomp_roots = self.step_decomposer.decompose_plan(
            #     steps,
            #     hybrid_retriever = self.hybrid_retriever,
            #     tool_registry    = _TOOL_REGISTRY,
            #     trajectory_db    = self.vector_db,
            # )
            # _plan.decomposition_roots = decomp_roots

            # all_leaves    = [node for root in decomp_roots for node in root.leaves()]
            # atomic_count  = sum(1 for n in all_leaves if n.atomicity == "atomic")
            # compound_count= sum(1 for n in all_leaves if n.atomicity == "compound")
            # branch_count  = sum(1 for n in all_leaves if n.atomicity == "branching")
            # print(f"  {len(steps)} top-level steps → {len(all_leaves)} leaf steps  "
            #       f"(atomic={atomic_count}  compound={compound_count}  branching={branch_count})")
            # for root in decomp_roots:
            #     print(root.pprint())

            # ── Formal correctness check: {P} π {Q} with typed Σ and α ────────
            # proof = validate_plan_tree(decomp_roots)
            # thin  = "─" * 70
            # print(f"\n{thin}")
            # print("  PLAN TREE CORRECTNESS  ({P} π {Q}  with Σ, α)")
            # print(thin)
            # st = proof["stats"]
            # print(f"  Roots            : {st['roots']}")
            # print(f"  Total nodes      : {st['total_nodes']}")
            # print(f"  Leaves           : {st['leaves']}")
            # print(f"  Leaves with Σ    : {st['leaves_with_state']}/{st['leaves']}"
            #       f"  (pre_state + post_state typed)")
            # print(f"  Leaves with α    : {st['leaves_with_action']}/{st['leaves']}"
            #       f"  (action_sig present)")
            # print(f"  Composition OK   : {st['composition_ok']} internal nodes")
            # print(f"  Composition FAIL : {st['composition_fail']} internal nodes")
            # if proof["valid"]:
            #     print(f"\n  ✓  ALL PROPERTIES HOLD — tree is a valid {'{P}'} π {'{Q}'}")
            # else:
            #     print(f"\n  ✗  {len(proof['violations'])} VIOLATION(S) FOUND:")
            #     for v in proof["violations"][:10]:
            #         print(f"     {v}")
            #     if len(proof["violations"]) > 10:
            #         print(f"     … and {len(proof['violations']) - 10} more")
            # print(thin)
            # print(sep)

            tree_before_path = self._log_task_tree_md(
                decomp_roots=decomp_roots,
                out_path="task_tree_before_verification.md",
                label="Before Verification",
            )
            print(f"  Task-tree leaf log (before verification) → {tree_before_path}")

        return _plan

    def refine_plan(self, plan_str: str, feedback: str) -> str:
        """Refine the research plan based on explicit human feedback.

        Unlike verify_kripke / verify_z3_pipeline which trigger refinement from
        solver-detected failures, this method accepts arbitrary human-readable
        feedback and produces an improved plan string.

        Args:
            plan_str: Current plan as a string (e.g. from _plan_to_str()).
            feedback: Human feedback or free-text description of what to improve.

        Returns:
            A refined plan string.
        """
        refinement = self.refiner(plan=plan_str, failed_properties=feedback)
        return refinement.refined_plan

    def verify_and_refine_plan(
        self,
        plan: "Plan",
        question: str,
        backend: str = "z3",
        max_retries: int = 3,
    ) -> tuple["Plan", "VerificationReport"]:
        """Verify a structured Plan with Z3 + Kripke + Lean and iteratively
        reformulate it until all verifiers pass or the retry budget is exhausted.

        Algorithm
        ---------
        For each attempt (up to *max_retries*):

          1. Serialize ``plan`` to text and run ``explain_and_verify_plan``.
          2. Collect Z3, Kripke, and Lean failure messages.
          3. If all verifiers pass → return the plan as-is.
          4. Otherwise call ``StructuredPlanRefiner`` with the structured plan
             fields and the precise error messages.
          5. Parse the refiner's ``refined_steps_json`` back into a new ``Plan``
             and continue to the next attempt.

        Parameters
        ----------
        plan:
            The initial ``Plan`` to verify and refine.
        question:
            Original research question (used by Z3 formalization).
        backend:
            Kripke verification backend — ``'z3'`` or ``'lean'``.
        max_retries:
            Maximum verify → refine iterations.

        Returns
        -------
        (refined_plan, report)
            ``refined_plan`` is the last plan that was verified (possibly still
            failing if the budget was exhausted).
            ``report`` is the ``VerificationReport`` from the final attempt.
        """
        import json as _json

        _SEP  = "═" * 68
        _sep  = "─" * 68

        current_plan = plan
        last_report: VerificationReport | None = None

        for attempt in range(1, max_retries + 1):
            print(f"\n{_SEP}")
            print(f"  VERIFY & REFINE  attempt {attempt}/{max_retries}")
            print(_SEP)

            plan_str   = current_plan.to_str()
            flat_steps = current_plan.to_instruction_steps()

            log.info(
                "verify_and_refine_plan: attempt=%d  steps=%d  backend=%s",
                attempt, len(flat_steps), backend,
            )

            # ── Step 1: full verification ──────────────────────────────────────
            (
                explanation,
                z3_result,
                kripke_result,
                lean_results,
                _step_fol, _chain, _type, _snippets,
            ) = self.explain_and_verify_plan(
                research_plan = flat_steps,
                plan_str      = plan_str,
                backend       = backend,
                run_optional  = False,
                question      = question,
            )

            verification = kripke_result["verification"]
            suggestion   = kripke_result["suggestion"]

            last_report = VerificationReport(
                z3          = z3_result,
                kripke      = verification,
                lean        = lean_results,
                chain       = [],
                type_checks = [],
                suggestion  = suggestion,
            )

            log.info("verify_and_refine_plan: attempt=%d  %s", attempt, last_report.summary)

            print(f"\n{_sep}")
            print(f"  STATUS  {last_report.summary}")
            print(_sep)

            if last_report.passed:
                print(f"  [✓] All verifiers passed on attempt {attempt}.")
                break

            if attempt == max_retries:
                print(f"  [✗] Retry budget exhausted — returning best available plan.")
                break

            # ── Step 2: collect structured error strings ───────────────────────
            z3_errors = self._format_z3_errors(z3_result)
            kripke_failures = self._format_kripke_failures(verification)
            lean_errors = self._format_lean_errors(lean_results)

            print(f"\n  Z3      : {z3_errors[:200] or '(none)'}")
            print(f"  Kripke  : {kripke_failures[:200] or '(none)'}")
            print(f"  Lean    : {lean_errors[:200] or '(none)'}")
            print(f"  Suggest : {suggestion[:200] or '(none)'}")

            # ── Step 3: reformulate via StructuredPlanRefiner ──────────────────
            steps_json = _json.dumps(
                [s.model_dump() for s in current_plan.all_steps()],
                ensure_ascii=False,
            )

            print(f"\n  Reformulating plan (attempt {attempt} → {attempt + 1})…")
            try:
                pred = self.plan_refiner(
                    goal             = current_plan.goal,
                    theory           = current_plan.theory,
                    hypothesis       = current_plan.hypothesis,
                    methodology      = current_plan.methodology,
                    plan_steps_json  = steps_json,
                    z3_errors        = z3_errors,
                    kripke_failures  = kripke_failures,
                    lean_errors      = lean_errors,
                    suggestion       = suggestion,
                )
            except Exception as exc:
                log.error("StructuredPlanRefiner failed: %s", exc)
                print(f"  [!] StructuredPlanRefiner failed: {exc} — stopping refinement loop.")
                break

            print(f"\n  Changes:\n    " + pred.change_summary.replace("\n", "\n    "))

            # ── Step 4: reconstruct Plan from refined_steps_json ──────────────
            try:
                raw_steps = _json.loads(pred.refined_steps_json)
                new_steps = [Step(**s) for s in raw_steps]
                # Re-nest sub_steps: keep only root-level steps (no parent implied)
                # The refiner emits a flat list; restore tree via sub_steps field
                current_plan = Plan(
                    goal        = pred.refined_goal or current_plan.goal,
                    theory      = current_plan.theory,
                    hypothesis  = pred.refined_hypothesis or current_plan.hypothesis,
                    methodology = pred.refined_methodology or current_plan.methodology,
                    tool_list   = sorted({s.tool_name for s in new_steps if s.tool_name}),
                    steps       = new_steps,
                )
                log.info(
                    "verify_and_refine_plan: plan reconstructed  steps=%d",
                    len(new_steps),
                )
            except Exception as exc:
                log.error("Plan reconstruction from refined_steps_json failed: %s", exc)
                print(f"  [!] Could not parse refined_steps_json: {exc} — keeping current plan.")

        return current_plan, last_report  # type: ignore[return-value]

    # ── Error formatting helpers ──────────────────────────────────────────────

    @staticmethod
    def _format_z3_errors(z3_result: dict) -> str:
        lines: list[str] = []
        for p in z3_result.get("z3_pipeline", {}).get("paths", []):
            if p.get("pruned") or "ERROR" in str(p.get("result", "")):
                lines.append(f"[{p['id']}] {p.get('result', 'unknown error')[:200]}")
        return "\n".join(lines) if lines else ""

    @staticmethod
    def _format_kripke_failures(verification: list) -> str:
        lines: list[str] = []
        for r in verification:
            if not r.holds and not r.error:
                ce = getattr(r, "counterexample", "") or ""
                lines.append(
                    f"PROPERTY {getattr(r, 'property_name', '?')} failed"
                    + (f" — counterexample: {ce[:150]}" if ce else "")
                )
            elif r.error:
                lines.append(f"ERROR: {r.error[:150]}")
        return "\n".join(lines) if lines else ""

    @staticmethod
    def _format_lean_errors(lean_results: list) -> str:
        lines: list[str] = []
        for r in lean_results:
            if not r.get("verified"):
                err = r.get("error") or r.get("compile_error") or "unknown lean error"
                sid = r.get("step", "?")
                lines.append(f"[step {sid}] {str(err)[:200]}")
        return "\n".join(lines) if lines else ""

    # ── Formalization API ────────────────────────────────────────────────────

    def formalize_plan(self, research_plan: str) -> dict[str, Any]:
        """Formalize the research plan into a Hoare triple and Kripke structure.

        Returns {"hoare": <HoarePrediction>, "kripke": <KripkePrediction>}.
        Both values are raw DSPy Prediction objects whose fields can be
        accessed as attributes (e.g. hoare.precondition, kripke.worlds).
        """
        print("  Extracting Hoare triple {P} C {Q}...")
        hoare = self.hoare_formalizer(research_plan)
        tid = self.vector_db.store_hoare(hoare)
        print(f"  Stored Hoare triple in VectorDB (id={tid[:8]}…, "
              f"total={self.vector_db.hoare_count()}).")
        print("  Extracting Kripke structure M = (S, R, V)...")
        kripke = self.kripke_formalizer(research_plan)
        return {"hoare": hoare, "kripke": kripke}

    # ── Verification API ─────────────────────────────────────────────────────

    def verify_kripke(
        self,
        research_plan: str,
        formalization: dict[str, Any],
        backend: str = "z3",
    ) -> tuple[list[VerificationResult], str]:
        """Verify the Kripke structure and suggest plan improvements.

        Parses the Kripke prediction from formalization, runs CTL verification
        with Z3 or Lean 4, then uses the LLM to propose a refined plan if
        any properties fail.

        Returns:
            verification  list of VerificationResult (one per CTL property)
            suggestion    LLM-generated refinement, or a "all clear" message
        """
        kripke_pred = formalization["kripke"]
        structure = parse_kripke_from_llm(
            worlds_str=kripke_pred.worlds,
            initial=kripke_pred.initial_world,
            relation_str=kripke_pred.accessibility_relation,
            propositions_str=kripke_pred.atomic_propositions,
            valuation_str=kripke_pred.valuation,
        )
        properties = [p.strip() for p in kripke_pred.modal_properties.split(",") if p.strip()]
        print(f"  Verifying {len(properties)} CTL properties with {backend.upper()}...")

        if backend == "lean":
            results = verify_kripke_lean(structure, properties)
            diag_report: Z3DiagnosticReport | None = None
        else:
            results, diag_report = verify_kripke_z3(
                structure, properties, with_diagnostics=True
            )

        failures = [r for r in results if not r.holds and not r.error]
        if failures:
            failed_desc = "; ".join(
                f"{r.property} fails at world '{r.witness or '?'}'"
                + (f" (path: {' → '.join(r.path)})" if r.path else "")
                for r in failures
            )
            print(f"  {len(failures)} {'property' if len(failures) == 1 else 'properties'} failed — generating refinement...")

            # Build rich diagnostic inputs for the refiner
            if diag_report is not None:
                kripke_str    = diag_report.structure_summary
                annot_str     = "\n\n".join(
                    d.to_str() for d in diag_report.failed
                )
                diag_str      = diag_report.to_str()
            else:
                kripke_str = annot_str = diag_str = ""

            refinement = self.refiner(
                plan=research_plan,
                failed_properties=failed_desc,
                kripke_structure=kripke_str,
                annotated_paths=annot_str,
                diagnostic_report=diag_str,
            )
            suggestion = refinement.refined_plan
        else:
            suggestion = "All CTL properties verified — no refinement needed."

        return results, suggestion

    def verify_with_agent(
        self,
        problem:      str,
        plan_str:     str,
        steps:        list[InstructionStep] | None = None,
        final_answer: str = "",
    ) -> tuple[list[AgentVerificationResult], str]:
        """Run agent-based verification over the plan.

        Executes three passes via PlanVerifierAgent:
          1. Per-step Hoare triple soundness  (when steps are provided)
          2. Consecutive chain entailment Q_i ⊨ P_{i+1}
          3. Overall solution correctness (adversarial LLM check)

        Return signature is identical to verify_kripke() — (results, suggestion)
        — so the two are interchangeable in the verify→refine loop.

        Args:
            problem:      The original research question or problem.
            plan_str:     Full plan serialised as a string.
            steps:        Optional list of InstructionStep objects for
                          per-step and chain checks.  When None only the
                          overall solution correctness check runs.
            final_answer: Claimed answer to inject into the solution check.

        Returns:
            results    list[AgentVerificationResult] — one per checked property
            suggestion human-readable refinement hint (empty string if all pass)
        """
        return self.plan_verifier_agent.verify(
            problem=problem,
            plan_str=plan_str,
            steps=steps,
            final_answer=final_answer,
        )

    # ── AST formalization ────────────────────────────────────────────────────

    def ast_formalization(
        self,
        research_plan: str,
        formalization: dict[str, Any],
    ) -> tuple[dict[str, str], str]:
        """AST-based formalization and validation path.

        Converts the research plan to a JSON AST (via Text2AST), then to
        executable Python (via AST2Python), and validates syntax.  Uses
        the Hoare triple from formalization to enrich the suggestion.

        Returns:
            artifact    dict with keys: ast, python_code, valid, compile_error
            suggestion  LLM-generated refinement or confirmation message
        """
        print("  Converting plan to Abstract Syntax Tree...")
        ast_pred = self.ast_agent(text=research_plan)
        ast_str = ast_pred.ast

        print("  Translating AST → Python code...")
        code_pred = self.code_agent(ast=ast_str)
        python_code = str(code_pred.python) if code_pred.python is not None else ""

        compile_error: str | None = None
        try:
            compile(python_code, "<ast_output>", "exec")
        except SyntaxError as exc:
            compile_error = str(exc)

        artifact: dict[str, str] = {
            "ast": ast_str,
            "python_code": python_code,
            "valid": "yes" if compile_error is None else "no",
            "compile_error": compile_error or "",
        }

        if compile_error:
            print("  Syntax error in generated code — generating refinement...")
            refinement = self.refiner(
                plan=research_plan,
                failed_properties=f"Python AST codegen failed: {compile_error}",
            )
            suggestion = refinement.refined_plan
        else:
            hoare = formalization.get("hoare")
            if hoare:
                suggestion = (
                    f"Code verified syntactically.\n"
                    f"  {{P}} {hoare.precondition}\n"
                    f"   C   {hoare.command}\n"
                    f"  {{Q}} {hoare.postcondition}"
                )
            else:
                suggestion = "Code verified syntactically — no refinement needed."

        return artifact, suggestion

    # ── Execution API ────────────────────────────────────────────────────────

    _MAX_STEP_RETRIES = 3

    def execute_plan_stepwise(
        self,
        question: str,
        research_plan: list[InstructionStep],
    ) -> list[dict[str, str]]:
        """Execute the research plan one step at a time, threading context forward.

        When a step fails (FAIL conditions or MISMATCH output) the refiner is
        invoked to diagnose the failure and retry with an improved method, up to
        _MAX_STEP_RETRIES attempts.  Every attempt — including failed ones — is
        printed so the full refinement trace is visible.

        Returns a list of result dicts — one per step — with keys:
            step, method, attempts, conditions_check, result,
            output_assessment, accumulated_context
        """
        sep  = "=" * 70
        thin = "-" * 70
        dash = "·" * 70

        print(f"\n{sep}")
        print(f"  STEP-BY-STEP EXECUTION  ({len(research_plan)} steps, max {self._MAX_STEP_RETRIES} retries)")
        print(sep)

        prior_results = "No prior steps completed yet."
        step_results: list[dict[str, str]] = []

        for s in research_plan:
            print(f"\n{thin}")
            print(f"  Step {s.step}  |  {s.command[:65]}")
            print(f"  Precondition:  {s.precondition[:110]}")
            print(f"  Postcondition: {s.postcondition[:110]}")
            print(thin)

            # ── Attempt 1: plain execution ────────────────────────────────────
            conditions_check    = ""
            result              = ""
            output_assessment   = ""
            accumulated_context = prior_results
            refined_command     = s.command

            try:
                out = self.step_executor(
                    question=question,
                    precondition=s.precondition,
                    command=s.command,
                    postcondition=s.postcondition,
                    prior_results=prior_results,
                )
                conditions_check    = out.conditions_check
                result              = out.result
                output_assessment   = out.output_assessment
                accumulated_context = out.accumulated_context
            except Exception as exc:
                conditions_check    = f"FAIL: exception — {exc}"
                output_assessment   = "MISMATCH: step execution raised an exception"

            passed  = conditions_check.upper().startswith("PASS")
            matched = output_assessment.upper().startswith("MATCH")

            print(f"\n  [Attempt 1]")
            print(f"  Conditions : {'[PASS]' if passed  else '[FAIL]'} {conditions_check}")
            print(f"  Result     :\n{result}")
            print(f"  Assessment : {'[MATCH]' if matched else '[MISMATCH]'} {output_assessment}")

            # ── Refinement loop ───────────────────────────────────────────────
            attempt = 1
            while (not passed or not matched) and attempt < self._MAX_STEP_RETRIES:
                attempt += 1
                prev_conditions_check = conditions_check
                prev_result           = result
                prev_assessment       = output_assessment

                try:
                    ref = self.step_refiner(
                        question=question,
                        precondition=s.precondition,
                        original_command=refined_command,
                        postcondition=s.postcondition,
                        prior_results=prior_results,
                        failed_conditions_check=prev_conditions_check,
                        failed_result=prev_result,
                        failed_assessment=prev_assessment,
                        attempt=str(attempt),
                    )
                    print(f"  Diagnosis        : {ref.diagnosis}")
                    print(f"  Refined command  : {ref.refined_command}")

                    refined_command     = ref.refined_command
                    conditions_check    = ref.conditions_check
                    result              = ref.result
                    output_assessment   = ref.output_assessment
                    accumulated_context = ref.accumulated_context
                except Exception as exc:
                    print(f"  Refiner error: {exc}")
                    conditions_check  = f"FAIL: refiner exception — {exc}"
                    output_assessment = "MISMATCH: refiner raised an exception"
                    break

                passed  = conditions_check.upper().startswith("PASS")
                matched = output_assessment.upper().startswith("MATCH")

                print(f"  Conditions : {'[PASS]' if passed  else '[FAIL]'} {conditions_check}")
                print(f"  Result     :\n{result}")
                print(f"  Assessment : {'[MATCH]' if matched else '[MISMATCH]'} {output_assessment}")

            # ── Final status ──────────────────────────────────────────────────
            final_status = "✓ OK" if (passed and matched) else f"✗ EXHAUSTED after {attempt} attempt(s)"

            prior_results = accumulated_context
            step_results.append({
                "step":                str(s.step),
                "command":             refined_command,
                "attempts":            str(attempt),
                "conditions_check":    conditions_check,
                "result":              result,
                "output_assessment":   output_assessment,
                "accumulated_context": accumulated_context,
            })

        return step_results

    # ── Executor: generate runnable script ────────────────────────────────────

    def execute_to_script(
        self,
        question: str,
        decomp_roots: list | None,
        research_plan: list[InstructionStep],
        framing: dict | None = None,
        output_dir: str = "tasks",
    ) -> Path:
        """Generate an executable Python script for the plan and save it under output_dir.

        The script is named ``tasks/<slug>.py`` where slug is the first 12 hex
        characters of the SHA-256 of the question string.

        Returns the Path to the saved script.
        """
        import hashlib
        from pathlib import Path as _Path

        sep  = "=" * 70
        thin = "-" * 70
        print(f"\n{sep}")
        print("  EXECUTOR  —  generating runnable script")
        print(sep)

        slug      = hashlib.sha256(question.encode()).hexdigest()[:12]
        out_dir   = _Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path  = out_dir / f"{slug}.py"

        tools_str = _tool_descriptions_str()

        # try:
        #     script = self.execution_agent.generate_script(
        #         question          = question,
        #         decomp_roots      = decomp_roots or [],
        #         research_plan     = research_plan,
        #         framing           = framing,
        #         available_tools_str = tools_str,
        #     )
        # except Exception as exc:
        #     print(f"  PlanExecutionAgent failed: {exc} — writing stub script")
        #     script = (
        #         f'"""Executor stub — PlanExecutionAgent failed: {exc}"""\n'
        #         f"# question: {question}\n"
        #         f"def run_task(question=None): return {{}}\n"
        #         f'if __name__ == "__main__": print(run_task())\n'
        #     )

        #out_path.write_text(script, encoding="utf-8")


        return out_path

    # ── Plan report ───────────────────────────────────────────────────────────

    @staticmethod
    def _step_needs_verification(
        step: "InstructionStep",
        exec_result: dict[str, str] | None,
        step_fol: dict[str, Any] | None,
        outgoing_chain: dict[str, Any] | None = None,
        incoming_chain: dict[str, Any] | None = None,
        outgoing_type: dict[str, Any] | None = None,
        incoming_type: dict[str, Any] | None = None,
    ) -> tuple[bool, list[str]]:
        """Return (needs_verification, reasons) for a single step.

        A step is flagged when any of the following hold:
          • execution conditions check failed (FAIL)
          • output assessment mismatched (MISMATCH)
          • step exhausted all retries
          • Z3 pipeline returned unsat or unknown
          • CEGIR found a genuine step inconsistency
          • WP check found the precondition is too weak
          • chain gap from this step to the next (outgoing) or from prior to this (incoming)
          • type mismatch between this step's output and the next step's input
        """
        reasons: list[str] = []

        if exec_result:
            if not exec_result.get("conditions_check", "PASS").upper().startswith("PASS"):
                reasons.append("conditions check FAIL")
            if not exec_result.get("output_assessment", "MATCH").upper().startswith("MATCH"):
                reasons.append("output assessment MISMATCH")
            if int(exec_result.get("attempts", "1")) >= SciTracer._MAX_STEP_RETRIES:
                reasons.append(f"exhausted {exec_result['attempts']} retries")

        if step_fol:
            pipeline = step_fol.get("z3_pipeline", {})
            verdict  = (pipeline.get("final_answer") or "").lower()
            conf     = (pipeline.get("confidence")   or "").lower()
            if "unsat" in verdict:
                reasons.append("Z3 pipeline: unsat")
            elif "unknown" in verdict or "undecided" in verdict:
                reasons.append("Z3 pipeline: unknown/undecided")
            if conf in ("low", "none"):
                reasons.append(f"Z3 confidence: {conf}")
            # CEGIR result
            cegir = step_fol.get("cegir", {})
            if cegir.get("fix_type") == "step_inconsistent":
                reasons.append("CEGIR: step is internally inconsistent")
            # WP check
            wp = step_fol.get("wp_result", "")
            if wp and "WP_GAP" in wp:
                reasons.append("WP check: precondition too weak")

        # Chain gaps (this step → next step)
        if outgoing_chain and outgoing_chain.get("entailment_holds") != "yes":
            gap = outgoing_chain.get("gap_description", "")
            reasons.append(f"chain gap → step {outgoing_chain['step_n1']}: {gap[:60]}")

        # Chain gaps (previous step → this step)
        if incoming_chain and incoming_chain.get("entailment_holds") != "yes":
            gap = incoming_chain.get("gap_description", "")
            reasons.append(f"chain gap from step {incoming_chain['step_n']}: {gap[:60]}")

        # Type mismatch (this step's output → next step)
        if outgoing_type and outgoing_type.get("types_compatible") != "yes":
            conv = outgoing_type.get("conversion_needed", "")
            reasons.append(
                f"type mismatch → step {outgoing_type['step_n1']}"
                + (f" (fix: {conv[:50]})" if conv else "")
            )

        # Type mismatch (previous step → this step's input)
        if incoming_type and incoming_type.get("types_compatible") != "yes":
            conv = incoming_type.get("conversion_needed", "")
            reasons.append(
                f"type mismatch from step {incoming_type['step_n']}"
                + (f" (fix: {conv[:50]})" if conv else "")
            )

        return bool(reasons), reasons

    def _save_plan_report(
        self,
        question: str,
        research_plan: list["InstructionStep"],
        step_results: list[dict[str, str]],
        step_fol_list: list[dict[str, Any]],
        chain_results: list[dict[str, Any]] | None = None,
        type_results: list[dict[str, Any]] | None = None,
        report_dir: str = "./sessions",
    ) -> Path:
        """Persist the detailed plan to Markdown + JSON and print the verification index.

        The Markdown report marks each step as ✓ OK or ⚠ REQUIRES VERIFICATION,
        lists verification reasons inline, and prints a consolidated
        "Verification Required" section at the end.

        Returns the path to the saved Markdown file.
        """
        import datetime
        import json

        out_dir = Path(report_dir)
        out_dir.mkdir(exist_ok=True)
        digest   = hashlib.sha256(question.encode()).hexdigest()[:12]
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path  = out_dir / f"plan_{ts}_{digest}.md"
        json_path = out_dir / f"plan_{ts}_{digest}.json"

        # Build lookup maps indexed by step number
        exec_map = {r["step"]: r for r in step_results}
        fol_map  = {r["step"]: r for r in step_fol_list}

        # Chain/type lookup: outgoing = (step_n→step_n1), incoming = (step_{n-1}→step_n)
        chain_list = chain_results or []
        type_list  = type_results  or []
        chain_out  = {r["step_n"]:  r for r in chain_list}  # step N as source
        chain_in   = {r["step_n1"]: r for r in chain_list}  # step N as target
        type_out   = {r["step_n"]:  r for r in type_list}
        type_in    = {r["step_n1"]: r for r in type_list}

        # ── Determine verification status per step ──────────────────────────
        to_verify: list[tuple[int, list[str]]] = []
        step_meta: list[dict[str, Any]] = []
        for s in research_plan:
            sid  = str(s.step)
            er   = exec_map.get(sid)
            sf   = fol_map.get(sid)
            needs, reasons = self._step_needs_verification(
                s, er, sf,
                outgoing_chain=chain_out.get(sid),
                incoming_chain=chain_in.get(sid),
                outgoing_type=type_out.get(sid),
                incoming_type=type_in.get(sid),
            )
            if needs:
                to_verify.append((s.step, reasons))
            step_meta.append({
                "step":               s.step,
                "task":               s.task,
                "dependencies":       s.dependencies,
                "needs_verification": needs,
                "reasons":            reasons,
                "input_type":         s.input_type,
                "frame":              s.frame,
                "precondition":       s.precondition,
                "tool":               s.tool,
                "tool_rationale":     s.tool_rationale,
                "command":            s.command,
                "postcondition":      s.postcondition,
                "reasoning":          s.reasoning,
                "exec_result":        er or {},
                "fol_axioms":         sf.get("fol_axioms", "") if sf else "",
                "wp_result":          sf.get("wp_result", "") if sf else "",
                "z3_final_answer":    (sf or {}).get("z3_pipeline", {}).get("final_answer", ""),
                "z3_confidence":      (sf or {}).get("z3_pipeline", {}).get("confidence", ""),
                "cegir":              (sf or {}).get("cegir", {}),
                "chain_out":          chain_out.get(sid, {}),
                "chain_in":           chain_in.get(sid, {}),
                "type_out":           type_out.get(sid, {}),
                "type_in":            type_in.get(sid, {}),
            })

        # ── Build Markdown ──────────────────────────────────────────────────
        lines: list[str] = [
            f"# SciTracer Research Plan Report",
            f"",
            f"**Question:** {question}",
            f"**Generated:** {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"**Steps:** {len(research_plan)}  |  "
            f"**Requiring verification:** {len(to_verify)}",
            f"",
            f"---",
            f"",
            f"## Research Plan",
            f"",
        ]

        for m in step_meta:
            flag = "⚠ REQUIRES VERIFICATION" if m["needs_verification"] else "✓ OK"
            deps_str = ", ".join(f"#{d}" for d in m.get("dependencies", [])) or "none"
            lines += [
                f"### Step {m['step']}  [{flag}]",
                f"",
                f"> **Task:** {m.get('task', '')}",
                f"> **Dependencies:** {deps_str}",
                f"",
            ]
            if m["needs_verification"]:
                lines += [
                    f"> **Verification required because:** "
                    + ", ".join(m["reasons"]),
                    f"",
                ]
            tool_display = f"`{m['tool']}`" if m.get("tool") else "*none (pure reasoning)*"
            lines += [
                f"| Field | Value |",
                f"|-------|-------|",
                f"| **Precondition {{P}}** | {m['precondition']} |",
                f"| **Input type** | `{m['input_type']}` |",
                f"| **ACTION (tool)** | {tool_display} |",
                f"| **Tool rationale** | {m.get('tool_rationale', '')} |",
                f"| **Command C** | {m['command']} |",
                f"| **Postcondition {{Q}}** | {m['postcondition']} |",
                f"| **Frame** | `{m['frame'] or 'unspecified'}` |",
                f"| **Reasoning** | {m['reasoning']} |",
            ]
            er = m["exec_result"]
            if er:
                lines += [
                    f"| **Execution result** | {er.get('result', '')[:200]} |",
                    f"| **Conditions check** | {er.get('conditions_check', '')} |",
                    f"| **Output assessment** | {er.get('output_assessment', '')} |",
                    f"| **Attempts** | {er.get('attempts', '1')} |",
                ]
            if m["fol_axioms"]:
                lines += [
                    f"",
                    f"**FOL Axioms:**",
                    f"```",
                    m["fol_axioms"],
                    f"```",
                ]
            if m["wp_result"]:
                wp_mark = "✓" if "WP_SUFFICIENT" in m["wp_result"] else "⚠"
                lines += [f"", f"**WP check:** {wp_mark} `{m['wp_result'][:120]}`"]
            if m["z3_final_answer"]:
                lines += [
                    f"",
                    f"**Z3 verdict:** {m['z3_final_answer']}  "
                    f"*(confidence: {m['z3_confidence']})*",
                ]
            cegir = m.get("cegir", {})
            if cegir.get("fix_type") == "step_inconsistent":
                lines += [
                    f"",
                    f"**CEGIR diagnosis:** {cegir.get('diagnosis', '')}",
                    f"",
                    f"> Refined {{P}}: {cegir.get('refined_precondition', '')}",
                    f"> Refined C:   {cegir.get('refined_command', '')}",
                    f"> Refined {{Q}}: {cegir.get('refined_postcondition', '')}",
                ]
            # Chain entailment results for outgoing edge
            co = m.get("chain_out", {})
            if co:
                chain_mark = "✓" if co.get("entailment_holds") == "yes" else "✗"
                lines += [
                    f"",
                    f"**Chain → Step {co.get('step_n1', '?')}:** "
                    f"{chain_mark} `{co.get('entailment_holds', 'unknown')}`"
                    + (f" — {co['gap_description'][:100]}" if co.get("gap_description") else ""),
                ]
            # Type consistency results for outgoing edge
            to_ = m.get("type_out", {})
            if to_:
                type_mark = "✓" if to_.get("types_compatible") == "yes" else "✗"
                lines += [
                    f"**Type → Step {to_.get('step_n1', '?')}:** "
                    f"{type_mark} `{to_.get('types_compatible', 'unknown')}`"
                    + (f" — {to_['mismatch'][:80]}" if to_.get("mismatch") else ""),
                ]
                if to_.get("conversion_needed"):
                    lines.append(f"> Conversion: `{to_['conversion_needed']}`")
            lines.append("")

        # ── Verification index ──────────────────────────────────────────────
        lines += [
            f"---",
            f"",
            f"## Verification Required",
            f"",
        ]
        if to_verify:
            for step_num, reasons in to_verify:
                lines.append(f"- **Step {step_num}** — {', '.join(reasons)}")
        else:
            lines.append("*All steps passed — no additional verification needed.*")
        lines.append("")

        # ── Step chain analysis table ───────────────────────────────────────
        if chain_list:
            lines += [
                f"---",
                f"",
                f"## Step Chain Analysis",
                f"",
                f"| Pair | LLM Entailment | Z3 Confirmed | Gap |",
                f"|------|---------------|-------------|-----|",
            ]
            for r in chain_list:
                llm = r.get("entailment_holds", "?")
                z3c = "✓" if r.get("confirmed_by_z3") else "✗"
                gap = r.get("gap_description", "")[:80]
                mark = "✓" if llm == "yes" and r.get("confirmed_by_z3") else "✗"
                lines.append(
                    f"| {mark} {r['step_n']}→{r['step_n1']} | {llm} | {z3c} | {gap} |"
                )
            lines.append("")

        # ── Type consistency table ──────────────────────────────────────────
        if type_list:
            lines += [
                f"---",
                f"",
                f"## Type Consistency",
                f"",
                f"| Pair | Compatible | Mismatch | Conversion needed |",
                f"|------|-----------|---------|------------------|",
            ]
            for r in type_list:
                compat = r.get("types_compatible", "?")
                mismatch = r.get("mismatch", "")[:60]
                conv = r.get("conversion_needed", "")[:50]
                mark = "✓" if compat == "yes" else "✗"
                lines.append(
                    f"| {mark} {r['step_n']}→{r['step_n1']} | {compat} | {mismatch} | {conv} |"
                )
            lines.append("")

        md_path.write_text("\n".join(lines), encoding="utf-8")

        # ── JSON dump ───────────────────────────────────────────────────────
        json_path.write_text(
            json.dumps({"question": question, "steps": step_meta}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return md_path


    def execute(self, question: str, plan: str) -> str:
        """Execute the verified research plan and produce a final answer."""
        print("  Executing plan...")
        prediction = self.executor(task=question, plan=plan)
        return prediction.answer

    # ── Knowledge & State API ────────────────────────────────────────────────

    def query_knowledge_graph(self, entity: str) -> list[dict[str, str]]:
        """Return all KG edges where source or target contains entity.

        Args:
            entity: A concept, method, or named entity to look up.

        Returns:
            List of dicts with keys: source, type, target.
        """
        q = entity.lower()
        return [
            {"source": r.source, "type": r.type, "target": r.target}
            for r in self.knowledge_graph.edges
            if q in r.source.lower() or q in r.target.lower()
        ]

    def query_similar(self, question: str, n: int = 5) -> list[str]:
        """Query the VectorDB for Hoare triples most similar to question.

        Useful for quickly surfacing related verified reasoning steps from
        previous pipeline runs that are semantically close to the given question.

        Args:
            question: A natural-language query.
            n:        Maximum number of results to return.

        Returns:
            List of formatted '{P} ... \\nC ... \\n{Q} ...' strings.
        """
        if self.vector_db.hoare_count() == 0:
            return []
        query_emb = self.vector_db.embed(question)
        return self.vector_db.query_hoare(query_emb, n_results=n)

    def status(self) -> dict[str, Any]:
        """Return the current state of all internal stores.

        Useful for monitoring store health before / after pipeline runs.

        Returns a dict with keys:
            trajectory_db_count   — trajectory points in Qdrant
            hoare_db_count        — Hoare triples in Qdrant
            knowledge_graph_nodes — unique entities in the KG
            knowledge_graph_edges — relationships in the KG
            ontology_vector_count — chunks in the ontology VectorStore
            ontology_graph_ready  — True when Neo4j GraphStore is connected
        """
        return {
            "trajectory_db_count":   self.vector_db.trajectory_count(),
            "hoare_db_count":        self.vector_db.hoare_count(),
            "knowledge_graph_nodes": len(self.knowledge_graph.nodes),
            "knowledge_graph_edges": len(self.knowledge_graph),
            "ontology_vector_count": self._ontology_vector.count(),
            "ontology_graph_ready":  self._ontology_graph is not None,
        }

    def reset(self) -> None:
        """Clear all in-memory state for a fresh pipeline run.

        Resets the knowledge graph and reinitialises the ontology VectorStore
        and its dependent objects (HybridRetriever, HoareOntologyPipeline).

        The TrajectoryVectorDB and Hoare triple store are intentionally NOT
        cleared — they accumulate cross-run knowledge for better prolongation.
        """
        self.knowledge_graph.reset()
        self._ontology_vector = VectorStore()
        self.hybrid_retriever = HybridRetriever(
            vector_store=self._ontology_vector,
            graph_store=self._ontology_graph,
        )
        self.ontology_pipeline = HoareOntologyPipeline(
            graph_store=self._ontology_graph,
            vector_store=self._ontology_vector,
        )
        print("  [reset] Knowledge graph and ontology VectorStore cleared.")

    # ── Session API ──────────────────────────────────────────────────────────

    @staticmethod
    def _session_path(question: str, local_path: str | None) -> Path:
        """Auto-generate a session file path from a hash of the inputs."""
        key = f"{question}|{local_path or ''}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        sessions_dir = Path("sessions")
        sessions_dir.mkdir(exist_ok=True)
        return sessions_dir / f"{digest}.pkl"

    _SESSION_VERSION = 2

    def save_session(
        self,
        path: str | Path,
        question: str,
        local_path: str | None,
        literature: list[Paper],
        trajectories: list[Any],
    ) -> None:
        """Pickle literature, trajectories, knowledge graph, and ontology index.

        Session payload (v2):
          version                — format version for forward-compatibility
          question / local_path  — key used to detect a matching session
          literature             — list[Paper] from collect_literature()
          trajectories           — list of DSPy Prediction objects
          knowledge_graph_edges  — list[dict] from KnowledgeGraph.serialize()
          trajectory_db_count    — int: number of points in TrajectoryVectorDB
          ontology_vector_points — list[dict] from VectorStore.dump_points()
        """
        kg_edges       = self.knowledge_graph.serialize()
        traj_count     = self.vector_db.trajectory_count()
        ontology_pts: list[dict] = []
        try:
            ontology_pts = self._ontology_vector.dump_points()
        except Exception:
            pass

        payload = {
            "version":                self._SESSION_VERSION,
            "question":               question,
            "local_path":             local_path,
            "literature":             literature,
            "trajectories":           trajectories,
            "knowledge_graph_edges":  kg_edges,
            "trajectory_db_count":    traj_count,
            "ontology_vector_points": ontology_pts,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(
            f"  Session saved → {path}  "
            f"(papers={len(literature)}, trajectories={len(trajectories)}, "
            f"kg_edges={len(kg_edges)}, traj_db={traj_count}, "
            f"ontology_chunks={len(ontology_pts)})"
        )

    @staticmethod
    def load_session(path: str | Path) -> dict | None:
        """Load a session from a pickle file.

        Returns a dict with keys:
          literature, trajectories, knowledge_graph_edges,
          trajectory_db_count, ontology_vector_points
        or None if the file does not exist or cannot be read.

        Transparently handles v1 sessions (no KG / ontology fields).
        """
        p = Path(path)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                payload = pickle.load(f)
            literature    = payload["literature"]
            trajectories  = payload["trajectories"]
            kg_edges      = payload.get("knowledge_graph_edges", [])
            traj_count    = payload.get("trajectory_db_count", 0)
            ontology_pts  = payload.get("ontology_vector_points", [])
            print(
                f"  Session loaded ← {p}  "
                f"(v{payload.get('version', 1)}: "
                f"papers={len(literature)}, trajectories={len(trajectories)}, "
                f"kg_edges={len(kg_edges)}, "
                f"traj_db_expected={traj_count}, "
                f"ontology_chunks={len(ontology_pts)})"
            )
            return {
                "literature":             literature,
                "trajectories":           trajectories,
                "knowledge_graph_edges":  kg_edges,
                "trajectory_db_count":    traj_count,
                "ontology_vector_points": ontology_pts,
            }
        except Exception as exc:
            print(f"  Could not load session {p}: {exc} — re-running extraction.")
            return None

    def make_research_plan_rhea(
        self,
        prompt: str,
        use_rag: bool = False,
        population_size: int = 8,
        horizon: int = 6,
        n_generations: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        commit_threshold: float = 0.8,
    ):
        """Build a research plan using RHEA (Rolling Horizon Evolutionary Algorithm).

        RHEA maintains a population of P action-sequence chromosomes and evolves
        them over G generations via mutation, crossover, and fitness-guided
        selection.  When use_rag=True, RAG serves two roles:
          • Action Space Buffer — top-K semantically relevant actions at each
            mutation locus (dynamic action pruning).
          • Agent Memory — historical execution logs for (state, action) pairs
            that ground the ForwardModel during fitness evaluation.

        Args:
            prompt:           Research question or problem statement.
            use_rag:          Wire RAG as action buffer + memory calibration.
            population_size:  Number of chromosomes P.
            horizon:          Action sequence length H.
            n_generations:    Evolutionary budget (number of generations).
            mutation_rate:    Per-gene mutation probability.
            crossover_rate:   Crossover probability between parent pairs.
            commit_threshold: Fitness threshold for rolling horizon commit.

        Returns:
            RHEAResult with .answer, .derivation, .best_chromosome, .evolution_stats
        """
        from agents.planning import RHEA
        rhea = RHEA(
            population_size=population_size,
            horizon=horizon,
            n_generations=n_generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            commit_threshold=commit_threshold,
            use_rag=use_rag,
            retriever=self.hybrid_retriever if use_rag else None,
            knowledge_graph=self.knowledge_graph if use_rag else None,
            trajectory_db=self.vector_db if use_rag else None,
            arxiv_n=3 if use_rag else 0,
            verbose=True,
        )
        return rhea.forward(prompt)

    def forward(
        self,
        question: str,
        backend: str = "z3",
        local_path: str | None = None,
        session_file: str | None = None,
        decomposition_mode: str = "tree",
        planner: str = "default",
        tasks_dir: str = "tasks",
    ) -> dict[str, Any]:
        """Run the complete SciTracer pipeline using all fine-grained steps.

        Args:
            question:           Research question or system requirement.
            backend:            Verification backend — 'z3' or 'lean'.
            local_path:         Optional local folder to read papers from.
            session_file:       Path to a pickle session file.  If the file exists,
            decomposition_mode: 'none' | 'tree' | 'htn'
                                'none' — flat plan, no decomposition
                                'tree' — binary Hoare tree (depth=2, branching=2)
                                'htn'  — HTN planning (iterative method application)
                          literature and trajectories are loaded from it and
                          steps 1-2 are skipped.  After a fresh run the results
                          are saved to this path (auto-generated if set to the
                          sentinel value 'auto').

        Returns a dict with keys:
            question, literature, trajectories, research_plan,
            formalization, verification, suggestion, answer
        """
        # Resolve session path
        resolved_session: Path | None = None
        if session_file == "auto":
            resolved_session = self._session_path(question, local_path)
        elif session_file:
            resolved_session = Path(session_file)

        # Try to restore from session (skips steps 1 & 2)
        cached = self.load_session(resolved_session) if resolved_session else None

        if cached:
            literature   = cached["literature"]
            trajectories = cached["trajectories"]
            print(f"\n[1/6] Literature collection  — skipped (session cache)")

            # ── Restore knowledge graph from session ──────────────────────────
            kg_edges = cached.get("knowledge_graph_edges", [])
            if kg_edges:
                self.knowledge_graph.restore(kg_edges)
                print(f"[2/6] Trajectory extraction  — skipped (session cache)")
                print(f"       KnowledgeGraph restored: {self.knowledge_graph}")
            else:
                print(f"[2/6] Trajectory extraction  — skipped (session cache)")
                print(f"       KnowledgeGraph: no edges in session (v1 session)")

            # ── Restore ontology VectorStore from session ─────────────────────
            ontology_pts = cached.get("ontology_vector_points", [])
            if ontology_pts and self._ontology_vector.count() == 0:
                try:
                    self._ontology_vector.restore_points(ontology_pts)
                    print(f"       Ontology VectorStore restored: "
                          f"{self._ontology_vector.count()} chunks")
                except Exception as exc:
                    print(f"       Ontology VectorStore restore failed: {exc}")
            elif self._ontology_vector.count() > 0:
                print(f"       Ontology VectorStore already populated: "
                      f"{self._ontology_vector.count()} chunks")

            # ── Check TrajectoryVectorDB ──────────────────────────────────────
            expected_traj = cached.get("trajectory_db_count", 0)
            actual_traj   = self.vector_db.trajectory_count()
            if actual_traj >= expected_traj > 0:
                print(f"       TrajectoryVectorDB: {actual_traj} points already on disk "
                      f"(expected ≥ {expected_traj}) — skip re-population")
            elif actual_traj < expected_traj:
                print(f"       TrajectoryVectorDB: {actual_traj} points on disk but "
                      f"{expected_traj} expected — re-populating from cached trajectories")
                for t in trajectories:
                    try:
                        self.vector_db.store_trajectory(t)
                    except Exception:
                        pass
        else:
            _lit_src = f"local: {local_path}" if local_path else "internet search"
            print(f"\n[1/6] Literature collection  ({_lit_src})")
            log.info("[1/6] Literature collection  source=%s", _lit_src)
            literature = self.collect_literature(question, local_path)
            log.info("[1/6] Collected %d papers", len(literature))

            print(f"\n[2/6] Trajectory extraction")
            log.info("[2/6] Trajectory extraction  papers=%d", len(literature))
            trajectories = self.make_trajectories(literature)
            log.info("[2/6] Extracted %d trajectories", len(trajectories))

        print(f"\n[2b/6] Tool discovery from trajectories")
        log.info("[2b/6] Tool discovery  trajectories=%d", len(trajectories))
        tools = self.discover_tools_from_trajectories(trajectories, question)
        log.info("[2b/6] Discovered %d tools", len(tools))

        print(f"\n[3/6] Research plan synthesis  (decomposition={decomposition_mode}, planner={planner})")
        log.info("[3/6] Plan synthesis  decomposition=%s  planner=%s", decomposition_mode, planner)

        _lats_result = None   # last LATSResult from orchestrator (for stats)
        _rhea_result = None   # set only when RHEA planner is active

        if planner in ("lats", "lats-rag"):
            # ── Two-phase orchestration: CoT macro + LATS per milestone ──────
            _use_rag = (planner == "lats-rag")

            # Build background context from trajectories for the macro planner
            _orch_context = "\n\n".join(
                str(t) for t in (trajectories or [])
                if t is not None
            )[:4000]

            _orch_result: OrchestratorResult = self.plan_orchestrator.forward(
                user_goal = question,
                context   = _orch_context,
                use_rag   = _use_rag,
            )

            n_steps = len(_orch_result.steps)
            n_milestones = len(_orch_result.lats_results)
            solved_count = sum(1 for r in _orch_result.lats_results if r.solved)
            print(
                f"  Orchestrator: {n_milestones} milestone(s) → {n_steps} step(s)  "
                f"LATS solved {solved_count}/{n_milestones}"
            )

            # Expose the last LATSResult for stats display in main.py
            _lats_result = _orch_result.lats_results[-1] if _orch_result.lats_results else None

            _rp = _orch_result
            plan_str = (
                f"ORCHESTRATED PLAN\n\n"
                f"Macro plan:\n{_orch_result.macro_plan}\n\n"
                + (_plan_to_str(_orch_result.steps) or question)
            )

        elif planner in ("rhea", "rhea-rag"):
            _rhea_result = self.make_research_plan_rhea(
                question,
                use_rag=(planner == "rhea-rag"),
            )
            ch = _rhea_result.best_chromosome
            stats = _rhea_result.evolution_stats
            print(f"  RHEA solved={_rhea_result.solved}  "
                  f"generations={_rhea_result.generations_run}  "
                  f"best_fitness={stats.get('best_fitness', '?'):.3f}  "
                  f"assessment={stats.get('final_assessment', '?')}")
            _search_plan_text = (
                f"RHEA SOLUTION\n\n"
                f"Chromosome: {ch.chromosome_str()}\n\n"
                f"Answer:\n{_rhea_result.answer}\n\n"
                f"Derivation:\n{_rhea_result.derivation}\n\n"
                f"Committed actions: {', '.join(stats.get('committed_actions', []))}"
            )
            _search_step = InstructionStep(
                step=1,
                task=_rhea_result.answer[:300] if _rhea_result.answer else question,
                dependencies=[],
                precondition="Problem statement is given.",
                input_type="str",
                tool="rhea",
                tool_rationale="Generated by Rolling Horizon Evolutionary Algorithm.",
                command=ch.chromosome_str()[:300],
                postcondition=f"Final answer produced: {_rhea_result.answer[:200]}",
                reasoning=_rhea_result.derivation[:300] if _rhea_result.derivation else "",
            )

            class _RheaPlanProxy:
                steps = [_search_step]
                theory = "Evolutionary planning"
                research_goal = question
                hypothesis = _rhea_result.answer[:200] if _rhea_result.answer else ""
                methodology = "Rolling Horizon Evolutionary Algorithm (RHEA)"
                architecture_type = "search"
                programming_language = "Python"
                dependencies: list = []
                data_sources: list = []
                additional_tools: list = []
                decomposition_roots: list = []
                htn_network = None

            _rp = _RheaPlanProxy()
            plan_str = _search_plan_text

        else:
            _rp = self.make_research_plan(
                question, trajectories, decomposition_mode=decomposition_mode
            )

        # Persist after plan synthesis so the session includes the ontology
        # VectorStore chunks (populated by make_research_plan's ontology indexing).
        if resolved_session and not cached:
            self.save_session(
                resolved_session, question, local_path, literature, trajectories
            )
        research_plan  = _rp.steps
        log.info("[3/6] Plan synthesised  steps=%d  planner=%s", len(research_plan), planner)
        plan_framing   = {
            "theory":               _rp.theory,
            "research_goal":        _rp.research_goal,
            "hypothesis":           _rp.hypothesis,
            "methodology":          _rp.methodology,
            "architecture_type":    _rp.architecture_type,
            "programming_language": _rp.programming_language,
            "dependencies":         ", ".join(_rp.dependencies),
            "data_sources":         ", ".join(_rp.data_sources),
            "additional_tools":     ", ".join(_rp.additional_tools),
        }
        if planner not in ("rhea", "rhea-rag"):
            _framing_header = (
                f"THEORY: {plan_framing['theory']}\n"
                f"RESEARCH GOAL: {plan_framing['research_goal']}\n"
                f"HYPOTHESIS: {plan_framing['hypothesis']}\n"
                f"METHODOLOGY: {plan_framing['methodology']}\n\n"
            )
            if planner not in ("lats", "lats-rag"):
                # For lats/lats-rag plan_str was already built by the orchestrator
                plan_str = _framing_header + (_plan_to_str(research_plan) or question)

        _MAX_VERIFY_RETRIES = 3

        print(f"\n[4/6] Verification: Z3 + Kripke + Lean (mandatory); FOL/chain/type (optional)")
        log.info("[4/6] Verification  backend=%s  steps=%d", backend, len(research_plan))
        explanation, z3_result, kripke_result, lean_results, step_fol, chain_results, type_results, code_snippets = (
            self.explain_and_verify_plan(
                research_plan, plan_str, backend,
                run_optional=False,
                decomp_roots=_rp.decomposition_roots or None,
                htn_network=_rp.htn_network,
                question=question,
            )
        )
        formalization = kripke_result["formalization"]
        verification  = kripke_result["verification"]
        suggestion    = kripke_result["suggestion"]
        fol           = z3_result

        _V_SEP  = "─" * 70
        _V_SEP2 = "═" * 70

        def _kripke_absent(results: list) -> bool:
            """True when every Kripke result is an error — formal checking unavailable."""
            return bool(results) and all(r.error for r in results)

        def _run_agent_verification(attempt: int, p_str: str, r_plan: list) -> tuple[list, str]:
            """Run PlanVerifierAgent and return (results, suggestion)."""
            print(f"\n{_V_SEP}")
            print(f"  AGENT VERIFICATION  (attempt {attempt})")
            print(_V_SEP)
            a_results, a_suggestion = self.verify_with_agent(
                problem=question,
                plan_str=p_str,
                steps=r_plan or None,
            )
            _print_verification(a_results, a_suggestion)
            return a_results, a_suggestion

        # Agent verification: always runs; replaces Kripke when all results errored
        if _kripke_absent(verification):
            print(f"  [!] Kripke verification unavailable — using agent verifier as primary gate")
            verification, suggestion = _run_agent_verification(1, plan_str, research_plan)
        else:
            _run_agent_verification(1, plan_str, research_plan)

        # ── Closed verify → refine → re-verify loop ───────────────────────────
        verify_attempts = 1
        kripke_failed   = sum(1 for r in verification if not r.holds and not r.error)

        for _attempt in range(2, _MAX_VERIFY_RETRIES + 1):
            if kripke_failed == 0:
                break
            if not suggestion:
                print(f"\n  [✗] {kripke_failed} propert{'y' if kripke_failed == 1 else 'ies'} failed — no suggestion, stopping loop")
                break

            plan_str = self.refine_plan(plan_str, feedback=suggestion)
            formalization_new = self.formalize_plan(plan_str)
            verification, suggestion = self.verify_kripke(
                research_plan=plan_str,
                formalization=formalization_new,
                backend=backend,
            )
            kripke_failed = sum(1 for r in verification if not r.holds and not r.error)
            formalization = formalization_new
            verify_attempts = _attempt

            print(f"\n{_V_SEP2}")
            print(f"  VERIFY LOOP  {_attempt}/{_MAX_VERIFY_RETRIES}")
            print(_V_SEP2)
            print(f"\n{_V_SEP}")
            print(f"  PLAN  (attempt {_attempt})")
            print(_V_SEP)
            for line in plan_str.splitlines():
                print(f"  {line}")
            print(f"\n{_V_SEP}")
            print(f"  FORMALIZATION  (attempt {_attempt})")
            print(_V_SEP)
            _print_hoare(formalization_new["hoare"])
            _, _ks = _parse_structure(formalization_new)
            _print_kripke(_ks)
            print(f"\n{_V_SEP}")
            print(f"  VERIFICATION  (attempt {_attempt}, Kripke/{backend.upper()})")
            print(_V_SEP)
            _print_verification(verification, suggestion)

            # Agent verification per retry iteration
            if _kripke_absent(verification):
                print(f"  [!] Kripke unavailable — agent verifier is primary gate")
                verification, suggestion = _run_agent_verification(_attempt, plan_str, research_plan)
                kripke_failed = sum(1 for r in verification if not r.holds and not r.error)
            else:
                _run_agent_verification(_attempt, plan_str, research_plan)

            if kripke_failed == 0:
                print(f"\n  [✓] All properties hold — exiting loop after attempt {_attempt}")
            elif _attempt == _MAX_VERIFY_RETRIES:
                print(f"\n  [✗] Budget exhausted — executing best available plan")

        if kripke_failed == 0:
            print(f"\n  All properties hold after {verify_attempts} attempt(s).")
            log.info("[4/6] Verification PASSED  attempts=%d", verify_attempts)
        else:
            print(
                f"\n  {kripke_failed} propert{'y' if kripke_failed == 1 else 'ies'} "
                f"still failing after {verify_attempts} attempt(s) — executing best plan."
            )
            log.warning(
                "[4/6] Verification PARTIAL  failed_props=%d  attempts=%d",
                kripke_failed, verify_attempts,
            )

        print(f"\n[5/6] Hoare formalization (VectorDB storage)")
        log.info("[5/6] Hoare formalization")
        hoare_formalization = self.formalize_plan(plan_str)
        formalization["hoare"] = hoare_formalization["hoare"]

        print(f"\n[6/6] Code generation  (Step2CodeExecutorAgent → one .py file per step)")
        log.info("[6/6] Code generation  steps=%d", len(research_plan))
        import hashlib as _hl2
        _q_slug = _hl2.sha256(question.encode()).hexdigest()[:12]
        exec_result = self.step2code_executor(
            question     = question,
            steps        = research_plan,
            output_dir   = tasks_dir,
            session_slug = _q_slug,
        )
        executor_script_path = exec_result["output_dir"]

        log.info(
            "[6/6] Code generation done  n_steps=%d  n_errors=%d  output_dir=%s",
            exec_result["n_steps"], exec_result["n_errors"], executor_script_path,
        )
        for a in exec_result["artifacts"]:
            if a["error"]:
                log.error("[6/6] Step %s codegen failed  task=%r", a["step"], a["task"])
            else:
                log.debug("[6/6] Step %s → %s", a["step"], a["path"])

        # Build step_results in the shape expected by main.py display logic
        step_results = [
            {
                "step":               str(a["step"]),
                "command":            a["task"],
                "conditions_check":   "ERROR" if a["error"] else "PASS",
                "output_assessment":  "ERROR" if a["error"] else "MATCH",
                "accumulated_context": a["path"],
                "attempts":           1,
            }
            for a in exec_result["artifacts"]
        ]
        answer = (
            "\n".join(a["path"] for a in exec_result["artifacts"])
            if exec_result["artifacts"] else plan_str
        )

        # ── Synthesis: collect all pipeline outputs → final answer ──────────────
        print(f"\n[7/6] Synthesis")
        synthesis_result: SynthesisResult = self.synthesis_agent.synthesize(
            question          = question,
            research_plan     = plan_str,
            step_results      = step_results,
            verification      = verification,
            suggestion        = suggestion,
            code_artifacts    = exec_result,
            execution_context = "",
        )
        print(synthesis_result.to_str())
        log.info(
            "[7/6] Synthesis done  confidence=%s  gaps=%s",
            synthesis_result.confidence_level,
            synthesis_result.gaps[:80],
        )

        report_path = self._save_plan_report(
            question=question,
            research_plan=research_plan,
            step_results=step_results,
            step_fol_list=step_fol,
            chain_results=chain_results,
            type_results=type_results,
        )

        plan_md_path = self._save_plan_md(
            question, research_plan, output_path="plan.md", framing=plan_framing
        )
        plan_py_path = self._save_plan_py(
            question, research_plan, code_snippets,
            output_path="plan_exec.py", framing=plan_framing,
        )
        print(f"  Instruction plan (Markdown) → {plan_md_path}")
        print(f"  Instruction plan (Python)   → {plan_py_path}")
        log.info("Pipeline complete  plan_md=%s  plan_py=%s  answer_lines=%d",
                 plan_md_path, plan_py_path, len(answer.splitlines()))

        # ── Final decomposition forest with verification badges ───────────────
        _print_final_decomposition(
            decomp_roots       = _rp.decomposition_roots,
            decomposition_mode = decomposition_mode,
            htn_network        = _rp.htn_network,
            lean_results       = lean_results,
            z3_result          = z3_result,
        )

        return {
            "question":              question,
            "literature":            literature,
            "trajectories":          trajectories,
            "knowledge_graph":       self.knowledge_graph,
            "research_plan":         research_plan,
            "plan_framing":          plan_framing,
            "explanation":           explanation,
            "z3_result":             z3_result,
            "kripke_result":         kripke_result,
            "lean_results":          lean_results,
            "fol":                   fol,
            "step_fol":              step_fol,
            "chain_results":         chain_results,
            "type_results":          type_results,
            "code_snippets":         code_snippets,
            "formalization":         formalization,
            "verification":          verification,
            "suggestion":            suggestion,
            "verify_attempts":       verify_attempts,
            "step_results":          step_results,
            "answer":                synthesis_result.final_answer,
            "synthesis":             synthesis_result,
            "report_path":           str(report_path),
            "plan_md_path":          str(plan_md_path),
            "plan_py_path":          str(plan_py_path),
            "executor_script_path":  str(executor_script_path),
            # LATS-specific (None when planner != lats/lats-rag)
            "lats_tree_stats":       _lats_result.tree_stats if _lats_result else None,
            "lats_rollouts":         _lats_result.rollouts if _lats_result else None,
            "lats_solved":           _lats_result.solved if _lats_result else None,
            # RHEA-specific (None when planner != rhea/rhea-rag)
            "rhea_evolution_stats":  _rhea_result.evolution_stats if _rhea_result else None,
            "rhea_generations":      _rhea_result.generations_run if _rhea_result else None,
            "rhea_solved":           _rhea_result.solved if _rhea_result else None,
        }