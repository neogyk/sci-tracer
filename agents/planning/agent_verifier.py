"""PlanVerifierAgent — Agent-driven plan and solution verification.

Complements (or replaces when absent) the formal Kripke/Z3 verifiers by
running three LLM-reasoning passes over a research plan:

  1. Step soundness    — {P_i} C_i {Q_i} is logically valid for every step
  2. Chain entailment  — Q_i ⊨ P_{i+1} for every consecutive pair
  3. Solution check    — adversarial LATS search for counterexamples

The return type mirrors tools.kripke.VerificationResult so agent results
plug directly into _print_verification and the verify→refine loop without
any changes to the display or gating logic.

Typical usage
-------------
    from agents.planning.agent_verifier import PlanVerifierAgent
    verifier = PlanVerifierAgent()
    results, suggestion = verifier.verify(problem, plan_str, steps)
    # results is list[AgentVerificationResult]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import dspy

from agents.planning.lats import LATS


# ── DSPy Signatures ───────────────────────────────────────────────────────────

class StepSoundnessSig(dspy.Signature):
    """Verify that a single Hoare-triple step is logically sound.

    A step {P} C {Q} is sound when:
      • Every variable mentioned in {Q} is either already in {P} or
        is produced by executing command C.
      • Command C is sufficient to transform any state satisfying {P}
        into a state satisfying {Q} — with no missing sub-steps.
      • There are no hidden assumptions in C that are not declared in {P}.

    Return PASS if the triple is sound, FAIL otherwise.  When FAIL, provide
    a concrete counterexample: a state that satisfies {P} but after executing
    C does NOT satisfy {Q}.
    """
    problem:       str = dspy.InputField(desc="The original research problem")
    step_number:   str = dspy.InputField(desc="Step index (e.g. '3')")
    precondition:  str = dspy.InputField(desc="{P}: what must hold before this step")
    command:       str = dspy.InputField(desc="C: the exact command / algorithm to execute")
    postcondition: str = dspy.InputField(desc="{Q}: what must hold after this step")

    verdict:        str = dspy.OutputField(desc="Exactly PASS or FAIL")
    reasoning:      str = dspy.OutputField(desc="Why the triple is sound or unsound")
    counterexample: str = dspy.OutputField(
        desc="Concrete state satisfying {P} that violates {Q} after C (empty if PASS)"
    )


class ChainEntailmentSig(dspy.Signature):
    """Verify that step N's postcondition entails step N+1's precondition.

    Q_N ⊨ P_{N+1} means: every state that satisfies Q_N also satisfies P_{N+1}.
    Equivalently, the output of step N provides everything step N+1 requires.

    Return YES when entailment holds, NO when there is a gap.
    When NO, name the specific variable or fact that Q_N does not provide
    but P_{N+1} requires.
    """
    problem:          str = dspy.InputField(desc="The original research problem")
    step_n:           str = dspy.InputField(desc="Step N index")
    postcondition_n:  str = dspy.InputField(desc="{Q_N}: postcondition of step N")
    step_m:           str = dspy.InputField(desc="Step N+1 index")
    precondition_m:   str = dspy.InputField(desc="{P_{N+1}}: precondition of step N+1")

    entails:  str = dspy.OutputField(desc="Exactly YES or NO")
    gap:      str = dspy.OutputField(
        desc="The specific variable/fact missing from Q_N that P_{N+1} needs (empty if YES)"
    )
    reasoning: str = dspy.OutputField(desc="Step-by-step justification of the entailment verdict")


class SolutionCorrectnessSig(dspy.Signature):
    """Perform an adversarial correctness check of the full solution.

    You are acting as a critical reviewer.  Given the problem and the complete
    solution plan, determine whether:
      (a) the plan actually solves the stated problem, and
      (b) the final answer follows necessarily from the plan steps.

    Try actively to find a flaw.  If you cannot find one after careful
    examination, conclude CORRECT.

    Confidence is your subjective certainty (0.0 = wild guess, 1.0 = certain).
    """
    problem:      str = dspy.InputField(desc="The original research problem")
    plan_str:     str = dspy.InputField(desc="The full plan or solution trace")
    final_answer: str = dspy.InputField(desc="The answer claimed by the plan (may be empty)")

    verdict:     str = dspy.OutputField(desc="One of: CORRECT, INCORRECT, UNCERTAIN")
    confidence:  str = dspy.OutputField(desc="Decimal confidence in [0.0, 1.0]")
    explanation: str = dspy.OutputField(desc="Detailed justification of the verdict")
    suggestion:  str = dspy.OutputField(
        desc="How to repair the plan/answer if INCORRECT or UNCERTAIN; empty if CORRECT"
    )


# ── Result types (mirror tools.kripke.VerificationResult interface) ───────────

@dataclass
class AgentVerificationResult:
    """Agent-produced verification result.

    Fields mirror tools.kripke.VerificationResult so agent results slot into
    _print_verification, the verify→refine loop, and VerificationReport.kripke
    without any downstream changes.
    """
    property:  str           # human-readable description of what was checked
    holds:     bool          # True = check passed
    witness:   str | None = None   # counterexample or gap (replaces Kripke witness)
    path:      list[str] | None = None  # step path relevant to the failure
    error:     str | None = None   # set if the agent call itself failed

    # Extra agent-only fields (ignored by generic display code)
    confidence:  float = 1.0
    verdict:     str   = ""    # PASS / FAIL / CORRECT / INCORRECT / UNCERTAIN / YES / NO
    reasoning:   str   = ""

    def __str__(self) -> str:
        status = "HOLDS" if self.holds else "FAILS"
        parts  = [f"{self.property}: {status}"]
        if self.witness:
            parts.append(f"witness={self.witness[:80]}")
        if self.error:
            parts.append(f"error={self.error}")
        return ", ".join(parts)


@dataclass
class AgentVerificationReport:
    """Full structured report from PlanVerifierAgent."""
    step_results:  list[AgentVerificationResult]   # per-step soundness
    chain_results: list[AgentVerificationResult]   # per-pair entailment
    solution:      AgentVerificationResult | None  # overall solution check
    lats_result:   Any | None = None               # LATSResult if adversarial ran
    suggestion:    str = ""

    @property
    def all_results(self) -> list[AgentVerificationResult]:
        out = list(self.step_results) + list(self.chain_results)
        if self.solution is not None:
            out.append(self.solution)
        return out

    @property
    def passed(self) -> bool:
        return all(r.holds for r in self.all_results if not r.error)

    @property
    def summary(self) -> str:
        total   = len(self.all_results)
        passed  = sum(1 for r in self.all_results if r.holds and not r.error)
        errors  = sum(1 for r in self.all_results if r.error)
        status  = "PASSED" if self.passed else "FAILED"
        return (f"[AGENT {status}] {passed}/{total} checks passed  errors={errors}")


# ── Main agent module ─────────────────────────────────────────────────────────

class PlanVerifierAgent(dspy.Module):
    """Agent-driven plan verifier.

    Runs three verification passes:
      1. step_soundness  — per-step Hoare triple check via StepSoundnessSig
      2. chain_entailment — consecutive Q_i ⊨ P_{i+1} check via ChainEntailmentSig
      3. solution_check  — overall correctness via SolutionCorrectnessSig, optionally
                           followed by an adversarial LATS counterexample search

    Parameters
    ----------
    run_lats          Run LATS adversarial search in addition to the LLM checks.
    lats_rollouts     Number of LATS rollouts for the adversarial check.
    lats_candidates   Branching factor for LATS.
    verbose           Print progress to stdout.
    """

    def __init__(
        self,
        run_lats:        bool = False,
        lats_rollouts:   int  = 10,
        lats_candidates: int  = 2,
        verbose:         bool = True,
    ) -> None:
        super().__init__()
        self.run_lats        = run_lats
        self.lats_rollouts   = lats_rollouts
        self.lats_candidates = lats_candidates
        self.verbose         = verbose

        self.step_checker    = dspy.ChainOfThought(StepSoundnessSig)
        self.chain_checker   = dspy.ChainOfThought(ChainEntailmentSig)
        self.solution_check  = dspy.ChainOfThought(SolutionCorrectnessSig)

        if run_lats:
            self.lats = LATS(
                max_depth=4,
                n_candidates=lats_candidates,
                max_rollouts=lats_rollouts,
                verbose=verbose,
            )
        else:
            self.lats = None

    # ── Public API ────────────────────────────────────────────────────────────

    def verify(
        self,
        problem:      str,
        plan_str:     str,
        steps:        list | None = None,
        final_answer: str = "",
    ) -> tuple[list[AgentVerificationResult], str]:
        """Run all verification passes and return (results, suggestion).

        Return signature matches verify_kripke() so the caller can use
        either interchangeably.

        Parameters
        ----------
        problem       The original research problem.
        plan_str      Full plan as a string (from _plan_to_str or raw text).
        steps         Optional list of InstructionStep objects.  When provided,
                      step-soundness and chain-entailment are checked per-step.
                      When None, only the overall solution check is run.
        final_answer  Claimed answer; injected into the solution correctness check.
        """
        report = self._run(problem, plan_str, steps, final_answer)
        suggestion = report.suggestion
        return report.all_results, suggestion

    def verify_full(
        self,
        problem:      str,
        plan_str:     str,
        steps:        list | None = None,
        final_answer: str = "",
    ) -> AgentVerificationReport:
        """Like verify() but returns the full structured AgentVerificationReport."""
        return self._run(problem, plan_str, steps, final_answer)

    # ── Internal pipeline ─────────────────────────────────────────────────────

    def _run(
        self,
        problem:      str,
        plan_str:     str,
        steps:        list | None,
        final_answer: str,
    ) -> AgentVerificationReport:
        _SEP  = "─" * 70

        self._log(f"\n{_SEP}")
        self._log(f"  AGENT VERIFICATION")
        self._log(_SEP)

        step_results:  list[AgentVerificationResult] = []
        chain_results: list[AgentVerificationResult] = []
        solution_res:  AgentVerificationResult | None = None
        lats_result    = None
        suggestions:   list[str] = []

        # ── Pass 1: Per-step Hoare triple soundness ───────────────────────────
        if steps:
            self._log(f"\n  [1/3] Step soundness  ({len(steps)} steps)")
            for step in steps:
                r = self._check_step(problem, step)
                step_results.append(r)
                mark = "✓" if r.holds else "✗"
                self._log(f"    [{mark}] Step {step.step}: {r.verdict}  {r.reasoning[:80]}")
                if not r.holds and r.witness:
                    suggestions.append(
                        f"Step {step.step} unsound: {r.witness[:200]}"
                    )
        else:
            self._log(f"\n  [1/3] Step soundness  — skipped (no structured steps)")

        # ── Pass 2: Chain entailment Q_i ⊨ P_{i+1} ───────────────────────────
        if steps and len(steps) > 1:
            self._log(f"\n  [2/3] Chain entailment  ({len(steps) - 1} pairs)")
            for i in range(len(steps) - 1):
                r = self._check_chain(problem, steps[i], steps[i + 1])
                chain_results.append(r)
                mark = "✓" if r.holds else "✗"
                self._log(
                    f"    [{mark}] Step {steps[i].step}→{steps[i+1].step}: "
                    f"{'entails' if r.holds else 'GAP — ' + r.witness[:60]}"
                )
                if not r.holds and r.witness:
                    suggestions.append(
                        f"Gap between step {steps[i].step} and {steps[i+1].step}: {r.witness[:200]}"
                    )
        else:
            self._log(f"\n  [2/3] Chain entailment  — skipped (fewer than 2 steps)")

        # ── Pass 3: Overall solution correctness ──────────────────────────────
        self._log(f"\n  [3/3] Solution correctness check")
        solution_res = self._check_solution(problem, plan_str, final_answer)
        mark = "✓" if solution_res.holds else "✗"
        self._log(
            f"    [{mark}] verdict={solution_res.verdict}  "
            f"confidence={solution_res.confidence:.2f}"
        )
        self._log(f"    explanation: {solution_res.reasoning[:120]}")
        if not solution_res.holds:
            suggestions.append(solution_res.witness or solution_res.reasoning[:300])

        # ── Optional: LATS adversarial counterexample search ─────────────────
        if self.lats is not None and not solution_res.holds:
            self._log(f"\n  [+] Adversarial LATS search for counterexample...")
            lats_problem = (
                f"Find a concrete counterexample that disproves the following solution:\n\n"
                f"Problem: {problem}\n\n"
                f"Claimed solution:\n{plan_str[:800]}\n\n"
                f"If no counterexample exists, answer FINISH[NO_COUNTEREXAMPLE_FOUND]. "
                f"If you find one, answer FINISH[COUNTEREXAMPLE: <describe it>]."
            )
            try:
                lats_result = self.lats.forward(lats_problem)
                ce_found = "COUNTEREXAMPLE" in lats_result.answer.upper()
                self._log(
                    f"    LATS: {'counterexample found' if ce_found else 'no counterexample found'}  "
                    f"solved={lats_result.solved}  rollouts={lats_result.rollouts}"
                )
                if ce_found:
                    suggestions.append(f"LATS counterexample: {lats_result.answer[:300]}")
            except Exception as exc:
                self._log(f"    [LATS error] {exc}")

        # ── Build suggestion ──────────────────────────────────────────────────
        if suggestions:
            suggestion = "Agent verification issues:\n" + "\n".join(
                f"  • {s}" for s in suggestions
            )
        else:
            suggestion = "Agent verification passed — no issues found."

        self._log(f"\n  Summary: {self._quick_summary(step_results, chain_results, solution_res)}")
        self._log(_SEP)

        return AgentVerificationReport(
            step_results=step_results,
            chain_results=chain_results,
            solution=solution_res,
            lats_result=lats_result,
            suggestion=suggestion,
        )

    # ── Per-check helpers ─────────────────────────────────────────────────────

    def _check_step(self, problem: str, step: Any) -> AgentVerificationResult:
        prop = f"step_{step.step}_soundness: {{{step.task[:60]}}}"
        try:
            pred = self.step_checker(
                problem=problem,
                step_number=str(step.step),
                precondition=step.precondition,
                command=step.command,
                postcondition=step.postcondition,
            )
            verdict = pred.verdict.strip().upper()
            holds   = verdict.startswith("PASS")
            return AgentVerificationResult(
                property=prop,
                holds=holds,
                witness=pred.counterexample.strip() or None,
                verdict=verdict,
                reasoning=pred.reasoning.strip(),
            )
        except Exception as exc:
            return AgentVerificationResult(
                property=prop,
                holds=False,
                error=str(exc),
                verdict="ERROR",
            )

    def _check_chain(self, problem: str, step_n: Any, step_m: Any) -> AgentVerificationResult:
        prop = f"chain_{step_n.step}→{step_m.step}_entailment"
        try:
            pred = self.chain_checker(
                problem=problem,
                step_n=str(step_n.step),
                postcondition_n=step_n.postcondition,
                step_m=str(step_m.step),
                precondition_m=step_m.precondition,
            )
            verdict = pred.entails.strip().upper()
            holds   = verdict.startswith("YES")
            return AgentVerificationResult(
                property=prop,
                holds=holds,
                witness=pred.gap.strip() or None,
                verdict=verdict,
                reasoning=pred.reasoning.strip(),
            )
        except Exception as exc:
            return AgentVerificationResult(
                property=prop,
                holds=False,
                error=str(exc),
                verdict="ERROR",
            )

    def _check_solution(
        self,
        problem:      str,
        plan_str:     str,
        final_answer: str,
    ) -> AgentVerificationResult:
        prop = "solution_correctness"
        try:
            pred = self.solution_check(
                problem=problem,
                plan_str=plan_str[:2000],
                final_answer=final_answer or "(not yet computed)",
            )
            verdict    = pred.verdict.strip().upper()
            holds      = verdict == "CORRECT"
            confidence = _parse_float(pred.confidence, default=0.5)
            return AgentVerificationResult(
                property=prop,
                holds=holds,
                witness=pred.suggestion.strip() or None,
                verdict=verdict,
                confidence=confidence,
                reasoning=pred.explanation.strip(),
            )
        except Exception as exc:
            return AgentVerificationResult(
                property=prop,
                holds=False,
                error=str(exc),
                verdict="ERROR",
            )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _quick_summary(
        self,
        step_res:  list[AgentVerificationResult],
        chain_res: list[AgentVerificationResult],
        sol:       AgentVerificationResult | None,
    ) -> str:
        step_p  = sum(1 for r in step_res if r.holds)
        chain_p = sum(1 for r in chain_res if r.holds)
        sol_s   = ("✓" if sol and sol.holds else "✗") if sol else "–"
        return (
            f"steps={step_p}/{len(step_res)} passed  "
            f"chain={chain_p}/{len(chain_res)} passed  "
            f"solution={sol_s}"
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# ── Helper ────────────────────────────────────────────────────────────────────

def _parse_float(text: str, default: float = 0.5) -> float:
    m = re.search(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)", text)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group())))
        except ValueError:
            pass
    return default


  # ── Z3 6-step pipeline ────────────────────────────────────────────────────

    def _z3_run_with_repair(
        self,
        z3_code: str,
        max_repairs: int = 3,
    ) -> tuple[str, str, list[dict[str, str]]]:
        """Step 3+4: execute z3_code; on failure call the repairer up to max_repairs times.

        Returns (final_code, result_text, repair_log).
        result_text is the captured stdout on success or an error string.
        """
        import contextlib
        import io

        repair_log: list[dict[str, str]] = []
        current_code = z3_code

        for attempt in range(max_repairs + 1):
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(current_code, {})  # noqa: S102
                result = buf.getvalue().strip() or "(Z3 ran but printed nothing)"
                return current_code, result, repair_log
            except Exception as exc:
                error_msg = str(exc)
                if attempt >= max_repairs:
                    break
                try:
                    ref = self.z3_repairer(
                        z3_code=current_code,
                        error=error_msg,
                        attempt=str(attempt + 1),
                    )
                    repair_log.append({
                        "attempt":   str(attempt + 1),
                        "error":     error_msg,
                        "rationale": ref.repair_rationale,
                    })
                    current_code = ref.repaired_code
                except Exception as repair_exc:
                    repair_log.append({
                        "attempt":   str(attempt + 1),
                        "error":     error_msg,
                        "rationale": f"Repairer failed: {repair_exc}",
                    })
                    break

        return current_code, f"ERROR (unrepaired after {len(repair_log)} attempt(s)): {error_msg}", repair_log

    def verify_z3_pipeline(
        self,
        question: str,
        plan_str: str,
        initial_z3_code: str | None = None,
        n_paths: int = 3,
        max_repairs: int = 3,
    ) -> dict[str, Any]:
        """6-step Z3 verification pipeline.

        (1) Plan π already provided as plan_str.
        (2) Generate N independent Z3 formalizations (paths); if initial_z3_code
            is given it becomes path_1 so the caller can reuse existing LLM output.
        (3) Repair each path using solver-feedback until it executes or the
            repair limit is reached.
        (4) Execute each repaired path to obtain a result string Si.
        (5) Prune ill-defined paths (trivial tautologies, unrepaired errors, …).
        (6) Aggregate surviving results by majority vote.

        Returns a dict with keys:
            paths         list of per-path dicts (id, z3_code, result, repair_log, pruned, rationale)
            surviving     list of surviving path ids
            vote_distribution, final_answer, confidence
        """
        import contextlib
        import io

        sep  = "=" * 70
        thin = "-" * 70

        print(f"\n{sep}")
        print(f"  Z3 PIPELINE  ({n_paths} path(s), max {max_repairs} repair(s) each)")
        print(sep)

        # ── Step 2: generate N paths ──────────────────────────────────────────
        paths: list[dict[str, Any]] = []
        for i in range(1, n_paths + 1):
            path_id = f"path_{i}"
            if i == 1 and initial_z3_code:
                z3_code = initial_z3_code
                print(f"\n  [{path_id}] using provided initial Z3 code")
            else:
                print(f"\n  [{path_id}] generating formalization…")
                try:
                    pred    = self.z3_formalizer(question=question, plan=plan_str, path_id=path_id)
                    z3_code = pred.z3_code
                except Exception as exc:
                    print(f"  [{path_id}] formalization failed: {exc}")
                    paths.append({"id": path_id, "z3_code": "", "result": f"ERROR: {exc}",
                                  "repair_log": [], "pruned": True, "rationale": "formalization failed"})
                    continue

            # ── Step 3+4: repair and execute ──────────────────────────────────
            final_code, result, repair_log = self._z3_run_with_repair(z3_code, max_repairs)
            if repair_log:
                print(f"  [{path_id}] {len(repair_log)} repair(s) applied")
                for r in repair_log:
                    print(f"    repair {r['attempt']}: {r['rationale'][:80]}")
            print(f"  [{path_id}] result: {result[:120]}")
            paths.append({"id": path_id, "z3_code": final_code, "result": result,
                          "repair_log": repair_log, "pruned": False, "rationale": ""})

        # ── Step 5: prune ill-defined paths ───────────────────────────────────
        print(f"\n{thin}")
        print("  PRUNING")
        print(thin)
        for p in paths:
            if p["pruned"]:
                continue
            try:
                verdict = self.z3_pruner(question=question, z3_result=p["result"])
                p["pruned"]    = verdict.is_valid.strip().lower() != "yes"
                p["rationale"] = verdict.rationale
            except Exception as exc:
                p["rationale"] = f"pruner error: {exc}"
            mark = "KEEP" if not p["pruned"] else "PRUNE"
            print(f"  [{p['id']}] {mark}: {p['rationale'][:100]}")

        surviving = [p for p in paths if not p["pruned"]]
        print(f"  Surviving paths: {len(surviving)} / {len(paths)}")

        # ── Step 6: majority vote ─────────────────────────────────────────────
        print(f"\n{thin}")
        print("  MAJORITY VOTE")
        print(thin)
        if not surviving:
            vote_dist = "no surviving paths"
            final_answer = "UNDECIDED — all paths were pruned"
            confidence = "none"
        else:
            surviving_text = "\n---\n".join(
                f"[{p['id']}] {p['result']}" for p in surviving
            )
            try:
                agg = self.z3_aggregator(question=question, surviving_results=surviving_text)
                vote_dist    = agg.vote_distribution
                final_answer = agg.final_answer
                confidence   = agg.confidence
            except Exception as exc:
                vote_dist    = "aggregation failed"
                final_answer = surviving[0]["result"] if surviving else "UNDECIDED"
                confidence   = f"low (aggregator error: {exc})"

        print(f"  Votes      : {vote_dist}")
        print(f"  Answer     : {final_answer}")
        print(f"  Confidence : {confidence}")
        print(f"{sep}\n")

        return {
            "paths":            paths,
            "surviving":        [p["id"] for p in surviving],
            "vote_distribution": vote_dist,
            "final_answer":     final_answer,
            "confidence":       confidence,
        }

    # ── CEGIR helper ──────────────────────────────────────────────────────────

    def _z3_cegir(
        self,
        z3_code: str,
        z3_result: str,
        question: str,
        precondition: str,
        command: str,
        postcondition: str,
    ) -> tuple[str, str, dict[str, Any]]:
        """Counterexample-guided instruction refinement for UNSAT Z3 results.

        When Z3 returns UNSAT, calls the LLM to diagnose whether the root cause
        is an encoding error (wrong Z3Py) or a genuine step inconsistency ({P}/{Q}
        contradict each other or the command is physically impossible).

        Returns (refined_z3_code, diagnosis_text, analysis_dict).
          - refined_z3_code: corrected Z3Py (may be identical to z3_code for step_inconsistent)
          - diagnosis_text:  one-line summary of the fix type and diagnosis
          - analysis_dict:   full LLM output fields for storage / reporting
        """
        try:
            analysis = self.unsat_analyzer(
                question=question,
                z3_code=z3_code,
                z3_result=z3_result,
                precondition=precondition,
                command=command,
                postcondition=postcondition,
            )
            diagnosis = f"[{analysis.fix_type.upper()}] {analysis.unsat_diagnosis}"
            return analysis.refined_z3_code or z3_code, diagnosis, {
                "fix_type":             analysis.fix_type,
                "diagnosis":            analysis.unsat_diagnosis,
                "refined_precondition": analysis.refined_precondition,
                "refined_command":      analysis.refined_command,
                "refined_postcondition": analysis.refined_postcondition,
            }
        except Exception as exc:
            return z3_code, f"(CEGIR failed: {exc})", {}

    # ── Step chain verification ───────────────────────────────────────────────

    def verify_step_chain(
        self,
        research_plan: list[InstructionStep],
    ) -> list[dict[str, Any]]:
        """Check postcondition_N → precondition_{N+1} for every consecutive step pair.

        For each pair, generates Z3Py asserting post_N ∧ ¬pre_{N+1} is UNSAT.
          UNSAT → entailment holds (chain is logically connected)
          SAT   → chain gap exists (step N+1 may be reached in an invalid state)

        Returns one dict per pair with keys:
            step_n, step_n1, entailment_holds, gap_description,
            z3_code, z3_result, confirmed_by_z3
        """
        sep  = "=" * 70
        thin = "-" * 70

        print(f"\n{sep}")
        print(f"  STEP CHAIN VERIFICATION  ({max(0, len(research_plan)-1)} consecutive pairs)")
        print(sep)

        results: list[dict[str, Any]] = []
        for i in range(len(research_plan) - 1):
            s_n  = research_plan[i]
            s_n1 = research_plan[i + 1]

            print(f"\n{thin}")
            print(f"  Step {s_n.step} → Step {s_n1.step}")
            print(thin)

            entailment_holds = "unknown"
            gap_description  = ""
            z3_code          = ""
            z3_result        = ""
            confirmed_by_z3  = False

            try:
                pred = self.step_chain_checker(
                    step_n=str(s_n.step),
                    postcondition_n=s_n.postcondition,
                    step_n1=str(s_n1.step),
                    precondition_n1=s_n1.precondition,
                )
                entailment_holds = pred.entailment_holds.strip().lower()
                gap_description  = pred.gap_description
                z3_code          = pred.z3_code
            except Exception as exc:
                gap_description = f"(chain checker failed: {exc})"

            if z3_code:
                _, z3_result, _ = self._z3_run_with_repair(z3_code, max_repairs=2)
                confirmed_by_z3 = "CHAIN_VALID" in z3_result

            llm_ok = entailment_holds == "yes"
            z3_ok  = confirmed_by_z3 or not z3_code
            mark   = "✓" if (llm_ok and z3_ok) else "✗"
            print(f"  [{mark}] LLM entailment: {entailment_holds}  "
                  f"Z3: {'CHAIN_VALID' if confirmed_by_z3 else z3_result[:80]}")
            if gap_description:
                print(f"  Gap: {gap_description}")

            results.append({
                "step_n":           str(s_n.step),
                "step_n1":          str(s_n1.step),
                "entailment_holds": entailment_holds,
                "gap_description":  gap_description,
                "z3_code":          z3_code,
                "z3_result":        z3_result,
                "confirmed_by_z3":  confirmed_by_z3,
            })

        chain_gaps = sum(
            1 for r in results
            if r["entailment_holds"] != "yes" or not r["confirmed_by_z3"]
        )
        print(f"\n  Chain gaps found: {chain_gaps} / {len(results)}")
        print(f"{sep}\n")
        return results

    # ── Type consistency verification ─────────────────────────────────────────

    def verify_type_consistency(
        self,
        research_plan: list[InstructionStep],
    ) -> list[dict[str, Any]]:
        """Check input_type compatibility across consecutive step pairs.

        Calls the LLM to compare the output type declared by step N with the
        input type expected by step N+1, and optionally runs a Z3 type encoding.

        Returns one dict per pair with keys:
            step_n, step_n1, types_compatible, mismatch, conversion_needed,
            z3_type_code, z3_type_result
        """
        sep  = "=" * 70
        thin = "-" * 70

        print(f"\n{sep}")
        print(f"  TYPE CONSISTENCY  ({max(0, len(research_plan)-1)} consecutive pairs)")
        print(sep)

        results: list[dict[str, Any]] = []
        for i in range(len(research_plan) - 1):
            s_n  = research_plan[i]
            s_n1 = research_plan[i + 1]

            print(f"\n{thin}")
            print(f"  Step {s_n.step} ({s_n.input_type[:40]}) → "
                  f"Step {s_n1.step} ({s_n1.input_type[:40]})")
            print(thin)

            types_compatible     = "unknown"
            mismatch_description = ""
            conversion_needed    = ""
            z3_type_code         = ""
            z3_type_result       = ""

            try:
                pred = self.type_checker(
                    step_n=str(s_n.step),
                    input_type_n=s_n.input_type,
                    postcondition_n=s_n.postcondition,
                    step_n1=str(s_n1.step),
                    input_type_n1=s_n1.input_type,
                    precondition_n1=s_n1.precondition,
                )
                types_compatible     = pred.types_compatible.strip().lower()
                mismatch_description = pred.mismatch_description
                conversion_needed    = pred.conversion_needed
                z3_type_code         = pred.z3_type_code
            except Exception as exc:
                mismatch_description = f"(type checker failed: {exc})"

            if z3_type_code:
                _, z3_type_result, _ = self._z3_run_with_repair(z3_type_code, max_repairs=1)

            mark = "✓" if types_compatible == "yes" else "✗"
            print(f"  [{mark}] Compatible: {types_compatible}")
            if mismatch_description:
                print(f"  Mismatch   : {mismatch_description}")
            if conversion_needed:
                print(f"  Conversion : {conversion_needed}")

            results.append({
                "step_n":            str(s_n.step),
                "step_n1":           str(s_n1.step),
                "types_compatible":  types_compatible,
                "mismatch":          mismatch_description,
                "conversion_needed": conversion_needed,
                "z3_type_code":      z3_type_code,
                "z3_type_result":    z3_type_result,
            })

        mismatches = sum(1 for r in results if r["types_compatible"] != "yes")
        print(f"\n  Type mismatches found: {mismatches} / {len(results)}")
        print(f"{sep}\n")
        return results

    # ── Step 4a ───────────────────────────────────────────────────────────────

    def translate_and_verify_fol(
        self,
        question: str,
        plan_str: str,
        n_paths: int = 3,
        max_repairs: int = 3,
    ) -> tuple[str, str, dict[str, Any]]:
        """Translate the research plan to FOL, print axioms, then run the 6-step Z3 pipeline.

        Returns:
            fol_axioms   human-readable FOL text
            z3_code      initial Z3 code from Plan2FOL (used as path_1)
            pipeline_result  full dict from verify_z3_pipeline
        """
        sep  = "=" * 70

        print(f"\n{sep}")
        print("  FIRST-ORDER LOGIC TRANSLATION")
        print(sep)

        pred = self.fol_formalizer(research_plan=plan_str)
        fol_axioms: str = pred.fol_axioms
        z3_code:    str = pred.z3_code

        print(fol_axioms)

        pipeline_result = self.verify_z3_pipeline(
            question=question,
            plan_str=plan_str,
            initial_z3_code=z3_code,
            n_paths=n_paths,
            max_repairs=max_repairs,
        )

        return fol_axioms, z3_code, pipeline_result

    def verify_steps_fol(
        self,
        research_plan: list[InstructionStep],
    ) -> list[dict[str, str]]:
        """Translate every research step to FOL, run Z3, and return per-step results.

        For each step the method:
          1. Calls the LLM to produce FOL axioms and matching Z3 Python code.
          2. Prints both.
          3. Executes the Z3 code and captures the verdict.
          4. Prints the verdict.

        Returns a list of dicts with keys: step, fol_axioms, z3_code, z3_result.
        """
        import contextlib
        import io

        sep  = "=" * 70
        thin = "-" * 70

        print(f"\n{sep}")
        print(f"  PER-STEP FOL + Z3 VERIFICATION  ({len(research_plan)} steps)")
        print(sep)

        results: list[dict[str, str]] = []
        for s in research_plan:
            print(f"\n{thin}")
            print(f"  Step {s.step}  |  {s.command[:65]}")
            print(thin)

            fol_axioms = ""
            z3_code    = ""
            try:
                pred = self.step_fol_formalizer(
                    step_number=str(s.step),
                    precondition=s.precondition,
                    command=s.command,
                    postcondition=s.postcondition,
                )
                fol_axioms = pred.fol_axioms
                z3_code    = pred.z3_code
            except Exception as exc:
                fol_axioms = f"(FOL extraction failed: {exc})"

            print(f"\n  FOL Axioms:\n{fol_axioms}")

            # Execute weakest-precondition check if available
            wp_result = ""
            if z3_code and hasattr(pred, "wp_z3_code") and pred.wp_z3_code:
                _, wp_result, _ = self._z3_run_with_repair(pred.wp_z3_code, max_repairs=1)
                mark = "✓" if "WP_SUFFICIENT" in wp_result else "⚠"
                print(f"  [{mark}] WP check: {wp_result[:120]}")

            step_desc = f"Step {s.step}: {s.command}"
            plan_str_step = (
                f"Precondition: {s.precondition}\n"
                f"Command: {s.command}\n"
                f"Postcondition: {s.postcondition}"
            )
            pipeline_result = self.verify_z3_pipeline(
                question=step_desc,
                plan_str=plan_str_step,
                initial_z3_code=z3_code or None,
            )

            # CEGIR: if the pipeline returned unsat, diagnose and attempt repair
            cegir_result: dict[str, Any] = {}
            final_answer = (pipeline_result.get("final_answer") or "").lower()
            if "unsat" in final_answer:
                print(f"\n  UNSAT detected — running CEGIR for step {s.step}...")
                refined_z3, diagnosis, cegir_result = self._z3_cegir(
                    z3_code=z3_code or "",
                    z3_result=final_answer,
                    question=step_desc,
                    precondition=s.precondition,
                    command=s.command,
                    postcondition=s.postcondition,
                )
                print(f"  CEGIR: {diagnosis}")
                if refined_z3 and refined_z3 != z3_code:
                    print(f"  Re-running with refined Z3 code...")
                    _, refined_result, _ = self._z3_run_with_repair(refined_z3, max_repairs=2)
                    cegir_result["refined_z3_result"] = refined_result
                    print(f"  Refined result: {refined_result[:120]}")

            results.append({
                "step":        str(s.step),
                "fol_axioms":  fol_axioms,
                "z3_code":     z3_code,
                "wp_result":   wp_result,
                "z3_pipeline": pipeline_result,
                "cegir":       cegir_result,
            })

        print(f"\n{sep}\n")
        return results

    # ── Step 4b ───────────────────────────────────────────────────────────────

    def explain_and_verify_plan(
        self,
        research_plan: list[InstructionStep],
        plan_str: str,
        backend: str = "z3",
        run_optional: bool = False,
        decomp_roots: list | None = None,
        htn_network: "HTNNetwork | None" = None,
        question: str = "",
    ) -> tuple[
        dict[str, str],           # explanation
        dict[str, str],           # z3_result  (mandatory)
        dict[str, Any],           # kripke_result  (mandatory)
        list[dict[str, Any]],     # lean_results   (mandatory, per-step)
        list[dict[str, Any]],     # step_fol_results  (optional)
        list[dict[str, Any]],     # chain_results     (optional)
        list[dict[str, Any]],     # type_results      (optional)
        list[str],                # code_snippets
    ]:
        """Verify the plan with Z3 + Kripke + Lean (mandatory) and optionally FOL/chain/type checks.

        Mandatory verification
        ----------------------
        Z3       multi-path encoding + majority vote (verify_z3_pipeline)
        Kripke   CTL/LTL model checking via z3 or lean backend
        Lean     per-step autoformalization → lean compiler verification loop

        Optional verification  (run_optional=True)
        -------------------------------------------
        FOL      whole-plan Plan2FOL axiom translation
        step-FOL per-step Step2FOL + weakest-precondition Z3
        chain    postcondition N ⊨ precondition N+1 checks
        type     output-type of step N compatible with input-type of step N+1
        """
        sep  = "=" * 70
        thin = "-" * 70

        # ── Explain the plan ──────────────────────────────────────────────────
        print(f"\n{sep}")
        print("  PLAN EXPLANATION")
        print(sep)
        exp = self.plan_explainer(research_plan=plan_str)
        print(f"\n  Problem:\n    {exp.problem}")
        print(f"\n  Motivation:\n    {exp.motivation}")
        print(f"\n  Methodology:\n    {exp.methodology}")
        print(f"\n  Expected Results:\n    {exp.expected_results}")
        explanation_text = (
            f"Problem: {exp.problem}\n"
            f"Motivation: {exp.motivation}\n"
            f"Methodology: {exp.methodology}\n"
            f"Expected Results: {exp.expected_results}"
        )
        explanation_dict = {
            "problem":          exp.problem,
            "motivation":       exp.motivation,
            "methodology":      exp.methodology,
            "expected_results": exp.expected_results,
        }

        # ══ MANDATORY: Z3 multi-path pipeline ════════════════════════════════
        print(f"\n{sep}")
        print("  Z3 VERIFICATION  (multi-path + majority vote)")
        print(sep)
        _, z3_initial, z3_pipeline = self.translate_and_verify_fol(
            question=plan_str[:200], plan_str=plan_str
        )
        z3_result = {"z3_code": z3_initial, "z3_pipeline": z3_pipeline}

        # ══ MANDATORY: Kripke model checking ═════════════════════════════════
        print(f"\n{thin}")
        print("  KRIPKE FORMALIZATION + VERIFICATION")
        print(thin)
        formalization = self.formalize_plan(explanation_text)
        kripke_pred   = formalization["kripke"]
        structure     = parse_kripke_from_llm(
            worlds_str=kripke_pred.worlds,
            initial=kripke_pred.initial_world,
            relation_str=kripke_pred.accessibility_relation,
            propositions_str=kripke_pred.atomic_propositions,
            valuation_str=kripke_pred.valuation,
        )
        _print_kripke(structure)
        verification, suggestion = self.verify_kripke(explanation_text, formalization, backend)
        _print_verification(verification, suggestion)
        kripke_result = {
            "formalization": formalization,
            "verification":  verification,
            "suggestion":    suggestion,
        }

        # ══ MANDATORY: Lean per-step autoformalization ════════════════════════
        print(f"\n{sep}")
        print(f"  LEAN AUTOFORMALIZATION  ({len(research_plan)} steps)")
        print(sep)
        lean_results: list[dict[str, Any]] = []
        _lean = self.lean_formalizer
        for step in research_plan:
            problem_text = f"{step.task}\nPrecondition: {step.precondition}\nCommand: {step.command}\nPostcondition: {step.postcondition}"
            result: dict[str, Any] = {"step": step.step, "task": step.task}
            if _lean is not None:
                try:
                    lr = _lean.forward(problem_text)
                    status = "VERIFIED" if lr.verified else f"UNVERIFIED ({lr.attempts} attempts)"
                    print(f"\n  Step {step.step}: {status}")
                    print(f"    Domain    : {lr.domain}")
                    print(f"    Compiler  : {lr.compiler_out[:120]}")
                    result.update({
                        "verified":      lr.verified,
                        "lean_code":     lr.lean_code,
                        "domain":        lr.domain,
                        "compiler_out":  lr.compiler_out,
                        "attempts":      lr.attempts,
                    })
                except Exception as exc:
                    print(f"\n  Step {step.step}: Lean failed — {exc}")
                    result["error"] = str(exc)
            else:
                print(f"\n  Step {step.step}: Lean unavailable (corpus not bootstrapped)")
                result["error"] = "lean_unavailable"
            lean_results.append(result)

        lean_verified = sum(1 for r in lean_results if r.get("verified"))
        print(f"\n  Lean: {lean_verified}/{len(lean_results)} steps verified")

        # ══ LEAF REFINEMENT — fix failed leaf nodes ═══════════════════════════
        if decomp_roots:
            failed_steps = {r["step"] for r in lean_results if not r.get("verified")}
            if failed_steps:
                print(f"\n{sep}")
                print(f"  LEAF REFINEMENT  ({len(failed_steps)} failed step(s))")
                print(sep)
                self._refine_failed_leaves(
                    decomp_roots=decomp_roots,
                    failed_steps=failed_steps,
                    lean_results=lean_results,
                    question=question,
                )

        # ══ OPTIONAL verifications ════════════════════════════════════════════
        step_fol_results: list[dict[str, Any]] = []
        chain_results:    list[dict[str, Any]] = []
        type_results:     list[dict[str, Any]] = []

        if run_optional:
            print(f"\n{sep}")
            print("  OPTIONAL: Per-step FOL + Z3")
            print(sep)
            step_fol_results = self.verify_steps_fol(research_plan)

            print(f"\n{sep}")
            print("  OPTIONAL: Step chain & type consistency")
            print(sep)
            chain_results = self.verify_step_chain(research_plan)
            type_results  = self.verify_type_consistency(research_plan)

        # ── Build ancestor-context map from decomp tree ───────────────────────
        # Maps step.step (int) → multiline ancestor chain string for Step2Code.
        # For depth-0 roots the context is empty; for leaves it includes every
        # ancestor triple from the root down to the immediate parent.
        _ancestor_ctx: dict[int, str] = {}

        def _build_ancestor_ctx(
            node: "HoareTreeNode",
            chain: list[str],
        ) -> None:
            s = node.step
            _ancestor_ctx[s.step] = "\n".join(chain)
            if not node.is_leaf:
                entry = (
                    f"depth={node.depth}  "
                    f"{{P}} {s.precondition[:120]}  "
                    f"C {s.command[:120]}  "
                    f"{{Q}} {s.postcondition[:120]}"
                )
                for child in node.children:
                    _build_ancestor_ctx(child, chain + [entry])

        if decomp_roots:
            for _root in decomp_roots:
                _build_ancestor_ctx(_root, [])

        # ── Generate Python code per research step ────────────────────────────
        import datetime as _dt
        import hashlib as _hl

        _q_slug   = _hl.sha256(question.encode()).hexdigest()[:12]
        _code_dir = Path("tasks") / "generated" / _q_slug
        _code_dir.mkdir(parents=True, exist_ok=True)

        # Write an index header once per question
        _index_path = _code_dir / "index.md"
        with _index_path.open("w", encoding="utf-8") as _idx:
            _idx.write(
                f"# Generated code for question\n\n"
                f"**Question**: {question}\n\n"
                f"**Generated**: {_dt.datetime.now().isoformat(timespec='seconds')}\n\n"
                f"## Steps\n\n"
            )

        print(f"\n{sep}")
        print(f"  GENERATED CODE  ({len(research_plan)} steps)  →  {_code_dir}")
        print(sep)
        code_snippets: list[str] = []
        for step in research_plan:
            result_code = self.step_coder(
                higher_node_context=_ancestor_ctx.get(step.step, ""),
                precondition=step.precondition,
                command=step.command,
                postcondition=step.postcondition,
            )
            code = str(result_code.python_code) if result_code.python_code is not None else ""
            code_snippets.append(code)

            # ── Save step file ────────────────────────────────────────────────
            _step_path = _code_dir / f"step_{step.step:03d}.py"
            _step_path.write_text(
                f'"""\nStep {step.step}: {step.task}\n\n'
                f'Precondition  {{P}}: {step.precondition}\n'
                f'Command       C  : {step.command}\n'
                f'Postcondition {{Q}}: {step.postcondition}\n'
                f'"""\n'
                + (
                    f'# Ancestor context:\n'
                    + "\n".join(
                        f"#   {ln}"
                        for ln in _ancestor_ctx.get(step.step, "").splitlines()
                    )
                    + "\n"
                    if _ancestor_ctx.get(step.step) else ""
                )
                + "\n"
                + code,
                encoding="utf-8",
            )

            # ── Append to index ───────────────────────────────────────────────
            with _index_path.open("a", encoding="utf-8") as _idx:
                _idx.write(
                    f"### Step {step.step}: {step.task}\n\n"
                    f"- **File**: `{_step_path.name}`\n"
                    f"- **Tool**: {step.tool or '(none)'}\n"
                    f"- **{{P}}**: {step.precondition[:120]}\n"
                    f"- **C**: {step.command[:120]}\n"
                    f"- **{{Q}}**: {step.postcondition[:120]}\n\n"
                )

            print(f"\n  -- Step {step.step}: {step.command[:60]}")
            print(f"     Precondition: {step.precondition[:100]}")
            print(f"     Saved → {_step_path}")
            print(code)
        print(f"{sep}\n")

        # ══ VERIFICATION SUMMARY ══════════════════════════════════════════════
        self._print_verification_summary(
            research_plan=research_plan,
            lean_results=lean_results,
            z3_result=z3_result,
            kripke_result=kripke_result,
        )

        # ══ VERIFIED PLAN PRINTOUT ════════════════════════════════════════════
        self._print_verified_plan(
            research_plan=research_plan,
            lean_results=lean_results,
            z3_result=z3_result,
        )

        # ── Log leaf/primitive layer AFTER verification ────────────────────────
        if decomp_roots:
            tree_after_path = self._log_task_tree_md(
                decomp_roots=decomp_roots,
                out_path="task_tree_after_verification.md",
                label="After Verification",
                lean_results=lean_results,
                z3_result=z3_result,
                kripke_result=kripke_result,
            )
            print(f"  Task-tree leaf log (after verification) → {tree_after_path}")

        if htn_network is not None:
            htn_after_path = self._log_htn_md(
                htn_network=htn_network,
                out_path="htn_network_after_verification.md",
                label="After Verification",
                lean_results=lean_results,
                z3_result=z3_result,
                kripke_result=kripke_result,
            )
            print(f"  HTN network log (after verification)   → {htn_after_path}")

        return (
            explanation_dict,
            z3_result,
            kripke_result,
            lean_results,
            step_fol_results,
            chain_results,
            type_results,
            code_snippets,
        )

    def verify(
        self,
        question: str,
        research_plan: list[InstructionStep],
        plan_str: str,
        backend: str = "z3",
    ) -> VerificationReport:
        """Run all mandatory verifiers and return a unified VerificationReport.

        Runs in order:
          1. Z3 multi-path pipeline          (verify_z3_pipeline)
          2. Kripke CTL model checking       (formalize_plan + verify_kripke)
          3. Step chain entailment checks    (verify_step_chain)
          4. Input/output type consistency   (verify_type_consistency)

        Lean per-step autoformalization is skipped here to keep the call fast;
        it is still run inside explain_and_verify_plan / forward.

        Args:
            question:       Original research question.
            research_plan:  Structured list of InstructionStep objects.
            plan_str:       Plan serialised as a string (e.g. from _plan_to_str).
            backend:        Kripke verification backend — 'z3' or 'lean'.

        Returns:
            VerificationReport with .passed, .summary, and per-verifier fields.
        """
        z3_result = self.verify_z3_pipeline(
            question=question,
            plan_str=plan_str,
        )

        formalization           = self.formalize_plan(plan_str)
        kripke_results, suggestion = self.verify_kripke(
            research_plan=plan_str,
            formalization=formalization,
            backend=backend,
        )

        chain_results = self.verify_step_chain(research_plan)
        type_results  = self.verify_type_consistency(research_plan)

        return VerificationReport(
            z3=z3_result,
            kripke=kripke_results,
            lean=[],
            chain=chain_results,
            type_checks=type_results,
            suggestion=suggestion,
        )

    @staticmethod
    def _print_verification_summary(
        research_plan: list["InstructionStep"],
        lean_results:  list[dict[str, Any]],
        z3_result:     dict[str, Any],
        kripke_result: dict[str, Any],
    ) -> None:
        """Print a structured per-step and plan-level verification summary table.

        Columns per step
        ----------------
        Step   — step number
        Task   — first 52 chars of task description
        Lean   — VERIFIED / UNVERIFIED / n/a
        Z3     — plan-level result applied to every step
        Kripke — plan-level result applied to every step
        Refined — whether any prover ran a repair/retry loop for this step
        """
        sep  = "═" * 80
        thin = "─" * 80

        print(f"\n{sep}")
        print("  VERIFICATION SUMMARY")
        print(sep)

        # ── Index lean results by step number ─────────────────────────────────
        lean_by_step: dict[int, dict] = {r["step"]: r for r in lean_results}

        # ── Z3 plan-level info ────────────────────────────────────────────────
        z3_pipe      = z3_result.get("z3_pipeline", {})
        z3_surviving = z3_pipe.get("surviving", [])
        z3_answer    = z3_pipe.get("final_answer", "")
        z3_conf      = z3_pipe.get("confidence", "")
        z3_vote      = z3_pipe.get("vote_distribution", "")
        z3_paths     = z3_pipe.get("paths", [])
        z3_verified  = bool(z3_surviving)
        z3_refined   = any(p.get("repair_log") for p in z3_paths)

        # ── Kripke plan-level info ─────────────────────────────────────────────
        kripke_checks: list["VerificationResult"] = kripke_result.get("verification", [])
        kripke_all_hold = all(r.holds for r in kripke_checks) if kripke_checks else False
        kripke_suggestion = kripke_result.get("suggestion", "")

        # ── Header row ────────────────────────────────────────────────────────
        h_step = "Step"
        h_task = f"{'Task':<54}"
        h_lean = f"{'Lean':<12}"
        h_z3   = f"{'Z3':<12}"
        h_kpke = f"{'Kripke':<10}"
        h_ref  = "Refined"
        print(f"\n  {h_step:<5} {h_task} {h_lean} {h_z3} {h_kpke} {h_ref}")
        print(f"  {thin}")

        lean_total    = 0
        lean_verified = 0

        for step in research_plan:
            lr = lean_by_step.get(step.step, {})

            # Lean column
            lean_total += 1
            if "error" in lr and lr["error"] == "lean_unavailable":
                lean_col = "n/a"
                lean_ref = False
                lean_attempts = 0
            elif "error" in lr:
                lean_col = "ERROR"
                lean_ref = False
                lean_attempts = 0
            elif lr.get("verified"):
                lean_col = "VERIFIED"
                lean_verified += 1
                lean_attempts = lr.get("attempts", 1)
                lean_ref = lean_attempts > 1
            else:
                lean_col = "UNVERIFIED"
                lean_attempts = lr.get("attempts", 1)
                lean_ref = lean_attempts > 1

            # Z3 and Kripke apply plan-wide — mark as inherited
            z3_col   = ("VERIFIED" if z3_verified else "UNVERIFIED") + " (plan)"
            kpke_col = ("HOLDS" if kripke_all_hold else "FAILS")    + " (plan)"

            # Refined: Lean repair + Z3 repair (plan-level)
            refined_parts = []
            if lean_ref:
                refined_parts.append(f"Lean×{lean_attempts}")
            if z3_refined:
                refined_parts.append("Z3")
            refined_col = ", ".join(refined_parts) if refined_parts else "no"

            task_preview = step.task[:52].replace("\n", " ")
            print(
                f"  {step.step:<5} {task_preview:<54} "
                f"{lean_col:<12} {z3_col:<18} {kpke_col:<14} {refined_col}"
            )

        print(f"\n  {thin}")
        print(f"  Lean   : {lean_verified}/{lean_total} steps verified by lean compiler")

        # ── Plan-level Z3 detail ──────────────────────────────────────────────
        z3_status = "VERIFIED" if z3_verified else "UNVERIFIED"
        surviving_count = len(z3_surviving)
        total_paths     = len(z3_paths)
        print(
            f"  Z3     : {z3_status}  "
            f"({surviving_count}/{total_paths} paths survived"
            + (f",  vote: {z3_vote}" if z3_vote else "")
            + (f",  answer: {z3_answer[:60]}" if z3_answer else "")
            + (f",  confidence: {z3_conf}" if z3_conf else "")
            + ")"
            + ("  [refined]" if z3_refined else "")
        )

        # ── Plan-level Kripke detail ──────────────────────────────────────────
        if kripke_checks:
            holds_n  = sum(1 for r in kripke_checks if r.holds)
            fails_n  = len(kripke_checks) - holds_n
            kpke_status = "ALL HOLD" if fails_n == 0 else f"{holds_n} hold / {fails_n} fail"
            print(f"  Kripke : {kpke_status}")
            for r in kripke_checks:
                mark = "✓" if r.holds else "✗"
                detail = f"  witness={r.witness}" if r.witness else ""
                detail += f"  error={r.error}" if r.error else ""
                print(f"    [{mark}] {r.property}{detail}")
        else:
            print("  Kripke : no properties checked")

        if kripke_suggestion and kripke_suggestion.strip().lower() not in ("", "all clear"):
            print(f"\n  Suggestion: {kripke_suggestion[:200]}")

        print(f"\n{sep}\n")

    @staticmethod
    def _print_verified_plan(
        research_plan: list["InstructionStep"],
        lean_results:  list[dict[str, Any]],
        z3_result:     dict[str, Any],
    ) -> None:
        """Print the full plan with per-step verification badges after verification.

        For every step the output shows:
          - [✓ VERIFIED] or [✗ UNVERIFIED] badge (Lean result when available,
            otherwise Z3 plan-level result)
          - Step number, task name, tool
          - {P} precondition, C command, {Q} postcondition
          - Completion metric (assert expression)
          - command_body snippet (first 6 lines)
        """
        sep  = "═" * 80
        thin = "─" * 80
        dash = "·" * 80

        lean_by_step: dict[int, dict] = {r["step"]: r for r in lean_results}
        z3_pipe     = z3_result.get("z3_pipeline", {})
        z3_verified = bool(z3_pipe.get("surviving", []))

        print(f"\n{sep}")
        print("  VERIFIED PLAN")
        print(sep)

        for idx, step in enumerate(research_plan, 1):
            lr = lean_by_step.get(step.step, {})

            if lr.get("verified"):
                badge   = "✓ LEAN VERIFIED"
                badge_c = "✓"
            elif "error" in lr and lr["error"] == "lean_unavailable":
                badge   = ("✓ Z3 VERIFIED" if z3_verified else "✗ Z3 UNVERIFIED") + " (plan)"
                badge_c = "✓" if z3_verified else "✗"
            elif lr:
                badge   = f"✗ LEAN UNVERIFIED (attempts={lr.get('attempts', 1)})"
                badge_c = "✗"
            else:
                badge   = ("✓ Z3 VERIFIED" if z3_verified else "✗ UNVERIFIED") + " (plan)"
                badge_c = "✓" if z3_verified else "✗"

            tool_str  = f"  ACTION: {step.tool}" if step.tool else "  ACTION: (none)"
            deps_str  = (
                "deps=" + ",".join(str(d) for d in step.dependencies)
                if step.dependencies else "deps=none"
            )

            print(f"\n{thin}")
            print(f"  [{badge_c}] Step {step.step}  [{badge}]{tool_str}  {deps_str}")
            print(thin)
            print(f"  Task  : {step.task}")
            print(f"  {{P}}   : {step.precondition}")
            print(f"  C     : {step.command}")
            print(f"  {{Q}}   : {step.postcondition}")

            if step.completion_metric:
                metric = step.completion_metric.strip()
                if not metric.startswith("assert "):
                    metric = f"assert {metric}"
                print(f"  Metric: {metric}")

            if getattr(step, "pre_state", None):
                names = ", ".join(
                    f"{sv.name}:{sv.python_type}"
                    for sv in step.pre_state
                )
                print(f"  Σ_pre : {names}")

            if getattr(step, "action_sig", None):
                sig = step.action_sig
                print(f"  α     : {sig.name}  reads={sig.reads}  writes={sig.writes}")

            if getattr(step, "post_state", None):
                names = ", ".join(
                    f"{sv.name}:{sv.python_type}"
                    for sv in step.post_state
                )
                print(f"  Σ_post: {names}")

            if step.command_body:
                body_lines = step.command_body.strip().splitlines()[:6]
                snippet    = "\n".join(f"      {ln}" for ln in body_lines)
                ellipsis   = "      ..." if len(step.command_body.strip().splitlines()) > 6 else ""
                print(f"  Body  :\n{snippet}")
                if ellipsis:
                    print(ellipsis)

            if lr.get("lean_statement"):
                print(f"  Lean  : {lr['lean_statement'][:120]}")

        print(f"\n{sep}\n")