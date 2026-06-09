import pdb
import time
import tqdm
import re
import sys
import subprocess
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import List, Optional
import dspy
from pydantic import BaseModel, Field
from agents.researcher import ResearchPlan
from agents.formalizer import KripkeFormalizer, Z3Formalization, NuSMVFormalization, Kripke
from memory.episodic_memory import VerificationAttempt


# ── Execution result ──────────────────────────────────────────────────────────

@dataclass
class Z3ExecutionResult:
    """Result of running a Z3Py program."""
    code: str
    stdout: str = ""
    stderr: str = ""
    success: bool = False

    @property
    def trace(self) -> str:
        """Full execution trace (stdout + stderr header when errors present)."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append("--- stderr ---\n" + self.stderr)
        return "\n".join(parts)

    @property
    def error(self) -> Optional[str]:
        return self.stderr if not self.success else None


# ── Z3 code executor ──────────────────────────────────────────────────────────

def _strip_code_fences(code: str) -> str:
    """Remove markdown ```python ... ``` fences if present."""
    code = code.strip()
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


def execute_z3_code(code: str, timeout: int = 30) -> Z3ExecutionResult:
    """Execute Z3Py code in an isolated subprocess and return trace/errors.

    Args:
        code:    Self-contained Python/Z3 source to execute.
        timeout: Maximum seconds to allow the process to run.

    Returns:
        Z3ExecutionResult with stdout trace and/or stderr error details.
    """
    code = _strip_code_fences(code)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        success = proc.returncode == 0
        return Z3ExecutionResult(
            code=code,
            stdout=proc.stdout,
            stderr="",
            success=success,
        )
    except subprocess.TimeoutExpired:
        return Z3ExecutionResult(
            code=code,
            stderr=f"TimeoutExpired: execution exceeded {timeout}s",
            success=False,
        )
    except Exception as exc:
        return Z3ExecutionResult(
            code=code,
            stderr="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            success=False,
        )
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def execute_nusmv_code(code: str, timeout: int = 30) -> Z3ExecutionResult:
    """Execute a NuSMV model in an isolated subprocess and return the output.

    Args:
        code:    Self-contained NuSMV source (.smv format) to verify.
        timeout: Maximum seconds to allow the process to run.

    Returns:
        Z3ExecutionResult with stdout trace and/or stderr error details.
        success=True only when nusmv runs without error AND all specs are verified
        (i.e. no 'is false' appears in the output).
    """
    code = _strip_code_fences(code)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".smv", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            ["nusmv", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout
        # NuSMV returns 0 even when specs fail — check output for falsified specs
        has_error = proc.returncode != 0
        spec_failed = "is false" in output.lower()
        success = not has_error and not spec_failed
        return Z3ExecutionResult(
            code=code,
            stdout=output,
            stderr="",
            success=success,
        )
    except FileNotFoundError:
        return Z3ExecutionResult(
            code=code,
            stderr="nusmv binary not found — install NuSMV and ensure it is on PATH",
            success=False,
        )
    except subprocess.TimeoutExpired:
        return Z3ExecutionResult(
            code=code,
            stderr=f"TimeoutExpired: execution exceeded {timeout}s",
            success=False,
        )
    except Exception as exc:
        return Z3ExecutionResult(
            code=code,
            stderr="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            success=False,
        )
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── DSPy signatures ───────────────────────────────────────────────────────────

class Z3FormalizationSig(dspy.Signature):
    """Generate a self-contained Z3Py program for a natural language plan.

    Each call produces ONE independent formalization path.
    The program must end with a print statement reporting sat/unsat/unknown
    and, when sat, the model values relevant to the question.
    """
    kripke: Kripke = dspy.InputField(
        desc="Kripke structure (S, R, V) derived from the plan"
    )
    proposed_plan: ResearchPlan = dspy.InputField(
        desc="Natural language instruction plan to encode as Z3 constraints"
    )
    z3_code: str = dspy.OutputField(
        desc=(
            "Self-contained executable Python using z3-solver. Must import z3, "
            "declare sorts/variables, assert all constraints from the plan, call "
            "solver.check(), and print a human-readable verdict + model when sat."
        )
    )


class PlanRefinement(dspy.Signature):
    """Refine the plan based on a detected execution error."""
    z3_code: str = dspy.InputField(
        desc="Z3 code that was executed"
    )
    execution_error: str = dspy.InputField(
        desc="Error message / stderr from the failed execution"
    )
    plan: ResearchPlan = dspy.InputField(
        desc="Solution plan — each step is an instruction leading to the solution"
    )
    kripke_structure: Kripke = dspy.InputField(
        desc="Kripke structure of the produced plan"
    )
    refined_plan: ResearchPlan = dspy.OutputField(
        desc="Refined plan that addresses the detected error"
    )


# ── Verifier module ───────────────────────────────────────────────────────────

@dataclass
class VerificationOutcome:
    """Full output of a Verifier.forward() call."""
    z3_code: str
    execution: Z3ExecutionResult
    refined_plan: Optional[ResearchPlan] = None
    # all per-attempt records for episodic memory
    attempts: list = field(default_factory=list)   # list[VerificationAttempt]

    @property
    def trace(self) -> str:
        return self.execution.trace

    @property
    def success(self) -> bool:
        return self.execution.success


class Verifier(dspy.Module):
    """Verify a ResearchPlan by:
      1. Extracting a Kripke structure from the plan.
      2. Generating Z3Py code that encodes the plan's constraints.
      3. Executing the Z3 code in a Python subprocess.
      4. Returning the full trace on success, or the error + a refined plan on failure.
    """

    def __init__(self):
        super().__init__()
        self.kripke_formalizer = dspy.ChainOfThought(KripkeFormalizer)
        self.z3_formalizer = dspy.ChainOfThought(Z3FormalizationSig)
        self.nusmv_formalizer = dspy.ChainOfThought(NuSMVFormalization)
        self.plan_refiner = dspy.ChainOfThought(PlanRefinement)

    def forward(
        self,
        proposed_plan: ResearchPlan,
        max_retries: int = 10,
        z3_timeout: int = 30,
    ) -> VerificationOutcome:
        """Formalize, execute, and optionally refine the plan.

        Args:
            proposed_plan: The plan produced by the Researcher module.
            max_retries:   How many times to retry after an execution error.
            z3_timeout:    Seconds before the subprocess is killed.

        Returns:
            VerificationOutcome containing the z3_code, execution result
            (stdout trace and/or stderr error), and an optional refined plan.
        """
        # Step 1 — extract Kripke structure
        kripke_pred = self.kripke_formalizer(proposed_plan=proposed_plan)
        kripke = kripke_pred.kripke

        plan = proposed_plan
        refined_plan: Optional[ResearchPlan] = None
        recorded_attempts: list[VerificationAttempt] = []

        for attempt in tqdm.tqdm(range(1 + max_retries)):
            # Step 2 — generate Z3Py code
            z3_pred = self.z3_formalizer(kripke=kripke, proposed_plan=plan)
            z3_code = z3_pred.z3_code

            # Step 3 — execute in isolated Python subprocess
            t0 = time.time()
            result = execute_z3_code(z3_code, timeout=z3_timeout)
            elapsed = round(time.time() - t0, 2)
            print(f"[verifier] attempt {attempt + 1}/{1 + max_retries} — {'OK' if result.success else 'FAILED'}")
            if result.stdout:
                print("[verifier] output:\n", result.stdout)

            is_unsat = "unsat" in result.stdout.lower()
            is_sat   = "sat"   in result.stdout.lower() and not is_unsat

            # Step 4 — on error or UNSAT, refine and record the refined plan
            next_refined: Optional[ResearchPlan] = None
            error_msg = result.stderr or (
                f"Z3 returned UNSAT: the plan constraints are unsatisfiable.\n{result.stdout}"
                if is_unsat else result.stdout
            )
            if (not result.success or is_unsat) and attempt < max_retries:
                refine_pred  = self.plan_refiner(
                    z3_code         = z3_code,
                    execution_error = error_msg,
                    plan            = plan,
                    kripke_structure= kripke,
                )
                next_refined  = refine_pred.refined_plan
                refined_plan  = next_refined

            # Record this attempt (refined_plan populated only when refinement happened)
            recorded_attempts.append(VerificationAttempt(
                attempt_number = attempt + 1,
                z3_code        = z3_code,
                stdout         = result.stdout,
                stderr         = result.stderr,
                success        = result.success and not is_unsat,
                is_sat         = is_sat,
                is_unsat       = is_unsat,
                refined_plan   = str(next_refined) if next_refined is not None else None,
                elapsed_s      = elapsed,
                backend        = "z3",
            ))

            if result.success and not is_unsat:
                return VerificationOutcome(
                    z3_code      = z3_code,
                    execution    = result,
                    refined_plan = refined_plan,
                    attempts     = recorded_attempts,
                )

            if next_refined is not None:
                plan = next_refined

        # All attempts exhausted — return last failure
        return VerificationOutcome(
            z3_code      = z3_code,
            execution    = result,
            refined_plan = refined_plan,
            attempts     = recorded_attempts,
        )

    def nusmv_verify(
        self,
        proposed_plan: ResearchPlan,
        max_retries: int = 10,
        nusmv_timeout: int = 30,
    ) -> VerificationOutcome:
        """Verify a ResearchPlan using NuSMV model checking.

        Steps:
          1. Extract a Kripke structure from the plan.
          2. Generate a NuSMV model encoding the plan's states and CTL/LTL specs.
          3. Execute nusmv in a subprocess and capture all output.
          4. If specs are falsified (UNSAT) or execution fails, refine the plan and retry.

        Args:
            proposed_plan: The plan produced by the Researcher module.
            max_retries:   How many times to retry after a failure or falsified spec.
            nusmv_timeout: Seconds before the subprocess is killed.

        Returns:
            VerificationOutcome containing the nusmv_code, execution result, and
            an optional refined plan if refinement was needed.
        """
        # Step 1 — extract Kripke structure
        kripke_pred = self.kripke_formalizer(proposed_plan=proposed_plan)
        kripke = kripke_pred.kripke

        plan = proposed_plan
        refined_plan: Optional[ResearchPlan] = None
        nusmv_code = ""
        result = None
        recorded_attempts: list[VerificationAttempt] = []

        for attempt in tqdm.tqdm(range(1 + max_retries)):
            # Step 2 — generate NuSMV model
            nusmv_pred = self.nusmv_formalizer(kripke=kripke, proposed_plan=plan)
            nusmv_code = nusmv_pred.nusmv_code

            # Step 3 — execute with nusmv CLI
            t0 = time.time()
            result = execute_nusmv_code(nusmv_code, timeout=nusmv_timeout)
            elapsed = round(time.time() - t0, 2)
            print(f"[nusmv_verifier] attempt {attempt + 1}/{1 + max_retries} — {'OK' if result.success else 'FAILED'}")
            if result.stdout:
                print("[nusmv_verifier] output:\n", result.stdout)

            # Check for UNSAT (falsified spec) in output
            is_unsat = "is false" in result.stdout.lower()
            is_sat = result.success and not is_unsat

            # Step 4 — refine plan on failure or falsified spec
            next_refined: Optional[ResearchPlan] = None
            error_msg = result.stderr or (
                f"NuSMV spec falsified (UNSAT): the plan does not satisfy the required properties.\n{result.stdout}"
                if is_unsat else result.stdout
            )
            if not is_sat and attempt < max_retries:
                refine_pred = self.plan_refiner(
                    z3_code=nusmv_code,
                    execution_error=error_msg,
                    plan=plan,
                    kripke_structure=kripke,
                )
                next_refined = refine_pred.refined_plan
                refined_plan = next_refined

            recorded_attempts.append(VerificationAttempt(
                attempt_number = attempt + 1,
                z3_code        = nusmv_code,
                stdout         = result.stdout,
                stderr         = result.stderr,
                success        = is_sat,
                is_sat         = is_sat,
                is_unsat       = is_unsat,
                refined_plan   = str(next_refined) if next_refined is not None else None,
                elapsed_s      = elapsed,
                backend        = "nusmv",
            ))

            if is_sat:
                return VerificationOutcome(
                    z3_code=nusmv_code,
                    execution=result,
                    refined_plan=refined_plan,
                    attempts=recorded_attempts,
                )

            if next_refined is not None:
                plan = next_refined

        # All attempts exhausted
        return VerificationOutcome(
            z3_code=nusmv_code,
            execution=result,
            refined_plan=refined_plan,
            attempts=recorded_attempts,
        )

    def verify(
        self,
        proposed_plan: ResearchPlan,
        z3_max_retries: int = 10,
        nusmv_max_retries: int = 2,
        z3_timeout: int = 30,
        nusmv_timeout: int = 30,
    ) -> VerificationOutcome:
        """Run Z3 verification loop then NuSMV verification loop (3 iterations).

        First runs the Z3 loop up to z3_max_retries times. If Z3 produces a
        refined plan, that refined plan is fed into the NuSMV loop. NuSMV runs
        up to nusmv_max_retries times (default 2 → 3 total iterations).

        Returns a combined VerificationOutcome whose attempts list contains
        all Z3 attempts followed by all NuSMV attempts, and whose refined_plan
        and execution reflect the final NuSMV result.
        """
        # ── Z3 loop ───────────────────────────────────────────────────────────
        print("[verify] starting Z3 verification loop…")
        z3_outcome = self.forward(
            proposed_plan=proposed_plan,
            max_retries=z3_max_retries,
            z3_timeout=z3_timeout,
        )
        all_attempts = list(z3_outcome.attempts)

        # Feed the best plan Z3 produced into NuSMV
        nusmv_input = z3_outcome.refined_plan or proposed_plan

        # ── NuSMV loop (3 iterations = max_retries=2) ─────────────────────────
        print("[verify] starting NuSMV verification loop (3 iterations)…")
        nusmv_outcome = self.nusmv_verify(
            proposed_plan=nusmv_input,
            max_retries=nusmv_max_retries,
            nusmv_timeout=nusmv_timeout,
        )
        all_attempts.extend(nusmv_outcome.attempts)

        # Final outcome: execution and refined_plan come from NuSMV (last stage)
        return VerificationOutcome(
            z3_code      = nusmv_outcome.z3_code,
            execution    = nusmv_outcome.execution,
            refined_plan = nusmv_outcome.refined_plan or z3_outcome.refined_plan,
            attempts     = all_attempts,
        )
