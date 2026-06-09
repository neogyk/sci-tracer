import ast
import json
import os
import re
import sys
import subprocess
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field, asdict
import dspy
from pydantic import BaseModel, Field

from agents.researcher import ResearchPlan, Task, Plan
from agents.experience import with_experience, default_store


# ── DSPy signatures ───────────────────────────────────────────────────────────

class SignatureExecution(dspy.Signature):
    research_plan: Plan = dspy.InputField(desc="Research Plan with initial conditions and goals")
    task: Task = dspy.InputField(desc="Task to execute")
    code: dspy.Code["python"] = dspy.OutputField(
        desc="Executable Python code for the input task, the code must have strong typing, 'don't put code=' at the beginn"
    )


class IntegrationSignature(dspy.Signature):
    """Connect multiple standalone Python modules into a single runnable program.

    Analyse all input files, resolve cross-module imports, fix missing imports,
    deduplicate definitions, and emit ONE orchestrator script that imports and
    calls each module in dependency order, then runs everything end-to-end.
    """
    research_plan: Plan = dspy.InputField(desc="Research Plan with initial conditions and goals")
    file_listing: str = dspy.InputField(
        desc=(
            "Newline-separated list of absolute file paths and their full source code, "
            "formatted as:\n  === path/to/file.py ===\n<source>\n=== ... ==="
        )
    )
    sandbox_dir: str = dspy.InputField(
        desc="Absolute path to the sandbox directory where all files live"
    )
    orchestrator_code: dspy.Code["python"] = dspy.OutputField(
        desc=(
            "Self-contained Python script (saved as run.py in sandbox_dir) that is "
            "the finalized version: "
            "1) adds sandbox_dir to sys.path, "
            "2) imports every module, "
            "3) adds logic that displays coherence between files in file_listing, "
            "4) runs the produced file script, "
            "5) prints all results. "
            "6) If script has to return an object it must return what was stated in goal."
        )
    )


class CodeRepairSignature(dspy.Signature):
    """Repair broken Python code for a single task step.

    You are given Python code that either failed AST validation (syntax error)
    or failed at runtime (exception / wrong output).  Fix ONLY the errors
    described in `error_message`.  Keep the logic intact; do not rewrite
    unnecessarily.  The repaired code must be self-contained and importable.
    """
    research_plan: Plan = dspy.InputField(
        desc="Research Plan providing context about what the code should accomplish"
    )
    task: Task = dspy.InputField(
        desc="The specific task this code implements"
    )
    broken_code: str = dspy.InputField(
        desc="The Python code that contains errors"
    )
    error_message: str = dspy.InputField(
        desc="The exact error: SyntaxError description or runtime traceback"
    )
    fixed_code: dspy.Code["python"] = dspy.OutputField(
        desc="The corrected, self-contained Python code with the errors resolved"
    )


class PostconditionCheckerSignature(dspy.Signature):
    """Generate a self-contained Python snippet that asserts the postcondition
    of a task step against the function(s) defined in the generated code.

    The snippet must:
      1. Import only stdlib / numpy / scipy — no project-specific modules.
      2. Define small, concrete test inputs (not random — use fixed literals).
      3. Call the function(s) defined in `generated_code` with those inputs.
      4. Assert the postcondition: type, shape, value range, sign, or exact value
         where the postcondition specifies it.
      5. Print a one-line confirmation on success, e.g.
         print("postcondition OK: <what was checked>")

    The snippet must NOT redefine the functions — it will be appended directly
    after the generated code and executed in the same namespace.
    If the postcondition cannot be turned into a concrete assertion
    (e.g. it is purely qualitative), output an empty string.
    """
    task:           Task = dspy.InputField(
        desc="Task whose postcondition must be verified"
    )
    generated_code: dspy.Code["python"]  = dspy.InputField(
        desc="The Python code that was generated for this task"
    )
    assertion_snippet: str = dspy.OutputField(
        desc=(
            "A short Python snippet (no function definitions) that imports what "
            "it needs, constructs minimal test inputs, calls the function, and "
            "asserts the postcondition.  Empty string if no assertion is possible."
        )
    )


class OrchestratorRepairSignature(dspy.Signature):
    """Repair the orchestrator script (run.py) after a failed dry run.

    You are given an orchestrator that failed either AST validation or at
    runtime.  Fix only the reported errors.  The repaired script must:
      1. Add sandbox_dir to sys.path.
      2. Import and call every step module in dependency order.
      3. Print all results.
    Do not change the high-level structure unless it is the root cause of the error.
    """
    research_plan: Plan = dspy.InputField(
        desc="Research Plan providing context about what the orchestrator should do"
    )
    file_listing: str = dspy.InputField(
        desc="The step files the orchestrator must integrate (paths + source)"
    )
    broken_code: str = dspy.InputField(
        desc="The orchestrator source that contains errors"
    )
    error_message: str = dspy.InputField(
        desc="The exact error: SyntaxError description or runtime traceback + stderr"
    )
    fixed_code: dspy.Code["python"] = dspy.OutputField(
        desc="The corrected orchestrator script with the errors resolved"
    )


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class StepFile:
    task_name: str
    file_path: str
    code: str


@dataclass
class StepLog:
    """Execution record for one plan step."""
    index:      int
    task_name:  str
    tool:       str
    reasoning:  str   # DSPy ChainOfThought rationale
    code:       str   # generated Python source
    file_path:  str
    stdout:     str
    success:    bool
    repairs:    int


@dataclass
class ExecutionResult:
    orchestrator_path: str
    stdout: str = ""
    stderr: str = ""
    success: bool = False
    repair_attempts: int = 0
    step_logs: list = field(default_factory=list)   # list[StepLog]

    @property
    def output(self) -> str:
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append("--- stderr ---\n" + self.stderr)
        return "\n".join(parts)


# ── AST verification ──────────────────────────────────────────────────────────

def _ast_verify(code: str) -> tuple[bool, str]:
    """Parse *code* with the Python AST.

    Returns ``(True, "")`` on success, or ``(False, error_message)`` when a
    ``SyntaxError`` is detected.  Other exceptions are treated as errors too.
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        msg = f"SyntaxError at line {exc.lineno}: {exc.msg}"
        if exc.text:
            msg += f"\n  source: {exc.text.rstrip()}"
        return False, msg
    except Exception as exc:
        return False, f"AST parse failed: {exc}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_fences(code) -> str:
    """Remove markdown fences and replace typographic Unicode characters that
    cause SyntaxError when pasted verbatim into Python source.

    Observed failures in the SciCode run (all 0% pass-rate):
      U+2011 non-breaking hyphen  → plain hyphen-minus
      U+2013 en dash              → plain hyphen-minus
      U+2014 em dash              → plain hyphen-minus
      U+2018/U+2019 curly quotes  → straight apostrophe
      U+201C/U+201D curly quotes  → straight double-quote
      U+2713 checkmark            → removed (comment decoration only)
      U+00A0 non-breaking space   → regular space
    """
    code = str(code).strip()
    # strip markdown fences
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    # replace typographic characters that break the Python parser
    _UNICODE_REPLACEMENTS = [
        ("‑", "-"),   # non-breaking hyphen
        ("–", "-"),   # en dash
        ("—", "-"),   # em dash
        ("‘", "'"),   # left single quotation mark
        ("’", "'"),   # right single quotation mark
        ("“", '"'),   # left double quotation mark
        ("”", '"'),   # right double quotation mark
        ("✓", ""),    # check mark — decoration only
        ("✔", ""),    # heavy check mark — decoration only
        (" ", " "),   # non-breaking space
    ]
    for bad, good in _UNICODE_REPLACEMENTS:
        code = code.replace(bad, good)
    return code.strip()


_CODE_WRAPPER = re.compile(r'^code=([\'"])(.*)\1\s*$', re.DOTALL)

def _unwrap_code_assignment(code: str) -> str:
    """Strip DSPy's code='...' wrapper when the LLM mimics its serialization format."""
    m = _CODE_WRAPPER.match(code.strip())
    if m:
        return m.group(2).replace("\\'", "'").replace('\\"', '"')
    return code


def _safe_filename(name: str, index: int) -> str:
    slug = re.sub(r"[^\w]+", "_", name.strip().lower())[:40] or "step"
    return f"step_{index:02d}_{slug}.py"


def _ensure_python_code(code: str) -> str:
    """Guarantee the string written to a .py file is actual Python source.

    Applies all cleaning passes in order:
      1. Strip markdown fences and typographic characters (_strip_fences).
      2. Unwrap DSPy's code='...' serialization wrapper (_unwrap_code_assignment).
      3. AST-parse the result; if it still fails to parse, attempt one more
         round of both passes in case the wrapper was nested or double-escaped.
    """
    cleaned = _unwrap_code_assignment(_strip_fences(code))
    ok, _ = _ast_verify(cleaned)
    if not ok:
        # second pass: the string inside may itself be fenced or wrapped again
        cleaned = _unwrap_code_assignment(_strip_fences(cleaned))
    return cleaned


def _write_step_file(sandbox: Path, index: int, task: Task, code: str) -> StepFile:
    """Write generated code to a .py file in the sandbox and return metadata."""
    filename = _safe_filename(task.state, index)
    file_path = sandbox / filename
    clean_code = _ensure_python_code(code)
    file_path.write_text(clean_code, encoding="utf-8")
    return StepFile(
        task_name=task.state,
        file_path=str(file_path),
        code=clean_code,
    )


def _build_file_listing(step_files: List[StepFile]) -> str:
    parts = []
    for sf in step_files:
        parts.append(f"=== {sf.file_path} ===")
        parts.append(sf.code)
    return "\n".join(parts)


def _run_script(script_path: str, timeout: int = 60) -> ExecutionResult:
    """Execute a Python script in a subprocess and capture its output."""
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ExecutionResult(
            orchestrator_path=script_path,
            stdout=proc.stdout,
            stderr=proc.stderr,
            success=proc.returncode == 0,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            orchestrator_path=script_path,
            stderr=f"TimeoutExpired: execution exceeded {timeout}s",
            success=False,
        )
    except Exception as exc:
        return ExecutionResult(
            orchestrator_path=script_path,
            stderr="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            success=False,
        )


# ── Per-run folder + history log ─────────────────────────────────────────────

class RunFolder:
    """Manages a per-problem directory under the sandbox root.

    Directory layout::

        sandbox/
          <run_id>/
            plan.json            ← ResearchPlan dict
            refined_plan.json    ← refined ResearchPlan (if verifier produced one)
            verification_trace.txt
            full_output.txt      ← concatenated pipeline output
            history.log          ← timestamped event log
            step_00_<name>.py
            step_01_<name>.py
            ...
            run.py               ← orchestrator
    """

    def __init__(self, root: Path, run_id: str) -> None:
        slug = re.sub(r"[^\w\-]", "_", run_id)[:80] or "run"
        self.path = root / slug
        self.path.mkdir(parents=True, exist_ok=True)
        self._events: list[str] = []

    # ── logging ───────────────────────────────────────────────────────────────

    def log(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        self._events.append(f"[{ts}] {msg}")

    def flush_history(self) -> None:
        (self.path / "history.log").write_text(
            "\n".join(self._events), encoding="utf-8"
        )

    # ── artifact writers ──────────────────────────────────────────────────────

    def save_text(self, filename: str, text: str) -> Path:
        p = self.path / filename
        p.write_text(str(text), encoding="utf-8")
        return p

    def save_json(self, filename: str, obj: object) -> Path:
        p = self.path / filename
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        return p

    def save_plan(self, plan: ResearchPlan, filename: str = "plan.json") -> Path:
        try:
            data = asdict(plan)
        except Exception:
            data = {"repr": str(plan)}
        return self.save_json(filename, data)

    def save_verification(
        self,
        trace: str,
        attempts: list,
        refined_plan=None,
    ) -> None:
        self.save_text("verification_trace.txt", trace)
        attempts_data = []
        for a in attempts:
            try:
                attempts_data.append({
                    "attempt_number": a.attempt_number,
                    "backend":        a.backend,
                    "success":        a.success,
                    "is_sat":         a.is_sat,
                    "is_unsat":       a.is_unsat,
                    "elapsed_s":      a.elapsed_s,
                    "stdout":         a.stdout,
                    "stderr":         a.stderr,
                    "z3_code":        a.z3_code,
                    "refined_plan":   a.refined_plan,
                })
            except Exception:
                attempts_data.append(str(a))
        self.save_json("verification_attempts.json", attempts_data)
        if refined_plan is not None:
            self.save_plan(refined_plan, "refined_plan.json")


# ── Executor module ───────────────────────────────────────────────────────────

class Executor(dspy.Module):
    """Generate, verify, repair, and execute the research plan as Python code.

    Pipeline
    --------
    For each task in the plan:
      1. Generate Python code via LLM (``SignatureExecution``).
      2. AST-verify the code; if invalid, repair with LLM (``CodeRepairSignature``)
         up to ``max_ast_retries`` times.
      3. Write the verified code to ``<sandbox>/step_NN_<name>.py``.
      3a. Execute the step file in a subprocess; on failure repair with
          ``CodeRepairSignature`` up to ``max_repair_retries`` times.

    Then:
      4. Call an integration LLM (``IntegrationSignature``) to build ``run.py``.
      5. AST-verify ``run.py``; if invalid, repair (``OrchestratorRepairSignature``).
      6. Dry-run ``run.py``; if it fails, repair the orchestrator and retry
         up to ``max_repair_retries`` times.
    """

    # Tools that are dispatched to SkillsReActAgent instead of code generation.
    _SKILL_TOOLS = {"list_skills", "activate_skill", "run_skill_script", "read_skill_resource"}

    def __init__(
        self,
        sandbox_dir: Optional[str] = None,
        run_timeout: int = 60,
        max_ast_retries: int = 3,
        max_repair_retries: int = 3,
        max_postcondition_retries: int = 2,
        skills_dir: Optional[str] = None,
    ):
        super().__init__()
        self.executor                   = dspy.ChainOfThought(SignatureExecution)
        self.postcondition_checker      = dspy.Predict(PostconditionCheckerSignature)
        self.integrator         = dspy.ChainOfThought(IntegrationSignature)
        self.code_repairer      = dspy.ChainOfThought(CodeRepairSignature)
        self.orch_repairer      = dspy.ChainOfThought(OrchestratorRepairSignature)
        self.sandbox_dir               = sandbox_dir
        self.run_timeout               = run_timeout
        self.max_ast_retries           = max_ast_retries
        self.max_repair_retries        = max_repair_retries
        self.max_postcondition_retries = max_postcondition_retries

        # ── dspy-skills integration ───────────────────────────────────────────
        self._skills_agent = None
        try:
            from dspy_skills import SkillsReActAgent, SkillsConfig
            cfg = SkillsConfig.from_yaml(skills_dir) if skills_dir else SkillsConfig()
            self._skills_agent = SkillsReActAgent(config=cfg)
            print("[executor] dspy-skills SkillsReActAgent loaded.")
        except ImportError:
            print("[executor] dspy-skills not installed — skill steps will fall back to code generation.")
        except Exception as exc:
            print(f"[executor] dspy-skills init failed ({exc}) — skill steps will fall back to code generation.")

    # ── dspy-skills dispatcher ────────────────────────────────────────────────

    def _run_skill_step(self, task: Task) -> str | None:
        """Run a skill-tool step via SkillsReActAgent.

        Returns the agent's stdout string on success, or None if the skills
        agent is unavailable (caller falls back to code generation).
        """
        if self._skills_agent is None:
            return None
        query = f"{task.state}\n{task.description}"
        try:
            result = self._skills_agent(query)
            return str(result)
        except Exception as exc:
            print(f"  [executor] skill step failed ({exc}); falling back to code generation.")
            return None

    # ── internal: run postcondition snippet in a subprocess ──────────────────

    @staticmethod
    def _run_postcondition(code: str, snippet: str) -> tuple[bool, str]:
        """Concatenate function code + assertion snippet and execute.

        Returns (passed, error_message).  error_message is empty on success.
        The snippet is appended after the function definitions so it runs in
        the same namespace — no imports or re-definitions needed.
        """
        if not snippet.strip():
            return True, ""

        combined = code + "\n\n# --- postcondition check ---\n" + snippet
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(combined)
                tmp = f.name

            proc = subprocess.run(
                [sys.executable, tmp],
                capture_output=True, text=True, timeout=15,
            )
            if proc.returncode == 0:
                return True, ""
            error = (proc.stderr or proc.stdout or "non-zero exit").strip()
            return False, error

        except subprocess.TimeoutExpired:
            return False, "PostconditionTimeout: assertion snippet exceeded 15 s"
        except Exception as exc:
            return False, f"PostconditionError: {exc}"
        finally:
            try:
                import os; os.unlink(tmp)
            except OSError:
                pass

    # ── internal: AST-verify + repair a single step file ─────────────────────

    def _verified_step_code(
        self,
        raw_code: str,
        task: Task,
        research_plan: ResearchPlan,
    ) -> str:
        """Return AST-valid, postcondition-passing code for *task*.

        Two-stage loop:
          Stage 1 — AST verification: syntax must parse cleanly.
          Stage 2 — Postcondition verification: the LLM generates a minimal
                    assertion snippet from the task's formal postcondition and
                    runs it against the generated code in a subprocess.
                    On failure the code repairer is called with the assertion
                    traceback, giving the model concrete feedback on what is
                    semantically wrong (wrong type, shape, sign, value range).
        """
        # ── Stage 1: AST ──────────────────────────────────────────────────────
        code = _strip_fences(raw_code)

        for attempt in range(1 + self.max_ast_retries):
            ok, err = _ast_verify(code)
            if ok:
                if attempt > 0:
                    print(f"    [ast] step repaired after {attempt} attempt(s)")
                break

            print(f"    [ast] step code invalid (attempt {attempt + 1}): {err}")
            if attempt == self.max_ast_retries:
                print("    [ast] max AST retries reached — using last code as-is")
                break

            pred = self.code_repairer(
                research_plan=research_plan,
                task=task,
                broken_code=code,
                error_message=err,
            )
            code = _strip_fences(pred.fixed_code)

        # ── Stage 2: Postcondition ────────────────────────────────────────────
        # Generate the assertion snippet once; reuse it across repair iterations.
        try:
            checker_pred = self.postcondition_checker(
                task=task,
                generated_code=code,
            )
            snippet = _strip_fences(checker_pred.assertion_snippet)
        except Exception as exc:
            print(f"    [postcondition] snippet generation failed: {exc}")
            snippet = ""

        for attempt in range(1 + self.max_postcondition_retries):
            passed, error = self._run_postcondition(code, snippet)
            if passed:
                if attempt > 0:
                    print(f"    [postcondition] passed after {attempt} repair(s)")
                break

            print(f"    [postcondition] FAILED (attempt {attempt + 1}): {error[:120]}")
            if attempt == self.max_postcondition_retries:
                print("    [postcondition] max retries reached — keeping last code")
                break

            # Feed the assertion traceback back to the code repairer so it
            # knows *what* is wrong semantically, not just syntactically.
            spec = task.hoare_triplet.postcondition
            error_msg = (
                f"Postcondition assertion failed.\n"
                f"Postcondition (formal spec): {spec}\n"
                f"Assertion snippet:\n{snippet}\n"
                f"Error:\n{error}"
            )
            pred = self.code_repairer(
                research_plan=research_plan,
                task=task,
                broken_code=code,
                error_message=error_msg,
            )
            code = _strip_fences(pred.fixed_code)

            # Re-generate the snippet against the repaired code so it matches
            # any renamed variables or changed signatures.
            try:
                checker_pred = self.postcondition_checker(
                    task=task,
                    generated_code=code,
                )
                snippet = _strip_fences(checker_pred.assertion_snippet)
            except Exception:
                pass  # keep the previous snippet

        return code

    # ── internal: AST-verify + repair the orchestrator ────────────────────────

    def _verified_orchestrator(
        self,
        raw_code: str,
        file_listing: str,
        research_plan: ResearchPlan,
    ) -> str:
        """Return AST-valid orchestrator code, repairing up to ``max_ast_retries`` times."""
        code = _ensure_python_code(raw_code)

        for attempt in range(1 + self.max_ast_retries):
            ok, err = _ast_verify(code)
            if ok:
                if attempt > 0:
                    print(f"  [ast] orchestrator repaired after {attempt} attempt(s)")
                return code

            print(f"  [ast] orchestrator invalid (attempt {attempt + 1}): {err}")
            if attempt == self.max_ast_retries:
                print("  [ast] max AST retries reached — using last orchestrator as-is")
                break

            pred = self.orch_repairer(
                research_plan=research_plan,
                file_listing=file_listing,
                broken_code=code,
                error_message=err,
            )
            code = _strip_fences(pred.fixed_code)

        return code

    # ── internal: dry-run repair loop ─────────────────────────────────────────

    def _run_with_repair(
        self,
        run_path: Path,
        file_listing: str,
        research_plan: ResearchPlan,
    ) -> ExecutionResult:
        """Execute run.py; on failure repair the orchestrator and retry."""
        result = _run_script(str(run_path), timeout=self.run_timeout)
        print(f"  [executor] run {'OK' if result.success else 'FAILED'}")

        repair_attempts = 0
        while not result.success and repair_attempts < self.max_repair_retries:
            repair_attempts += 1
            error_msg = (result.stderr or result.stdout or "unknown error")[:2000]
            print(
                f"  [executor] repair attempt {repair_attempts}/{self.max_repair_retries} "
                f"— error: {error_msg[:120]}"
            )

            broken_code = run_path.read_text(encoding="utf-8")

            # AST-check first: if it's a syntax error, repair from there
            ast_ok, ast_err = _ast_verify(broken_code)
            repair_error = ast_err if not ast_ok else error_msg

            pred = self.orch_repairer(
                research_plan=research_plan,
                file_listing=file_listing,
                broken_code=broken_code,
                error_message=repair_error,
            )
            fixed_code = _ensure_python_code(pred.fixed_code)

            # Validate the repaired code before writing
            ast_ok, ast_err = _ast_verify(fixed_code)
            if not ast_ok:
                print(f"  [executor] repaired code still has AST error: {ast_err}")
                # Try once more to fix the syntax
                pred2 = self.orch_repairer(
                    research_plan=research_plan,
                    file_listing=file_listing,
                    broken_code=fixed_code,
                    error_message=ast_err,
                )
                fixed_code = _ensure_python_code(pred2.fixed_code)

            run_path.write_text(fixed_code, encoding="utf-8")
            print(f"  [executor] repaired orchestrator written: {run_path}")

            result = _run_script(str(run_path), timeout=self.run_timeout)
            print(f"  [executor] re-run {'OK' if result.success else 'FAILED'}")

        result.repair_attempts = repair_attempts
        if result.stdout:
            print("[executor] stdout:\n", result.stdout)
        if result.stderr:
            print("[executor] stderr:\n", result.stderr)
        return result

    # ── public forward ────────────────────────────────────────────────────────

    def forward(
        self,
        instructions:  Plan,
        research_plan: ResearchPlan,
        run_id:        str = "",
    ) -> ExecutionResult:
        # ── resolve sandbox and create per-run folder ──────────────────────────
        _project_root   = Path(__file__).resolve().parent.parent
        sandbox_root    = Path(self.sandbox_dir) if self.sandbox_dir else _project_root / "sandbox"
        run_folder      = RunFolder(sandbox_root, run_id or "default")
        sandbox         = run_folder.path   # all step files live inside the run folder

        run_folder.log(f"run_id={run_id or 'default'}")
        run_folder.log(f"sandbox={sandbox}")
        run_folder.save_plan(research_plan, "plan.json")
        run_folder.log("plan.json saved")

        # ── steps 1–3: generate, AST-verify/repair, persist each task ─────────
        step_files: List[StepFile] = []
        step_logs:  List[StepLog]  = []
        for idx, task in enumerate(instructions.steps.instructions):
            print(f"[executor] generating step {idx:02d}: {task.state[:60]}")
            run_folder.log(f"step {idx:02d} start: {task.state[:80]}")

            # ── dispatch skill-tool steps to SkillsReActAgent ─────────────────
            if task.tool in self._SKILL_TOOLS:
                skill_output = self._run_skill_step(task)
                if skill_output is not None:
                    raw_code = (
                        f"# skill step: {task.tool}\n"
                        f"def step_{idx:02d}(context: dict) -> dict:\n"
                        f"    return {{'output': {repr(skill_output)}}}\n"
                    )
                    sf = _write_step_file(sandbox, idx, task, raw_code)
                    step_files.append(sf)
                    step_logs.append(StepLog(
                        index=idx, task_name=task.state, tool=task.tool,
                        reasoning="", code=raw_code, file_path=sf.file_path,
                        stdout=skill_output, success=True, repairs=0,
                    ))
                    continue

            pred     = self.executor(task=task, research_plan=research_plan)
            reasoning = getattr(pred, "rationale", "") or getattr(pred, "reasoning", "")
            try:
                raw_code = pred.code.code
            except Exception:
                raw_code = str(pred.code)
            raw_code = _unwrap_code_assignment(raw_code)

            verified_code = self._verified_step_code(raw_code, task, research_plan)
            sf = _write_step_file(sandbox, idx, task, verified_code)

            # ── step execution test + repair ──────────────────────────────────
            run_result = _run_script(sf.file_path, timeout=self.run_timeout)
            step_repairs = 0
            if run_result.success:
                print(f"[executor] step {idx:02d} executed OK")
                run_folder.log(f"step {idx:02d} execution OK")
            else:
                print(f"[executor] step {idx:02d} execution FAILED — attempting repair")
                for repair in range(self.max_repair_retries):
                    step_repairs += 1
                    error_msg = (run_result.stderr or run_result.stdout or "unknown error")[:2000]
                    run_folder.log(f"step {idx:02d} repair {repair + 1}: {error_msg[:120]}")
                    print(
                        f"[executor] step {idx:02d} repair {repair + 1}/{self.max_repair_retries}"
                        f" — {error_msg[:80]}"
                    )

                    ast_ok, ast_err = _ast_verify(sf.code)
                    repair_error = ast_err if not ast_ok else error_msg

                    pred2 = self.code_repairer(
                        research_plan=research_plan,
                        task=task,
                        broken_code=sf.code,
                        error_message=repair_error,
                    )
                    repaired = _strip_fences(_unwrap_code_assignment(
                        pred2.fixed_code.code
                        if hasattr(pred2.fixed_code, "code")
                        else str(pred2.fixed_code)
                    ))

                    Path(sf.file_path).write_text(repaired, encoding="utf-8")
                    sf = StepFile(task_name=sf.task_name, file_path=sf.file_path, code=repaired)

                    run_result = _run_script(sf.file_path, timeout=self.run_timeout)
                    if run_result.success:
                        print(f"[executor] step {idx:02d} repaired and executed OK (repair {repair + 1})")
                        run_folder.log(f"step {idx:02d} repaired OK after {repair + 1} attempt(s)")
                        break
                else:
                    print(f"[executor] step {idx:02d} could not be repaired — keeping last version")
                    run_folder.log(f"step {idx:02d} repair exhausted after {self.max_repair_retries} attempt(s)")

            step_files.append(sf)
            step_logs.append(StepLog(
                index=idx, task_name=task.state, tool=task.tool,
                reasoning=reasoning, code=sf.code, file_path=sf.file_path,
                stdout=run_result.stdout or "", success=run_result.success,
                repairs=step_repairs,
            ))
            run_folder.log(f"step {idx:02d} written: {sf.file_path}")
            print(f"[executor] wrote step {idx:02d}: {sf.file_path}")

        # ── step 4: build orchestrator via integration LLM ────────────────────
        file_listing     = _build_file_listing(step_files)
        integration_pred = self.integrator(
            file_listing=file_listing,
            sandbox_dir=str(sandbox),
            research_plan=research_plan,
        )
        raw_orch = (
            integration_pred.orchestrator_code.code
            if hasattr(integration_pred.orchestrator_code, "code")
            else str(integration_pred.orchestrator_code)
        )
        raw_orch = _unwrap_code_assignment(raw_orch)

        # ── step 5: AST-verify/repair orchestrator before writing ─────────────
        verified_orch = self._verified_orchestrator(raw_orch, file_listing, research_plan)
        run_path = sandbox / "run.py"
        run_path.write_text(_ensure_python_code(verified_orch), encoding="utf-8")
        run_folder.log(f"orchestrator written: {run_path}")
        print(f"[executor] orchestrator written: {run_path}")

        # ── step 6: dry-run with repair loop ──────────────────────────────────
        result = self._run_with_repair(run_path, file_listing, research_plan)

        # ── step 7: flush history log ─────────────────────────────────────────
        run_folder.log(
            f"execution {'OK' if result.success else 'FAILED'} "
            f"(repairs={result.repair_attempts})"
        )
        if result.stdout:
            run_folder.save_text("stdout.txt", result.stdout)
        if result.stderr:
            run_folder.save_text("stderr.txt", result.stderr)
        run_folder.flush_history()
        print(f"[executor] run artifacts saved → {run_folder.path}")

        result.step_logs = step_logs
        return result

    #@with_experience(store=default_store, enable=True)
    def execute(
        self,
        task_summary:  str,
        instructions:  Plan         = None,
        research_plan: ResearchPlan = None,
        run_id:        str          = "",
    ) -> ExecutionResult:
        """Experience-tracked entry point for forward().

        ``task_summary`` is a short string logged in the experience store.
        Pass the same ``instructions`` and ``research_plan`` you would pass
        to ``forward()`` directly.
        """
        return self.forward(
            instructions=instructions,
            research_plan=research_plan,
            run_id=run_id,
        )
