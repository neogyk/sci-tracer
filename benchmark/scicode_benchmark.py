"""SciCode benchmark for SciTracer.

Evaluates the SciTracer agent pipeline on SciCode — a scientist-curated coding
benchmark (paper: https://huggingface.co/papers/2407.13168) covering 80 main
research problems decomposed into 338 subproblems across mathematics, physics,
chemistry, biology, and materials science.

Reference
---------
"SciCode: A Research Coding Benchmark Curated by Scientists"
Minyang Tian et al., 2024.  https://scicode-bench.github.io/

Evaluation modes
----------------
1. **Background** (``--background``): provide the optional scientific background
   text together with the problem description.
2. **Gold context** (default) / **Generated context** (``--no-gold-context``):
   when evaluating subproblem *k*, pass the gold-standard solutions for
   subproblems 1…k-1 as context (gold) or use the agent's own prior solutions.

Metric
------
Pass@1 at **subproblem level** and **main problem level**.
A main problem is solved only if *all* of its subproblems pass.

Usage
-----
::

    python benchmark/scicode_benchmark.py

    # Use scientific background text:
    python benchmark/scicode_benchmark.py --background

    # Use agent-generated (not gold) solutions for prior subproblems:
    python benchmark/scicode_benchmark.py --no-gold-context

    # Limit to first N main problems, skip Executor:
    python benchmark/scicode_benchmark.py --limit 5 --no-executor

    # Custom dataset split and output paths:
    python benchmark/scicode_benchmark.py \\
        --split test \\
        --results-path benchmark/results/scicode_results.json \\
        --experience-path benchmark/results/scicode_experience.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

# ── path bootstrap ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dspy
from datasets import load_dataset

from benchmark.experience import Experience, MistakeCategory

log = logging.getLogger("benchmark.scicode")

# HuggingFace dataset identifier
_HF_DATASET = "SciCode1/SciCode"

# Code-execution timeout per subproblem (seconds)
_EXEC_TIMEOUT = 60


# ══════════════════════════════════════════════════════════════════════════════
# Code execution
# ══════════════════════════════════════════════════════════════════════════════

def _run_code(code: str, test_code: str, timeout: int = _EXEC_TIMEOUT) -> tuple[bool, str]:
    """Execute ``code`` followed by ``test_code`` in a subprocess.

    Returns (passed, stderr_or_output).  *passed* is True when the process
    exits with return-code 0.
    """
    combined = "\n\n".join([
        "import sys, math, json",
        "try:",
        "    import numpy as np",
        "    import scipy",
        "    import sympy",
        "except ImportError:",
        "    pass",
        "",
        code,
        "",
        test_code,
    ])

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(combined)
        tmp = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = proc.returncode == 0
        output = proc.stdout + proc.stderr
        return passed, output.strip()
    except subprocess.TimeoutExpired:
        return False, f"TimeoutExpired ({timeout}s)"
    except Exception as exc:
        return False, str(exc)
    finally:
        try:
            Path(tmp).unlink()
        except OSError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Code extraction from LLM / agent output
# ══════════════════════════════════════════════════════════════════════════════

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_code(text: str) -> str:
    """Extract the first fenced Python code block from *text*.

    Falls back to the raw text if no fence is found.
    """
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


class SciCodeExtractor(dspy.Signature):
    """Extract a complete, self-contained Python solution from agent output.

    The solution must implement every function mentioned in the subproblem and
    must be executable without interactive input.  Output ONLY the Python code
    — no markdown fences, no explanations.
    """

    subproblem_description: str = dspy.InputField(
        desc="The scientific subproblem to solve."
    )
    prior_context: str = dspy.InputField(
        desc="Code and results from previous subproblems (may be empty)."
    )
    agent_output: str = dspy.InputField(
        desc="Full output produced by the SciTracer pipeline."
    )
    python_code: str = dspy.OutputField(
        desc=(
            "Complete, runnable Python code that solves the subproblem. "
            "Include all necessary imports.  Do NOT include markdown fences."
        )
    )


class LLMCodeExtractor(dspy.Module):
    """LLM-based Python code extractor for SciCode subproblems."""

    def __init__(self) -> None:
        super().__init__()
        self._extract = dspy.Predict(SciCodeExtractor)

    def forward(
        self,
        subproblem_description: str,
        prior_context: str,
        agent_output: str,
    ) -> str:
        """Returns extracted Python code string, or empty string on failure."""
        # First try cheap regex extraction
        code = _extract_code(agent_output)
        if code and len(code) > 30:
            return code

        try:
            pred = self._extract(
                subproblem_description=subproblem_description[:800],
                prior_context=prior_context[:600],
                agent_output=agent_output[:2000],
            )
            return pred.python_code.strip()
        except Exception as exc:
            log.warning("LLM code extraction failed: %s", exc)
            return ""


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ══════════════════════════════════════════════════════════════════════════════

def _build_subproblem_prompt(
    main_problem_description: str,
    main_background: str,
    main_io: str,
    required_dependencies: str,
    step: dict[str, Any],
    *,
    use_background: bool,
    prior_solutions: list[tuple[str, str]],   # [(description, code), ...]
) -> str:
    """Construct the prompt for a single subproblem step."""
    parts: list[str] = []

    # Required imports
    deps = (required_dependencies or "").strip()
    if deps:
        parts.append(f"## Required Imports\n\n```python\n{deps}\n```")

    # Optional scientific background (main + step level)
    if use_background:
        main_bg = (main_background or "").strip()
        step_bg = (step.get("step_background") or "").strip()
        bg = "\n\n".join(filter(None, [main_bg, step_bg]))
        if bg:
            parts.append(f"## Scientific Background\n\n{bg}")

    # Main problem description and I/O spec
    if main_problem_description:
        parts.append(f"## Problem\n\n{main_problem_description}")
    if main_io:
        parts.append(f"## Input / Output Specification\n\n{main_io}")

    # Prior subproblem solutions (gold or agent-generated)
    if prior_solutions:
        ctx_parts = []
        for idx, (desc, code) in enumerate(prior_solutions, 1):
            ctx_parts.append(
                f"### Step {idx} (already solved)\n"
                f"{desc}\n\n```python\n{code}\n```"
            )
        parts.append("## Context from Previous Steps\n\n" + "\n\n".join(ctx_parts))

    # Subproblem / step description
    sub_desc = (step.get("step_description_prompt") or "").strip()
    if sub_desc:
        parts.append(f"## Current Step\n\n{sub_desc}")

    # Required function signature
    func_header = (step.get("function_header") or "").strip()
    if func_header:
        parts.append(
            f"## Required Function Signature\n\n"
            f"Your solution must implement:\n```python\n{func_header}\n```"
        )

    parts.append(
        "## Instructions\n\n"
        "Write complete, self-contained Python code that solves the current step. "
        "Include all necessary imports.  Your code will be executed and tested."
    )

    return "\n\n".join(parts)


def load_scicode(
    split: str = "validation",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load SciCode1/SciCode from HuggingFace.

    Each row in the dataset is a **main problem** with subproblems nested in
    the ``sub_steps`` list.  Returns a list of normalised main-problem dicts::

        {
            "problem_id":              str,
            "problem_name":            str,
            "problem_description":     str,   # problem_description_main
            "problem_background":      str,   # problem_background_main
            "problem_io":              str,   # problem_io
            "required_dependencies":   str,
            "general_solution":        str,
            "general_tests":           list[str],
            "subproblems": [
                {
                    "step_number":            str,   # e.g. "1.1"
                    "step_description_prompt": str,
                    "step_background":        str,
                    "ground_truth_code":      str,
                    "function_header":        str,
                    "test_cases":             list[str],
                    "return_line":            str,
                },
                ...
            ],
        }

    Available splits: ``"validation"`` (15 problems) and ``"test"`` (65 problems).
    """
    log.info("Loading %s (split=%s)…", _HF_DATASET, split)
    try:
        dataset = load_dataset(_HF_DATASET, split=split, trust_remote_code=True)
    except Exception as exc:
        log.error("Failed to load dataset: %s", exc)
        raise

    result: list[dict[str, Any]] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break

        sub_steps = row.get("sub_steps") or []
        subproblems = []
        for step in sub_steps:
            if not isinstance(step, dict):
                continue
            # test_cases may be a list of strings or a single string
            tc = step.get("test_cases") or []
            if isinstance(tc, str):
                tc = [tc]
            subproblems.append({
                "step_number":             str(step.get("step_number", "")),
                "step_description_prompt": str(step.get("step_description_prompt", "")),
                "step_background":         str(step.get("step_background", "")),
                "ground_truth_code":       str(step.get("ground_truth_code", "")),
                "function_header":         str(step.get("function_header", "")),
                "test_cases":              [str(t) for t in tc],
                "return_line":             str(step.get("return_line", "")),
            })

        general_tests = row.get("general_tests") or []
        if isinstance(general_tests, str):
            general_tests = [general_tests]

        result.append({
            "problem_id":            str(row.get("problem_id", f"p{i}")),
            "problem_name":          str(row.get("problem_name", "")),
            "problem_description":   str(row.get("problem_description_main", "")),
            "problem_background":    str(row.get("problem_background_main", "")),
            "problem_io":            str(row.get("problem_io", "")),
            "required_dependencies": str(row.get("required_dependencies", "")),
            "general_solution":      str(row.get("general_solution", "")),
            "general_tests":         [str(t) for t in general_tests],
            "subproblems":           subproblems,
        })

    n_sub = sum(len(p["subproblems"]) for p in result)
    log.info("Loaded %d main problems (%d subproblems).", len(result), n_sub)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline runner
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    prompt:        str,
    researcher:    Any,
    formalizer:    Any,
    verifier:      Any,
    executor:      Any | None,
    code_extractor: LLMCodeExtractor,
    prior_context:  str,
    subproblem_description: str,
    run_id:        str = "",
) -> tuple[str, str]:
    """Run SciTracer on a single subproblem prompt.

    Returns (extracted_code, full_output_text).
    All pipeline artifacts (plan, verification trace, refined plan, scripts)
    are written to sandbox/<run_id>/ automatically.
    """
    from pathlib import Path
    from agents.executor import RunFolder

    # Resolve run folder (mirrors Executor's sandbox root logic)
    _project_root = Path(__file__).resolve().parent.parent
    sandbox_root  = _project_root / "sandbox"
    run_folder    = RunFolder(sandbox_root, run_id or "default")

    output_parts: list[str] = []
    import pdb;
    pdb.set_trace()
    # ── Researcher ──────────────────────────────────────────────────────────────
    run_folder.log("researcher start")
    plan = researcher.forward(prompt)
    run_folder.save_plan(plan, "plan.json")
    run_folder.log("plan.json saved")
    output_parts.append(f"[plan]\n{plan}")

    # ── Formalizer ─────────────────────────────────────────────────────────────
    run_folder.log("formalizer start")
    formalized_plan = formalizer.forward(plan)
    run_folder.save_text("formalized_plan.txt", str(formalized_plan))
    run_folder.log("formalized_plan.txt saved")
    output_parts.append(f"[formalized_plan]\n{formalized_plan}")

    # ── Verifier ───────────────────────────────────────────────────────────────
    run_folder.log("verifier start")
    verification_results = verifier.forward(formalized_plan)
    run_folder.save_verification(
        trace        = verification_results.trace,
        attempts     = getattr(verification_results, "attempts", []),
        refined_plan = verification_results.refined_plan,
    )
    run_folder.log(
        f"verification done — success={verification_results.success} "
        f"attempts={len(getattr(verification_results, 'attempts', []))}"
    )
    output_parts.append(f"[verification_trace]\n{verification_results.trace}")

    # ── Executor (optional) ────────────────────────────────────────────────────
    exec_stdout = ""
    if executor is not None:
        run_folder.log("executor start")
        refined     = verification_results.refined_plan or plan
        # Executor creates its own run folder at the same path via the same run_id
        code_result = executor.forward(refined, research_plan=plan, run_id=run_id)
        exec_stdout = code_result.output
        output_parts.append(f"[executor_output]\n{exec_stdout}")

    full_output = "\n\n".join(output_parts)
    run_folder.save_text("full_output.txt", full_output)

    # ── flush history ──────────────────────────────────────────────────────────
    run_folder.log("pipeline complete")
    run_folder.flush_history()

    # Code extraction: executor output first, then full pipeline text
    source = exec_stdout or full_output
    code = code_extractor.forward(
        subproblem_description=subproblem_description,
        prior_context=prior_context,
        agent_output=source,
    )
    return code, full_output


# ══════════════════════════════════════════════════════════════════════════════
# Mistake categorisation
# ══════════════════════════════════════════════════════════════════════════════

class MistakeCategorizer(dspy.Signature):
    """Categorise a scientific coding mistake into exactly one category.

    Categories:
      - Reasoning Error
      - Conceptual Misunderstanding
      - Missed Insight
      - Incomplete or Unclear Answer
      - Calculation Error
      - Invalid Approach
      - Question Misunderstanding
      - Notation Error
      - Hallucination

    Return the category name verbatim and a one-sentence explanation.
    """

    problem_description: str = dspy.InputField(desc="The original subproblem.")
    agent_output:        str = dspy.InputField(desc="The agent's generated code.")
    test_output:         str = dspy.InputField(desc="Test execution output (stderr/stdout).")
    category:    str = dspy.OutputField(desc="Exactly one category name from the list.")
    explanation: str = dspy.OutputField(desc="One sentence explaining the mistake.")


_CATEGORY_MAP: dict[str, MistakeCategory] = {
    cat.value.lower(): cat for cat in MistakeCategory
}


def _parse_category(raw: str) -> MistakeCategory:
    raw_lower = raw.strip().lower()
    for key, cat in _CATEGORY_MAP.items():
        if key in raw_lower:
            return cat
    return MistakeCategory.REASONING_ERROR


class ExperienceRecorder:
    """Records benchmark results into an Experience log with LLM categorisation."""

    def __init__(self, experience: Experience, categorize: bool = True) -> None:
        self.experience = experience
        self.categorize = categorize
        if categorize:
            self._categorizer = dspy.Predict(MistakeCategorizer)

    def record(
        self,
        question:     str,
        agent_output: str,
        test_output:  str,
        is_correct:   bool,
        run_id:       str,
        metadata:     dict[str, Any] | None = None,
    ) -> None:
        meta = metadata or {}
        if is_correct:
            self.experience.add_correct(
                question        = question,
                agent_output    = agent_output,
                expected_output = "pass",
                run_id          = run_id,
                metadata        = meta,
            )
            return

        category    = MistakeCategory.REASONING_ERROR
        explanation = ""
        if self.categorize and agent_output.strip():
            try:
                pred = self._categorizer(
                    problem_description = question[:600],
                    agent_output        = agent_output[:600],
                    test_output         = test_output[:400],
                )
                category    = _parse_category(pred.category)
                explanation = pred.explanation
            except Exception as exc:
                log.warning("Categorisation failed: %s", exc)

        self.experience.add_mistake(
            question        = question,
            agent_output    = agent_output,
            category        = category,
            expected_output = "pass",
            explanation     = explanation,
            run_id          = run_id,
            metadata        = meta,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main benchmark loop
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    *,
    split:           str        = "validation",
    limit:           int | None = None,
    use_background:  bool       = False,
    gold_context:    bool       = True,
    run_executor:    bool       = True,
    experience_path: Path       = Path("benchmark/results/scicode_experience.jsonl"),
    results_path:    Path       = Path("benchmark/results/scicode_results.json"),
    categorize:      bool       = True,
) -> dict[str, Any]:
    """Run the full SciCode benchmark and return a results summary dict.

    Parameters
    ----------
    split:
        HuggingFace split to use (``"dev"`` or ``"test"``).
    limit:
        Maximum number of main problems to evaluate.
    use_background:
        If True, prepend the optional scientific background text to each prompt.
    gold_context:
        If True (default), supply the gold-standard solution for each completed
        subproblem as context for the next one.  If False, use the agent's own
        generated code instead.
    run_executor:
        If False, skip the Executor agent and rely on the Researcher output for
        code extraction.
    experience_path / results_path:
        Output file locations.
    categorize:
        Whether to classify mistakes using an LLM.
    """

    # ── Deferred project imports ─────────────────────────────────────────────────
    from core.lm_setup import _setup_lm
    from core.log_setup import setup_logging
    from agents.researcher import Researcher
    from agents.formalizer import Formalizer
    from agents.verifier import Verifier
    from agents.executor import Executor
    from agents.rag import RAG
    from storage.vector_db import TrajectoryStore
    from storage.knowledge_graph import KnowledgeGraph

    setup_logging(log_dir="logs", console_level=logging.INFO)
    lm = _setup_lm()
    dspy.configure(lm=lm)

    # ── Initialise shared agents ─────────────────────────────────────────────────
    log.info("Initialising agents…")
    vector_db     = TrajectoryStore()
    graph_storage = KnowledgeGraph()

    rag        = RAG(vector_db, graph_storage)
    researcher = Researcher(rag=rag)
    formalizer = Formalizer()
    verifier   = Verifier()
    executor   = Executor() if run_executor else None

    code_extractor = LLMCodeExtractor()

    # ── Load dataset ─────────────────────────────────────────────────────────────
    main_problems = load_scicode(split=split, limit=limit)
    log.info("Running benchmark on %d main problems…", len(main_problems))

    experience    = Experience()
    recorder      = ExperienceRecorder(experience, categorize=categorize)
    per_main:  list[dict[str, Any]] = []
    per_sub:   list[dict[str, Any]] = []

    # ── Per-main-problem loop ────────────────────────────────────────────────────
    for main_idx, main_problem in enumerate(main_problems):
        pid          = main_problem["problem_id"]
        subject      = main_problem["problem_name"]
        subproblems  = main_problem["subproblems"]

        log.info("[%d/%d] problem_id=%s  name=%s  (#sub=%d)",
                 main_idx + 1, len(main_problems), pid, subject, len(subproblems))

        # Accumulated context for sequential subproblem evaluation
        prior_solutions: list[tuple[str, str]] = []   # [(desc, code), ...]
        main_pass = True

        for sub_idx, sub in enumerate(subproblems):
            sid        = sub["step_number"]
            sub_desc   = sub["step_description_prompt"]
            gold_code  = sub["ground_truth_code"]
            # Each step may have multiple test-case strings; join them
            test_code  = "\n\n".join(sub["test_cases"])
            run_id     = f"scicode-{pid}-{sid}-{int(time.time())}"

            # Build prompt
            prompt = _build_subproblem_prompt(
                main_problem_description = main_problem["problem_description"],
                main_background          = main_problem["problem_background"],
                main_io                  = main_problem["problem_io"],
                required_dependencies    = main_problem["required_dependencies"],
                step                     = sub,
                use_background           = use_background,
                prior_solutions          = prior_solutions,
            )
            prior_ctx_str = "\n\n".join(
                f"# Subproblem {i+1}\n{code}"
                for i, (_, code) in enumerate(prior_solutions)
            )

            t0 = time.time()
            generated_code = ""
            full_output    = ""
            passed         = False
            test_output    = ""
            error_msg      = ""

            try:
                generated_code, full_output = run_pipeline(
                    prompt                 = prompt,
                    researcher             = researcher,
                    formalizer             = formalizer,
                    verifier               = verifier,
                    executor               = executor,
                    code_extractor         = code_extractor,
                    prior_context          = prior_ctx_str,
                    subproblem_description = sub_desc,
                    run_id                 = run_id,
                )
                if generated_code and test_code:
                    passed, test_output = _run_code(generated_code, test_code)
                elif not test_code:
                    # No test case — treat as passed if code was generated
                    passed = bool(generated_code.strip())
                    test_output = "(no test case)"
            except Exception as exc:
                error_msg   = traceback.format_exc()
                full_output = f"ERROR: {exc}"
                log.error("Pipeline failed (run=%s): %s", run_id, exc)

            elapsed = time.time() - t0

            if not passed:
                main_pass = False

            recorder.record(
                question     = sub_desc,
                agent_output = generated_code or full_output,
                test_output  = test_output,
                is_correct   = passed,
                run_id       = run_id,
                metadata     = {
                    "problem_id":  pid,
                    "step_number": sid,
                    "problem_name": subject,
                    "elapsed_s":   round(elapsed, 2),
                },
            )

            sub_result = {
                "problem_id":   pid,
                "step_number":  sid,
                "problem_name": subject,
                "step_description": sub_desc[:120],
                "passed":       passed,
                "elapsed_s":    round(elapsed, 2),
                "test_output":  test_output[:300],
                "error":        error_msg[:300] if error_msg else "",
                "run_id":       run_id,
            }
            per_sub.append(sub_result)

            log.info("  [%s/%s] passed=%s  (%.1fs)", sub_idx + 1, len(subproblems), passed, elapsed)

            # Update prior-solutions context for the next subproblem
            if gold_context:
                context_code = gold_code        # always use reference
            else:
                context_code = generated_code   # use what the agent produced
            prior_solutions.append((sub_desc, context_code))

        per_main.append({
            "problem_id":    pid,
            "problem_name":  subject,
            "n_subproblems": len(subproblems),
            "passed":        main_pass,
        })

    # ── Aggregate metrics ────────────────────────────────────────────────────────
    n_main_total   = len(per_main)
    n_main_correct = sum(1 for r in per_main if r["passed"])
    n_sub_total    = len(per_sub)
    n_sub_correct  = sum(1 for r in per_sub  if r["passed"])

    main_pass_at1  = n_main_correct / n_main_total if n_main_total else 0.0
    sub_pass_at1   = n_sub_correct  / n_sub_total  if n_sub_total  else 0.0

    # Per-problem breakdown (keyed by problem_name)
    by_subject: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in per_sub:
        s = r["problem_name"] or "unknown"
        by_subject[s]["total"]   += 1
        by_subject[s]["correct"] += int(r["passed"])
    subject_metrics = {
        s: {
            "total":       v["total"],
            "correct":     v["correct"],
            "pass_at_1":   round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
        }
        for s, v in by_subject.items()
    }

    summary = {
        "dataset":           _HF_DATASET,
        "split":             split,
        "use_background":    use_background,
        "gold_context":      gold_context,
        "run_executor":      run_executor,
        # Subproblem-level
        "subproblem_total":   n_sub_total,
        "subproblem_correct": n_sub_correct,
        "subproblem_pass_at_1": round(sub_pass_at1, 4),
        # Main-problem-level
        "main_problem_total":   n_main_total,
        "main_problem_correct": n_main_correct,
        "main_problem_pass_at_1": round(main_pass_at1, 4),
        # Subject breakdown (subproblem level)
        "by_subject":    subject_metrics,
        "mistake_counts": experience.mistake_counts(),
        # Raw results
        "per_main":      per_main,
        "per_subproblem": per_sub,
    }

    # ── Persist ──────────────────────────────────────────────────────────────────
    results_path.parent.mkdir(parents=True, exist_ok=True)
    experience_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    log.info("Results saved → %s", results_path)

    experience.save(experience_path)
    log.info("Experience log → %s", experience_path)

    # ── Print summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  SCICODE BENCHMARK RESULTS")
    print("=" * 64)
    print(f"  Dataset          : {_HF_DATASET}  (split={split})")
    print(f"  Background text  : {'yes' if use_background else 'no'}")
    print(f"  Prior context    : {'gold' if gold_context else 'generated'}")
    print()
    print(f"  Subproblem Pass@1 : {sub_pass_at1:.2%}  "
          f"({n_sub_correct}/{n_sub_total})")
    print(f"  Main Prob  Pass@1 : {main_pass_at1:.2%}  "
          f"({n_main_correct}/{n_main_total})")
    print()
    print("  By subject (subproblem level):")
    for subj, m in sorted(subject_metrics.items()):
        print(f"    {subj:<30} {m['pass_at_1']:.2%}  ({m['correct']}/{m['total']})")
    print()
    print(experience.summary())

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark SciTracer on SciCode "
            "(https://huggingface.co/papers/2407.13168)"
        )
    )
    p.add_argument(
        "--split", default="validation",
        help="HuggingFace dataset split: 'validation' (15 problems) or 'test' (65 problems). "
             "Default: validation",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Max main problems to evaluate (default: all).",
    )
    p.add_argument(
        "--background", action="store_true",
        help="Provide the optional scientific background text in the prompt.",
    )
    p.add_argument(
        "--no-gold-context", action="store_true",
        help="Use the agent's own generated code (not gold) as context for "
             "subsequent subproblems.",
    )
    p.add_argument(
        "--no-executor", action="store_true",
        help="Skip the Executor agent; extract code from Researcher/Verifier output.",
    )
    p.add_argument(
        "--no-categorize", action="store_true",
        help="Disable LLM-based mistake categorisation.",
    )
    p.add_argument(
        "--results-path",
        default="benchmark/results/scicode_results.json",
        help="Output path for the JSON results summary.",
    )
    p.add_argument(
        "--experience-path",
        default="benchmark/results/scicode_experience.jsonl",
        help="Output path for the JSONL experience log.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        split           = args.split,
        limit           = args.limit,
        use_background  = args.background,
        gold_context    = not args.no_gold_context,
        run_executor    = not args.no_executor,
        experience_path = Path(args.experience_path),
        results_path    = Path(args.results_path),
        categorize      = not args.no_categorize,
    )
