"""TPBench benchmark for SciTracer.

Loads the ZhiqiGao/TPBench dataset from HuggingFace (10 theoretical-physics
problems across QM, Cosmology, Stat-Mech, Classical Mechanics, HET), runs each
problem through the SciTracer pipeline, and evaluates both the free-form answer
and the produced Python code against the reference implementation.

Dataset fields used
-------------------
    problem_id               — unique identifier
    domain                   — physics subdiscipline
    difficulty_level         — "1"–"5"
    problem                  — full problem statement
    answer                   — reference answer (string, may contain LaTeX)
    code_answer_requirements — specification the produced code must satisfy
    reference_implementation — reference Python implementation (for comparison)

Pipeline (mirrors aime_2026.py)
---------------------------------
    rag        = RAG(vector_db, graph_storage)
    researcher = Researcher(rag=rag)
    plan       = researcher.forward(prompt)

    formalizer        = Formalizer()
    formalized_plan   = formalizer.forward(plan)

    verifier             = Verifier()
    verification_results = verifier.forward(formalized_plan)

    executor    = Executor()
    code_result = executor.execute(task_summary, instructions=plan, research_plan=plan)
    # answer extracted from code_result.stdout or verification_results.trace

Usage
-----
::

    python benchmark/tpbench.py

    # Limit to first N problems:
    python benchmark/tpbench.py --limit 3

    # Skip executor:
    python benchmark/tpbench.py --no-executor

    # Filter by domain:
    python benchmark/tpbench.py --domain "QM"

    # Filter by max difficulty:
    python benchmark/tpbench.py --max-difficulty 3
"""
from __future__ import annotations
import pdb
import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── path bootstrap ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dspy
from datasets import load_dataset

from benchmark.experience import Experience, MistakeCategory

log = logging.getLogger("benchmark.tpbench")


# ══════════════════════════════════════════════════════════════════════════════
# Answer extraction
# ══════════════════════════════════════════════════════════════════════════════

class TPAnswerExtractor(dspy.Signature):
    """Extract the final physics answer from agent output and judge correctness.

    Physics answers may be symbolic expressions, numerical values with units,
    or short prose.  Compare the agent answer with the reference and return
    'correct' or 'incorrect' with a one-sentence justification.
    """
    problem:          str = dspy.InputField(desc="The original physics problem.")
    reference_answer: str = dspy.InputField(desc="The reference answer from the dataset.")
    agent_output:     str = dspy.InputField(desc="The full output produced by the agent pipeline.")
    extracted_answer: str = dspy.OutputField(desc="The answer found in agent output (verbatim or 'UNKNOWN').")
    verdict:          str = dspy.OutputField(desc="'correct' or 'incorrect'.")
    justification:    str = dspy.OutputField(desc="One sentence explaining the verdict.")


class LLMAnswerJudge(dspy.Module):
    """LLM-based answer extraction and correctness judgment for physics answers."""

    def __init__(self) -> None:
        super().__init__()
        self._judge = dspy.ChainOfThought(TPAnswerExtractor)

    def forward(
        self,
        problem:          str,
        reference_answer: str,
        agent_output:     str,
    ) -> tuple[str, bool, str]:
        """Return (extracted_answer, is_correct, justification)."""
        if not agent_output.strip():
            return "UNKNOWN", False, "No output produced."
        try:
            pred = self._judge(
                problem          = problem[:600],
                reference_answer = reference_answer[:400],
                agent_output     = agent_output[:3000],
            )
            extracted  = pred.extracted_answer.strip()
            is_correct = pred.verdict.strip().lower() == "correct"
            return extracted, is_correct, pred.justification
        except Exception as exc:
            log.warning("LLM judge failed: %s", exc)
            return "UNKNOWN", False, f"Judge error: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ══════════════════════════════════════════════════════════════════════════════

def load_tpbench(
    split:          str        = "public",
    limit:          int | None = None,
    domain_filter:  str | None = None,
    max_difficulty: int | None = None,
) -> list[dspy.Example]:
    """Load ZhiqiGao/TPBench from HuggingFace as DSPy Examples.

    Each Example has:
      - problem_id              (str)
      - domain                  (str)
      - difficulty_level        (str, "1"–"5")
      - question                (str)  — the problem statement
      - answer                  (str)  — reference answer
      - code_requirements       (str)  — what the code must implement
      - reference_implementation (str) — reference Python code
    """
    log.info("Loading ZhiqiGao/TPBench (split=%s)…", split)
    dataset = load_dataset("ZhiqiGao/TPBench", split=split, trust_remote_code=True)

    examples: list[dspy.Example] = []
    for row in dataset:
        difficulty = str(row.get("difficulty_level", "0"))
        domain     = str(row.get("domain", ""))

        if domain_filter and domain_filter.lower() not in domain.lower():
            continue
        if max_difficulty is not None:
            try:
                if int(difficulty) > max_difficulty:
                    continue
            except ValueError:
                pass

        ex = dspy.Example(
            problem_id             = str(row.get("problem_id", "")),
            domain                 = domain,
            difficulty_level       = difficulty,
            question               = str(row.get("problem", "")),
            answer                 = str(row.get("answer", "")),
            code_requirements      = str(row.get("code_answer_requirements", "")),
            reference_implementation = str(row.get("reference_implementation", "")),
        ).with_inputs("question")

        examples.append(ex)
        if limit is not None and len(examples) >= limit:
            break

    log.info("Loaded %d TPBench examples.", len(examples))
    return examples


# ══════════════════════════════════════════════════════════════════════════════
# Prompt builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt(example: dspy.Example) -> str:
    """Build a full task prompt from a TPBench example."""
    parts = [
        f"Domain: {example.domain}  |  Difficulty: {example.difficulty_level}/5",
        "",
        example.question,
    ]
    if example.code_requirements.strip():
        parts += [
            "",
            "Code requirements:",
            example.code_requirements,
        ]
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline runner
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    extracted_answer:     str
    is_correct:           bool
    justification:        str
    full_output:          str
    plan:                 Any
    verification_results: Any
    step_logs:            list


def run_pipeline(
    prompt:    str,
    researcher: Any,
    formalizer: Any,
    verifier:   Any,
    executor:   Any | None,
    judge:      LLMAnswerJudge,
    example:    dspy.Example,
    builder:    Any | None = None,
) -> PipelineResult:
    """Run the full agent pipeline on one TPBench problem."""
    from dataclasses import fields as _fields
    output_parts: list[str] = []

    plan = researcher.forward(prompt)
    output_parts.append(f"[plan]\n{plan}")
    if builder is not None:
        builder.set_plan(plan)

    formalized_plan = formalizer.forward(plan)
    output_parts.append(f"[formalized_plan]\n{formalized_plan}")
    if builder is not None:
        builder.set_formalized_code(str(formalized_plan))

    verification_results = verifier.forward(formalized_plan)
    output_parts.append(f"[verification_trace]\n{verification_results.trace}")
    if builder is not None:
        for att in getattr(verification_results, "attempts", []):
            builder.add_attempt(att)

    exec_output = ""
    step_logs:  list = []
    if executor is not None:
        refined     = getattr(verification_results, "refined_plan", None) or plan
        code_result = executor.execute(
            task_summary  = prompt[:200],
            instructions  = refined,
            research_plan = plan,
        )
        exec_output = code_result.output
        step_logs   = code_result.step_logs or []
        output_parts.append(f"[executor_output]\n{exec_output}")

    full_output   = "\n\n".join(output_parts)
    answer_source = exec_output or verification_results.trace

    extracted, is_correct, justification = judge.forward(
        problem          = example.question,
        reference_answer = example.answer,
        agent_output     = answer_source,
    )

    return PipelineResult(
        extracted_answer     = extracted,
        is_correct           = is_correct,
        justification        = justification,
        full_output          = full_output,
        plan                 = plan,
        verification_results = verification_results,
        step_logs            = step_logs,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Mistake categorisation  (reuses AIME pattern)
# ══════════════════════════════════════════════════════════════════════════════

class TPMistakeCategorizer(dspy.Signature):
    """Categorise a physics reasoning mistake into exactly one category.

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
    """
    problem:         str = dspy.InputField(desc="The original physics problem.")
    agent_output:    str = dspy.InputField(desc="The agent's full output text.")
    expected_answer: str = dspy.InputField(desc="The correct reference answer.")
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
    def __init__(self, experience: Experience, categorize: bool = True) -> None:
        self.experience = experience
        self.categorize = categorize
        if categorize:
            self._categorizer = dspy.Predict(TPMistakeCategorizer)

    def record(
        self,
        question:     str,
        agent_output: str,
        expected:     str,
        is_correct:   bool,
        run_id:       str,
        metadata:     dict[str, Any] | None = None,
    ) -> None:
        meta = metadata or {}

        if is_correct:
            self.experience.add_correct(
                question=question, agent_output=agent_output,
                expected_output=expected, run_id=run_id, metadata=meta,
            )
            return

        category    = MistakeCategory.REASONING_ERROR
        explanation = ""
        if self.categorize and agent_output.strip():
            try:
                pred        = self._categorizer(
                    problem         = question[:800],
                    agent_output    = agent_output[:800],
                    expected_answer = expected[:200],
                )
                category    = _parse_category(pred.category)
                explanation = pred.explanation
            except Exception as exc:
                log.warning("Categorisation failed: %s", exc)

        self.experience.add_mistake(
            question=question, agent_output=agent_output, category=category,
            expected_output=expected, explanation=explanation,
            run_id=run_id, metadata=meta,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main benchmark loop
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("benchmark/results/tpbench")


def run_benchmark(
    *,
    split:           str        = "public",
    limit:           int | None = None,
    domain_filter:   str | None = None,
    max_difficulty:  int | None = None,
    run_executor:    bool       = True,
    results_dir:     Path       = RESULTS_DIR,
    categorize:      bool       = True,
) -> dict[str, Any]:
    """Run the full TPBench benchmark and return a results summary dict."""

    from core.lm_setup import _setup_lm
    from core.log_setup import setup_logging
    from agents.researcher import Researcher
    from agents.formalizer import Formalizer
    from agents.verifier import Verifier
    from agents.executor import Executor
    from agents.reporter import Reporter
    from memory.episodic_memory import EpisodicMemory, EpisodeBuilder

    results_dir.mkdir(parents=True, exist_ok=True)
    experience_path = results_dir / "experience.jsonl"
    episodes_path   = results_dir / "episodes.jsonl"
    results_path    = results_dir / "results.json"

    setup_logging(log_dir="logs", console_level=logging.INFO)
    lm = _setup_lm()
    dspy.configure(lm=lm, enable_disk_cache=False, enable_memory_cache=False)

    log.info("Initialising agents…")
    judge = LLMAnswerJudge()

    examples = load_tpbench(
        split=split, limit=limit,
        domain_filter=domain_filter, max_difficulty=max_difficulty,
    )
    log.info("Running benchmark on %d problems…", len(examples))

    experience = Experience()
    recorder   = ExperienceRecorder(experience, categorize=categorize)
    ep_memory  = EpisodicMemory()
    per_example_results: list[dict[str, Any]] = []

    # Track last run's artifacts for the end-of-benchmark report
    _last_plan:       Any = None
    _last_exec_out:   str = ""
    _last_verif:      Any = None
    _last_step_logs:  list | None = None

    for i, example in enumerate(examples):
        pdb.set_trace()
        prompt   = _build_prompt(example)
        expected = example.answer
        run_id   = f"tpbench-{example.problem_id}-{int(time.time())}"

        log.info(
            "[%d/%d] id=%s  domain=%s  diff=%s",
            i + 1, len(examples), example.problem_id, example.domain, example.difficulty_level,
        )
        t0 = time.time()
        #No external knowledge usage;
        researcher = Researcher()
        formalizer = Formalizer()
        verifier   = Verifier()
        executor   = Executor() if run_executor else None
        builder    = EpisodeBuilder(run_id=run_id, question=example.question)

        extracted_answer = "UNKNOWN"
        is_correct       = False
        justification    = ""
        full_output      = ""

        #try:
        pr = run_pipeline(
                prompt     = prompt,
                researcher = researcher,
                formalizer = formalizer,
                verifier   = verifier,
                executor   = executor,
                judge      = judge,
                example    = example,
                builder    = builder,
            )
        extracted_answer = pr.extracted_answer
        is_correct       = pr.is_correct
        justification    = pr.justification
        full_output      = pr.full_output
        _last_plan       = pr.plan
        _last_verif      = pr.verification_results
        _last_exec_out   = pr.full_output
        _last_step_logs  = pr.step_logs
#    except Exception as exc:
#        log.error("Pipeline failed (run=%s): %s", run_id, exc)
    #full_output = f"ERROR: {exc}"

    elapsed = time.time() - t0

    recorder.record(
        question     = example.question,
        agent_output = full_output,
        expected     = expected,
        is_correct   = is_correct,
        run_id       = run_id,
        metadata     = {
            "problem_id":      example.problem_id,
            "domain":          example.domain,
            "difficulty_level": example.difficulty_level,
            "elapsed_s":       round(elapsed, 2),
            "predicted":       extracted_answer,
            "expected":        expected,
            "justification":   justification,
        },
    )

    category    = None
    explanation = ""
    if not is_correct:
        entry = experience.mistakes[-1] if experience.mistakes else None
        if entry:
            category    = entry.mistake_category.value if entry.mistake_category else None
            explanation = entry.explanation

    builder.set_executor_output(full_output)
    builder.finish(
        final_answer        = extracted_answer,
        expected_answer     = expected,
        is_correct          = is_correct,
        mistake_category    = category,
        mistake_explanation = explanation,
        metadata            = {
            "problem_id":       example.problem_id,
            "domain":           example.domain,
            "difficulty_level": example.difficulty_level,
            "elapsed_s":        round(elapsed, 2),
            "justification":    justification,
        },
    )
    ep_memory.add(builder.build())

    per_example_results.append({
        "index":            i,
        "problem_id":       example.problem_id,
        "domain":           example.domain,
        "difficulty_level": example.difficulty_level,
        "expected":         expected,
        "predicted":        extracted_answer,
        "correct":          is_correct,
        "justification":    justification,
        "elapsed_s":        round(elapsed, 2),
        "run_id":           run_id,
    })

    log.info(
        "  → correct=%s  predicted=%.60s  (%.1fs)",
        is_correct, extracted_answer, elapsed,
    )

    # ── Aggregate ──────────────────────────────────────────────────────────────
    n_correct = sum(1 for r in per_example_results if r["correct"])
    n_total   = len(per_example_results)
    accuracy  = n_correct / n_total if n_total else 0.0

    by_domain: dict[str, dict] = {}
    for r in per_example_results:
        d = r["domain"]
        if d not in by_domain:
            by_domain[d] = {"n": 0, "correct": 0}
        by_domain[d]["n"]       += 1
        by_domain[d]["correct"] += int(r["correct"])
    for d, stats in by_domain.items():
        stats["accuracy"] = round(stats["correct"] / stats["n"], 4) if stats["n"] else 0.0

    summary = {
        "dataset":        "ZhiqiGao/TPBench",
        "split":          split,
        "n_total":        n_total,
        "n_correct":      n_correct,
        "accuracy":       round(accuracy, 4),
        "run_executor":   run_executor,
        "by_domain":      by_domain,
        "mistake_counts": experience.mistake_counts(),
        "per_example":    per_example_results,
    }

    # ── Persist ────────────────────────────────────────────────────────────────
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Results saved → %s", results_path)

    experience.save(experience_path)
    log.info("Experience log → %s", experience_path)

    ep_memory.save(episodes_path)
    log.info("Episodic memory → %s  (%d episodes)", episodes_path, len(ep_memory))

    # ── Report ─────────────────────────────────────────────────────────────────
    try:
        Reporter(output_dir=results_dir).forward(
            question             = f"TPBench — {n_total} theoretical physics problems",
            research_plan        = _last_plan,
            execution_trace      = _last_exec_out,
            verification_summary = str(_last_verif) if _last_verif else "",
            step_logs            = _last_step_logs or [],
        )
    except Exception as exc:
        log.warning("Reporter failed: %s", exc)

    # ── Print ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TPBENCH RESULTS")
    print("=" * 60)
    print(f"  Dataset  : ZhiqiGao/TPBench  (split={split})")
    print(f"  Problems : {n_total}")
    print(f"  Correct  : {n_correct}")
    print(f"  Accuracy : {accuracy:.2%}")
    print()
    print("  By domain:")
    for domain, stats in by_domain.items():
        print(f"    {domain:<20} {stats['correct']}/{stats['n']}  ({stats['accuracy']:.2%})")
    print()
    print(experience.summary())

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark SciTracer on ZhiqiGao/TPBench (theoretical physics)"
    )
    p.add_argument("--split",           default="public",
                   help="HuggingFace dataset split (default: public)")
    p.add_argument("--limit",           type=int, default=None,
                   help="Max problems to evaluate (default: all 10)")
    p.add_argument("--domain",          default=None,
                   help="Filter by domain substring, e.g. 'QM', 'Cosmology'")
    p.add_argument("--max-difficulty",  type=int, default=None,
                   help="Only include problems with difficulty_level <= N")
    p.add_argument("--no-executor",   action="store_true",
                   help="Skip Executor step; use verifier trace for answer extraction")
    p.add_argument("--results-dir",  default=str(RESULTS_DIR),
                   help="Directory for results.json, experience.jsonl, episodes.jsonl, REPORT.md "
                        f"(default: {RESULTS_DIR})")
    p.add_argument("--no-categorize", action="store_true",
                   help="Disable LLM-based mistake categorisation")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        split          = args.split,
        limit          = args.limit,
        domain_filter  = args.domain,
        max_difficulty = args.max_difficulty,
        run_executor   = not args.no_executor,
        results_dir    = Path(args.results_dir),
        categorize     = not args.no_categorize,
    )
