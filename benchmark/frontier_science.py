"""Frontier Science benchmark for SciTracer.

Loads the openai/frontierscience dataset from HuggingFace (160 expert-level
physics, chemistry, and biology problems), runs each problem through the
SciTracer agent pipeline (Researcher → Formalizer → Verifier → Executor), and
evaluates answer quality using an LLM judge.  Mistakes are recorded in an
Experience store.

Dataset: https://huggingface.co/datasets/openai/frontierscience
  - Split  : test (160 rows)
  - Fields : problem, answer, subject, task_group_id
  - Domains: physics, chemistry, biology
  - Answers: free-form expert text (not numeric/multiple-choice)

Usage
-----
::

    # From the project root:
    python benchmark/frontier_science.py

    # Limit to first N problems (faster dev loop):
    python benchmark/frontier_science.py --limit 5

    # Filter by subject:
    python benchmark/frontier_science.py --subject physics

    # Skip executor step:
    python benchmark/frontier_science.py --no-executor --limit 10

    # Save to custom paths:
    python benchmark/frontier_science.py \\
        --experience-path results/fs_exp.jsonl \\
        --results-path results/fs_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# ── path bootstrap ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dspy
from datasets import load_dataset

from benchmark.experience import Experience, MistakeCategory

log = logging.getLogger("benchmark.frontier_science")


# ══════════════════════════════════════════════════════════════════════════════
# LLM-based answer judge  (free-form answers need semantic comparison)
# ══════════════════════════════════════════════════════════════════════════════

class FrontierScienceJudge(dspy.Signature):
    """Judge whether an agent's scientific answer is correct.

    Compare the agent answer against the reference answer for a graduate-level
    science problem.  An answer is CORRECT if it captures the key scientific
    concepts, mechanisms, and conclusions — exact wording is not required.
    An answer is INCORRECT if it is missing critical points, factually wrong,
    or off-topic.

    Output exactly one word: CORRECT or INCORRECT.
    """

    problem:          str = dspy.InputField(desc="The original scientific problem.")
    reference_answer: str = dspy.InputField(desc="The ground-truth expert answer.")
    agent_answer:     str = dspy.InputField(desc="The agent's produced answer.")
    judgement:        str = dspy.OutputField(
        desc="CORRECT if the agent answer captures the key scientific content, else INCORRECT."
    )


class AnswerJudge(dspy.Module):
    """Wraps FrontierScienceJudge with a safe boolean output."""

    def __init__(self) -> None:
        super().__init__()
        self._judge = dspy.Predict(FrontierScienceJudge)

    def is_correct(self, problem: str, reference: str, agent_answer: str) -> bool:
        if not agent_answer.strip():
            return False
        try:
            pred = self._judge(
                problem=problem[:1200],
                reference_answer=reference[:1200],
                agent_answer=agent_answer[:1200],
            )
            return pred.judgement.strip().upper().startswith("CORRECT")
        except Exception as exc:
            log.warning("Judge call failed: %s", exc)
            return False


# ══════════════════════════════════════════════════════════════════════════════
# DSPy metric
# ══════════════════════════════════════════════════════════════════════════════

# Module-level judge instance (initialised lazily after dspy.configure)
_judge: AnswerJudge | None = None


def _get_judge() -> AnswerJudge:
    global _judge
    if _judge is None:
        _judge = AnswerJudge()
    return _judge


def _frontier_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Any = None,
) -> bool:
    """DSPy metric: True iff the LLM judge rates the prediction as CORRECT."""
    reference    = example.get("answer", "")
    agent_answer = getattr(prediction, "answer", "") or ""
    problem      = example.get("question", "")
    return _get_judge().is_correct(problem, reference, agent_answer)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ══════════════════════════════════════════════════════════════════════════════

def load_frontier_science(
    subject: str | None = None,
    limit: int | None = None,
) -> list[dspy.Example]:
    """Load openai/frontierscience from HuggingFace as DSPy Examples.

    Each Example has:
      - question (str)  : the problem statement
      - answer   (str)  : the expert reference answer
      - subject  (str)  : physics | chemistry | biology
      - metadata (dict) : all original dataset fields

    Args:
        subject: optional filter — one of 'physics', 'chemistry', 'biology'.
        limit:   cap on the number of examples returned.
    """
    log.info("Loading openai/frontierscience (split=test)…")
    dataset = load_dataset("openai/frontierscience", split="test")

    examples: list[dspy.Example] = []
    for row in dataset:
        subj = (row.get("subject") or "").lower()
        if subject and subj != subject.lower():
            continue
        if limit is not None and len(examples) >= limit:
            break

        ex = dspy.Example(
            question=str(row.get("problem", "")),
            answer=str(row.get("answer", "")),
            subject=subj,
            task_group_id=str(row.get("task_group_id", "")),
            metadata={k: v for k, v in row.items()},
        ).with_inputs("question")
        examples.append(ex)

    log.info("Loaded %d examples (subject=%s).", len(examples), subject or "all")
    return examples


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline runner
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    prompt:     str,
    researcher: Any,
    formalizer: Any,
    verifier:   Any,
    executor:   Any | None,
) -> tuple[str, str]:
    """Run the SciTracer pipeline on a single Frontier Science problem.

    Returns (answer_str, full_output_text).  answer_str is the final free-form
    answer extracted from executor stdout (or verifier trace as fallback).
    """
    output_parts: list[str] = []

    # ── Researcher ─────────────────────────────────────────────────────────────
    plan = researcher.forward(prompt)
    output_parts.append(f"[plan]\n{plan}")

    # ── Formalizer ────────────────────────────────────────────────────────────
    formalized_plan = formalizer.forward(plan)
    output_parts.append(f"[formalized_plan]\n{formalized_plan}")

    # ── Verifier ──────────────────────────────────────────────────────────────
    verification_results = verifier.forward(formalized_plan)
    output_parts.append(f"[verification_trace]\n{verification_results.trace}")

    # ── Executor (optional) ───────────────────────────────────────────────────
    exec_output = ""
    if executor is not None:
        refined     = verification_results.refined_plan or plan
        code_result = executor.forward(refined, research_plan=plan)
        exec_output = code_result.output
        output_parts.append(f"[executor_output]\n{exec_output}")

    full_output  = "\n\n".join(output_parts)
    answer_str   = exec_output.strip() or verification_results.trace.strip()
    return answer_str, full_output


# ══════════════════════════════════════════════════════════════════════════════
# Mistake categorisation
# ══════════════════════════════════════════════════════════════════════════════

class MistakeCategorizer(dspy.Signature):
    """Categorise a scientific reasoning mistake into exactly one category.

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

    question:        str = dspy.InputField(desc="The original science problem.")
    agent_output:    str = dspy.InputField(desc="The agent's full output text.")
    expected_answer: str = dspy.InputField(desc="The correct expert answer.")
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
        expected:     str,
        is_correct:   bool,
        run_id:       str,
        metadata:     dict[str, Any] | None = None,
    ) -> None:
        meta = metadata or {}

        if is_correct:
            self.experience.add_correct(
                question        = question,
                agent_output    = agent_output,
                expected_output = expected,
                run_id          = run_id,
                metadata        = meta,
            )
            return

        category    = MistakeCategory.REASONING_ERROR
        explanation = ""
        if self.categorize and agent_output.strip():
            try:
                pred        = self._categorizer(
                    question        = question[:800],
                    agent_output    = agent_output[:800],
                    expected_answer = expected[:800],
                )
                category    = _parse_category(pred.category)
                explanation = pred.explanation
            except Exception as exc:
                log.warning("Categorisation failed: %s", exc)

        self.experience.add_mistake(
            question        = question,
            agent_output    = agent_output,
            category        = category,
            expected_output = expected,
            explanation     = explanation,
            run_id          = run_id,
            metadata        = meta,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main benchmark loop
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    *,
    subject:         str | None = None,
    limit:           int | None = None,
    run_executor:    bool       = True,
    experience_path: Path       = Path("benchmark/results/experience.jsonl"),
    results_path:    Path       = Path("benchmark/results/frontier_science_results.json"),
    categorize:      bool       = True,
) -> dict[str, Any]:
    """Run the full Frontier Science benchmark and return a results summary dict."""

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
    dspy.configure(lm=lm, enable_disk_cache=False, enable_memory_cache=False)

    log.info("Initialising agents…")
    vector_db     = TrajectoryStore()
    graph_storage = KnowledgeGraph()

    examples = load_frontier_science(subject=subject, limit=limit)
    log.info("Running benchmark on %d problems…", len(examples))

    experience = Experience()
    recorder   = ExperienceRecorder(experience, categorize=categorize)
    per_example_results: list[dict[str, Any]] = []

    for i, example in enumerate(examples):
        question = example.question
        expected = example.answer
        subj     = example.subject
        run_id   = f"frontierscience-{i:04d}-{int(time.time())}"

        log.info("[%d/%d] [%s] %s…", i + 1, len(examples), subj, question[:80])
        t0 = time.time()

        predicted_str = ""
        full_output   = ""

        try:
            rag        = RAG(vector_db, graph_storage)
            researcher = Researcher(rag=rag)
            formalizer = Formalizer()
            verifier   = Verifier()
            executor   = Executor() if run_executor else None

            predicted_str, full_output = run_pipeline(
                prompt     = question,
                researcher = researcher,
                formalizer = formalizer,
                verifier   = verifier,
                executor   = executor,
            )
        except Exception as exc:
            log.error("Pipeline failed (run=%s): %s", run_id, exc)
            full_output = f"ERROR: {exc}"

        elapsed = time.time() - t0

        correct = _get_judge().is_correct(question, expected, predicted_str)

        recorder.record(
            question     = question,
            agent_output = full_output,
            expected     = expected,
            is_correct   = correct,
            run_id       = run_id,
            metadata     = {
                "subject":        subj,
                "task_group_id":  example.task_group_id,
                "problem_index":  i,
                "elapsed_s":      round(elapsed, 2),
            },
        )

        per_example_results.append({
            "index":     i,
            "subject":   subj,
            "question":  question[:120],
            "correct":   correct,
            "elapsed_s": round(elapsed, 2),
            "run_id":    run_id,
        })

        log.info(
            "  → correct=%s  subject=%-12s  (%.1fs)",
            correct, subj, elapsed,
        )

    # ── Aggregate ──────────────────────────────────────────────────────────────
    n_correct = sum(1 for r in per_example_results if r["correct"])
    n_total   = len(per_example_results)
    accuracy  = n_correct / n_total if n_total else 0.0

    # Per-subject breakdown
    subjects = sorted({r["subject"] for r in per_example_results})
    subject_breakdown: dict[str, dict[str, Any]] = {}
    for subj in subjects:
        rows    = [r for r in per_example_results if r["subject"] == subj]
        n_s     = len(rows)
        n_s_cor = sum(1 for r in rows if r["correct"])
        subject_breakdown[subj] = {
            "n_total":   n_s,
            "n_correct": n_s_cor,
            "accuracy":  round(n_s_cor / n_s, 4) if n_s else 0.0,
        }

    summary = {
        "dataset":           "openai/frontierscience",
        "split":             "test",
        "subject_filter":    subject,
        "n_total":           n_total,
        "n_correct":         n_correct,
        "accuracy":          round(accuracy, 4),
        "run_executor":      run_executor,
        "subject_breakdown": subject_breakdown,
        "mistake_counts":    experience.mistake_counts(),
        "per_example":       per_example_results,
    }

    # ── Persist ────────────────────────────────────────────────────────────────
    results_path.parent.mkdir(parents=True, exist_ok=True)
    experience_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Results saved → %s", results_path)

    experience.save(experience_path)
    log.info("Experience log → %s", experience_path)

    # ── Print ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FRONTIER SCIENCE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Dataset  : openai/frontierscience  (split=test)")
    print(f"  Filter   : {subject or 'all subjects'}")
    print(f"  Problems : {n_total}")
    print(f"  Correct  : {n_correct}")
    print(f"  Accuracy : {accuracy:.2%}")
    print()
    print("  Per-subject breakdown:")
    for subj, stats in subject_breakdown.items():
        print(
            f"    {subj:<12} : {stats['n_correct']:>3}/{stats['n_total']:<3}"
            f"  ({stats['accuracy']:.2%})"
        )
    print()
    print(experience.summary())

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark SciTracer on openai/frontierscience"
    )
    p.add_argument(
        "--subject", default=None,
        choices=["physics", "chemistry", "biology"],
        help="Filter by subject (default: all)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Max problems to evaluate (default: all 160)",
    )
    p.add_argument(
        "--no-executor", action="store_true",
        help="Skip Executor step; use verifier trace for answer",
    )
    p.add_argument(
        "--experience-path",
        default="benchmark/results/experience.jsonl",
        help="Path for experience JSONL log",
    )
    p.add_argument(
        "--results-path",
        default="benchmark/results/frontier_science_results.json",
        help="Path for results JSON summary",
    )
    p.add_argument(
        "--no-categorize", action="store_true",
        help="Disable LLM-based mistake categorisation",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        subject         = args.subject,
        limit           = args.limit,
        run_executor    = not args.no_executor,
        experience_path = Path(args.experience_path),
        results_path    = Path(args.results_path),
        categorize      = not args.no_categorize,
    )
