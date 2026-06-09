"""AIME 2026 benchmark for SciTracer.

Loads the MathArena/aime_2026 dataset from HuggingFace, runs each problem
through the SciTracer agent pipeline (Researcher → Formalizer → Verifier →
Executor), and evaluates answer accuracy using DSPy's built-in evaluation
harness.  Mistakes are recorded in an Experience store.

Pipeline (mirrors test_plan_verifier.py)
-----------------------------------------
    rag        = RAG(vector_db, graph_storage)
    researcher = Researcher(rag=rag)
    plan       = researcher.forward(prompt)

    formalizer        = Formalizer()
    formalized_plan   = formalizer.forward(plan)

    verifier             = Verifier()
    verification_results = verifier.forward(formalized_plan)

    executor = Executor()
    code     = executor.forward(plan, research_plan=plan)
    # answer extracted from code.stdout

Usage
-----
::

    # From the project root:
    python benchmark/aime_2026.py

    # Limit to first N problems (faster dev loop):
    python benchmark/aime_2026.py --limit 5

    # Skip executor step (faster; uses verifier trace for answer):
    python benchmark/aime_2026.py --no-executor --limit 10

    # Save experience log to a custom path:
    python benchmark/aime_2026.py --experience-path results/exp.jsonl

    # Use a different HuggingFace split:
    python benchmark/aime_2026.py --split test
"""
from __future__ import annotations
import pdb
import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

# ── path bootstrap ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dspy
from datasets import load_dataset
from agents.knowledge_constructor import KnowledgeConsructor

from benchmark.experience import Experience, MistakeCategory

log = logging.getLogger("benchmark.aime_2026")


# ══════════════════════════════════════════════════════════════════════════════
# Answer extraction helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extract_integer(text: str) -> int | None:
    """Extract the last integer from text (handles LaTeX \\boxed{} and prose).

    AIME answers are integers in [0, 999].  The agent may wrap the answer in
    prose or LaTeX; we try \\boxed{} first, then fall back to the last
    standalone integer we find.
    """
    boxed = re.findall(r"\\boxed\{(\d+)\}", text)
    if boxed:
        return int(boxed[-1])
    numbers = re.findall(r"\b(\d{1,3})\b", text)
    if numbers:
        return int(numbers[-1])
    return None


class AIMEAnswerExtractor(dspy.Signature):
    """Extract the final integer answer from a free-form solution text.

    The answer must be a single integer in [0, 999] (AIME range).
    Output -1 if no answer can be determined.
    """

    solution_text: str = dspy.InputField(
        desc="Full solution or execution output produced by the agent."
    )
    extracted_answer: int = dspy.OutputField(
        desc="Final integer answer in [0, 999]. Output -1 if unknown."
    )


class LLMAnswerExtractor(dspy.Module):
    """LLM fallback for answer extraction when regex fails."""

    def __init__(self) -> None:
        super().__init__()
        self._extract = dspy.Predict(AIMEAnswerExtractor)

    def forward(self, solution_text: str) -> int:
        try:
            pred = self._extract(solution_text=solution_text)
            return int(pred.extracted_answer)
        except (ValueError, TypeError):
            return -1


def extract_answer(text: str, llm_extractor: LLMAnswerExtractor | None = None) -> int | None:
    """Try regex first, then LLM fallback. Returns None if both fail."""
    result = _extract_integer(text)
    if result is not None:
        return result
    if llm_extractor and text.strip():
        val = llm_extractor(solution_text=text)
        return val if val != -1 else None
    return None


# ══════════════════════════════════════════════════════════════════════════════
# DSPy metric
# ══════════════════════════════════════════════════════════════════════════════

def _aime_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Any = None,
) -> bool:
    """DSPy metric: True iff predicted integer == expected integer."""
    expected_raw  = example.get("answer", example.get("Answer", ""))
    predicted_raw = getattr(prediction, "answer", "") or ""

    try:
        expected = int(str(expected_raw).strip())
    except (ValueError, TypeError):
        return False

    predicted = _extract_integer(str(predicted_raw))
    return predicted is not None and predicted == expected


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ══════════════════════════════════════════════════════════════════════════════

def load_aime_2026(split: str = "train", limit: int | None = None) -> list[dspy.Example]:
    """Load MathArena/aime_2026 from HuggingFace as DSPy Examples.

    Each Example has:
      - question (str): the problem statement
      - answer   (int): the expected integer answer
      - metadata (dict): all original dataset fields
    """
    log.info("Loading MathArena/aime_2026 (split=%s)…", split)
    dataset = load_dataset("MathArena/aime_2026", split=split)

    examples: list[dspy.Example] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break

        problem = (
            row.get("problem") or row.get("Problem")
            or row.get("question") or row.get("Question") or ""
        )
        answer = (
            row.get("answer") or row.get("Answer")
            or row.get("solution") or ""
        )

        ex = dspy.Example(
            question=str(problem),
            answer=answer,
            metadata={k: v for k, v in row.items()},
        ).with_inputs("question")
        examples.append(ex)

    log.info("Loaded %d examples.", len(examples))
    return examples


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline runner  (matches test_plan_verifier.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    prompt:        str,
    researcher:    Any,
    formalizer:    Any,
    verifier:      Any,
    executor:      Any | None,
    llm_extractor: LLMAnswerExtractor,
    builder:       Any | None = None,   # EpisodeBuilder | None
) -> tuple[str, str, Any]:
    """Run the full agent pipeline on a single problem.

    Returns (answer_str, full_output_text, verification_outcome).
    """
    output_parts: list[str] = []

    # ── Researcher ─────────────────────────────────────────────────────────────
    plan = researcher.forward(prompt)
    output_parts.append(f"[plan]\n{plan}")
    if builder is not None:
        builder.set_plan(plan)

    # ── Formalizer ────────────────────────────────────────────────────────────
    formalized_plan = formalizer.forward(plan)
    output_parts.append(f"[formalized_plan]\n{formalized_plan}")
    if builder is not None:
        builder.set_formalized_code(str(formalized_plan))

    # ── Verifier ──────────────────────────────────────────────────────────────
    verification_results = verifier.forward(formalized_plan)
    output_parts.append(f"[verification_trace]\n{verification_results.trace}")
    if builder is not None:
        for att in verification_results.attempts:
            builder.add_attempt(att)

    # ── Executor (optional) ───────────────────────────────────────────────────
    exec_output = ""
    if executor is not None:
        refined     = verification_results.refined_plan or plan
        code_result = executor.forward(refined, research_plan=plan)
        exec_output = code_result.output
        output_parts.append(f"[executor_output]\n{exec_output}")

    full_output = "\n\n".join(output_parts)

    answer_source = exec_output or verification_results.trace
    answer_int    = extract_answer(answer_source, llm_extractor)
    answer_str    = str(answer_int) if answer_int is not None else ""

    return answer_str, full_output, verification_results


# ══════════════════════════════════════════════════════════════════════════════
# Mistake categorisation
# ══════════════════════════════════════════════════════════════════════════════

class MistakeCategorizer(dspy.Signature):
    """Categorise a mathematical reasoning mistake into exactly one category.

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

    question:        str = dspy.InputField(desc="The original AIME problem.")
    agent_output:    str = dspy.InputField(desc="The agent's full output text.")
    expected_answer: str = dspy.InputField(desc="The correct integer answer.")
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
                    expected_answer = str(expected),
                )
                category    = _parse_category(pred.category)
                explanation = pred.explanation
            except Exception as exc:
                log.warning("Categorisation failed: %s", exc)

        self.experience.add_mistake(
            question        = question,
            agent_output    = agent_output,
            category        = category,
            expected_output = str(expected),
            explanation     = explanation,
            run_id          = run_id,
            metadata        = meta,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main benchmark loop
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    *,
    split:           str        = "train",
    limit:           int | None = None,
    run_executor:    bool       = True,
    experience_path: Path       = Path("benchmark/results/experience.jsonl"),
    episodes_path:   Path       = Path("benchmark/results/episodes.jsonl"),
    results_path:    Path       = Path("benchmark/results/aime_2026_results.json"),
    categorize:      bool       = True,
) -> dict[str, Any]:
    """Run the full AIME 2026 benchmark and return a results summary dict."""

    # ── Deferred project imports ───────────────────────────────────────────────
    from core.lm_setup import _setup_lm
    from core.log_setup import setup_logging
    from agents.researcher import Researcher
    from agents.formalizer import Formalizer
    from agents.verifier import Verifier
    from agents.executor import Executor
    from agents.rag import RAG
    from storage.vector_db import TrajectoryStore
    from storage.knowledge_graph import KnowledgeGraph
    from memory.episodic_memory import EpisodicMemory, EpisodeBuilder

    setup_logging(log_dir="logs", console_level=logging.INFO)
    lm = _setup_lm()
    dspy.configure(lm=lm, enable_disk_cache=False, enable_memory_cache=False)

    # ── Initialise agents (shared across all problems) ─────────────────────────
    log.info("Initialising agents…")
    
    vector_db     = TrajectoryStore()
    graph_storage = KnowledgeGraph()

    #TODO add the
    #kc = KnowledgeConsructor(storage=vector_db, kg=graph_storage, question=prompt)
    #kc.collect_literature(prompt=prompt)

    #rag        = RAG(vector_db, graph_storage)
    #researcher = Researcher(rag=rag)
    #formalizer = Formalizer()
    #verifier   = Verifier()
    #executor   = Executor() if run_executor else None

    #TODO add the experience or memory for the improvement
    llm_extractor = LLMAnswerExtractor()

    # ── Load dataset ───────────────────────────────────────────────────────────
    examples = load_aime_2026(split=split, limit=limit)
    log.info("Running benchmark on %d problems…", len(examples))

    #TODO resolve the experience
    experience = Experience()
    recorder   = ExperienceRecorder(experience, categorize=categorize)
    ep_memory  = EpisodicMemory()
    per_example_results: list[dict[str, Any]] = []

    # ── Per-problem loop ───────────────────────────────────────────────────────
    for i, example in enumerate(examples):
        question = example.question
        expected = example.answer
        run_id   = f"aime2026-{i:04d}-{int(time.time())}"

        log.info("[%d/%d] %s…", i + 1, len(examples), question[:80])
        t0 = time.time()

        predicted_str = ""
        full_output   = ""
        builder = EpisodeBuilder(run_id=run_id, question=question)
        #kc = KnowledgeConsructor(question=question, vector_db=vector_db, kg=graph_storage)
        #kc.collect_literature
        rag        = RAG(vector_db, graph_storage)
        researcher = Researcher(rag=rag)
        formalizer = Formalizer()
        verifier   = Verifier()
        executor   = Executor() if run_executor else None
        llm_extractor = LLMAnswerExtractor()

        try:
            predicted_str, full_output, verification_outcome = run_pipeline(
                prompt        = question,
                researcher    = researcher,
                formalizer    = formalizer,
                verifier      = verifier,
                executor      = executor,
                llm_extractor = llm_extractor,
                builder       = builder,
            )
        except Exception as exc:
            log.error("Pipeline failed (run=%s): %s", run_id, exc)
            full_output          = f"ERROR: {exc}"
            verification_outcome = None

        elapsed = time.time() - t0

        # DSPy-style metric check
        fake_pred = dspy.Prediction(answer=predicted_str)
        correct   = _aime_metric(example, fake_pred)

        # ── ExperienceRecorder (flat log, backwards-compatible) ────────────────
        recorder.record(
            question     = question,
            agent_output = full_output,
            expected     = str(expected),
            is_correct   = correct,
            run_id       = run_id,
            metadata     = {
                "problem_index": i,
                "elapsed_s":     round(elapsed, 2),
                "predicted":     predicted_str,
                "expected":      str(expected),
            },
        )

        # ── EpisodicMemory (rich per-attempt log) ──────────────────────────────
        category    = None
        explanation = ""
        if not correct:
            entry = experience.mistakes[-1] if experience.mistakes else None
            if entry:
                category    = entry.mistake_category.value if entry.mistake_category else None
                explanation = entry.explanation

        builder.set_executor_output(full_output)
        builder.finish(
            final_answer        = predicted_str,
            expected_answer     = str(expected),
            is_correct          = correct,
            mistake_category    = category,
            mistake_explanation = explanation,
            metadata            = {"problem_index": i, "elapsed_s": round(elapsed, 2)},
        )
        ep_memory.add(builder.build())

        per_example_results.append({
            "index":     i,
            "question":  question[:120],
            "expected":  str(expected),
            "predicted": predicted_str,
            "correct":   correct,
            "elapsed_s": round(elapsed, 2),
            "run_id":    run_id,
        })

        log.info(
            "  → predicted=%-5s  expected=%-5s  correct=%s  (%.1fs)",
            predicted_str or "?", expected, correct, elapsed,
        )

    # ── Aggregate ──────────────────────────────────────────────────────────────
    n_correct = sum(1 for r in per_example_results if r["correct"])
    n_total   = len(per_example_results)
    accuracy  = n_correct / n_total if n_total else 0.0

    summary = {
        "dataset":        "MathArena/aime_2026",
        "split":          split,
        "n_total":        n_total,
        "n_correct":      n_correct,
        "accuracy":       round(accuracy, 4),
        "run_executor":   run_executor,
        "mistake_counts": experience.mistake_counts(),
        "per_example":    per_example_results,
    }

    # ── Persist ────────────────────────────────────────────────────────────────
    results_path.parent.mkdir(parents=True, exist_ok=True)
    experience_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Results saved → %s", results_path)

    experience.save(experience_path)
    log.info("Experience log → %s", experience_path)

    ep_memory.save(episodes_path)
    log.info("Episodic memory → %s  (%d episodes, %d with refinements)",
             episodes_path, len(ep_memory),
             len(ep_memory.with_refinements()))

    # ── Print ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  AIME 2026 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Dataset  : MathArena/aime_2026  (split={split})")
    print(f"  Problems : {n_total}")
    print(f"  Correct  : {n_correct}")
    print(f"  Accuracy : {accuracy:.2%}")
    print()
    print(experience.summary())

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark SciTracer on MathArena/aime_2026"
    )
    p.add_argument("--split",           default="train",
                   help="HuggingFace dataset split (default: train)")
    p.add_argument("--limit",           type=int, default=None,
                   help="Max problems to evaluate (default: all)")
    p.add_argument("--no-executor",     action="store_true",
                   help="Skip Executor step; use verifier trace for answer extraction")
    p.add_argument("--experience-path", default="benchmark/results/experience.jsonl",
                   help="Path for experience JSONL log")
    p.add_argument("--episodes-path",   default="benchmark/results/episodes.jsonl",
                   help="Path for rich episodic memory JSONL log")
    p.add_argument("--results-path",    default="benchmark/results/aime_2026_results.json",
                   help="Path for results JSON summary")
    p.add_argument("--no-categorize",   action="store_true",
                   help="Disable LLM-based mistake categorisation")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        split           = args.split,
        limit           = args.limit,
        run_executor    = not args.no_executor,
        experience_path = Path(args.experience_path),
        episodes_path   = Path(args.episodes_path),
        results_path    = Path(args.results_path),
        categorize      = not args.no_categorize,
    )
