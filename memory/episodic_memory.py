"""Episodic memory for the SciTracer agent pipeline.

Stores the *complete* reasoning chain for every problem run:
  plan → Kripke formalization → verification attempts (all retries + refinements)
  → executor output → final answer

Unlike the flat Experience log, each ReasoningEpisode keeps every intermediate
artifact separately so you can later ask:
  - Which problems required plan refinement?
  - What did the Z3 code look like before vs. after refinement?
  - How many verification attempts did the worst-case problem need?
  - Show me all episodes where the first attempt returned UNSAT.

Usage
-----
::
    from memory.episodic_memory import EpisodicMemory, EpisodeBuilder
    from benchmark.experience import MistakeCategory

    # ── build incrementally during a pipeline run ──────────────────────────────
    builder = EpisodeBuilder(run_id="aime2026-0007", question=problem_text)

    plan = researcher.forward(question)
    builder.set_plan(plan)

    formalized = formalizer.forward(plan)
    builder.set_formalized(formalized)

    outcome = verifier.forward(plan)           # verifier records its own attempts
    for att in outcome.attempts:               # see VerificationAttempt below
        builder.add_attempt(att)

    builder.set_executor_output(exec_result.output)
    builder.finish(
        final_answer   = "277",
        expected_answer= "277",
        is_correct     = True,
    )

    mem = EpisodicMemory()
    mem.add(builder.build())
    mem.save("memory/episodes.jsonl")

    # ── query interface ────────────────────────────────────────────────────────
    mem = EpisodicMemory.load("memory/episodes.jsonl")

    # All episodes that needed plan refinement
    refined = mem.with_refinements()

    # All episodes where the first Z3 attempt failed
    hard = mem.failed_first_attempt()

    # Free-text search
    hits = mem.search("orthogonality character sum")

    # Flexible filter
    wrong = mem.where(is_correct=False, min_attempts=2)

    print(mem.stats())
"""
from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerificationAttempt:
    """One Z3 (or NuSMV) verification attempt, including any plan refinement.

    Attributes
    ----------
    attempt_number : 1-based index within the episode.
    z3_code        : the generated Z3Py (or NuSMV .smv) source that was run.
    stdout         : captured standard output from the subprocess.
    stderr         : captured standard error (non-empty on failure).
    success        : True iff the subprocess exited cleanly and output was SAT.
    is_sat         : True iff 'sat' appeared in stdout and 'unsat' did not.
    is_unsat       : True iff 'unsat' appeared in stdout.
    refined_plan   : string representation of the refined plan produced after
                     this attempt failed, or None if no refinement occurred.
    elapsed_s      : wall-clock seconds for this attempt.
    backend        : 'z3' | 'nusmv' | 'lean'
    """
    attempt_number: int
    z3_code:        str
    stdout:         str
    stderr:         str
    success:        bool
    is_sat:         bool
    is_unsat:       bool
    refined_plan:   Optional[str] = None
    elapsed_s:      float = 0.0
    backend:        str   = "z3"

    @property
    def had_refinement(self) -> bool:
        return self.refined_plan is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "z3_code":        self.z3_code,
            "stdout":         self.stdout,
            "stderr":         self.stderr,
            "success":        self.success,
            "is_sat":         self.is_sat,
            "is_unsat":       self.is_unsat,
            "refined_plan":   self.refined_plan,
            "elapsed_s":      self.elapsed_s,
            "backend":        self.backend,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VerificationAttempt":
        return cls(
            attempt_number = d["attempt_number"],
            z3_code        = d.get("z3_code", ""),
            stdout         = d.get("stdout", ""),
            stderr         = d.get("stderr", ""),
            success        = d.get("success", False),
            is_sat         = d.get("is_sat", False),
            is_unsat       = d.get("is_unsat", False),
            refined_plan   = d.get("refined_plan"),
            elapsed_s      = d.get("elapsed_s", 0.0),
            backend        = d.get("backend", "z3"),
        )


@dataclass
class ReasoningEpisode:
    """Complete reasoning chain for one problem run.

    Attributes
    ----------
    episode_id        : unique UUID for this episode.
    run_id            : benchmark run identifier (e.g. 'aime2026-0007-1780229519').
    timestamp         : UTC ISO-8601 creation time.
    question          : the original problem text.
    plan              : string representation of the ResearchPlan.
    kripke_structure  : string representation of the extracted Kripke structure.
    formalized_code   : the raw Z3/NuSMV code before the first execution attempt.
    verification_attempts : ordered list of ALL attempts, including retried ones.
    executor_output   : stdout from the Executor step.
    final_answer      : answer string produced by the pipeline.
    expected_answer   : ground-truth answer.
    is_correct        : whether final_answer matched expected_answer.
    mistake_category  : taxonomy category if is_correct=False.
    mistake_explanation : one-sentence LLM explanation of the mistake.
    metadata          : arbitrary extra context (subject, elapsed_s, model, …).
    """
    episode_id:            str
    run_id:                str
    timestamp:             str
    question:              str

    # ── pipeline stages ───────────────────────────────────────────────────────
    plan:                  str = ""
    kripke_structure:      str = ""
    formalized_code:       str = ""
    verification_attempts: list[VerificationAttempt] = field(default_factory=list)
    executor_output:       str = ""

    # ── outcome ───────────────────────────────────────────────────────────────
    final_answer:          str  = ""
    expected_answer:       str  = ""
    is_correct:            bool = False
    mistake_category:      Optional[str] = None
    mistake_explanation:   str  = ""

    metadata:              dict[str, Any] = field(default_factory=dict)

    # ── derived properties ────────────────────────────────────────────────────

    @property
    def n_attempts(self) -> int:
        return len(self.verification_attempts)

    @property
    def had_refinement(self) -> bool:
        return any(a.had_refinement for a in self.verification_attempts)

    @property
    def first_attempt_failed(self) -> bool:
        return bool(self.verification_attempts) and not self.verification_attempts[0].success

    @property
    def final_attempt(self) -> Optional[VerificationAttempt]:
        return self.verification_attempts[-1] if self.verification_attempts else None

    @property
    def refinement_pairs(self) -> list[tuple[VerificationAttempt, VerificationAttempt]]:
        """Return (failed_attempt, next_attempt) pairs where refinement occurred."""
        pairs = []
        for i, att in enumerate(self.verification_attempts[:-1]):
            if att.had_refinement:
                pairs.append((att, self.verification_attempts[i + 1]))
        return pairs

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id":             self.episode_id,
            "run_id":                 self.run_id,
            "timestamp":              self.timestamp,
            "question":               self.question,
            "plan":                   self.plan,
            "kripke_structure":       self.kripke_structure,
            "formalized_code":        self.formalized_code,
            "verification_attempts":  [a.to_dict() for a in self.verification_attempts],
            "executor_output":        self.executor_output,
            "final_answer":           self.final_answer,
            "expected_answer":        self.expected_answer,
            "is_correct":             self.is_correct,
            "mistake_category":       self.mistake_category,
            "mistake_explanation":    self.mistake_explanation,
            "metadata":               self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReasoningEpisode":
        return cls(
            episode_id           = d.get("episode_id", str(uuid.uuid4())),
            run_id               = d.get("run_id", ""),
            timestamp            = d.get("timestamp", datetime.now(timezone.utc).isoformat()),
            question             = d.get("question", ""),
            plan                 = d.get("plan", ""),
            kripke_structure     = d.get("kripke_structure", ""),
            formalized_code      = d.get("formalized_code", ""),
            verification_attempts= [VerificationAttempt.from_dict(a)
                                    for a in d.get("verification_attempts", [])],
            executor_output      = d.get("executor_output", ""),
            final_answer         = d.get("final_answer", ""),
            expected_answer      = d.get("expected_answer", ""),
            is_correct           = d.get("is_correct", False),
            mistake_category     = d.get("mistake_category"),
            mistake_explanation  = d.get("mistake_explanation", ""),
            metadata             = d.get("metadata", {}),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Builder  — incremental construction during a live pipeline run
# ══════════════════════════════════════════════════════════════════════════════

class EpisodeBuilder:
    """Build a ReasoningEpisode incrementally as the pipeline executes.

    Example
    -------
        builder = EpisodeBuilder(run_id="aime2026-0007", question=text)

        builder.set_plan(str(researcher.forward(text)))
        builder.set_kripke(str(kripke_pred.kripke))
        builder.set_formalized_code(z3_code)

        for attempt_num, (code, result, refined) in enumerate(verifier_attempts, 1):
            builder.add_attempt(VerificationAttempt(
                attempt_number = attempt_num,
                z3_code        = code,
                stdout         = result.stdout,
                stderr         = result.stderr,
                success        = result.success,
                is_sat         = "sat" in result.stdout.lower() and "unsat" not in result.stdout.lower(),
                is_unsat       = "unsat" in result.stdout.lower(),
                refined_plan   = str(refined) if refined else None,
            ))

        builder.set_executor_output(exec_output)
        builder.finish(
            final_answer    = predicted_str,
            expected_answer = str(expected),
            is_correct      = correct,
            mistake_category    = category.value if not correct else None,
            mistake_explanation = explanation,
            metadata        = {"subject": "math", "elapsed_s": elapsed},
        )

        episode = builder.build()
    """

    def __init__(self, run_id: str, question: str) -> None:
        self._run_id    = run_id
        self._question  = question
        self._plan      = ""
        self._kripke    = ""
        self._formalized= ""
        self._attempts: list[VerificationAttempt] = []
        self._executor  = ""
        self._final_answer    = ""
        self._expected_answer = ""
        self._is_correct      = False
        self._mistake_category: Optional[str] = None
        self._mistake_explanation = ""
        self._metadata: dict[str, Any] = {}

    def set_plan(self, plan: Any) -> "EpisodeBuilder":
        self._plan = str(plan)
        return self

    def set_kripke(self, kripke: Any) -> "EpisodeBuilder":
        self._kripke = str(kripke)
        return self

    def set_formalized_code(self, code: str) -> "EpisodeBuilder":
        self._formalized = code
        return self

    def add_attempt(self, attempt: VerificationAttempt) -> "EpisodeBuilder":
        self._attempts.append(attempt)
        return self

    def set_executor_output(self, output: str) -> "EpisodeBuilder":
        self._executor = output
        return self

    def finish(
        self,
        final_answer:        str,
        expected_answer:     str,
        is_correct:          bool,
        mistake_category:    Optional[str] = None,
        mistake_explanation: str = "",
        metadata:            dict[str, Any] | None = None,
    ) -> "EpisodeBuilder":
        self._final_answer        = final_answer
        self._expected_answer     = expected_answer
        self._is_correct          = is_correct
        self._mistake_category    = mistake_category
        self._mistake_explanation = mistake_explanation
        self._metadata            = metadata or {}
        return self

    def build(self) -> ReasoningEpisode:
        return ReasoningEpisode(
            episode_id            = str(uuid.uuid4()),
            run_id                = self._run_id,
            timestamp             = datetime.now(timezone.utc).isoformat(),
            question              = self._question,
            plan                  = self._plan,
            kripke_structure      = self._kripke,
            formalized_code       = self._formalized,
            verification_attempts = list(self._attempts),
            executor_output       = self._executor,
            final_answer          = self._final_answer,
            expected_answer       = self._expected_answer,
            is_correct            = self._is_correct,
            mistake_category      = self._mistake_category,
            mistake_explanation   = self._mistake_explanation,
            metadata              = self._metadata,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Episodic Memory store + query interface
# ══════════════════════════════════════════════════════════════════════════════

class EpisodicMemory:
    """Persistent store of ReasoningEpisodes with a rich query interface.

    Persistence
    -----------
    Episodes are stored as newline-delimited JSON (one episode per line) so the
    file can be streamed, appended to, and read without loading everything into
    memory at once.

    Query methods
    -------------
    mem.all()                → list[ReasoningEpisode]
    mem.get(episode_id)      → ReasoningEpisode | None
    mem.where(...)           → filtered list (see method docstring)
    mem.search(text)         → full-text search across question + plan + traces
    mem.with_refinements()   → episodes where at least one plan was refined
    mem.failed_first_attempt() → episodes where attempt 1 failed
    mem.by_question(q)       → all episodes for a given question
    mem.by_run(run_id)       → all episodes from one benchmark run
    mem.stats()              → summary dict
    """

    def __init__(self) -> None:
        self._episodes: list[ReasoningEpisode] = []
        self._index:    dict[str, ReasoningEpisode] = {}   # episode_id → episode

    # ── write ──────────────────────────────────────────────────────────────────

    def add(self, episode: ReasoningEpisode) -> None:
        """Append a new episode to the in-memory store."""
        self._episodes.append(episode)
        self._index[episode.episode_id] = episode

    # ── basic retrieval ────────────────────────────────────────────────────────

    def all(self) -> list[ReasoningEpisode]:
        return list(self._episodes)

    def get(self, episode_id: str) -> Optional[ReasoningEpisode]:
        return self._index.get(episode_id)

    def by_run(self, run_id: str) -> list[ReasoningEpisode]:
        return [e for e in self._episodes if e.run_id == run_id]

    def by_question(self, question: str) -> list[ReasoningEpisode]:
        """Return all episodes whose question equals or contains the given text."""
        q = question.strip()
        return [e for e in self._episodes if q in e.question or e.question in q]

    # ── semantic filters ───────────────────────────────────────────────────────

    def with_refinements(self) -> list[ReasoningEpisode]:
        """Episodes where at least one verification attempt triggered plan refinement."""
        return [e for e in self._episodes if e.had_refinement]

    def failed_first_attempt(self) -> list[ReasoningEpisode]:
        """Episodes where the first Z3 attempt failed (UNSAT or subprocess error)."""
        return [e for e in self._episodes if e.first_attempt_failed]

    def correct(self) -> list[ReasoningEpisode]:
        return [e for e in self._episodes if e.is_correct]

    def incorrect(self) -> list[ReasoningEpisode]:
        return [e for e in self._episodes if not e.is_correct]

    def by_mistake_category(self, category: str) -> list[ReasoningEpisode]:
        return [e for e in self._episodes if e.mistake_category == category]

    # ── flexible filter ────────────────────────────────────────────────────────

    def where(
        self,
        *,
        is_correct:        Optional[bool] = None,
        had_refinement:    Optional[bool] = None,
        min_attempts:      Optional[int]  = None,
        max_attempts:      Optional[int]  = None,
        first_failed:      Optional[bool] = None,
        mistake_category:  Optional[str]  = None,
        run_id:            Optional[str]  = None,
        backend:           Optional[str]  = None,
    ) -> list[ReasoningEpisode]:
        """Return episodes matching all supplied (non-None) criteria.

        Parameters
        ----------
        is_correct       : filter by final answer correctness.
        had_refinement   : filter by whether plan refinement occurred.
        min_attempts     : minimum number of verification attempts.
        max_attempts     : maximum number of verification attempts.
        first_failed     : filter by whether the first attempt failed.
        mistake_category : filter by mistake taxonomy string.
        run_id           : filter to a single benchmark run.
        backend          : filter by verifier backend ('z3' | 'nusmv' | 'lean').
        """
        result = self._episodes
        if is_correct is not None:
            result = [e for e in result if e.is_correct == is_correct]
        if had_refinement is not None:
            result = [e for e in result if e.had_refinement == had_refinement]
        if min_attempts is not None:
            result = [e for e in result if e.n_attempts >= min_attempts]
        if max_attempts is not None:
            result = [e for e in result if e.n_attempts <= max_attempts]
        if first_failed is not None:
            result = [e for e in result if e.first_attempt_failed == first_failed]
        if mistake_category is not None:
            result = [e for e in result if e.mistake_category == mistake_category]
        if run_id is not None:
            result = [e for e in result if e.run_id == run_id]
        if backend is not None:
            result = [
                e for e in result
                if any(a.backend == backend for a in e.verification_attempts)
            ]
        return result

    # ── full-text search ───────────────────────────────────────────────────────

    def search(self, text: str) -> list[ReasoningEpisode]:
        """Return episodes where `text` appears in question, plan, kripke, or
        any verification attempt (code or stdout)."""
        t = text.lower()
        hits = []
        for e in self._episodes:
            haystack = " ".join([
                e.question, e.plan, e.kripke_structure, e.formalized_code,
                e.executor_output,
                *[a.z3_code + " " + a.stdout + " " + (a.refined_plan or "")
                  for a in e.verification_attempts],
            ]).lower()
            if t in haystack:
                hits.append(e)
        return hits

    # ── statistics ─────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return a summary statistics dict."""
        n = len(self._episodes)
        if n == 0:
            return {"total": 0}

        attempts_list = [e.n_attempts for e in self._episodes]
        category_counts = Counter(
            e.mistake_category
            for e in self._episodes
            if not e.is_correct and e.mistake_category
        )

        return {
            "total":                   n,
            "correct":                 sum(1 for e in self._episodes if e.is_correct),
            "incorrect":               sum(1 for e in self._episodes if not e.is_correct),
            "accuracy":                round(sum(1 for e in self._episodes if e.is_correct) / n, 4),
            "with_refinements":        sum(1 for e in self._episodes if e.had_refinement),
            "first_attempt_failed":    sum(1 for e in self._episodes if e.first_attempt_failed),
            "avg_attempts":            round(sum(attempts_list) / n, 2),
            "max_attempts":            max(attempts_list, default=0),
            "total_attempts":          sum(attempts_list),
            "mistake_category_counts": dict(category_counts.most_common()),
        }

    def summary(self) -> str:
        """Human-readable summary table."""
        s = self.stats()
        if s["total"] == 0:
            return "EpisodicMemory: empty"

        lines = [
            "=" * 60,
            "  EPISODIC MEMORY SUMMARY",
            "=" * 60,
            f"  Episodes       : {s['total']}",
            f"  Correct        : {s['correct']}",
            f"  Incorrect      : {s['incorrect']}",
            f"  Accuracy       : {s['accuracy']:.2%}",
            "",
            f"  With refinements    : {s['with_refinements']}",
            f"  First attempt failed: {s['first_attempt_failed']}",
            f"  Avg attempts/episode: {s['avg_attempts']}",
            f"  Max attempts        : {s['max_attempts']}",
            f"  Total attempts      : {s['total_attempts']}",
        ]
        if s["mistake_category_counts"]:
            lines += ["", "  MISTAKE BREAKDOWN", "-" * 60]
            for cat, cnt in s["mistake_category_counts"].items():
                pct = cnt / s["incorrect"] * 100 if s["incorrect"] else 0
                lines.append(f"    {cat:<35} {cnt:>4}  ({pct:5.1f}%)")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ── persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path, mode: str = "a") -> None:
        """Persist episodes to a JSONL file.

        Parameters
        ----------
        path : destination file path.
        mode : 'a' (append, default) or 'w' (overwrite).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode, encoding="utf-8") as f:
            for ep in self._episodes:
                f.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "EpisodicMemory":
        """Load episodes from a JSONL file produced by :meth:`save`."""
        mem = cls()
        path = Path(path)
        if not path.exists():
            return mem
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    mem.add(ReasoningEpisode.from_dict(json.loads(line)))
        return mem

    def __len__(self) -> int:
        return len(self._episodes)

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"EpisodicMemory(total={s['total']}, correct={s.get('correct',0)}, "
            f"with_refinements={s.get('with_refinements',0)}, "
            f"avg_attempts={s.get('avg_attempts',0)})"
        )
