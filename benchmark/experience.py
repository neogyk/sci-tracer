"""Experience — stores agent reasoning steps, mistakes and their categorisation.

An Experience object accumulates everything that happened during a single run
(or across multiple runs) so that post-hoc analysis can identify systematic
weaknesses and feed correction examples back into the training loop.

Mistake taxonomy
----------------
Each mistake must belong to exactly one of the categories below (chosen to
cover the full space of failure modes observed in scientific / mathematical
reasoning agents):

    REASONING_ERROR             — valid premises, invalid logical inference
    CONCEPTUAL_MISUNDERSTANDING — incorrect domain knowledge assumed as true
    MISSED_INSIGHT              — correct direction but key step skipped/ignored
    INCOMPLETE_OR_UNCLEAR       — answer is partially right but ambiguous/truncated
    CALCULATION_ERROR           — arithmetic / algebraic / numerical mistake
    INVALID_APPROACH            — wrong method chosen for the problem type
    QUESTION_MISUNDERSTANDING   — agent answered a different question than asked
    NOTATION_ERROR              — symbol/variable/unit used incorrectly
    HALLUCINATION               — fabricated fact with no grounding in context
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ── Mistake taxonomy ───────────────────────────────────────────────────────────

class MistakeCategory(str, Enum):
    REASONING_ERROR            = "Reasoning Error"
    CONCEPTUAL_MISUNDERSTANDING = "Conceptual Misunderstanding"
    MISSED_INSIGHT             = "Missed Insight"
    INCOMPLETE_OR_UNCLEAR      = "Incomplete or Unclear Answer"
    CALCULATION_ERROR          = "Calculation Error"
    INVALID_APPROACH           = "Invalid Approach"
    QUESTION_MISUNDERSTANDING  = "Question Misunderstanding"
    NOTATION_ERROR             = "Notation Error"
    HALLUCINATION              = "Hallucination"


# ── Step result ────────────────────────────────────────────────────────────────

class StepResult(str, Enum):
    CORRECT   = "correct"
    MISTAKE   = "mistake"
    SKIPPED   = "skipped"
    UNCERTAIN = "uncertain"


# ── Single experience entry ────────────────────────────────────────────────────

@dataclass
class ExperienceEntry:
    """Represents one agent step — either a correct step or a mistake.

    Parameters
    ----------
    step_id:
        Unique identifier for this step within a run.
    question:
        The original question or sub-question this step was addressing.
    agent_output:
        What the agent actually produced at this step (reasoning / answer text).
    result:
        Whether this step was correct, a mistake, skipped, or uncertain.
    mistake_category:
        If ``result == StepResult.MISTAKE``, the taxonomy category.
        None for correct/skipped steps.
    expected_output:
        Ground-truth or reference output for this step (if known).
    explanation:
        Human- or LLM-generated explanation of why this is a mistake (or why
        it is correct).
    metadata:
        Arbitrary dict for run-specific context (e.g. agent name, step index,
        model temperature, token counts).
    run_id:
        Groups entries that belong to the same pipeline run.
    timestamp:
        UTC ISO-8601 when this entry was recorded.
    """

    step_id:          str
    question:         str
    agent_output:     str
    result:           StepResult
    mistake_category: MistakeCategory | None = None
    expected_output:  str = ""
    explanation:      str = ""
    metadata:         dict[str, Any] = field(default_factory=dict)
    run_id:           str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:        str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Validation ─────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if self.result == StepResult.MISTAKE and self.mistake_category is None:
            raise ValueError(
                "ExperienceEntry: `mistake_category` must be set when result=MISTAKE."
            )
        if self.result != StepResult.MISTAKE and self.mistake_category is not None:
            raise ValueError(
                "ExperienceEntry: `mistake_category` should only be set for MISTAKE entries."
            )

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id":          self.step_id,
            "run_id":           self.run_id,
            "timestamp":        self.timestamp,
            "question":         self.question,
            "agent_output":     self.agent_output,
            "expected_output":  self.expected_output,
            "result":           self.result.value,
            "mistake_category": self.mistake_category.value if self.mistake_category else None,
            "explanation":      self.explanation,
            "metadata":         self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperienceEntry":
        return cls(
            step_id          = d["step_id"],
            run_id           = d.get("run_id", str(uuid.uuid4())),
            timestamp        = d.get("timestamp", datetime.now(timezone.utc).isoformat()),
            question         = d["question"],
            agent_output     = d["agent_output"],
            expected_output  = d.get("expected_output", ""),
            result           = StepResult(d["result"]),
            mistake_category = MistakeCategory(d["mistake_category"]) if d.get("mistake_category") else None,
            explanation      = d.get("explanation", ""),
            metadata         = d.get("metadata", {}),
        )


# ── Experience store ───────────────────────────────────────────────────────────

class Experience:
    """Persistent log of agent reasoning steps, mistakes, and correct moves.

    Designed to be accumulated across many benchmark runs and later used for:
      - Mistake frequency analysis by category
      - Few-shot correction example selection
      - DSPy BootstrapFewShotWithRandomSearch training sets
      - Human-in-the-loop auditing

    Usage
    -----
    ::

        exp = Experience()

        # Record a correct step
        exp.add_correct(
            question     = "What is 2 + 2?",
            agent_output = "4",
            expected_output = "4",
            run_id       = "run-001",
        )

        # Record a mistake
        exp.add_mistake(
            question         = "Integrate x^2 dx",
            agent_output     = "x^3 + C",
            expected_output  = "x^3/3 + C",
            category         = MistakeCategory.CALCULATION_ERROR,
            explanation      = "Forgot to divide by 3 when integrating.",
            run_id           = "run-001",
        )

        # Summary
        print(exp.summary())

        # Persist
        exp.save("experience.jsonl")

        # Reload
        exp2 = Experience.load("experience.jsonl")
    """

    def __init__(self) -> None:
        self._entries: list[ExperienceEntry] = []

    # ── Core add methods ───────────────────────────────────────────────────────

    def add_entry(self, entry: ExperienceEntry) -> None:
        """Append a pre-built ExperienceEntry."""
        self._entries.append(entry)

    def add_correct(
        self,
        question:        str,
        agent_output:    str,
        expected_output: str = "",
        explanation:     str = "",
        run_id:          str = "",
        metadata:        dict[str, Any] | None = None,
    ) -> ExperienceEntry:
        """Record a correct agent step. Returns the created entry."""
        entry = ExperienceEntry(
            step_id         = str(uuid.uuid4()),
            run_id          = run_id or str(uuid.uuid4()),
            question        = question,
            agent_output    = agent_output,
            expected_output = expected_output,
            result          = StepResult.CORRECT,
            explanation     = explanation,
            metadata        = metadata or {},
        )
        self._entries.append(entry)
        return entry

    def add_mistake(
        self,
        question:        str,
        agent_output:    str,
        category:        MistakeCategory,
        expected_output: str = "",
        explanation:     str = "",
        run_id:          str = "",
        metadata:        dict[str, Any] | None = None,
    ) -> ExperienceEntry:
        """Record a mistake. Returns the created entry."""
        entry = ExperienceEntry(
            step_id          = str(uuid.uuid4()),
            run_id           = run_id or str(uuid.uuid4()),
            question         = question,
            agent_output     = agent_output,
            expected_output  = expected_output,
            result           = StepResult.MISTAKE,
            mistake_category = category,
            explanation      = explanation,
            metadata         = metadata or {},
        )
        self._entries.append(entry)
        return entry

    def add_skipped(
        self,
        question:    str,
        run_id:      str = "",
        explanation: str = "",
        metadata:    dict[str, Any] | None = None,
    ) -> ExperienceEntry:
        """Record a skipped step (e.g. agent timed out or refused)."""
        entry = ExperienceEntry(
            step_id      = str(uuid.uuid4()),
            run_id       = run_id or str(uuid.uuid4()),
            question     = question,
            agent_output = "",
            result       = StepResult.SKIPPED,
            explanation  = explanation,
            metadata     = metadata or {},
        )
        self._entries.append(entry)
        return entry

    # ── Queries ────────────────────────────────────────────────────────────────

    @property
    def entries(self) -> list[ExperienceEntry]:
        return list(self._entries)

    @property
    def mistakes(self) -> list[ExperienceEntry]:
        return [e for e in self._entries if e.result == StepResult.MISTAKE]

    @property
    def correct_steps(self) -> list[ExperienceEntry]:
        return [e for e in self._entries if e.result == StepResult.CORRECT]

    def by_category(self, category: MistakeCategory) -> list[ExperienceEntry]:
        """Return all mistake entries with the given category."""
        return [e for e in self.mistakes if e.mistake_category == category]

    def by_run(self, run_id: str) -> list[ExperienceEntry]:
        """Return all entries from a specific run."""
        return [e for e in self._entries if e.run_id == run_id]

    def mistake_counts(self) -> dict[str, int]:
        """Return a dict mapping category name → count, sorted descending."""
        counts: dict[str, int] = {cat.value: 0 for cat in MistakeCategory}
        for e in self.mistakes:
            if e.mistake_category:
                counts[e.mistake_category.value] += 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    # ── Statistics ─────────────────────────────────────────────────────────────

    def accuracy(self) -> float:
        """Fraction of non-skipped entries that are correct."""
        evaluated = [e for e in self._entries if e.result != StepResult.SKIPPED]
        if not evaluated:
            return 0.0
        return sum(1 for e in evaluated if e.result == StepResult.CORRECT) / len(evaluated)

    def summary(self) -> str:
        """Return a human-readable summary table."""
        total     = len(self._entries)
        n_correct = len(self.correct_steps)
        n_mistake = len(self.mistakes)
        n_skipped = sum(1 for e in self._entries if e.result == StepResult.SKIPPED)
        n_uncert  = sum(1 for e in self._entries if e.result == StepResult.UNCERTAIN)

        lines = [
            "=" * 60,
            "  EXPERIENCE SUMMARY",
            "=" * 60,
            f"  Total entries : {total}",
            f"  Correct       : {n_correct}",
            f"  Mistakes      : {n_mistake}",
            f"  Skipped       : {n_skipped}",
            f"  Uncertain     : {n_uncert}",
            f"  Accuracy      : {self.accuracy():.2%}",
            "",
            "  MISTAKE BREAKDOWN",
            "-" * 60,
        ]
        for cat, count in self.mistake_counts().items():
            if count > 0:
                pct = count / n_mistake * 100 if n_mistake else 0
                lines.append(f"    {cat:<35} {count:>4}  ({pct:5.1f}%)")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Append all entries to a JSONL file (one JSON object per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def save_json(self, path: str | Path) -> None:
        """Write all entries to a single JSON array file (overwrites)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._entries], f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "Experience":
        """Load entries from a JSONL file produced by :meth:`save`."""
        path = Path(path)
        exp = cls()
        if not path.exists():
            return exp
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    exp.add_entry(ExperienceEntry.from_dict(json.loads(line)))
        return exp

    @classmethod
    def load_json(cls, path: str | Path) -> "Experience":
        """Load entries from a JSON array file produced by :meth:`save_json`."""
        path = Path(path)
        exp = cls()
        if not path.exists():
            return exp
        with path.open("r", encoding="utf-8") as f:
            items = json.load(f)
        for d in items:
            exp.add_entry(ExperienceEntry.from_dict(d))
        return exp

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"Experience(total={len(self)}, correct={len(self.correct_steps)}, "
            f"mistakes={len(self.mistakes)}, accuracy={self.accuracy():.2%})"
        )
