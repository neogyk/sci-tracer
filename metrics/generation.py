"""Generation quality metrics for RAG evaluation.

Functions
---------
faithfulness       — is every claim in the answer supported by the retrieved context?
answer_relevance   — does the answer directly address the user's question?

Both metrics use DSPy LLM-as-judge signatures and return a float in [0, 1].
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import dspy

log = logging.getLogger("scitracer.metrics.generation")


# ── DSPy signatures ────────────────────────────────────────────────────────────

class ClaimExtractor(dspy.Signature):
    """Extract every atomic factual claim made in the answer as a numbered list.

    Each claim should be a single, self-contained statement that can be
    independently verified against an external source.
    """

    answer: str = dspy.InputField(desc="Generated answer to decompose into atomic claims")

    claims: str = dspy.OutputField(
        desc="Newline-separated list of atomic claims, one per line, no bullets or numbers"
    )


class ClaimVerifier(dspy.Signature):
    """Decide whether a single claim is fully supported by the provided context.

    A claim is supported when the context explicitly states or logically implies it.
    A claim is NOT supported when it requires outside knowledge not present in the context,
    even if the claim happens to be true in the real world.
    """

    context: str = dspy.InputField(desc="Retrieved document passages used to generate the answer")
    claim:   str = dspy.InputField(desc="A single atomic claim to verify")

    is_supported: bool  = dspy.OutputField(desc="True if the claim is directly supported by the context")
    evidence:     str   = dspy.OutputField(desc="Verbatim quote or paraphrase from the context that supports the claim, or 'none'")


class AnswerRelevanceJudge(dspy.Signature):
    """Score how directly and completely the answer addresses the user's question.

    A score of 1.0 means the answer is laser-focused on the question with no
    irrelevant content.  A score of 0.0 means the answer is entirely off-topic.
    Penalise answers that are evasive, rambling, or answer a different question.
    """

    question: str = dspy.InputField(desc="User's original question")
    answer:   str = dspy.InputField(desc="Generated answer to evaluate")

    relevance_score: float = dspy.OutputField(
        desc="Float in [0, 1] measuring how directly the answer addresses the question"
    )
    explanation: str = dspy.OutputField(
        desc="One-sentence explanation of the score"
    )


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class FaithfulnessResult:
    score:           float = 0.0
    n_claims:        int   = 0
    n_supported:     int   = 0
    unsupported:     list  = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Faithfulness(score={self.score:.3f}  "
            f"supported={self.n_supported}/{self.n_claims}  "
            f"unsupported={self.unsupported})"
        )


@dataclass
class AnswerRelevanceResult:
    score:       float = 0.0
    explanation: str   = ""

    def __str__(self) -> str:
        return f"AnswerRelevance(score={self.score:.3f}  explanation={self.explanation!r})"


@dataclass
class GenerationMetrics:
    faithfulness:    FaithfulnessResult    = field(default_factory=FaithfulnessResult)
    answer_relevance: AnswerRelevanceResult = field(default_factory=AnswerRelevanceResult)

    @property
    def faithfulness_score(self) -> float:
        return self.faithfulness.score

    @property
    def answer_relevance_score(self) -> float:
        return self.answer_relevance.score

    def __str__(self) -> str:
        return (
            f"GenerationMetrics("
            f"faithfulness={self.faithfulness.score:.3f}  "
            f"answer_relevance={self.answer_relevance.score:.3f})"
        )


# ── Individual metric functions ────────────────────────────────────────────────

def faithfulness(
    answer: str,
    context: str | Sequence[str],
) -> FaithfulnessResult:
    """Measure whether every claim in the answer is supported by the context.

    Algorithm
    ---------
    1. Use an LLM to decompose the answer into atomic claims.
    2. For each claim, use an LLM verifier to decide if the context supports it.
    3. Score = supported_claims / total_claims.

    Parameters
    ----------
    answer:
        The generated answer to evaluate.
    context:
        Retrieved document text(s) used to generate the answer.
        If a list is provided the passages are concatenated.

    Returns
    -------
    FaithfulnessResult
    """
    if isinstance(context, (list, tuple)):
        context = "\n\n---\n\n".join(str(c)[:1500] for c in context)

    if not answer.strip():
        return FaithfulnessResult()

    log.info("faithfulness: extracting claims from answer (%d chars)", len(answer))

    extractor = dspy.ChainOfThought(ClaimExtractor)
    verifier  = dspy.ChainOfThought(ClaimVerifier)

    # Step 1 — extract claims
    try:
        extracted = extractor(answer=answer)
        raw_claims = [c.strip() for c in extracted.claims.splitlines() if c.strip()]
    except Exception as exc:
        log.warning("ClaimExtractor failed: %s", exc)
        return FaithfulnessResult()

    if not raw_claims:
        log.warning("ClaimExtractor returned no claims")
        return FaithfulnessResult()

    log.info("faithfulness: verifying %d claims against context", len(raw_claims))

    # Step 2 — verify each claim
    supported:   list[str] = []
    unsupported: list[str] = []

    for claim in raw_claims:
        try:
            pred = verifier(context=context[:4000], claim=claim)
            if pred.is_supported:
                supported.append(claim)
                log.debug("  [SUPPORTED]   %s", claim)
            else:
                unsupported.append(claim)
                log.debug("  [UNSUPPORTED] %s", claim)
        except Exception as exc:
            log.warning("ClaimVerifier failed for claim %r: %s", claim[:60], exc)
            unsupported.append(claim)

    n_claims    = len(raw_claims)
    n_supported = len(supported)
    score       = n_supported / n_claims if n_claims > 0 else 0.0

    result = FaithfulnessResult(
        score=score,
        n_claims=n_claims,
        n_supported=n_supported,
        unsupported=unsupported,
    )
    log.info("faithfulness result: %s", result)
    return result


def answer_relevance(
    question: str,
    answer: str,
) -> AnswerRelevanceResult:
    """Measure how directly the answer addresses the user's question.

    Uses an LLM judge that scores the answer on a continuous [0, 1] scale
    and provides a one-sentence explanation.

    Parameters
    ----------
    question:
        The user's original research question.
    answer:
        The generated answer to evaluate.

    Returns
    -------
    AnswerRelevanceResult
    """
    if not answer.strip() or not question.strip():
        return AnswerRelevanceResult()

    log.info("answer_relevance: question=%r  answer_len=%d", question[:80], len(answer))

    judge = dspy.ChainOfThought(AnswerRelevanceJudge)

    try:
        pred  = judge(question=question, answer=answer[:3000])
        score = max(0.0, min(1.0, float(pred.relevance_score)))
        result = AnswerRelevanceResult(score=score, explanation=pred.explanation)
        log.info("answer_relevance result: %s", result)
        return result
    except Exception as exc:
        log.warning("AnswerRelevanceJudge failed: %s", exc)
        return AnswerRelevanceResult()


# ── Convenience: compute all generation metrics at once ───────────────────────

def evaluate_generation(
    question: str,
    answer: str,
    context: str | Sequence[str],
) -> GenerationMetrics:
    """Compute faithfulness and answer relevance in one call.

    Parameters
    ----------
    question:
        The user's original research question.
    answer:
        The generated answer to evaluate.
    context:
        Retrieved document text(s) used to produce the answer.

    Returns
    -------
    GenerationMetrics dataclass
    """
    log.info(
        "evaluate_generation  question=%r  answer_len=%d",
        question[:80], len(answer),
    )

    faith = faithfulness(answer=answer, context=context)
    rel   = answer_relevance(question=question, answer=answer)

    result = GenerationMetrics(faithfulness=faith, answer_relevance=rel)
    log.info("generation result: %s", result)
    return result
