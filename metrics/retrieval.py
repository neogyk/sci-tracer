"""Retrieval quality metrics for RAG evaluation.

Functions
---------
context_precision   — fraction of retrieved docs that are relevant to the query
context_recall      — fraction of ground-truth information covered by the retrieved context
hit_rate            — was at least one relevant doc in the top-k results?
mrr                 — Mean Reciprocal Rank of the first relevant result
ndcg                — Normalized Discounted Cumulative Gain at k

All metrics return a float in [0, 1].

When ``relevance_labels`` (list[int], 1=relevant / 0=not) are not supplied,
the semantic metrics fall back to an LLM judge via a DSPy Signature so that
a ground-truth corpus is not strictly required.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Sequence

import dspy
import numpy as np

log = logging.getLogger("scitracer.metrics.retrieval")


# ── DSPy signatures ────────────────────────────────────────────────────────────

class DocumentRelevanceJudge(dspy.Signature):
    """Decide whether a document is useful for answering the query.

    Return is_relevant=True only when the document contains substantive
    information that directly helps address the query.  Tangentially related
    content counts as not relevant.
    """

    query: str    = dspy.InputField(desc="User research query")
    document: str = dspy.InputField(desc="Retrieved document text (may be truncated)")

    is_relevant:     bool  = dspy.OutputField(desc="True if the document is useful for answering the query")
    relevance_score: float = dspy.OutputField(desc="Continuous relevance score between 0.0 (irrelevant) and 1.0 (highly relevant)")


class RecallCoverageJudge(dspy.Signature):
    """Estimate what fraction of the ground-truth answer's key facts are present
    in the retrieved context.

    Score 1.0 when every key fact in the reference answer can be found in the
    context; 0.0 when none can be found.
    """

    query:            str = dspy.InputField(desc="User research query")
    retrieved_context:str = dspy.InputField(desc="Concatenated retrieved document passages")
    reference_answer: str = dspy.InputField(desc="Ground-truth answer or key facts that should be present")

    coverage_score: float = dspy.OutputField(
        desc="Float in [0, 1]: fraction of key facts from the reference answer that appear in the retrieved context"
    )
    missing_facts: str = dspy.OutputField(
        desc="Comma-separated list of key facts not found in the retrieved context, or 'none'"
    )


# ── Dataclass ──────────────────────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    context_precision: float = 0.0
    context_recall:    float = 0.0
    hit_rate:          float = 0.0
    mrr:               float = 0.0
    ndcg:              float = 0.0
    details:           dict  = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"RetrievalMetrics("
            f"precision={self.context_precision:.3f}  "
            f"recall={self.context_recall:.3f}  "
            f"hit_rate={self.hit_rate:.3f}  "
            f"MRR={self.mrr:.3f}  "
            f"NDCG={self.ndcg:.3f})"
        )


# ── Individual metric functions ────────────────────────────────────────────────

def context_precision(
    query: str,
    retrieved_docs: Sequence[str],
    relevance_labels: Sequence[int] | None = None,
) -> float:
    """Precision@k — fraction of retrieved docs that are relevant.

    Parameters
    ----------
    query:
        The user's search query.
    retrieved_docs:
        Ordered list of retrieved document texts.
    relevance_labels:
        Optional binary labels (1=relevant, 0=not relevant), aligned with
        ``retrieved_docs``.  When absent an LLM judge is used.

    Returns
    -------
    float in [0, 1]
    """
    if not retrieved_docs:
        return 0.0

    if relevance_labels is not None:
        if len(relevance_labels) != len(retrieved_docs):
            raise ValueError("relevance_labels must have the same length as retrieved_docs")
        return float(sum(relevance_labels)) / len(retrieved_docs)

    judge = dspy.ChainOfThought(DocumentRelevanceJudge)
    n_relevant = 0
    per_doc: list[dict] = []

    for i, doc in enumerate(retrieved_docs):
        try:
            pred = judge(query=query, document=doc[:2000])
            relevant = bool(pred.is_relevant)
            score    = float(pred.relevance_score)
        except Exception as exc:
            log.warning("DocumentRelevanceJudge failed for doc %d: %s", i, exc)
            relevant, score = False, 0.0

        if relevant:
            n_relevant += 1
        per_doc.append({"rank": i + 1, "is_relevant": relevant, "score": score})
        log.debug("  doc[%d] relevant=%s score=%.3f", i, relevant, score)

    precision = n_relevant / len(retrieved_docs)
    log.info("context_precision=%.3f  (relevant=%d / total=%d)", precision, n_relevant, len(retrieved_docs))
    return precision


def context_recall(
    query: str,
    retrieved_docs: Sequence[str],
    reference_answer: str,
) -> float:
    """Coverage recall — fraction of ground-truth facts found in the retrieved context.

    Uses an LLM judge to compare the concatenated retrieved context against
    the reference answer and estimate how much of the reference is covered.

    Parameters
    ----------
    query:
        The user's search query.
    retrieved_docs:
        Retrieved document texts.
    reference_answer:
        Ground-truth answer or the list of facts that should be retrievable.

    Returns
    -------
    float in [0, 1]
    """
    if not retrieved_docs:
        return 0.0
    if not reference_answer:
        return 1.0

    context = "\n\n---\n\n".join(doc[:1500] for doc in retrieved_docs)
    judge   = dspy.ChainOfThought(RecallCoverageJudge)

    try:
        pred  = judge(
            query=query,
            retrieved_context=context,
            reference_answer=reference_answer,
        )
        score = max(0.0, min(1.0, float(pred.coverage_score)))
        log.info("context_recall=%.3f  missing=%r", score, pred.missing_facts)
        return score
    except Exception as exc:
        log.warning("RecallCoverageJudge failed: %s", exc)
        return 0.0


def hit_rate(relevance_labels: Sequence[int], k: int | None = None) -> float:
    """Hit Rate@k — 1 if any relevant document appears in the top-k results.

    Parameters
    ----------
    relevance_labels:
        Binary labels in retrieval rank order (1=relevant, 0=not relevant).
    k:
        Cutoff.  Defaults to the full list length.

    Returns
    -------
    float: 1.0 or 0.0
    """
    labels = list(relevance_labels)[:k] if k else list(relevance_labels)
    result = 1.0 if any(r > 0 for r in labels) else 0.0
    log.debug("hit_rate@%s=%.1f", k or len(relevance_labels), result)
    return result


def mrr(relevance_labels: Sequence[int]) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first relevant document.

    Parameters
    ----------
    relevance_labels:
        Binary labels in retrieval rank order (1=relevant, 0=not relevant).

    Returns
    -------
    float in [0, 1]  (0 when no relevant document was retrieved)
    """
    for rank, label in enumerate(relevance_labels, start=1):
        if label > 0:
            score = 1.0 / rank
            log.debug("MRR=%.4f  (first relevant at rank %d)", score, rank)
            return score
    log.debug("MRR=0.0  (no relevant document found)")
    return 0.0


def ndcg(relevance_labels: Sequence[int], k: int | None = None) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Supports graded relevance (e.g. 0 / 1 / 2) as well as binary labels.

    Parameters
    ----------
    relevance_labels:
        Relevance scores in retrieval rank order (higher = more relevant).
    k:
        Cutoff.  Defaults to the full list length.

    Returns
    -------
    float in [0, 1]
    """
    labels = list(relevance_labels)
    if k is not None:
        labels = labels[:k]

    if not labels or max(labels) == 0:
        return 0.0

    def _dcg(rels: list[int]) -> float:
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))

    dcg  = _dcg(labels)
    idcg = _dcg(sorted(labels, reverse=True))
    score = dcg / idcg if idcg > 0 else 0.0
    log.debug("NDCG@%s=%.4f  DCG=%.4f  IDCG=%.4f", k or len(labels), score, dcg, idcg)
    return score


# ── Convenience: compute all retrieval metrics at once ────────────────────────

def evaluate_retrieval(
    query: str,
    retrieved_docs: Sequence[str],
    relevance_labels: Sequence[int] | None = None,
    reference_answer: str | None = None,
    k: int | None = None,
) -> RetrievalMetrics:
    """Compute all retrieval metrics in one call.

    Parameters
    ----------
    query:
        User research query.
    retrieved_docs:
        Ordered list of retrieved document texts.
    relevance_labels:
        Optional binary/graded labels aligned with ``retrieved_docs``.
        When omitted, LLM-based judges fill in context_precision and
        context_recall; hit_rate / MRR / NDCG are skipped (set to 0.0).
    reference_answer:
        Ground-truth text for context_recall computation.  Required when
        ``relevance_labels`` is None.
    k:
        Rank cutoff for hit_rate and NDCG.

    Returns
    -------
    RetrievalMetrics dataclass
    """
    log.info(
        "evaluate_retrieval  query=%r  n_docs=%d  has_labels=%s  k=%s",
        query[:80], len(retrieved_docs), relevance_labels is not None, k,
    )

    cp = context_precision(query, retrieved_docs, relevance_labels)

    cr = 0.0
    if reference_answer:
        cr = context_recall(query, retrieved_docs, reference_answer)
    elif relevance_labels is not None:
        # Approximate recall as ratio of retrieved relevant to total relevant known
        total_relevant = sum(relevance_labels)
        cr = float(sum(1 for l in relevance_labels if l > 0)) / total_relevant if total_relevant else 0.0

    hr = mval = ng = 0.0
    if relevance_labels is not None:
        hr   = hit_rate(relevance_labels, k)
        mval = mrr(relevance_labels)
        ng   = ndcg(relevance_labels, k)

    result = RetrievalMetrics(
        context_precision=cp,
        context_recall=cr,
        hit_rate=hr,
        mrr=mval,
        ndcg=ng,
        details={"n_docs": len(retrieved_docs), "k": k},
    )
    log.info("retrieval result: %s", result)
    return result
