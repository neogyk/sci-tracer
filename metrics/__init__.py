"""SciTracer metrics — RAG and agent evaluation.

Retrieval metrics
-----------------
context_precision   fraction of retrieved docs that are relevant to the query
context_recall      fraction of ground-truth facts covered by the retrieved context
hit_rate            was at least one relevant doc in the top-k results?
mrr                 Mean Reciprocal Rank of the first relevant result
ndcg                Normalized Discounted Cumulative Gain at k

Generation metrics
------------------
faithfulness        every claim in the answer is grounded in the retrieved context
answer_relevance    the answer directly addresses the user's question

Convenience
-----------
RAGMetricsEvaluator     orchestrates all metrics for a turn or a batch
EvaluationReport        structured result with .summary() and .save()
BatchEvaluationReport   aggregated results over multiple turns
"""

from .retrieval import (
    RetrievalMetrics,
    context_precision,
    context_recall,
    hit_rate,
    mrr,
    ndcg,
    evaluate_retrieval,
)
from .generation import (
    FaithfulnessResult,
    AnswerRelevanceResult,
    GenerationMetrics,
    faithfulness,
    answer_relevance,
    evaluate_generation,
)
from .evaluator import (
    EvaluationReport,
    BatchEvaluationReport,
    RAGMetricsEvaluator,
)

__all__ = [
    # retrieval
    "RetrievalMetrics",
    "context_precision",
    "context_recall",
    "hit_rate",
    "mrr",
    "ndcg",
    "evaluate_retrieval",
    # generation
    "FaithfulnessResult",
    "AnswerRelevanceResult",
    "GenerationMetrics",
    "faithfulness",
    "answer_relevance",
    "evaluate_generation",
    # evaluator
    "EvaluationReport",
    "BatchEvaluationReport",
    "RAGMetricsEvaluator",
]
