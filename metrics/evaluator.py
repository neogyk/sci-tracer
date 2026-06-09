"""RAGMetricsEvaluator — unified evaluation entry point.

Runs all retrieval and generation metrics for a single RAG turn or a batch
of turns, logs results, and returns a structured EvaluationReport.

Typical usage
-------------
::

    from metrics import RAGMetricsEvaluator

    evaluator = RAGMetricsEvaluator()
    report = evaluator.evaluate(
        question        = "How does attention work in transformers?",
        retrieved_docs  = [...],      # list[str] from your RAG pipeline
        generated_answer= "...",      # str from your LLM
        reference_answer= "...",      # optional ground-truth
        relevance_labels= [1, 0, 1],  # optional binary labels
    )
    print(report.summary())
    report.save("logs/eval_report.json")
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .retrieval import RetrievalMetrics, evaluate_retrieval
from .generation import GenerationMetrics, evaluate_generation

log = logging.getLogger("scitracer.metrics.evaluator")


# ── Report dataclass ───────────────────────────────────────────────────────────

@dataclass
class EvaluationReport:
    """Complete evaluation report for a single RAG turn."""

    question:        str
    generated_answer: str
    retrieval:       RetrievalMetrics
    generation:      GenerationMetrics
    timestamp:       str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    metadata:        dict = field(default_factory=dict)

    # ── Aggregate score ────────────────────────────────────────────────────────

    @property
    def aggregate_score(self) -> float:
        """Weighted average of all five metrics.

        Weights
        -------
        context_precision  0.20
        context_recall     0.20
        faithfulness       0.25
        answer_relevance   0.25
        ndcg               0.10
        """
        weights = {
            "context_precision":  0.20,
            "context_recall":     0.20,
            "faithfulness":       0.25,
            "answer_relevance":   0.25,
            "ndcg":               0.10,
        }
        scores = {
            "context_precision":  self.retrieval.context_precision,
            "context_recall":     self.retrieval.context_recall,
            "faithfulness":       self.generation.faithfulness_score,
            "answer_relevance":   self.generation.answer_relevance_score,
            "ndcg":               self.retrieval.ndcg,
        }
        return sum(weights[k] * scores[k] for k in weights)

    # ── Human-readable summary ─────────────────────────────────────────────────

    def summary(self) -> str:
        sep = "─" * 60
        lines = [
            "",
            sep,
            f"  EVALUATION REPORT  {self.timestamp}",
            sep,
            f"  Question : {self.question[:120]}",
            "",
            "  ── Retrieval ─────────────────────────────────────────────",
            f"  Context Precision : {self.retrieval.context_precision:.3f}",
            f"  Context Recall    : {self.retrieval.context_recall:.3f}",
            f"  Hit Rate          : {self.retrieval.hit_rate:.3f}",
            f"  MRR               : {self.retrieval.mrr:.3f}",
            f"  NDCG              : {self.retrieval.ndcg:.3f}",
            "",
            "  ── Generation ────────────────────────────────────────────",
            f"  Faithfulness      : {self.generation.faithfulness_score:.3f}",
            f"    supported claims : {self.generation.faithfulness.n_supported}/{self.generation.faithfulness.n_claims}",
        ]
        if self.generation.faithfulness.unsupported:
            for claim in self.generation.faithfulness.unsupported[:3]:
                lines.append(f"    ✗ {claim[:100]}")
        lines += [
            f"  Answer Relevance  : {self.generation.answer_relevance_score:.3f}",
            f"    {self.generation.answer_relevance.explanation}",
            "",
            f"  ── Aggregate Score : {self.aggregate_score:.3f} ──────────────────────",
            sep,
        ]
        return "\n".join(lines)

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "timestamp":          self.timestamp,
            "question":           self.question,
            "generated_answer":   self.generated_answer,
            "aggregate_score":    self.aggregate_score,
            "retrieval": {
                "context_precision": self.retrieval.context_precision,
                "context_recall":    self.retrieval.context_recall,
                "hit_rate":          self.retrieval.hit_rate,
                "mrr":               self.retrieval.mrr,
                "ndcg":              self.retrieval.ndcg,
            },
            "generation": {
                "faithfulness":       self.generation.faithfulness_score,
                "n_claims":           self.generation.faithfulness.n_claims,
                "n_supported":        self.generation.faithfulness.n_supported,
                "unsupported_claims": self.generation.faithfulness.unsupported,
                "answer_relevance":   self.generation.answer_relevance_score,
                "relevance_explanation": self.generation.answer_relevance.explanation,
            },
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("EvaluationReport saved → %s", path)
        return path


# ── Batch result ───────────────────────────────────────────────────────────────

@dataclass
class BatchEvaluationReport:
    """Aggregated metrics over a list of EvaluationReports."""

    reports: list[EvaluationReport] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.reports)

    def _mean(self, attr_chain: str) -> float:
        if not self.reports:
            return 0.0
        vals = []
        for r in self.reports:
            obj = r
            for attr in attr_chain.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                vals.append(float(obj))
        return sum(vals) / len(vals) if vals else 0.0

    def summary(self) -> str:
        sep = "═" * 60
        return "\n".join([
            "",
            sep,
            f"  BATCH EVALUATION  n={self.n}",
            sep,
            "  ── Retrieval (mean) ──────────────────────────────────────",
            f"  Context Precision : {self._mean('retrieval.context_precision'):.3f}",
            f"  Context Recall    : {self._mean('retrieval.context_recall'):.3f}",
            f"  Hit Rate          : {self._mean('retrieval.hit_rate'):.3f}",
            f"  MRR               : {self._mean('retrieval.mrr'):.3f}",
            f"  NDCG              : {self._mean('retrieval.ndcg'):.3f}",
            "",
            "  ── Generation (mean) ─────────────────────────────────────",
            f"  Faithfulness      : {self._mean('generation.faithfulness.score'):.3f}",
            f"  Answer Relevance  : {self._mean('generation.answer_relevance.score'):.3f}",
            "",
            f"  ── Mean Aggregate  : {self._mean('aggregate_score'):.3f} ─────────────────────",
            sep,
        ])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "n": self.n,
            "mean_aggregate_score": self._mean("aggregate_score"),
            "mean_retrieval": {
                "context_precision": self._mean("retrieval.context_precision"),
                "context_recall":    self._mean("retrieval.context_recall"),
                "hit_rate":          self._mean("retrieval.hit_rate"),
                "mrr":               self._mean("retrieval.mrr"),
                "ndcg":              self._mean("retrieval.ndcg"),
            },
            "mean_generation": {
                "faithfulness":      self._mean("generation.faithfulness.score"),
                "answer_relevance":  self._mean("generation.answer_relevance.score"),
            },
            "reports": [r.to_dict() for r in self.reports],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("BatchEvaluationReport saved → %s", path)
        return path


# ── Main evaluator ─────────────────────────────────────────────────────────────

class RAGMetricsEvaluator:
    """Evaluate a single RAG turn or a batch of turns.

    Parameters
    ----------
    k:
        Default rank cutoff for hit_rate and NDCG.
    log_dir:
        When set, each ``evaluate()`` call saves a JSON report here.
    """

    def __init__(self, k: int = 5, log_dir: str | Path | None = "logs/metrics") -> None:
        self.k = k
        self.log_dir = Path(log_dir) if log_dir else None

    def evaluate(
        self,
        question:         str,
        retrieved_docs:   Sequence[str],
        generated_answer: str,
        reference_answer: str | None = None,
        relevance_labels: Sequence[int] | None = None,
        metadata:         dict | None = None,
        save_report:      bool = True,
    ) -> EvaluationReport:
        """Run all metrics for one RAG turn.

        Parameters
        ----------
        question:
            User's research query.
        retrieved_docs:
            Ordered list of texts returned by the retriever.
        generated_answer:
            Final LLM answer to evaluate.
        reference_answer:
            Optional ground-truth for context recall.
        relevance_labels:
            Optional binary/graded relevance labels aligned with
            ``retrieved_docs`` (enables hit_rate, MRR, NDCG).
        metadata:
            Arbitrary key/value pairs stored in the report.
        save_report:
            Write the report to ``log_dir`` if one was configured.

        Returns
        -------
        EvaluationReport
        """
        log.info(
            "RAGMetricsEvaluator.evaluate  question=%r  n_docs=%d",
            question[:80], len(retrieved_docs),
        )

        retrieval_metrics = evaluate_retrieval(
            query=question,
            retrieved_docs=retrieved_docs,
            relevance_labels=relevance_labels,
            reference_answer=reference_answer,
            k=self.k,
        )

        generation_metrics = evaluate_generation(
            question=question,
            answer=generated_answer,
            context=retrieved_docs,
        )

        report = EvaluationReport(
            question=question,
            generated_answer=generated_answer,
            retrieval=retrieval_metrics,
            generation=generation_metrics,
            metadata=metadata or {},
        )

        print(report.summary())

        if save_report and self.log_dir:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = "".join(c if c.isalnum() else "_" for c in question[:40]).strip("_")
            self.log_dir.mkdir(parents=True, exist_ok=True)
            report.save(self.log_dir / f"eval_{ts}_{slug}.json")

        return report

    def evaluate_batch(
        self,
        items: Sequence[dict],
        save_report: bool = True,
    ) -> BatchEvaluationReport:
        """Evaluate a list of RAG turns.

        Each item in ``items`` is a dict with the same keys as
        :meth:`evaluate` (``question``, ``retrieved_docs``,
        ``generated_answer``, and optionally ``reference_answer``,
        ``relevance_labels``, ``metadata``).

        Returns
        -------
        BatchEvaluationReport
        """
        log.info("RAGMetricsEvaluator.evaluate_batch  n=%d", len(items))
        reports = []
        for i, item in enumerate(items):
            log.info("Batch item %d/%d", i + 1, len(items))
            report = self.evaluate(**item, save_report=False)
            reports.append(report)

        batch = BatchEvaluationReport(reports=reports)
        print(batch.summary())

        if save_report and self.log_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir.mkdir(parents=True, exist_ok=True)
            batch.save(self.log_dir / f"eval_batch_{ts}.json")

        return batch
