"""Example evaluation script using the SciTracer metrics module.

Run
---
    python evaluate/eval.py

The script runs a small hard-coded devset through RAGMetricsEvaluator and
prints a summary table.  Replace the devset with your own examples or load
from a JSONL file.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from core.lm_setup import _setup_lm
from metrics import RAGMetricsEvaluator

# ── Configure LM ──────────────────────────────────────────────────────────────
lm = _setup_lm()
dspy.configure(lm=lm)

# ── Minimal devset ─────────────────────────────────────────────────────────────
DEVSET = [
    {
        "question": "How does the attention mechanism work in transformers?",
        "retrieved_docs": [
            "The attention mechanism computes a weighted sum of values, where the weight "
            "assigned to each value is determined by a compatibility function of the query "
            "with the corresponding key.  In transformers, multi-head attention applies this "
            "operation h times in parallel with different learned projections.",
            "Convolutional neural networks apply fixed-size filters across the input to "
            "detect local patterns, independent of their position.",
            "The scaled dot-product attention formula is: Attention(Q,K,V) = softmax(QK^T/√d_k)V.",
        ],
        "generated_answer": (
            "The attention mechanism in transformers allows each token to attend to all "
            "other tokens in the sequence.  It computes query, key, and value vectors and "
            "uses scaled dot-product attention: softmax(QK^T/√d_k)V.  "
            "Multi-head attention runs this in parallel with different projections."
        ),
        "reference_answer": (
            "Attention computes a weighted sum of value vectors.  "
            "Weights come from softmax(QK^T / sqrt(d_k)).  "
            "Multi-head attention applies this h times with different projections."
        ),
        "relevance_labels": [1, 0, 1],
    },
    {
        "question": "What is the main contribution of the BERT paper?",
        "retrieved_docs": [
            "BERT (Bidirectional Encoder Representations from Transformers) introduced "
            "deep bidirectional pre-training for language understanding.  It uses masked "
            "language modelling and next-sentence prediction as pre-training objectives.",
            "GPT uses a left-to-right language model objective for pre-training, seeing "
            "only left context in the self-attention layers.",
        ],
        "generated_answer": (
            "BERT's main contribution is deep bidirectional pre-training using masked "
            "language modelling, allowing the model to condition on both left and right "
            "context simultaneously.  This significantly improved NLP benchmarks."
        ),
        "reference_answer": (
            "BERT introduced bidirectional pre-training via masked language modelling "
            "and next-sentence prediction."
        ),
        "relevance_labels": [1, 0],
    },
]

# ── Run evaluation ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from core.log_setup import setup_logging
    setup_logging(log_dir="logs", console_level=30)  # 30 = WARNING

    evaluator = RAGMetricsEvaluator(k=3, log_dir="logs/metrics")
    batch     = evaluator.evaluate_batch(DEVSET, save_report=True)
    print(batch.summary())
