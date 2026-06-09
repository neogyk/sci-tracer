"""Run a single ColliderBench task through the SciTracer pipeline.

Usage
-----
    python benchmark/collider/run_collider.py

The paper (CMS-SUS-16-034, arXiv:1709.08908) is downloaded from ArXiv into a
local papers/ directory, then ingested into the RAG via _collect_from_local.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dspy
from agents.knowledge_constructor import KnowledgeConsructor
from agents.researcher import Researcher
from agents.formalizer import Formalizer
from agents.verifier import Verifier
from agents.executor import Executor
from agents.rag import RAG
from agents.experience import ExperienceStore, with_experience, default_store
from agents.reporter import Reporter
from storage.vector_db import TrajectoryStore
from storage.knowledge_graph import KnowledgeGraph
from core.lm_setup import _setup_lm

PAPER_URL  = "https://arxiv.org/pdf/1709.08908"
PAPERS_DIR = Path(__file__).parent / "papers"

AGENT_CONTEXT_DIR = Path(__file__).parent / "agent_context"

TASK_PROMPT = (
    "You are a LHC experimentalist with expertise in Standard Model and BSM search strategies, "
    "event generation tools, event selection design, and statistical interpretation of collider data. "
    "Read these in order: "
    "1. agent_context/TASK.md — the benchmark's task for this run. "
    "2. agent_context/AGENTS.md — your role and how to work. "
    "3. agent_context/TOOLS.md — CLI tool reference. "
    "Everything you need is in this directory.\n\n"
    + (AGENT_CONTEXT_DIR / "TASK.md").read_text()
)


if __name__ == "__main__":
    lm = _setup_lm()
    dspy.configure(lm=lm)

    vector_db     = TrajectoryStore()
    graph_storage = KnowledgeGraph()

    # ── Experience store — shared across all pipeline steps ───────────────────
    # default_store is already wired into Researcher.forward via @with_experience.
    # Use it directly to inspect or print the log after the run.
    store = default_store

    # Toggle the decorator on/off at runtime (True = on, False = pass-through):
    #   Researcher.forward.experience_enabled = False

    # ── Download paper, then ingest via _collect_from_local ───────────────────
    kc = KnowledgeConsructor(question=TASK_PROMPT, storage=vector_db, kg=graph_storage)
    try:
        kc.download_paper(PAPER_URL, str(PAPERS_DIR))
        kc._collect_from_local(TASK_PROMPT, str(PAPERS_DIR))
        print("[collider] Paper ingested into RAG.")
    except Exception as exc:
        print(f"[collider] WARNING: RAG ingestion failed ({exc}); continuing without it.")

    # ── Pipeline ────────────────────────────────────────────────────────────────
    rag        = RAG(vector_db, graph_storage)
    researcher = Researcher(rag=rag)
    formalizer = Formalizer()
    verifier   = Verifier()
    executor   = Executor()

    # Researcher.forward already carries @with_experience — no extra call needed.
    print("\n[collider] Running Researcher …")
    plan = researcher.forward(TASK_PROMPT)
    print(plan)

    # Formalizer/Verifier/Executor take structured objects, not string prompts.
    # Decorate their forward calls inline with a one-off wrapper:
    @with_experience(store=store, enable=True)
    def run_formalizer(self, prompt: str):
        return formalizer.forward(plan)

    @with_experience(store=store, enable=True)
    def run_verifier(self, prompt: str):
        return verifier.forward(formalized_plan)

    print("\n[collider] Running Formalizer …")
    formalized_plan = run_formalizer(None, str(plan)[:200])
    print(formalized_plan)

    print("\n[collider] Running Verifier …")
    verification_results = run_verifier(None, str(formalized_plan)[:200])
    print(verification_results)

    print("\n[collider] Running Executor …")
    refined_plan = verification_results.refined_plan or plan
    code_result  = executor.execute(
        task_summary  = TASK_PROMPT[:200],
        instructions  = refined_plan,
        research_plan = plan,
    )
    print(code_result)

    # ── Generate REPORT.md ───────────────────────────────────────────────────
    reporter = Reporter(output_dir=Path(__file__).parent)
    reporter.forward(
        question             = TASK_PROMPT,
        research_plan        = plan,
        execution_trace      = code_result.output,
        verification_summary = str(verification_results),
        step_logs            = code_result.step_logs,
    )

    # ── Print experience log ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EXPERIENCE LOG")
    print("=" * 60)
    for i, entry in enumerate(store.recent(n=20), 1):
        status = "OK" if entry.value.startswith("SUCCESS") else "FAIL"
        print(f"  [{i:02d}] [{status}] {entry.name}")
        print(f"        task : {entry.task[:100]}")
        print(f"        value: {entry.value[:120]}")
    print(f"\n  Total recorded: {len(store)}")
