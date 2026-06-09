"""SciTracer — Scientific Research Assistant.

Pipeline (mirrors test_plan_verifier.py)
-----------------------------------------
1. Researcher   : builds a research plan via RAG + trajectory context
2. Formalizer   : extracts a Kripke structure + Z3 formalization
3. Verifier     : runs Z3 (+ NuSMV) verification with iterative refinement
4. Executor     : generates one .py file per plan step, builds run.py,
                  and executes the orchestrator
                  → ALL artifacts go into  sandbox/<run_id>/

Usage
-----
    python main.py "Please solve this QM problem …"
    python main.py "…" --sandbox my_runs --max-retries 5
"""
from __future__ import annotations

import argparse
import configparser
import logging
from pathlib import Path
import pdb
import dspy

from core.lm_setup import _setup_lm
from core.log_setup import setup_logging

from storage.vector_db import TrajectoryStore
from storage.knowledge_graph import KnowledgeGraph

from agents.knowledge_constructor import KnowledgeConsructor
from agents.rag import RAG
from agents.researcher import Researcher
from agents.formalizer import Formalizer
from agents.verifier import Verifier
from agents.executor import Executor

_CONFIG_PATH = Path(__file__).parent / "config.ini"


def _load_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if _CONFIG_PATH.exists():
        cfg.read(_CONFIG_PATH)
    return cfg


def main() -> None:
    cfg = _load_config()

    def _str(section: str, key: str, fallback: str = "") -> str:
        return cfg.get(section, key, fallback=fallback).strip()

    def _bool(section: str, key: str, fallback: bool = False) -> bool:
        return cfg.getboolean(section, key, fallback=fallback)

    parser = argparse.ArgumentParser(
        description="SciTracer — AI-powered scientific research assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Defaults are read from {_CONFIG_PATH} (CLI flags always override).",
    )
    parser.add_argument(
        "--question",
        help="Research question or problem to investigate",
    )
    parser.add_argument(
        "--sandbox",
        type=str,
        default=_str("executor", "sandbox_dir", "sandbox"),
        metavar="DIR",
        help="Root sandbox directory for run artifacts (default: sandbox/)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=_str("literature", "local_path") or None,
        metavar="FOLDER",
        help="Local folder to read papers from instead of searching online",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(_str("verification", "max_retries") or "3"),
        metavar="N",
        help="Verification retry budget (default: 3)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_CONFIG_PATH),
        metavar="FILE",
        help="Path to config.ini (default: %(default)s)",
    )

    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────────
    log_dir = _str("logging", "log_dir", "logs")
    setup_logging(log_dir=log_dir)
    log = logging.getLogger("scitracer")
    log.info("SciTracer started  question=%r", args.question)

    # ── LM ────────────────────────────────────────────────────────────────────
    lm = _setup_lm()
    dspy.configure(lm=lm)
    print(f"LM: {lm.model}")
    log.info("LM configured: %s", lm.model)

    # ── Per-run sandbox folder ────────────────────────────────────────────────
    _words  = args.question.split()
    run_id  = "".join(w[0].lower() for w in _words[:10] if w)
    sandbox_root = Path(args.sandbox)
    run_dir      = sandbox_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "question.txt").write_text(args.question, encoding="utf-8")
    print(f"Run artifacts → {run_dir}")
    log.info("Run dir: %s", run_dir)

    # Add a per-run file handler so all log output is also captured in run_dir
    run_log_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    run_log_handler.setLevel(logging.DEBUG)
    run_log_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    )
    logging.getLogger().addHandler(run_log_handler)

    # ── [0/4] Knowledge construction (literature → Hoare triplets → VectorDB) ──
    print("\n[0/4] Knowledge construction…")
    log.info("[0/4] Knowledge construction")

    vector_db     = TrajectoryStore()
    graph_storage = KnowledgeGraph()

    kc = KnowledgeConsructor(
        question = args.question,
        storage  = vector_db,
        kg       = graph_storage,
    )
    chunks = kc.collect_literature(args.question, local_path=args.path)
    log.info("[0/4] Knowledge construction done  chunks=%d", len(chunks))
    print(f"  Literature indexed  (chunks={len(chunks)})")

    # ── [1/4] Research plan ───────────────────────────────────────────────────
    print("\n[1/4] Research plan synthesis…")
    log.info("[1/4] Research plan synthesis")
    pdb.set_trace()
    rag        = RAG(vector_db, graph_storage)
    researcher = Researcher(rag=rag)

    plan = researcher.forward(args.question)

    n_steps = (
        len(plan.steps.instructions)
        if hasattr(plan.steps, "instructions")
        else len(plan.steps) if isinstance(plan.steps, list) else 0
    )
    log.info("[1/4] Plan synthesised  steps=%d", n_steps)
    print(f"  Plan ready  (steps={n_steps})")

    # ── [2/4] Formalization ───────────────────────────────────────────────────
    print("\n[2/4] Formalization (Kripke + Z3)…")
    log.info("[2/4] Formalization")

    formalizer     = Formalizer()
    formalized_z3  = formalizer.forward(plan)   # returns z3_code string

    z3_path = run_dir / "z3_formalization.py"
    z3_path.write_text(str(formalized_z3), encoding="utf-8")
    log.info("[2/4] Formalization done  z3_path=%s", z3_path)
    print(f"  Z3 formalization → {z3_path}")

    # ── [3/4] Verification ────────────────────────────────────────────────────
    print("\n[3/4] Verification (Z3 + NuSMV with refinement)…")
    log.info("[3/4] Verification  max_retries=%d", args.max_retries)

    verifier            = Verifier()
    verification_result = verifier.forward(plan, max_retries=args.max_retries)
    refined_plan        = verification_result.refined_plan or plan

    # Persist verification artefacts
    (run_dir / "verification_trace.txt").write_text(
        verification_result.trace, encoding="utf-8"
    )
    log.info(
        "[3/4] Verification done  success=%s  attempts=%d",
        verification_result.success,
        len(verification_result.attempts),
    )
    print(
        f"  Verification {'OK' if verification_result.success else 'FAILED'}"
        f"  (attempts={len(verification_result.attempts)})"
    )

    # ── [4/4] Execution ───────────────────────────────────────────────────────
    print("\n[4/4] Execution (code generation + orchestrator run)…")
    log.info("[4/4] Execution  run_id=%s  sandbox_root=%s", run_id, sandbox_root)

    # Executor places every step file and run.py inside sandbox_root/<run_id>/
    executor   = Executor(sandbox_dir=str(sandbox_root))
    exec_result = executor.forward(
        instructions  = refined_plan,
        research_plan = refined_plan,
        run_id        = run_id,
    )

    log.info(
        "[4/4] Execution done  success=%s  repairs=%d  run_dir=%s",
        exec_result.success,
        exec_result.repair_attempts,
        run_dir,
    )
    print(
        f"\n  Execution {'OK' if exec_result.success else 'FAILED'}"
        f"  (repairs={exec_result.repair_attempts})"
    )
    print(f"  All artifacts → {run_dir}")

    if exec_result.stdout:
        print("\n--- stdout ---")
        print(exec_result.stdout)

    log.info("SciTracer complete  run_dir=%s", run_dir)


if __name__ == "__main__":
    main()
