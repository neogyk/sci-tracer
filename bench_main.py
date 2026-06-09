"""SciTracer — FrontierScience Benchmark Runner.

Dataset
-------
  openai/frontierscience  (160 frontier science problems)
  Fields: problem (str), answer (str), subject (str), task_group_id (str)
  Subjects: physics, chemistry, biology

Pipeline modes
--------------
Without --logic_verification  (fast baseline)
    Steps 1-3:  literature → trajectories → plan
    Execution:  direct single-shot answer from the plan (no formalization)

With --logic_verification  (full pipeline)
    Steps 1-5:  literature → trajectories → plan → formalize → verify
    Execution:  answer produced after Kripke/Z3 verification + refinement

Usage
-----
    # Quick sanity check — 5 problems, no RAG, no verification
    python bench_main.py --n 5 --no-rag

    # Full benchmark on physics with formal verification
    python bench_main.py --subject physics --logic_verification

    # Resume a previous partial run
    python bench_main.py --resume results/bench_20240101.json

    # All 160 problems, baseline mode, parallel-friendly
    python bench_main.py --no-rag --decompose none --output results/baseline.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy

from core.lm_setup import _setup_lm
from core.plan_types import _plan_to_str
from core.print_helpers import _parse_structure, _print_hoare, _print_kripke, _print_verification
from core.tracer import SciTracer


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_frontierscience(split: str = "test") -> list[dict[str, str]]:
    """Load openai/frontierscience via HuggingFace datasets.

    Falls back to ``data/frontierscience.json`` when the hub is unreachable.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
        # trust_remote_code removed — not supported by this dataset anymore
        ds = load_dataset("openai/frontierscience", split=split)
        return [
            {
                "problem":       row["problem"],
                "answer":        row["answer"],
                "subject":       row["subject"],
                "task_group_id": row["task_group_id"],
            }
            for row in ds
        ]
    except Exception as exc:
        print(f"  [dataset] HuggingFace load failed ({exc}) — trying local fallback.")
        local = Path("data/frontierscience.json")
        if local.exists():
            with open(local, encoding="utf-8") as f:
                return json.load(f)
        raise RuntimeError(
            "Could not load from HuggingFace and no local cache at "
            "data/frontierscience.json.\n"
            "Run once to cache: python -c \""
            "from datasets import load_dataset; import json; "
            "d=load_dataset('openai/frontierscience',split='test'); "
            "json.dump(list(d), open('data/frontierscience.json','w'))\""
        ) from exc


# ── Text normalisation ────────────────────────────────────────────────────────

def _strip_latex(text: str) -> str:
    """Remove LaTeX delimiters and normalise common math macros to plain text.

    Dataset answers arrive as LaTeX strings like ``\\(U = CT'\\)`` or
    ``\\[ \\frac{\\Delta T}{\\Delta h} = ... \\]``.  Stripping the wrappers
    lets exact-match and numeric-match operate on the actual value.
    """
    # Remove inline / display math delimiters
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)
    text = re.sub(r"\$\$|\$", "", text)
    # Backtick fences (dataset quirk)
    text = text.replace("`", "")
    # Common macro → symbol
    text = (
        text.replace(r"\times", "×")
            .replace(r"\cdot", "·")
            .replace(r"\approx", "≈")
            .replace(r"\infty", "∞")
            .replace(r"\prime", "'")
    )
    # \cmd{arg} → arg
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    # remaining \cmd → space
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    # stray braces
    text = re.sub(r"[{}]", "", text)
    # collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


def _normalise(text: str) -> str:
    """Strip LaTeX, collapse whitespace, lowercase."""
    return _strip_latex(text).lower()


# ── Answer scoring ────────────────────────────────────────────────────────────

def exact_match(predicted: str, reference: str) -> bool:
    """Normalised string equality after LaTeX stripping."""
    return _normalise(predicted) == _normalise(reference)


def _extract_number(text: str) -> float | None:
    """Extract the first numeric value from *text*, handling:
    - plain integers / floats  (``2.31``, ``-3``)
    - scientific notation      (``2.31e6``, ``2.31E+6``)
    - cross-product notation   (``2.31 × 10^6``, ``2.31 x 10^6``)
    - LaTeX scientific         (``2.31 \\times 10^{6}``)

    Returns None when no number is found.
    """
    text = _strip_latex(text)

    # ── Try "mantissa × 10^exp" first (highest specificity) ──────────────────
    cross = re.compile(
        r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"  # mantissa
        r"\s*[×x\*·]\s*10\s*\^?\s*\{?\s*([-+]?\d+)\}?", # × 10^N
    )
    m = cross.search(text)
    if m:
        try:
            return float(m.group(1)) * (10 ** int(m.group(2)))
        except (ValueError, OverflowError):
            pass

    # ── Fall back to plain / e-notation number ────────────────────────────────
    plain = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
    m = plain.search(text)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass

    return None


def numeric_match(predicted: str, reference: str, rtol: float = 0.05) -> bool:
    """True when both strings contain the same leading numeric value
    within relative tolerance *rtol* (default 5 %).

    Returns False when either string contains no parseable number.
    """
    a = _extract_number(predicted)
    b = _extract_number(reference)
    if a is None or b is None:
        return False
    denom = max(abs(b), 1e-30)
    return abs(a - b) / denom <= rtol


class LLMJudge:
    """Ask the LLM whether a predicted answer is equivalent to the reference.

    Caches verdicts so the same (predicted, reference) pair is never re-scored.
    """

    _PROMPT = (
        "You are a science grader. Decide whether the PREDICTED answer is "
        "mathematically or scientifically equivalent to the REFERENCE answer. "
        "Reply with exactly one word: YES or NO.\n\n"
        "REFERENCE: {reference}\n"
        "PREDICTED: {predicted}\n\n"
        "Are they equivalent?"
    )

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], bool] = {}

    def score(self, predicted: str, reference: str) -> bool:
        key = (_normalise(predicted)[:300], _normalise(reference)[:300])
        if key in self._cache:
            return self._cache[key]
        try:
            prompt = self._PROMPT.format(
                reference=_strip_latex(reference).strip(),
                predicted=predicted.strip(),
            )
            resp = dspy.settings.lm(messages=[{"role": "user", "content": prompt}])
            verdict = (resp[0] if isinstance(resp, list) else str(resp)).strip().upper()
            result = verdict.startswith("YES")
        except Exception as exc:
            print(f"  [judge] scoring failed: {exc}")
            result = False
        self._cache[key] = result
        return result


# ── Plan extraction helper ────────────────────────────────────────────────────

def _get_plan_str(plan_obj: Any, fallback: str) -> str:
    """Safely extract a plan string from make_research_plan's return value.

    make_research_plan returns either:
      - a ResearchPlan/dspy.Prediction object with a .steps list, or
      - the raw prompt string when no trajectories were available.
    """
    if isinstance(plan_obj, str):
        return plan_obj or fallback
    steps = getattr(plan_obj, "steps", None)
    if steps:
        return _plan_to_str(steps)
    return fallback


# ── Single-problem evaluation ─────────────────────────────────────────────────

def run_one(
    tracer: SciTracer,
    judge: LLMJudge,
    row: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate a single benchmark row; return a result dict."""
    problem       = row["problem"]
    reference     = row["answer"]
    subject       = row["subject"]
    task_group_id = row["task_group_id"]

    t0        = time.monotonic()
    predicted = ""
    error_msg = ""
    pipeline  = "logic_verification" if args.logic_verification else "direct"
    extra: dict[str, Any] = {}

    # Reset tracer state between problems so the KG / ontology store from
    # the previous row doesn't bleed into the current one.
    tracer.reset()

    try:
        if args.logic_verification:
            predicted, extra = _run_with_verification(tracer, problem, args)
        else:
            predicted = _run_direct(tracer, problem, args)
    except Exception as exc:
        error_msg = str(exc)
        print(f"  [ERROR] {task_group_id[:8]}: {exc}")

    elapsed = time.monotonic() - t0

    em  = exact_match(predicted, reference)
    nm  = numeric_match(predicted, reference)
    llm = judge.score(predicted, reference) if predicted else False

    return {
        "task_group_id":  task_group_id,
        "subject":        subject,
        "problem":        problem[:300],
        "reference":      reference,
        "predicted":      predicted,
        "pipeline":       pipeline,
        "exact_match":    em,
        "numeric_match":  nm,
        "llm_equiv":      llm,
        "elapsed_s":      round(elapsed, 2),
        "error":          error_msg,
        **extra,
    }


def _run_direct(tracer: SciTracer, problem: str, args: argparse.Namespace) -> str:
    """Baseline path — no formalization, no verification.

    When --no-rag:  passes the problem directly as the plan.
    Otherwise:      literature → trajectories → plan → execute.
    """
    if args.no_rag:
        return tracer.execute(problem, problem)

    literature   = tracer.collect_literature(problem, local_path=args.path)
    trajectories = tracer.make_trajectories(literature)
    plan_obj     = tracer.make_research_plan(
        problem, trajectories, decomposition_mode=args.decompose
    )
    plan_str = _get_plan_str(plan_obj, fallback=problem)
    return tracer.execute(problem, plan_str)


_MAX_VERIFY_RETRIES = 3  # maximum verify → refine iterations


def _run_with_verification(
    tracer: SciTracer,
    problem: str,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    """Full pipeline — plan → formalize → verify → refine → re-verify → execute.

    Runs a closed feedback loop: after each verification failure the plan is
    refined and re-verified up to _MAX_VERIFY_RETRIES times.  Execution uses
    whichever plan_str passed the last verification round (or the best partial
    plan if the loop exhausts its budget).

    Returns (answer_str, extras_dict).
    extras contains verification metadata written to the result record.
    """
    if args.no_rag:
        plan_str = problem
    else:
        literature   = tracer.collect_literature(problem, local_path=args.path)
        trajectories = tracer.make_trajectories(literature)
        plan_obj     = tracer.make_research_plan(
            problem, trajectories, decomposition_mode=args.decompose
        )
        plan_str = _get_plan_str(plan_obj, fallback=problem)

    # ── Closed verify → refine loop ───────────────────────────────────────────
    _SEP  = "─" * 70
    _SEP2 = "═" * 70
    verification: list = []
    suggestion   = ""
    kripke_failed = 0
    all_verified  = False

    def _kripke_absent(results: list) -> bool:
        return bool(results) and all(r.error for r in results)

    def _run_agent_check(attempt: int, p_str: str) -> tuple[list, str]:
        print(f"\n{_SEP}")
        print(f"  AGENT VERIFICATION  (attempt {attempt})")
        print(_SEP)
        a_results, a_suggestion = tracer.verify_with_agent(
            problem=problem,
            plan_str=p_str,
        )
        _print_verification(a_results, a_suggestion)
        return a_results, a_suggestion

    for attempt in range(1, _MAX_VERIFY_RETRIES + 1):
        print(f"\n{_SEP2}")
        print(f"  VERIFY LOOP  {attempt}/{_MAX_VERIFY_RETRIES}")
        print(_SEP2)

        # ── Plan (instruction) ────────────────────────────────────────────────
        print(f"\n{_SEP}")
        print(f"  PLAN  (attempt {attempt})")
        print(_SEP)
        for line in plan_str.splitlines():
            print(f"  {line}")

        # ── Formalization ─────────────────────────────────────────────────────
        print(f"\n{_SEP}")
        print(f"  FORMALIZATION  (attempt {attempt})")
        print(_SEP)
        formalization = tracer.formalize_plan(plan_str)
        _, kripke_structure = _parse_structure(formalization)
        _print_hoare(formalization["hoare"])
        _print_kripke(kripke_structure)

        # ── Kripke verification ───────────────────────────────────────────────
        print(f"\n{_SEP}")
        print(f"  VERIFICATION  (attempt {attempt}, Kripke/{args.backend})")
        print(_SEP)
        verification, suggestion = tracer.verify_kripke(
            research_plan=plan_str,
            formalization=formalization,
            backend=args.backend,
        )
        _print_verification(verification, suggestion)

        # ── Agent verification ────────────────────────────────────────────────
        # Always runs as a complementary check.
        # Becomes the primary gate when Kripke is fully unavailable (all errors).
        if _kripke_absent(verification):
            print(f"  [!] Kripke unavailable — agent verifier is primary gate")
            verification, suggestion = _run_agent_check(attempt, plan_str)
        else:
            _run_agent_check(attempt, plan_str)

        kripke_failed = sum(1 for r in verification if not r.holds and not r.error)
        all_verified  = kripke_failed == 0

        if all_verified:
            print(f"\n  [✓] All properties hold — exiting loop after attempt {attempt}")
            break

        print(
            f"\n  [✗] {kripke_failed} propert{'y' if kripke_failed == 1 else 'ies'} failed"
            + (f" — refining plan for attempt {attempt + 1}..."
               if attempt < _MAX_VERIFY_RETRIES
               else " — budget exhausted, executing best available plan")
        )
        if attempt < _MAX_VERIFY_RETRIES and suggestion:
            plan_str = tracer.refine_plan(plan_str, feedback=suggestion)

    answer = tracer.execute(problem, plan_str)

    extras: dict[str, Any] = {
        "kripke_properties_checked": len(verification),
        "kripke_failed":             kripke_failed,
        "kripke_verified":           all_verified,
        "verify_attempts":           attempt,
        "verification_suggestion":   suggestion[:200] if suggestion else "",
    }
    return answer, extras


# ── Results I/O ───────────────────────────────────────────────────────────────

def _load_partial(path: str) -> tuple[list[dict], set[str]]:
    """Load a partial results file; return (results, seen_task_ids)."""
    p = Path(path)
    if not p.exists():
        return [], set()
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    seen    = {r["task_group_id"] for r in results}
    print(f"  Resuming from {path}: {len(seen)} rows already done.")
    return results, seen


def _save(results: list[dict], metrics: dict, output: str) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    tmp = output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    Path(tmp).replace(output)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0}

    def _agg(rows: list[dict]) -> dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {"n": 0}
        return {
            "n":             n,
            "exact_match":   round(sum(r["exact_match"]  for r in rows) / n, 4),
            "numeric_match": round(sum(r["numeric_match"] for r in rows) / n, 4),
            "llm_equiv":     round(sum(r["llm_equiv"]     for r in rows) / n, 4),
            "errors":        sum(1 for r in rows if r["error"]),
            "avg_elapsed_s": round(sum(r["elapsed_s"] for r in rows) / n, 2),
        }

    metrics: dict[str, Any] = {"total": total, "overall": _agg(results)}
    for subj in ("physics", "chemistry", "biology"):
        rows = [r for r in results if r["subject"] == subj]
        if rows:
            metrics[subj] = _agg(rows)
    return metrics


def _print_metrics(metrics: dict[str, Any]) -> None:
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  BENCHMARK RESULTS  ({metrics['total']} problems)")
    print(sep)
    for key, m in metrics.items():
        if key == "total" or not isinstance(m, dict) or "n" not in m:
            continue
        print(
            f"  {key:12}  n={m['n']:4}  "
            f"exact={m['exact_match']:.1%}  "
            f"numeric={m['numeric_match']:.1%}  "
            f"llm_equiv={m['llm_equiv']:.1%}  "
            f"errors={m['errors']}  "
            f"avg_t={m['avg_elapsed_s']:.1f}s"
        )
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SciTracer — FrontierScience benchmark runner.\n\n"
            "Without --logic_verification: fast direct-answer path, no formalization.\n"
            "With    --logic_verification: Kripke/Z3 verification applied before "
            "the final answer."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Dataset selection ─────────────────────────────────────────────────────
    grp = parser.add_argument_group("Dataset")
    grp.add_argument("--split",   default="test",
                     help="HuggingFace dataset split (default: test)")
    grp.add_argument("--subject", choices=["physics", "chemistry", "biology", "all"],
                     default="all",
                     help="Restrict evaluation to one subject (default: all)")
    grp.add_argument("--n",       type=int, default=None, metavar="N",
                     help="Evaluate only the first N rows (default: all)")
    grp.add_argument("--start",   type=int, default=0, metavar="IDX",
                     help="Start from this 0-based row index (default: 0)")

    # ── Pipeline control ──────────────────────────────────────────────────────
    grp = parser.add_argument_group("Pipeline")
    grp.add_argument(
        "--logic_verification",
        action="store_true",
        help=(
            "Enable formal logic verification: formalize the research plan into a "
            "Hoare triple + Kripke structure, run CTL model checking with Z3/Lean, "
            "and refine before executing. "
            "Without this flag the pipeline skips all formalization."
        ),
    )
    grp.add_argument("--backend",  choices=["z3", "lean"], default="z3",
                     help="Verification backend — only with --logic_verification (default: z3)")
    grp.add_argument("--no-rag",   action="store_true",
                     help="Skip literature retrieval; use the problem text directly as the plan")
    grp.add_argument("--path",     type=str, default=None, metavar="FOLDER",
                     help="Local paper folder forwarded to collect_literature()")
    grp.add_argument("--decompose", choices=["none", "tree", "htn"], default="none",
                     help="Plan decomposition mode (default: none for benchmark speed)")

    # ── Output / resume ───────────────────────────────────────────────────────
    grp = parser.add_argument_group("Output")
    grp.add_argument("--output",     type=str, default=None, metavar="FILE",
                     help="JSON output file (auto-named under results/ when omitted)")
    grp.add_argument("--resume",     type=str, default=None, metavar="FILE",
                     help="Resume from a partial results JSON, skipping evaluated rows")
    grp.add_argument("--save_every", type=int, default=5, metavar="N",
                     help="Checkpoint every N problems (default: 5)")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    # ── LM setup ──────────────────────────────────────────────────────────────
    lm = _setup_lm()
    dspy.configure(lm=lm)
    mode = "logic_verification ON" if args.logic_verification else "direct (no formalization)"
    print(f"LM   : {lm.model}")
    print(f"Mode : {mode}")
    print(f"RAG  : {'off (--no-rag)' if args.no_rag else 'on'}")

    # ── Output path ───────────────────────────────────────────────────────────
    if args.output is None:
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "verified" if args.logic_verification else "direct"
        args.output = f"results/bench_{suffix}_{ts}.json"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading openai/frontierscience [{args.split}]...")
    rows = load_frontierscience(args.split)

    if args.subject != "all":
        rows = [r for r in rows if r["subject"] == args.subject]
    rows = rows[args.start:]
    if args.n is not None:
        rows = rows[: args.n]

    subj_counts = {}
    for r in rows:
        subj_counts[r["subject"]] = subj_counts.get(r["subject"], 0) + 1
    print(f"  {len(rows)} problems: {subj_counts}")

    # ── Resume ────────────────────────────────────────────────────────────────
    results, seen = _load_partial(args.resume) if args.resume else ([], set())
    rows_todo = [r for r in rows if r["task_group_id"] not in seen]
    print(f"  {len(rows_todo)} to evaluate  ({len(seen)} already done)")

    if not rows_todo:
        print("  Nothing to do.")
        _print_metrics(compute_metrics(results))
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    tracer = SciTracer()
    judge  = LLMJudge()

    for i, row in enumerate(rows_todo, 1):
        short_prob = row["problem"][:70].replace("\n", " ")
        print(
            f"\n[{i:3}/{len(rows_todo)}] {row['subject']:10}  "
            f"id={row['task_group_id'][:8]}  {short_prob}…"
        )

        result = run_one(tracer, judge, row, args)
        results.append(result)

        mark = "✓" if result["llm_equiv"] else ("~" if result["numeric_match"] else "✗")
        print(
            f"  [{mark}] exact={result['exact_match']}  "
            f"numeric={result['numeric_match']}  "
            f"llm_equiv={result['llm_equiv']}  "
            f"t={result['elapsed_s']}s"
        )
        if result["predicted"]:
            print(f"  predicted : {result['predicted'][:100]}")
        print(f"  reference : {_strip_latex(result['reference'])[:100]}")
        if result["error"]:
            print(f"  ERROR     : {result['error']}")

        if i % args.save_every == 0:
            metrics = compute_metrics(results)
            _save(results, metrics, args.output)
            print(f"  ── checkpoint saved → {args.output}  ({len(results)} rows)")

    # ── Final save + report ───────────────────────────────────────────────────
    metrics = compute_metrics(results)
    _save(results, metrics, args.output)
    _print_metrics(metrics)
    print(f"\nResults → {args.output}")


if __name__ == "__main__":
    main()
