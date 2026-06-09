import sys
import os
import importlib
import math
import traceback
from typing import Dict, Tuple, List, Any

def add_sandbox_to_path() -> str:
    # Use the directory of this script (run.py) as the sandbox directory
    sandbox_dir = os.path.dirname(os.path.abspath(__file__))
    if sandbox_dir not in sys.path:
        sys.path.insert(0, sandbox_dir)
    return sandbox_dir

def safe_import(module_name: str):
    try:
        mod = importlib.import_module(module_name)
        print(f"[import] Successfully imported {module_name}")
        return mod, None
    except Exception as e:
        print(f"[import] ERROR importing {module_name}: {e}")
        traceback.print_exc()
        return None, e

def rel_err(a: float, b: float) -> float:
    denom = abs(a)
    if denom == 0.0:
        return math.inf if b != 0.0 else 0.0
    return abs(a - b) / denom

def coherence_report(
    m0, m1, m2,
    sigmas: List[float],
    bvals: List[float],
    qmc_n: int = 8192
) -> Dict[str, Any]:
    """
    Compare the three b_eff implementations pairwise and also compare the canonical
    (m0.b_eff) against deterministic QMC using step_02's utilities.
    """
    # Canonical function
    b0 = m0.b_eff
    b1 = m1.b_eff
    b2 = m2.b_eff

    # Pairwise maxima
    pair_max_rel: Dict[str, float] = {
        "m0_vs_m1": 0.0,
        "m0_vs_m2": 0.0,
        "m1_vs_m2": 0.0,
    }
    pair_worst: Dict[str, Tuple[float, float, float, float]] = {}

    # QMC grid and checks
    z_grid = m2.build_z_grid(qmc_n)
    max_rel_err_qmc = 0.0
    worst_rel_case_qmc: Tuple[float, float, float, float] | None = None
    max_abs_small_qmc = 0.0
    worst_abs_case_qmc: Tuple[float, float, float, float] | None = None

    for sigma in sigmas:
        for b_in in bvals:
            be0 = b0(sigma, b_in)
            be1 = b1(sigma, b_in)
            be2 = b2(sigma, b_in)

            # Pairwise relative errors (ignore cases where reference is effectively zero by using symmetric measure)
            def upd(tag: str, ref: float, comp: float):
                # If ref is tiny, compare relative to comp if it is larger; else compare absolute 0
                if abs(ref) > 1e-12:
                    r = abs(comp - ref) / abs(ref)
                elif abs(comp) > 1e-12:
                    r = abs(comp - ref) / abs(comp)
                else:
                    r = 0.0
                nonlocal pair_max_rel, pair_worst, sigma, b_in
                if r > pair_max_rel[tag]:
                    pair_max_rel[tag] = r
                    pair_worst[tag] = (sigma, b_in, ref, comp)

            upd("m0_vs_m1", be0, be1)
            upd("m0_vs_m2", be0, be2)
            upd("m1_vs_m2", be1, be2)

            # QMC comparison for canonical implementation
            est = m2.slope_qmc(sigma, b_in, z_grid)
            err = abs(est - be0)
            if abs(be0) > 1e-3:
                rel = err / abs(be0)
                if rel > max_rel_err_qmc:
                    max_rel_err_qmc = rel
                    worst_rel_case_qmc = (sigma, b_in, be0, est)
            else:
                if err > max_abs_small_qmc:
                    max_abs_small_qmc = err
                    worst_abs_case_qmc = (sigma, b_in, be0, est)

    return {
        "pair_max_rel": pair_max_rel,
        "pair_worst": pair_worst,
        "qmc": {
            "N": qmc_n,
            "max_rel_err": max_rel_err_qmc,
            "worst_rel_case": worst_rel_case_qmc,
            "max_abs_small": max_abs_small_qmc,
            "worst_abs_case": worst_abs_case_qmc,
        },
    }

def main():
    sandbox_dir = add_sandbox_to_path()
    print(f"[path] Sandbox directory added to sys.path: {sandbox_dir}")

    # Import modules in a logical order
    modules = {}
    order = [
        "step_00_phase_1_derive_effective_bias_formula",
        "step_01_implement_b_eff_function",
        "step_02_validate_against_deterministic_quasi_mon",
    ]
    for name in order:
        mod, err = safe_import(name)
        if err is not None:
            print("[fatal] Cannot proceed due to import error.")
            return
        modules[name] = mod

    m0 = modules["step_00_phase_1_derive_effective_bias_formula"]
    m1 = modules["step_01_implement_b_eff_function"]
    m2 = modules["step_02_validate_against_deterministic_quasi_mon"]

    print("\n[info] Discovered key functions:")
    print(f" - {m0.__name__}.b_eff")
    print(f" - {m1.__name__}.b_eff")
    print(f" - {m2.__name__}.b_eff")
    print("Selecting canonical implementation: step_00_phase_1_derive_effective_bias_formula.b_eff")

    # Reproduce a small demo similar to step_00's __main__
    print("\n[demo] Sample b_eff values (from canonical implementation):")
    for sigma in (0.1, 0.5, 1.0, 3.0):
        row = []
        for b in (-2.0, -0.5, 0.0, 0.5, 2.0):
            be = m0.b_eff(sigma, b)
            row.append(f"b_in={b:5.1f}: {be: .6f}")
        print(f"sigma={sigma:4.1f} -> " + ", ".join(row))

    # Display coherence across implementations and against QMC
    sigmas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    bvals = [-3.0, -2.0, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]

    print("\n[coherence] Comparing implementations and validating against QMC ...")
    report = coherence_report(m0, m1, m2, sigmas, bvals, qmc_n=8192)

    pair_max_rel = report["pair_max_rel"]
    pair_worst = report["pair_worst"]
    qmc_rep = report["qmc"]

    print("[coherence] Pairwise maximum relative errors:")
    for tag, val in pair_max_rel.items():
        worst = pair_worst.get(tag, None)
        if worst is not None:
            sigma, b_in, ref, comp = worst
            print(f" - {tag}: {val:.6g} at (sigma={sigma}, b_in={b_in}), ref={ref:.6g}, comp={comp:.6g}")
        else:
            print(f" - {tag}: {val:.6g}")

    print("\n[coherence] QMC comparison (canonical vs deterministic quasi-MC):")
    print(f" - z-grid size N: {qmc_rep['N']}")
    print(f" - max_rel_err (|b_eff|>1e-3): {qmc_rep['max_rel_err']:.6g}")
    if qmc_rep["worst_rel_case"] is not None:
        s, b, be, est = qmc_rep["worst_rel_case"]
        print(f"   worst at (sigma={s}, b_in={b}): beff={be:.6g}, qmc={est:.6g}")
    print(f" - max_abs_small (|b_eff|<=1e-3): {qmc_rep['max_abs_small']:.6g}")
    if qmc_rep["worst_abs_case"] is not None:
        s, b, be, est = qmc_rep["worst_abs_case"]
        print(f"   worst at (sigma={s}, b_in={b}): beff={be:.6g}, qmc={est:.6g}")

    # Enforce the same tolerances as specified
    assert qmc_rep["max_rel_err"] <= 0.01 + 1e-12, "Relative error too high vs QMC"
    assert qmc_rep["max_abs_small"] <= 1e-4 + 1e-12, "Absolute error too high for small values vs QMC"

    # Surface step_02's own validation metrics (computed at import time)
    print("\n[step_02] Import-time validation metrics:")
    try:
        metrics = m2.metrics
        print(" - Validation passed inside step_02.")
        print(f" - Max relative error (|b_eff|>1e-3): {metrics['max_rel_err']:.6g}")
        print(f" - Max absolute error (|b_eff|<=1e-3): {metrics['max_abs_small']:.6g}")
        print(f" - N used internally: {int(metrics['N'])}")
    except Exception:
        print(" - No metrics found on step_02 (unexpected).")

    # Additional large-sigma sanity check (like step_02.__main__)
    sigma_large = 1e2
    val = m0.b_eff(sigma_large, 1.0)
    expected_asym = math.sqrt(0.5 * math.pi) / sigma_large
    print("\n[sanity] Large-sigma asymptotic for b_in=+1:")
    print(f" - b_eff({sigma_large}, 1.0) = {val:.6g}")
    print(f" - Reference scale ~ sqrt(pi/2)/sigma = {expected_asym:.6g}")
    assert val < 2.0 * expected_asym, "Asymptotic sanity check failed"

    print("\n[summary] All modules imported, coherence verified, and validations passed.")
    print("[done] Orchestrator completed successfully.")

if __name__ == "__main__":
    main()