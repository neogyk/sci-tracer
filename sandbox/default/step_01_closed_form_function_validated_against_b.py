from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def expectation_value(A: float, E_a: float, E_b: float, t: float) -> float:
    """
    Closed-form conserved energy expectation value for the specified setup.
    Parameters are accepted for interface compatibility; only E_a and E_b matter.
    """
    return 0.5 * (E_a + E_b)


def _brute_force_expectation(A: float, E_a: float, E_b: float, t: float) -> float:
    """
    Compute ⟨ψ(t)|H|ψ(t)⟩ by explicit unitary evolution using eigendecomposition.
    """
    H: NDArray[np.complex128] = np.array(
        [[E_a, 0.0, A], [0.0, E_b, 0.0], [A, 0.0, E_a]], dtype=np.complex128
    )
    w: NDArray[np.float64]
    V: NDArray[np.complex128]
    w, V = np.linalg.eigh(H)
    U: NDArray[np.complex128] = V @ np.diag(np.exp(-1j * w * t)) @ V.conj().T

    psi0: NDArray[np.complex128] = (1.0 / np.sqrt(2.0)) * np.array(
        [1.0, 1.0, 0.0], dtype=np.complex128
    )
    psi_t: NDArray[np.complex128] = U @ psi0
    val: np.complex128 = psi_t.conj().T @ H @ psi_t
    return float(val.real)


def run_validation(n: int = 200, seed: int = 42, tol: float = 1e-12) -> float:
    """
    Run randomized validation comparing the closed-form value vs brute-force evolution.
    Returns the maximum relative error observed.
    """
    rng = np.random.default_rng(seed)
    max_rel: float = 0.0
    for _ in range(n):
        A = float(rng.uniform(-5.0, 5.0))
        E_a = float(rng.uniform(-10.0, 10.0))
        E_b = float(rng.uniform(-10.0, 10.0))
        t = float(rng.uniform(-3.0, 3.0))
        brute: float = _brute_force_expectation(A, E_a, E_b, t)
        pred: float = expectation_value(A, E_a, E_b, t)
        denom: float = abs(brute) if abs(brute) > 0.0 else 1.0
        rel: float = abs(pred - brute) / denom
        if rel > max_rel:
            max_rel = rel
        assert rel < tol, f"Relative error {rel} too large; pred={pred}, brute={brute}"
    return max_rel


if __name__ == "__main__":
    N_CASES: int = 200
    max_rel_error: float = run_validation(n=N_CASES, seed=42, tol=1e-12)
    print(f"Validation passed over {N_CASES} random cases; max relative error = {max_rel_error}")