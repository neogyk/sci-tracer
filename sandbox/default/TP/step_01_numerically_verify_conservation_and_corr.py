from typing import Final
import numpy as np
from numpy.typing import NDArray


def expectation_value(A: float, E_a: float, E_b: float, t: float) -> float:
    """
    Analytic expectation value of the Hamiltonian ⟨H⟩(t) for a time-independent H
    and initial state (|1> + |2>)/sqrt(2), given the specified model.
    By conservation of energy, it is time-independent and equals (E_a + E_b)/2.
    Parameters A and t are unused in the analytic value but are part of the interface.
    """
    return 0.5 * (E_a + E_b)


def unitary_time_evolution_operator(H: NDArray[np.float64], t: float) -> NDArray[np.complex128]:
    """
    Compute U(t) = exp(-i H t) via eigen-decomposition for a real-symmetric H.
    """
    w, V = np.linalg.eigh(H)  # V is real orthogonal, H is real symmetric
    phases: NDArray[np.complex128] = np.exp(-1j * w * t)
    U: NDArray[np.complex128] = V @ np.diag(phases) @ V.T  # V.T = V^{-1} for orthogonal V
    return U


def compute_energy(H: NDArray[np.float64], psi: NDArray[np.complex128]) -> float:
    """
    Compute ⟨psi| H |psi⟩, returning the real part as a float.
    """
    return float(np.vdot(psi, H @ psi).real)


def run_verification(trials: int = 100, seed: int = 42, tol: float = 1e-9) -> None:
    """
    Run randomized verification that the numerical energy expectation matches
    the analytic value within tolerance tol for the specified number of trials.
    """
    # Initial state: (|1> + |2>)/sqrt(2)
    psi0: NDArray[np.complex128] = (1.0 / np.sqrt(2.0)) * np.array([1.0, 1.0, 0.0], dtype=np.complex128)

    rng = np.random.default_rng(seed)
    for _ in range(trials):
        A: float = float(rng.uniform(-5.0, 5.0))
        E_a: float = float(rng.uniform(-3.0, 3.0))
        E_b: float = float(rng.uniform(-3.0, 3.0))
        t: float = float(rng.uniform(0.0, 10.0))

        H: NDArray[np.float64] = np.array(
            [[E_a, 0.0, A],
             [0.0, E_b, 0.0],
             [A,   0.0, E_a]],
            dtype=np.float64
        )

        U: NDArray[np.complex128] = unitary_time_evolution_operator(H, t)
        psit: NDArray[np.complex128] = U @ psi0

        exp_num: float = compute_energy(H, psit)
        exp_ana: float = expectation_value(A, E_a, E_b, t)

        diff: float = abs(exp_num - exp_ana)
        assert diff <= tol, f"Energy mismatch: num={exp_num}, ana={exp_ana}, diff={diff}"

    print(f"All {trials} tests passed with tolerance ≤ {tol}.")


if __name__ == "__main__":
    run_verification(trials=100, seed=42, tol=1e-9)