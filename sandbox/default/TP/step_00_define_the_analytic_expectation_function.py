from typing import Final


def expectation_value(A: float, E_a: float, E_b: float, t: float) -> float:
    """
    Analytic expectation value of the Hamiltonian for the specified setup.

    Parameters:
        A (float): Off-diagonal coupling parameter (unused analytically).
        E_a (float): Energy parameter for level a.
        E_b (float): Energy parameter for level b.
        t (float): Time parameter (unused due to conservation for time-independent H).

    Returns:
        float: The conserved expectation value, (E_a + E_b) / 2.
    """
    half: Final[float] = 0.5
    return half * (E_a + E_b)