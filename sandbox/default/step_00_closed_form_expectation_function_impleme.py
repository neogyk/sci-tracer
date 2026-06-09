from __future__ import annotations


def expectation_value(A: float, E_a: float, E_b: float, t: float) -> float:
    """Return the conserved energy expectation (E_a + E_b)/2.

    Parameters
    ----------
    A : float
        Off-diagonal coupling (unused; expectation independent of A).
    E_a : float
        Energy level a.
    E_b : float
        Energy level b.
    t : float
        Time parameter (unused; expectation is time-independent).

    Returns
    -------
    float
        (E_a + E_b) / 2
    """
    return 0.5 * (E_a + E_b)