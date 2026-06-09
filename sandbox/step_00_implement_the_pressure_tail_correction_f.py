import math

def P_tail(N: int, L: float, sigma: float, epsilon: float, rc: float) -> float:
    """
    Compute the tail correction to the pressure for a truncated-and-shifted
    Lennard-Jones system.

    Parameters
    ----------
    N : int
        Number of particles.
    L : float
        Box side length (nm). Must satisfy L >= 2*rc.
    sigma : float
        Lennard-Jones sigma (nm).
    epsilon : float
        Lennard-Jones epsilon (zJ).
    rc : float
        Cutoff radius (nm). Must be > 0 and <= L/2.

    Returns
    -------
    float
        Pressure tail correction in bar.
    """
    V = L ** 3
    rho = N / V

    # Compute the analytic tail correction in energy/volume [zJ/nm^3]
    # Formula: p_tail = (16/3)*π*ε*σ^3*ρ^2 * [ (2/3)*(σ/rc)^9 - (σ/rc)^3 ]
    s_over_rc = sigma / rc
    term = (2.0 / 3.0) * (s_over_rc ** 9) - (s_over_rc ** 3)
    p_tail_energy_vol = (16.0 / 3.0) * math.pi * epsilon * (sigma ** 3) * (rho ** 2) * term

    # Convert to bar: 1 zJ/nm^3 = 10 bar
    p_tail_bar = p_tail_energy_vol * 10.0

    return p_tail_bar