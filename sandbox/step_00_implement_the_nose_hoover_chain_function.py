import numpy as np

def nose_hoover_chain(
    x0: float,
    v0: float,
    T: float,
    M: int,
    m: float,
    omega: float,
    dt: float,
    nsteps: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrates a 1D harmonic oscillator coupled to M Nosé–Hoover chains using
    a symplectic, fourth‑order (Yoshida) splitting of the NHC Liouville
    operator, combined with velocity‑Verlet for the physical oscillator.

    Parameters
    ----------
    x0 : float
        Initial position of the oscillator.
    v0 : float
        Initial velocity of the oscillator.
    T : float
        Desired temperature of the canonical ensemble (k_B = 1).
    M : int
        Number of thermostats in the chain (M >= 1).
    m : float
        Mass of the oscillator.
    omega : float
        Angular frequency of the harmonic oscillator.
    dt : float
        Total time step for the overall integrator.
    nsteps : int
        Number of integration steps.

    Returns
    -------
    x_traj : np.ndarray, shape (nsteps, 1)
        Trajectory of positions.
    v_traj : np.ndarray, shape (nsteps, 1)
        Trajectory of velocities.
    """
    k_B = 1.0
    # Mass parameters for the thermostat chain (one per degree of freedom)
    Q = np.full(M, k_B * T / omega**2, dtype=np.float64)

    xi = np.zeros(M, dtype=np.float64)          # thermostat “positions”
    v_xi = np.zeros(M, dtype=np.float64)        # thermostat “velocities”

    x = np.float64(x0)
    v = np.float64(v0)

    x_traj = np.zeros((nsteps, 1), dtype=np.float64)
    v_traj = np.zeros((nsteps, 1), dtype=np.float64)

    # Force on the first thermostat (from the physical particle)
    def G1(v_val: np.float64) -> np.float64:
        return (m * v_val**2 - k_B * T) / Q[0]

    # Force on the k‑th thermostat (k = 1, …, M‑1, 0‑based index)
    def Gk(k: int, v_xi_arr: np.ndarray) -> np.float64:
        return (Q[k-1] * v_xi_arr[k-1]**2 - k_B * T) / Q[k]

    # --------------------------------------------------------------
    # Second‑order reversible NHC integrator (one full step of size tau)
    # Based on Martyna, Tuckerman & Klein (1992)
    # --------------------------------------------------------------
    def base_nhc(tau: np.float64) -> None:
        nonlocal x, v, xi, v_xi

        # 1. Damp thermostat momenta from top down (i = M‑1 down to 1)
        for i in range(M-1, 0, -1):
            v_xi[i] *= np.exp(-v_xi[i+1] * tau / 2.0)

        # 2. Half update of particle velocity
        v *= np.exp(-v_xi[0] * tau / 2.0)

        # 3. Half update of thermostat positions
        xi += v_xi * (tau / 2.0)

        # 4. Force update on first thermostat
        v_xi[0] += G1(v) * (tau / 2.0)

        # 5. Force updates on remaining thermostats (i = 1, …, M‑1)
        for i in range(1, M):
            v_xi[i] += Gk(i, v_xi) * (tau / 2.0)

        # 6. Second half update of particle velocity
        v *= np.exp(-v_xi[0] * tau / 2.0)

        # 7. Second half update of thermostat positions
        xi += v_xi * (tau / 2.0)

        # 8. Second damping of thermostat momenta from top down
        for i in range(M-1, 0, -1):
            v_xi[i] *= np.exp(-v_xi[i+1] * tau / 2.0)

    # --------------------------------------------------------------
    # Yoshida’s fourth‑order composition coefficients
    # --------------------------------------------------------------
    theta = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 - theta)
    w2 = -theta / (2.0 - theta)

    def yoshida_nhc(dt_half: np.float64) -> None:
        """Advance the NHC chain by a half‑step using Yoshida’s triple‑
        composition of the reversible base integrator."""
        base_nhc(w1 * dt_half)
        base_nhc(w2 * dt_half)
        base_nhc(w1 * dt_half)

    # --------------------------------------------------------------
    # Main time‑stepping loop: split integration of the full Liouville
    # operator into NHC half‑steps (Yoshida) and velocity‑Verlet for
    # the harmonic oscillator.
    # --------------------------------------------------------------
    for step in range(nsteps):
        # First half: NHC (fourth‑order)
        yoshida_nhc(dt / 2.0)

        # Velocity‑Verlet for the harmonic oscillator
        a = -omega**2 * x
        v += a * (dt / 2.0)
        x += v * dt
        a = -omega**2 * x
        v += a * (dt / 2.0)

        # Second half: NHC (fourth‑order)
        yoshida_nhc(dt / 2.0)

        # Store the state at this full step
        x_traj[step, 0] = x
        v_traj[step, 0] = v

    return x_traj, v_traj