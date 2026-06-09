import numpy as np

def harmonic_mannella_leapfrog(
    x0: float,
    v0: float,
    t0: float,
    steps: int,
    taup: float,
    omega0: float,
    vrms: float,
) -> float:
    """
    Integrate the Langevin equation for a harmonic oscillator using Mannella's leapfrog scheme.

    Parameters
    ----------
    x0 : float
        Initial position.
    v0 : float
        Initial velocity.
    t0 : float
        Total integration time.
    steps : int
        Number of time steps (must be > 0).
    taup : float
        Momentum relaxation time (tau_p).
    omega0 : float
        Natural frequency of the harmonic trap.
    vrms : float
        Root‑mean‑square velocity (sqrt(kT/m)).

    Returns
    -------
    float
        Final position after time t0.
    """
    dt = t0 / steps
    gamma = 1.0 / taup                     # damping coefficient
    noise_amp = np.sqrt(2.0 * gamma * vrms**2 * dt)   # sqrt(2*dt/taup)*vrms

    x = float(x0)
    v = float(v0)

    for _ in range(steps):
        g = np.random.normal(0.0, 1.0)

        # Position update: explicit in position (uses current velocity)
        x_new = x + dt * v

        # Velocity update: implicit in damping via the denominator 1 + dt/(2*tau_p)
        v_new = (
            v * (1.0 - 0.5 * gamma * dt)
            - omega0**2 * x_new * dt
            + noise_amp * g
        ) / (1.0 + 0.5 * gamma * dt)

        x, v = x_new, v_new

    return x