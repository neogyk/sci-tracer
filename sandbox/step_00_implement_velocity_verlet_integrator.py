from typing import Tuple
import numpy as np

# Assumes existence of forces(positions, N, L, rc, sigma, epsilon, m) that returns acceleration array of shape (N,3)
# Also assumes min_image or wrapping is handled inside forces; we wrap positions explicitly.

def velocity_verlet(
    N: int,
    sigma: float,
    epsilon: float,
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
    m: float,
    L: float,
    rc: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one step of the velocity Verlet integrator.

    Parameters
    ----------
    N : int
        Number of particles.
    sigma, epsilon : float
        Lennard-Jones parameters.
    positions : np.ndarray, shape (N,3)
        Current positions.
    velocities : np.ndarray, shape (N,3)
        Current velocities.
    dt : float
        Time step.
    m : float
        Particle mass.
    L : float
        Box length.
    rc : float
        Cutoff radius.

    Returns
    -------
    new_positions : np.ndarray, shape (N,3)
        Positions after one step.
    new_velocities : np.ndarray, shape (N,3)
        Velocities after one step.
    """
    # Compute current accelerations
    acc = forces(positions, N, L, rc, sigma, epsilon, m)  # Already forces / m? We assume forces returns acceleration.
    # Alternatively, if forces returns force, then acc = forces / m. We'll assume it returns acceleration for clarity.

    # Half-step velocity update
    v_half = velocities + 0.5 * dt * acc

    # Full-step position update and wrap into box (periodic boundary conditions)
    new_positions = positions + dt * v_half
    new_positions = new_positions % L  # wrap to [0, L)

    # Compute new accelerations at updated positions
    new_acc = forces(new_positions, N, L, rc, sigma, epsilon, m)

    # Final velocity update
    new_velocities = v_half + 0.5 * dt * new_acc

    return new_positions, new_velocities