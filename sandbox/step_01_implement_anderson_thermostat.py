import numpy as np
import scipy.constants as const

def anderson_thermostat(
    velocities: np.ndarray,
    dt: float,
    nu: float,
    T: float,
    m: float
) -> np.ndarray:
    """
    Apply the Andersen thermostat to the velocity array.

    Parameters
    ----------
    velocities : np.ndarray
        Array of shape (N, 3) containing velocities in nm/ps.
    dt : float
        Integration time step in ps.
    nu : float
        Collision frequency in 1/ps.
    T : float
        Target temperature in K.
    m : float
        Particle mass in g/mol (e.g., 39.948 for argon).

    Returns
    -------
    np.ndarray
        Modified velocities (same object) with stochastic collisions applied
        and center-of-mass motion removed.

    Notes
    -----
    Each particle is selected with probability nu*dt. Selected particles
    receive new velocities drawn from the Maxwell–Boltzmann distribution
    at temperature T. After reassignment, the centre-of-mass velocity is
    subtracted to conserve total linear momentum.

    Unit conversions:
        mass_kg = m * 1.66053906660e-27  (g/mol -> kg per particle)
        kB = 1.380649e-23  (J/K)
        sigma_v_m_s = sqrt(kB * T / mass_kg)  (m/s)
        sigma_v_nm_ps = sigma_v_m_s * 1e-3   (nm/ps)
    """
    N = velocities.shape[0]
    p = nu * dt                     # selection probability per particle

    # Convert mass to kg per particle and compute velocity standard deviation
    mass_kg = m * 1.66053906660e-27
    kB = const.Boltzmann            # 1.380649e-23 J/K
    sigma_v_m_s = np.sqrt(kB * T / mass_kg)   # in m/s
    sigma_v_nm_ps = sigma_v_m_s * 1e-3        # convert to nm/ps

    # Generate random velocities for all particles (only a subset will be used)
    new_velocities = np.random.normal(0.0, sigma_v_nm_ps, size=(N, 3))

    # Select particles to collide
    mask = np.random.random(N) < p
    velocities[mask] = new_velocities[mask]

    # Remove centre-of-mass velocity (conserves total linear momentum)
    com_velocity = np.mean(velocities, axis=0)
    velocities -= com_velocity

    return velocities