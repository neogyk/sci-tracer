import numpy as np
from typing import List

def generate_Hamiltonian(
    P: List[float],
    phi: float,
    R: float,
    l: float,
    w: float,
    a: float,
    n: float,
    h: float,
    N: int,
    rho: float
) -> np.ndarray:
    """
    Constructs the Hamiltonian matrix H for N coupled harmonic oscillators
    under optical binding forces.

    Parameters
    ----------
    P : list of float
        Positions of the N particles (assumed along x-axis).
    phi : float
        Laser power (in watts).
    R : float
        Particle radius (in meters).
    l : float
        Wavelength of trapping laser (in meters).
    w : float
        Beam waist (in meters).
    a : float
        (Unused in current model) Additional parameter.
    n : float
        Refractive index of particle material.
    h : float
        Step size for finite difference derivative (in meters).
    N : int
        Number of particles.
    rho : float
        Mass density of particle material (kg/m^3).

    Returns
    -------
    H : ndarray of shape (N, N)
        Symmetric matrix with diagonal entries Ω_i (natural frequencies)
        and off-diagonal entries g_{ij} (coupling strengths).
    """
    # Physical constants
    epsilon0 = 8.854187817e-12          # Vacuum permittivity [F/m]
    c = 299792458.0                     # Speed of light [m/s]

    # 1. Mass of a single spherical particle
    m = (4.0 / 3.0) * np.pi * R**3 * rho   # kg

    # 2. Polarizability (Clausius-Mossotti relation, SI units)
    alpha = 4.0 * np.pi * epsilon0 * R**3 * (n**2 - 1.0) / (n**2 + 2.0)

    # 3. Trap stiffness constant
    #    Approximate expression for a Gaussian beam trap (Rayleigh regime)
    #    k = (2 * phi) / (pi * w**4 * c) * (alpha / epsilon0)
    k_trap = (2.0 * phi) / (np.pi * w**4 * c) * (alpha / epsilon0)

    # 4. Optical binding force function
    #    Simplified expression for two equal dipoles driven by a plane wave,
    #    polarization along the line joining them.
    def binding_force(r: float) -> float:
        """Optical binding force (positive = repulsive?) as a function of separation r."""
        if r == 0.0:
            return 0.0
        kr = 2.0 * np.pi / l * r
        # Force magnitude, arbitrary scaling for demonstration
        F = - (alpha**2 * phi) / (4.0 * np.pi * epsilon0 * r**4) * (1.0 + kr**2) * np.cos(kr)
        return F

    # 5. Compute diagonal entries Ω_i
    omega_i = np.sqrt(k_trap / m)          # same for all (could be generalized)

    # 6. Compute off-diagonal couplings g_{ij}
    g = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            r_ij = abs(P[i] - P[j])
            # Derivative using finite difference
            F_r = binding_force(r_ij)
            F_r_plus = binding_force(r_ij + h)
            dF = (F_r_plus - F_r) / h     # derivative of force w.r.t. separation
            # Coupling stiffness k_ij = dF, then g_ij = k_ij / m
            g_ij = dF / m
            g[i, j] = g_ij
            g[j, i] = g_ij

    # 7. Construct Hamiltonian matrix H
    H = np.diag(omega_i)
    H += g

    return H