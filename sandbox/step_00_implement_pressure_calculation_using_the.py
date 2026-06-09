def pressure(N: int, L: float, T: float, xyz: 'np.ndarray', sigma: float, epsilon: float, rc: float) -> tuple:
    """
    Compute kinetic and virial pressure contributions and total pressure.

    Parameters:
        N (int): Number of particles.
        L (float): Box length (nm).
        T (float): Temperature (K).
        xyz (np.ndarray): Positions array of shape (N, 3) (nm).
        sigma (float): LJ sigma parameter (nm).
        epsilon (float): LJ epsilon parameter (zJ).
        rc (float): Cutoff distance (nm).

    Returns:
        tuple: (P_kin, P_vir, P_total) in bar.
    """
    import numpy as np
    kB = 0.0138064852  # zJ/K
    V = L**3
    # Kinetic pressure in zJ/nm^3 -> convert to bar (factor 10)
    P_kin = (N * kB * T / V) * 10.0
    virial_sum = 0.0
    for i in range(N):
        for j in range(i+1, N):
            r_ij = dist_v(xyz[i], xyz[j], L)
            f_ij = f_ij(r_ij, sigma, epsilon, rc)
            virial_sum += np.dot(f_ij, r_ij)
    P_vir = (virial_sum / (3.0 * V)) * 10.0
    P_total = P_kin + P_vir
    return (P_kin, P_vir, P_total)