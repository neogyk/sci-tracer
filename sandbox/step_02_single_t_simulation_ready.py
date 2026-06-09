import numpy as np

def simulate(N, T, nsweeps, burn_frac=0.2):
    """
    Run Metropolis Monte Carlo simulation for a single temperature.

    Parameters:
    N (int): Linear size of the square lattice.
    T (float): Temperature.
    nsweeps (int): Total number of sweeps.
    burn_frac (float): Fraction of sweeps to discard as burn-in.

    Returns:
    tuple: (mean_mag, std_mag, mean_energy, std_energy)
    """
    lattice = np.random.choice([-1, 1], size=(N, N))
    burn = int(nsweeps * burn_frac)
    m_vals = []
    e_vals = []
    for step in range(nsweeps):
        lattice = metropolis_sweep(lattice, T)  # assumed defined
        if step >= burn:
            m = np.abs(np.mean(lattice))
            e = energy(lattice) / (N * N)  # assumed defined
            m_vals.append(m)
            e_vals.append(e)
    return np.mean(m_vals), np.std(m_vals), np.mean(e_vals), np.std(e_vals)