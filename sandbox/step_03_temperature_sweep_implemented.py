import numpy as np

def scan_T(N: int, T_range: np.ndarray, nsweeps: int = 5000):
    """
    Perform a temperature sweep for the 2D Ising model.
    
    Parameters:
        N (int): Linear lattice size (lattice has NxN spins).
        T_range (np.ndarray): Array of temperatures to simulate.
        nsweeps (int): Number of Monte Carlo sweeps per temperature.
    
    Returns:
        tuple: (T, mag, chi, binder, energy) arrays of length len(T_range).
            T : temperatures
            mag : mean absolute magnetization per spin
            chi : magnetic susceptibility = N^2 * var(|m|) / T
            binder : Binder cumulant = 1 - <m^4> / (3 <m^2>^2)
            energy : mean energy per spin
    """
    mags = []
    chis = []
    binders = []
    energies = []
    
    for T in T_range:
        # Use the simulate function to get average magnetization, its std, and average energy.
        m_mean, m_std, e_mean, e_std = simulate(N, T, nsweeps)
        
        # Magnetic susceptibility
        chi = (N * N) * (m_std ** 2) / T
        chis.append(chi)
        
        # Binder cumulant requires second and fourth moments of |m|.
        # We run a separate simulation (could be combined but kept separate for clarity)
        m_sq_vals = []
        m_4_vals = []
        lattice = np.random.choice([-1, 1], size=(N, N))
        burn = int(nsweeps * 0.2)
        for step in range(nsweeps):
            lattice = metropolis_sweep(lattice, T)
            if step >= burn:
                m = np.abs(np.mean(lattice))
                m_sq_vals.append(m ** 2)
                m_4_vals.append(m ** 4)
        avg_m2 = np.mean(m_sq_vals)
        avg_m4 = np.mean(m_4_vals)
        binder = 1.0 - avg_m4 / (3.0 * avg_m2 ** 2) if avg_m2 > 0 else 0.0
        binders.append(binder)
        
        mags.append(m_mean)
        energies.append(e_mean)
    
    return np.array(T_range), np.array(mags), np.array(chis), np.array(binders), np.array(energies)


# The following functions are assumed to be already defined in the environment.
# For completeness, we include their signatures (they should be defined before calling scan_T).

def energy(lattice: np.ndarray) -> float:
    """Compute total energy of the 2D Ising lattice (J=1, periodic boundaries)."""
    N = lattice.shape[0]
    total = 0.0
    for i in range(N):
        for j in range(N):
            # Four neighbors (periodic)
            neighbors = [((i-1)%N, j), (i, (j+1)%N), ((i+1)%N, j), (i, (j-1)%N)]
            for ni, nj in neighbors:
                total += -1.0 * lattice[i, j] * lattice[ni, nj]
    return total / 2.0  # each bond counted twice


def metropolis_sweep(lattice: np.ndarray, T: float) -> np.ndarray:
    """Perform one full Metropolis sweep over all spins."""
    N = lattice.shape[0]
    for i in range(N):
        for j in range(N):
            s = lattice[i, j]
            neighbors = [((i-1)%N, j), (i, (j+1)%N), ((i+1)%N, j), (i, (j-1)%N)]
            deltaE = 2.0 * s * sum(lattice[ni, nj] for ni, nj in neighbors)
            if deltaE < 0 or np.random.random() < np.exp(-deltaE / T):
                lattice[i, j] = -s
    return lattice


def simulate(N: int, T: float, nsweeps: int, burn_frac: float = 0.2):
    """
    Run Metropolis simulation for one temperature.
    Returns (mean_mag, std_mag, mean_energy, std_energy).
    """
    lattice = np.random.choice([-1, 1], size=(N, N))
    burn = int(nsweeps * burn_frac)
    m_vals = []
    e_vals = []
    for step in range(nsweeps):
        lattice = metropolis_sweep(lattice, T)
        if step >= burn:
            m = np.abs(np.mean(lattice))
            e = energy(lattice) / (N * N)
            m_vals.append(m)
            e_vals.append(e)
    return np.mean(m_vals), np.std(m_vals), np.mean(e_vals), np.std(e_vals)


if __name__ == "__main__":
    # Example: run a small temperature sweep for N=16
    T_range = np.linspace(2.0, 2.5, 5)
    T_arr, mag_arr, chi_arr, binder_arr, energy_arr = scan_T(16, T_range, nsweeps=2000)
    print("T:", T_arr)
    print("mag:", mag_arr)
    print("chi:", chi_arr)
    print("binder:", binder_arr)
    print("energy:", energy_arr)