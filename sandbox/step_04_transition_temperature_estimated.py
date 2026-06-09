import numpy as np

def estimate_Tc(N_list=[16, 32, 64, 128],
                T_range=np.linspace(2.0, 2.5, 50),
                nsweeps=5000):
    """
    Estimate the critical temperature Tc of the 2D Ising model from Binder
    cumulant crossing and finite-size scaling.

    Parameters
    ----------
    N_list : list of int
        Lattice sizes (linear dimension). Must contain at least two sizes.
    T_range : 1D array
        Temperatures at which the simulation is performed.
    nsweeps : int
        Number of Monte Carlo sweeps for each temperature.

    Returns
    -------
    Tc_est : float
        Estimated critical temperature.
    """
    if len(N_list) < 2:
        raise ValueError("Need at least two lattice sizes.")

    # Run simulation for each size and store binder cumulant curves
    Ts = T_range
    binder_curves = {}
    for N in N_list:
        _, _, _, binders, _ = scan_T(N, Ts, nsweeps)
        binder_curves[N] = np.array(binders)

    # Find crossing temperatures for consecutive size pairs
    crossing_Ts = []
    eff_lengths = []
    for i in range(len(N_list) - 1):
        N1 = N_list[i]
        N2 = N_list[i+1]
        b1 = binder_curves[N1]
        b2 = binder_curves[N2]
        diff = b1 - b2

        # Detect sign changes in the difference
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) == 0:
            print(f"No crossing found between N={N1} and N={N2}.")
            continue

        # Use the first crossing (the one closest to the critical region)
        idx = sign_changes[0]
        T_low, T_high = Ts[idx], Ts[idx+1]
        d_low, d_high = diff[idx], diff[idx+1]

        # Linear interpolation to find root
        T_cross = T_low - d_low * (T_high - T_low) / (d_high - d_low)
        crossing_Ts.append(T_cross)
        eff_lengths.append(np.sqrt(N1 * N2))   # geometric mean

    if len(crossing_Ts) < 2:
        raise ValueError("Insufficient crossing points for a reliable fit.")

    # Finite-size scaling: Tc_est = intercept of linear fit in 1/L_eff^2
    x = 1.0 / np.array(eff_lengths)**2
    y = np.array(crossing_Ts)
    coeffs = np.polyfit(x, y, deg=1)
    Tc_est = coeffs[1]   # intercept
    return Tc_est


# Example usage (if run as main)
if __name__ == "__main__":
    # Ensure that the required functions (energy, metropolis_sweep, simulate, scan_T)
    # are already defined in the execution environment.
    try:
        Tc = estimate_Tc(N_list=[16, 32, 64], T_range=np.linspace(2.0, 2.5, 30), nsweeps=5000)
        print(f"Estimated Tc = {Tc:.4f} (expected Onsager value: 2.269)")
    except Exception as e:
        print(f"Error: {e}")