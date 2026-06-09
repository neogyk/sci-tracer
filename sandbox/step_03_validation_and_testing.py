import numpy as np
from scipy.constants import k_B

def initialize_fcc_lattice(n_atoms: int, density: float) -> tuple:
    """Generate positions for an FCC lattice of n_atoms (must be 4*m^3)."""
    m = int(round((n_atoms / 4) ** (1/3)))
    a = (n_atoms / density) ** (1/3)  # box length in nm
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    positions = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for b in basis:
                    pos = (np.array([i, j, k], dtype=np.float64) + b) * a / m
                    positions.append(pos)
    return np.array(positions), a

def run_validation():
    # Argon parameters
    sigma = 0.3405          # nm
    epsilon = 1.654         # kJ/mol
    kB = 0.008314462618     # kJ/(mol·K)
    m = 39.948              # g/mol
    rc = 2.5 * sigma
    T_target = 100.0        # K
    nu = 0.1                # 1/ps
    dt = 0.001              # ps
    num_steps = 1000
    density = 0.021         # atoms/nm^3
    N = 108

    # Initialize positions
    initial_positions, L = initialize_fcc_lattice(N, density)

    # Initialize velocities from Maxwell-Boltzmann at target temperature
    sigma_v = np.sqrt(kB * T_target / m) * 0.001  # nm/ps
    initial_velocities = np.random.normal(0, sigma_v, (N, 3))
    initial_velocities -= np.mean(initial_velocities, axis=0)  # remove COM drift

    # Storage for temperature history
    temperature_trace = np.empty(num_steps)

    # Simulation loop using low‑level functions (precondition says they exist)
    positions = initial_positions.copy()
    velocities = initial_velocities.copy()
    for step in range(num_steps):
        # Velocity Verlet step
        positions, velocities = velocity_verlet(
            N, sigma, epsilon, positions, velocities, dt, m, L, rc
        )
        # Anderson thermostat
        velocities = anderson_thermostat(velocities, dt, nu, T_target, m)
        # Compute instantaneous temperature
        kinetic_energy = 0.5 * m * np.sum(velocities ** 2)   # kJ/mol
        T_inst = (2.0 / 3.0) * kinetic_energy / (N * kB)
        temperature_trace[step] = T_inst

    # Check average temperature over last 500 steps
    avg_T_last_500 = np.mean(temperature_trace[-500:])
    assert abs(avg_T_last_500 - T_target) < 5.0, \
        f"Average temperature {avg_T_last_500:.2f} K not within 5% of {T_target} K"
    print(f"Validation passed: average temperature = {avg_T_last_500:.2f} K")

if __name__ == "__main__":
    run_validation()