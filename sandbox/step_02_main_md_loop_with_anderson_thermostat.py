import numpy as np
from typing import Tuple, Optional

def compute_temperature(velocities: np.ndarray, m: float) -> float:
    """
    Compute instantaneous temperature from velocities.
    Units: velocities in nm/ps, mass in g/mol -> convert to kg for energy.
    kB = 0.008314462618 kJ/(mol*K) but we use velocities in nm/ps.
    For consistency, we use kB_nm_ps = 0.008314462618 * (1e-12 / (1e-9)**2) ??? 
    Simpler: use standard kB in J/K, convert mass to kg.
    mass_kg = m * 1.66053906660e-27
    kB = 1.380649e-23
    Then kinetic energy per particle = 0.5 * m_kg * (v_nm_ps * 1e3)^2 (since 1 nm/ps = 1000 m/s)
    => KE = 0.5 * m_kg * (v * 1000)^2 = 0.5 * m_kg * 1e6 * v^2
    Temperature = (2/3) * KE_avg / kB  (per particle, averaged over N)
    """
    N = velocities.shape[0]
    m_kg = m * 1.66053906660e-27  # mass per particle in kg
    kB = 1.380649e-23            # Boltzmann constant in J/K
    # Convert velocities from nm/ps to m/s
    v_m_per_s = velocities * 1000.0
    # Kinetic energy per particle in J
    KE_per_particle = 0.5 * m_kg * np.sum(v_m_per_s**2, axis=1)
    # Average kinetic energy
    KE_avg = np.mean(KE_per_particle)
    # Temperature: KE_avg = (3/2) kB T
    T = (2.0/3.0) * KE_avg / kB
    return T

def run_simulation(
    N: int,
    init_positions: np.ndarray,
    init_velocities: np.ndarray,
    L: float,
    sigma: float,
    epsilon: float,
    rc: float,
    m: float,
    dt: float,
    num_steps: int,
    T: float,
    nu: float,
    store_energies: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run molecular dynamics simulation with Anderson thermostat.
    
    Parameters:
    - N: number of particles
    - init_positions: (N,3) array of initial positions (nm)
    - init_velocities: (N,3) array of initial velocities (nm/ps)
    - L: side length of cubic box (nm)
    - sigma, epsilon: Lennard-Jones parameters (nm, kJ/mol)
    - rc: cutoff distance (nm)
    - m: mass per particle (g/mol)
    - dt: time step (ps)
    - num_steps: number of integration steps
    - T: target temperature (K)
    - nu: collision frequency for Anderson thermostat (1/ps)
    - store_energies: if True, store potential and kinetic energy histories
    
    Returns:
    - final_positions: (N,3) array
    - final_velocities: (N,3) array
    - energy_history: (num_steps, 3) if store_energies else None, columns: [time, potential, kinetic]
    """
    positions = init_positions.copy()
    velocities = init_velocities.copy()
    
    if store_energies:
        energy_history = np.zeros((num_steps, 3))
    else:
        energy_history = None
    
    for step in range(num_steps):
        # Velocity Verlet step
        positions, velocities = velocity_verlet(N, sigma, epsilon, positions, velocities, dt, m, L, rc)
        
        # Anderson thermostat
        velocities = anderson_thermostat(velocities, dt, nu, T, m)
        
        if store_energies:
            # Compute potential energy
            pot_energy = total_energy(positions, N, L, rc, sigma, epsilon, m)
            # Compute kinetic energy from velocities
            # Using same temperature formula: KE_total = (3/2)*N*kB*T
            # But we can compute directly (in kJ/mol)
            # Kinetic energy per particle in kJ/mol: 0.5 * m * v^2 (with v in nm/ps, m g/mol)
            # But kB in kJ/(mol*K) is 0.008314462618, so we can avoid mass conversion.
            # Actually, for consistency with potential (kJ/mol), use mass in g/mol and velocities in nm/ps.
            # Then KE in kJ/mol = 0.5 * m * v^2, but need factor? 
            # Standard conversion: 1 J/mol = 1e-3 kJ/mol, and 1 (m/s)^2 = 1e12 (nm/ps)^2? Better to use standard constants.
            # Since we have temperature already, we can compute KE_avg from temperature.
            # But we can compute directly using m and v: in molecular dynamics units, 
            # if mass is in g/mol, velocities in nm/ps, then KE = 0.5 * m * sum(v**2) (kJ/mol)?
            # Check: m in g/mol, v in nm/ps: energy in (g/mol)*(nm/ps)^2 = (1e-3 kg/mol)*(1e-9 m / 1e-12 s)^2 = (1e-3)*(1e3)^2 = 1e3 kg m^2/s^2 / mol = 1e3 J/mol = 1 kJ/mol. 
            # So indeed KE in kJ/mol = 0.5 * m * sum(v**2) (for each particle). To get total, sum over particles.
            # We'll use this to avoid temperature conversion.
            v_sq = np.sum(velocities**2, axis=1)
            kin_energy_per_particle = 0.5 * m * v_sq  # kJ/mol per particle
            kin_energy_total = np.sum(kin_energy_per_particle)  # kJ/mol total
            # Potential energy already in kJ/mol from total_energy
            # Convert total kinetic to kJ/mol (already)
            energy_history[step, 0] = step * dt
            energy_history[step, 1] = pot_energy
            energy_history[step, 2] = kin_energy_total
    
    return positions, velocities, energy_history


# Example test call (optional, for validation)
if __name__ == "__main__":
    # Test parameters (Argon-like)
    N = 108
    # Generate simple cubic initial positions (for testing) or FCC? Use simple cubic for simplicity.
    L = 3.0  # nm
    # Place particles on a 3x3x3 grid (27 points, not 108). Need 108: 6x6x3? Actually 108 = 4*27? For demonstration, use random positions.
    np.random.seed(42)
    positions = np.random.uniform(0, L, (N,3))
    # Initial velocities from Maxwell-Boltzmann at T=100K
    m_Ar = 39.948  # g/mol
    kB = 0.008314462618  # kJ/(mol*K)
    sigma = 3.405e-1  # nm? Actually 0.3405 nm
    epsilon = 0.996  # kJ/mol
    sigma = 0.345  # nm (approximate)
    epsilon = 1.0  # for test
    rc = 2.5 * sigma
    dt = 0.001  # ps
    T_target = 100.0  # K
    nu = 0.1  # 1/ps
    # Generate initial velocities: v = sqrt(kB*T/m) * random Gaussian
    sigma_v = np.sqrt(kB * T_target / m_Ar)  # nm/ps? Check units: kB in kJ/(mol*K), m in g/mol -> nm^2/ps^2? 
    # Actually kB * T = energy per mol in kJ/mol, divide by m (g/mol) gives (kJ/g) but need J/kg for sqrt.
    # Simpler: use the earlier conversion with m_kg. We'll just assign random velocities with appropriate scale.
    # For this test, just set small random velocities.
    velocities = np.random.normal(0, 0.1, (N,3))
    
    # We need to have velocity_verlet and anderson_thermostat defined.
    # Assuming they are defined elsewhere; if not, define dummy for test.
    # For completeness, we provide dummy implementations.
    def velocity_verlet(N, sigma, epsilon, positions, velocities, dt, m, L, rc):
        # Dummy: just return unchanged (test only)
        return positions, velocities
    
    def anderson_thermostat(velocities, dt, nu, T, m):
        # Dummy
        return velocities
    
    def total_energy(positions, N, L, rc, sigma, epsilon, m):
        # Dummy
        return 0.0
    
    # Run simulation
    final_positions, final_velocities, energy_hist = run_simulation(
        N, positions, velocities, L, sigma, epsilon, rc, m_Ar, dt, num_steps=100, T=T_target, nu=nu, store_energies=True
    )
    print("Simulation completed.")
    print("Final positions shape:", final_positions.shape)
    print("Final velocities shape:", final_velocities.shape)
    if energy_hist is not None:
        print("Energy history shape:", energy_hist.shape)