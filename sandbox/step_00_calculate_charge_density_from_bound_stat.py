import numpy as np

def sort_states(states):
    """Sort bound states by energy (ascending) then by angular momentum l (ascending)."""
    # Placeholder: actual implementation should follow the provided sort_states function.
    # This function must be defined elsewhere in the larger codebase.
    return sorted(states, key=lambda x: (x[0], x[1]))

def compute_Schrod(r_grid, energy, l):
    """
    Solve radial Schrödinger equation for given energy and l using Numerov method.
    Returns normalized u(r) = r * R(r).
    """
    # Placeholder: actual implementation should follow the previously defined compute_Schrod function.
    # This function must be defined elsewhere in the larger codebase.
    # For demonstration, return zeros.
    return np.zeros_like(r_grid)

def calculate_charge_density(bound_states, r_grid, Z):
    """
    Calculate spherically averaged electron charge density from bound states.
    
    Parameters:
        bound_states : list of (energy, l) tuples
        r_grid       : 1D numpy array of radial coordinates
        Z            : atomic number (total number of electrons)
    
    Returns:
        charge_density : 1D numpy array of same length as r_grid, n(r)
    """
    sorted_states = sort_states(bound_states)
    charge_density = np.zeros_like(r_grid)
    total_filled = 0
    
    for state in sorted_states:
        energy = state[0]
        l = int(state[1])
        deg = 2 * (2 * l + 1)          # degeneracy of subshell
        remaining = Z - total_filled
        
        if remaining <= 0:
            break
        
        # Determine filling number (Fermi factor: partially fill if remaining < deg)
        if remaining >= deg:
            fill = deg
            total_filled += deg
        else:
            fill = remaining
            total_filled += remaining
        
        # Compute radial wavefunction u(r) = r * R(r)
        u = compute_Schrod(r_grid, energy, l)
        
        # Contribution to spherically averaged electron density:
        # n(r) = fill * |R(r)|^2 / (4π) = fill * u^2 / (4π r^2)
        with np.errstate(divide='ignore', invalid='ignore'):
            contrib = fill * (u ** 2) / (4.0 * np.pi * r_grid ** 2)
            contrib[r_grid == 0] = 0.0   # u(0)=0 for all l, avoid 0/0
        
        charge_density += contrib
    
    return charge_density