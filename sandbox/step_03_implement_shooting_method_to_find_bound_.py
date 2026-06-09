import numpy as np

def find_bound_states(x, Emax, Estep):
    """
    Vary energy E from 0 to Emax with step Estep and check boundary conditions to find eigenvalues.
    
    Parameters:
    x (array): spatial grid
    Emax (float): maximum energy to search
    Estep (float): energy step
    
    Returns:
    list of (n_nodes, energy) tuples for eigenvalues found.
    """
    # Set boundary conditions: u(-xmax)=0, u'(-xmax)=1 (arbitrary)
    xmax = x[-1]
    E_vals = np.arange(0, Emax, Estep)
    bound_states = []
    for En in E_vals:
        u_norm = Solve_Schrod(x, En, 0.0, 1.0, x[1]-x[0])
        # Check if wavefunction goes to zero at the far right boundary
        if abs(u_norm[-1]) < 1e-6:
            # Count sign changes to determine number of nodes
            signs = np.sign(u_norm)
            n_nodes = np.sum(signs[1:] != signs[:-1])
            bound_states.append((n_nodes, En))
    # Remove duplicates and sort by energy
    bound_states = sorted(set(bound_states), key=lambda x: x[1])
    return bound_states