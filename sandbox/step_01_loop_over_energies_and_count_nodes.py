import numpy as np
from typing import Tuple, Callable

# Precondition: energies (np.ndarray), x (np.ndarray), 
# solve_schrodinger(x, E, u_init) -> np.ndarray, 
# count_zero_crossings(u) -> int are defined.

def loop_and_count_nodes(
    energies: np.ndarray,
    x: np.ndarray,
    solve_schrodinger: Callable[[np.ndarray, float, Tuple[float, float]], np.ndarray],
    count_zero_crossings: Callable[[np.ndarray], int]
) -> np.ndarray:
    """
    For each trial energy, compute the wavefunction using Numerov integration
    and count its zero crossings (node count).
    The boundary condition at x=0 is chosen based on the expected parity,
    which alternates as node count increases.
    """
    node_counts = np.zeros_like(energies, dtype=int)
    # Start with even parity (ground state, 0 nodes)
    expected_nodes = 0  

    for i, E in enumerate(energies):
        # Set initial condition: even -> u(0)=1, u'(0)=0; odd -> u(0)=0, u'(0)=1
        if expected_nodes % 2 == 0:
            u_init = (1.0, 0.0)
        else:
            u_init = (0.0, 1.0)

        # Compute wavefunction
        u = solve_schrodinger(x, E, u_init)

        # Count nodes (zero crossings, excluding near-zero endpoints)
        nodes = count_zero_crossings(u)
        node_counts[i] = nodes

        # Update expected node count for the next trial energy
        expected_nodes = nodes

    return node_counts