import numpy as np
from functools import reduce

def multi_rail_encoding_state(rails: int) -> np.ndarray:
    """
    Construct the density matrix of a multi-rail encoding state for m rails.

    The state is defined as:
        |ψ_m⟩ = (1/√m) Σ_{k=0}^{m-1} |e_k⟩_A ⊗ |e_k⟩_B,
    where |e_k⟩ is an m-qubit basis vector with a single excitation at qubit k.

    Parameters
    ----------
    rails : int
        Number of rails (m). Must be a positive integer.

    Returns
    -------
    np.ndarray
        A 2^(2m) x 2^(2m) numpy array (float64) representing the density matrix.
    """
    m = rails
    if m <= 0:
        raise ValueError("Number of rails must be a positive integer.")

    # Single-qubit basis states
    zero = np.array([1.0, 0.0])
    one  = np.array([0.0, 1.0])

    # Accumulator for the unnormalized state vector
    state_vec = np.zeros(2**(2*m), dtype=np.float64)

    for k in range(m):
        # Build the m-qubit basis vector |e_k> for one side
        qubits = [one if i == k else zero for i in range(m)]
        e_k = reduce(np.kron, qubits)   # shape (2^m,)
        # Tensor product of two copies: |e_k>_A ⊗ |e_k>_B
        bipartite = np.kron(e_k, e_k)   # shape (2^(2m),)
        state_vec += bipartite

    # Normalize
    state_vec /= np.sqrt(m)

    # Outer product to get density matrix
    rho = np.outer(state_vec, state_vec)
    return rho