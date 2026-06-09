import numpy as np

def multi_rail_encoding_state(rails: int) -> np.ndarray:
    """
    Construct the density matrix of a multi‑rail encoding state for 'rails' qudits.

    Parameters
    ----------
    rails : int
        Number of rails (dimension of the logical qudit). Must be positive.

    Returns
    -------
    np.ndarray
        Density matrix ρ of shape (2^(2*rails), 2^(2*rails)) and dtype float64.
    """
    m = rails
    if m <= 0:
        raise ValueError("rails must be a positive integer")

    # Single‑qubit basis vectors
    zero = np.array([1.0, 0.0], dtype=np.float64)  # |0⟩
    one  = np.array([0.0, 1.0], dtype=np.float64)  # |1⟩

    # Build all |e_k⟩ basis vectors (m‑qubit states with a single excitation)
    e_vectors = []
    for k in range(m):
        states = [one if i == k else zero for i in range(m)]
        ek = states[0]
        for s in states[1:]:
            ek = np.kron(ek, s)
        e_vectors.append(ek)

    # Bipartite entangled state |ψ⟩ = (1/√m) Σ_k |e_k⟩⊗|e_k⟩
    psi = sum(np.kron(ek, ek) for ek in e_vectors)
    psi /= np.sqrt(m)

    # Density matrix ρ = |ψ⟩⟨ψ|
    rho = np.outer(psi, psi)
    return rho