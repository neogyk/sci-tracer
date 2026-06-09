import numpy as np
from scipy.optimize import fminbound

# ----- Partial trace and permutation -----
def syspermute(X, perm, dims):
    """
    Permute subsystems of a density matrix X.
    X: (total_dim, total_dim) matrix
    perm: list of new order for subsystems
    dims: dimensions of each subsystem
    Returns permuted matrix.
    """
    n = len(dims)
    total_dim = int(np.prod(dims))
    X = X.reshape(dims + dims)  # shape (d0,d1,...,d0,d1,...)
    # Transpose indices: first n indices go to new order for rows, next n for cols
    axes = perm + [p + n for p in perm]
    X_perm = np.transpose(X, axes)
    # Reshape back to matrix
    new_dims = [dims[i] for i in perm]
    new_total = int(np.prod(new_dims))
    X_perm = X_perm.reshape(new_total, new_total)
    return X_perm

def partial_trace(X, sys, dim):
    """
    Partial trace over subsystems specified by sys.
    X: (total_dim, total_dim) matrix
    sys: list of subsystem indices to trace (0-based)
    dim: list of subsystem dimensions
    Returns reduced density matrix.
    """
    n = len(dim)
    keep = [i for i in range(n) if i not in sys]
    trace = [i for i in range(n) if i in sys]
    order = keep + trace
    dims_perm = [dim[i] for i in order]
    X_perm = syspermute(X, order, dim)
    dim_keep = int(np.prod(dims_perm[:len(keep)]))
    dim_trace = int(np.prod(dims_perm[len(keep):]))
    X_reshaped = X_perm.reshape(dim_keep, dim_trace, dim_keep, dim_trace)
    result = np.trace(X_reshaped, axis1=1, axis2=3)
    return result


# ----- GADC Kraus operators -----
def gadc_kraus(gamma, nbar):
    if nbar == 0:
        p = 0.0
    else:
        p = nbar / (2 * nbar + 1)
    sq_g = np.sqrt(1 - gamma)
    K0 = np.sqrt(1 - p) * np.array([[1, 0], [0, sq_g]])
    K1 = np.sqrt(1 - p) * np.array([[0, np.sqrt(gamma)], [0, 0]])
    K2 = np.sqrt(p) * np.array([[sq_g, 0], [0, 1]])
    K3 = np.sqrt(p) * np.array([[0, 0], [np.sqrt(gamma), 0]])
    return [K0, K1, K2, K3]


# ----- Helper functions -----
def entropy(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-12]
    return -np.sum(evals * np.log2(evals))

def apply_channel(rho, kraus):
    out = np.zeros_like(rho)
    for K in kraus:
        out += K @ rho @ K.conj().T
    return out

def purification(rho):
    """
    Purify a single‑qubit state ρ using a reference qubit.
    |ψ⟩ = (√ρ ⊗ I) |Φ⁺⟩, where |Φ⁺⟩ = (|00⟩+|11⟩)/√2.
    Returns the 4×4 projector |ψ⟩⟨ψ|.
    """
    evals, evecs = np.linalg.eigh(rho)
    sqrt_rho = evecs @ np.diag(np.sqrt(np.maximum(evals, 0))) @ evecs.T
    sqrt_rho_I = np.kron(sqrt_rho, np.eye(2))
    # Projector onto |Φ⁺⟩ (not normalised to 1, but the formula works)
    Omega = np.array([[0.5, 0, 0, 0.5],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0.5, 0, 0, 0.5]], dtype=complex)
    psi_mat = sqrt_rho_I @ Omega @ sqrt_rho_I.T
    return psi_mat


# ----- Coherent information maximisation -----
def gadc_coh_info(gamma, nbar):
    kraus = gadc_kraus(gamma, nbar)
    def f(lam):
        rho = np.array([[1 - lam, 0], [0, lam]], dtype=complex)
        rho_out = apply_channel(rho, kraus)
        S_out = entropy(rho_out)
        psi = purification(rho)
        sigma = np.zeros((4, 4), dtype=complex)
        for K in kraus:
            K_I = np.kron(K, np.eye(2))
            sigma += K_I @ psi @ K_I.conj().T
        S_sigma = entropy(sigma)
        return -(S_out - S_sigma)   # minimise negative coherent information
    res = fminbound(f, 0, 1, full_output=1)
    max_coh = -res[1]
    return max_coh


# ----- Example computation -----
if __name__ == "__main__":
    gamma_example = 0.3
    nbar_example = 1.0
    result = gadc_coh_info(gamma_example, nbar_example)
    print(f"Coherent information (γ={gamma_example}, n̄={nbar_example}): {result:.6f} bits")