import numpy as np

def ham_eig(k_input, latvecs, basis):
    """
    Compute the sorted eigenvalues of the tight-binding Hamiltonian at a given k-point.

    Parameters:
    -----------
    k_input : array_like, shape (2,)
        The k-point in reciprocal space (kx, ky).
    latvecs : ndarray, shape (3, 3)
        Lattice vectors in real space, each row is a vector.
    basis : ndarray, shape (n, 3)
        Basis atom positions in fractional coordinates? Actually in Cartesian coordinates
        relative to the unit cell origin.

    Returns:
    --------
    eigenvalues : ndarray, shape (4,)
        Sorted eigenvalues in ascending order.
    """
    H = build_hamiltonian(k_input, latvecs, basis)
    eigvals = np.linalg.eigh(H)[0]
    return np.sort(eigvals)