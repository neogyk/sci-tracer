import numpy as np

def compute_B(pa):
    """
    Compute the B matrix from lattice parameters.
    B is defined such that reciprocal lattice vector in Cartesian coordinates
    (without 2π factor) is given by q_cart = B * [h, k, l]^T,
    and d* = |B * [h,k,l]|.
    
    Parameters
    ----------
    pa : tuple
        (a, b, c, alpha, beta, gamma) where a, b, c are in angstroms,
        and angles in degrees.
    
    Returns
    -------
    B : ndarray of shape (3,3)
    """
    a, b, c, alpha, beta, gamma = pa
    
    # Convert angles to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    
    # Direct lattice vectors in Cartesian coordinates (columns of A)
    a_vec = np.array([a, 0.0, 0.0])
    
    b_vec = np.array([b * np.cos(gamma),
                      b * np.sin(gamma),
                      0.0])
    
    # Components of c vector
    c1 = c * np.cos(beta)
    c2 = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    c3 = c * np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 +
                     2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    c_vec = np.array([c1, c2, c3])
    
    # Direct lattice matrix (columns are direct basis vectors)
    A = np.column_stack((a_vec, b_vec, c_vec))
    
    # B matrix: B = (A^{-1})^T
    B = np.linalg.inv(A).T
    
    return B