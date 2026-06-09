import numpy as np

def hopping(d):
    """
    Moon–Koshino hopping integral for a displacement vector d (in bohr).
    Returns -t(d) in eV.
    """
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-12:
        return 0.0  # on-site energy assumed zero
    dz = d[2]
    ratio = (dz / d_norm) ** 2
    Vpppi = -2.7 * np.exp(-1.0 * (d_norm - 2.68))
    Vppsigma = 0.48 * np.exp(-1.0 * (d_norm - 6.33))
    return Vpppi * (1 - ratio) + Vppsigma * ratio

def build_hamiltonian(k, latvecs, basis):
    """
    Construct the 4x4 tight-binding Hamiltonian for AA-stacked bilayer graphene.
    
    Parameters:
        k : (kx, ky) tuple of floats – in-plane wavevector (units: bohr^-1).
        latvecs : 3x3 array – lattice vectors (bohr). Only latvecs[0] and latvecs[1] are used.
        basis : Nx3 array – atomic positions within the unit cell (bohr).
    
    Returns:
        H : 4x4 complex Hermitian matrix.
    """
    n = len(basis)
    H = np.zeros((n, n), dtype=complex)
    k3d = np.array([k[0], k[1], 0.0])
    cutoff = 7.0  # bohr
    
    for i in range(n):
        for j in range(n):
            for n1 in range(-1, 2):
                for n2 in range(-1, 2):
                    R = n1 * latvecs[0] + n2 * latvecs[1]
                    d = basis[j] + R - basis[i]
                    dist = np.linalg.norm(d)
                    if dist < cutoff and dist > 1e-6:
                        t = hopping(d)
                        phase = np.exp(1j * np.dot(k3d, d))
                        H[i, j] += t * phase
    # Enforce Hermiticity as specified in the plan
    return H + H.conj().T - np.diag(H.diagonal())