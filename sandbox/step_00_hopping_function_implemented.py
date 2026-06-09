import numpy as np

def hopping(d):
    """
    Compute the hopping integral -t(d) using the Moon-Koshino parameterization.

    Parameters
    ----------
    d : numpy.ndarray of shape (3,)
        Displacement vector between two atoms in bohr.

    Returns
    -------
    float
        Hopping energy in eV.
    """
    d_norm = np.linalg.norm(d)
    dz = d[2]
    ratio = (dz / d_norm) ** 2
    Vpppi = -2.7 * np.exp(-1.0 * (d_norm - 2.68))      # V_{ppπ} in eV
    Vppsigma = 0.48 * np.exp(-1.0 * (d_norm - 6.33))   # V_{ppσ} in eV
    return Vpppi * (1 - ratio) + Vppsigma * ratio