import numpy as np

def potential_attractive(rnorm, z0, A):
    """
    Compute the attractive contribution of the Kolmogorov–Crespi interlayer potential.
    
    Parameters
    ----------
    rnorm : float or np.ndarray
        Distances between atoms.
    z0 : float
        Equilibrium distance.
    A : float
        Dispersion coefficient.
    
    Returns
    -------
    float or np.ndarray
        The attractive potential value(s) = -A * (z0 / rnorm)^6.
    """
    return -A * (z0 / rnorm) ** 6