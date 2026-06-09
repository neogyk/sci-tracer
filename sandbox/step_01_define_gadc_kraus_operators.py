import numpy as np

def gadc_kraus(gamma, nbar):
    """
    Return the four Kraus operators for the generalized amplitude damping channel.
    Parameters:
    gamma : float, damping parameter in [0,1]
    nbar  : float, mean thermal photon number (>=0)
    Returns:
    list of 2x2 numpy arrays (Kraus operators)
    """
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