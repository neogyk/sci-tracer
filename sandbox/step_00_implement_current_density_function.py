import numpy as np

def current_density(gw: float, g0: float, J0: float) -> float:
    """
    Calculates the current density J_w as a function of the gain coefficient g_w.

    Parameters
    ----------
    gw : float
        Gain coefficient.
    g0 : float
        Empirical gain coefficient.
    J0 : float
        Empirical factor in the current density equation.

    Returns
    -------
    Jw : float
        Current density J_w.
    """
    return J0 * np.exp((gw / g0) - 1)