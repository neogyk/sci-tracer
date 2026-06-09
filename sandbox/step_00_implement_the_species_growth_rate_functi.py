import numpy as np

def SpeciesGrowth(spc, res, b, c, w, m):
    """Calculate species growth rate.
    Inputs:
    spc: current species abundance (unused but included for interface consistency)
    res: resource level, 1D array of length R
    b: inverse timescale of species dynamics, 1D array of length N
    c: consumer-resource conversion matrix, 2D array of shape [N, R]
    w: value/efficiency of resource, 1D array of length R
    m: species maintenance cost, 1D array of length N
    Outputs:
    g_spc: growth rate of species, 1D array of length N
    """
    # Compute the weighted sum of resources: c @ (w * res) -> (N,)
    weighted_res = np.dot(c, w * res)
    # Subtract maintenance cost and multiply by inverse timescale
    g_spc = b * (weighted_res - m)
    return g_spc