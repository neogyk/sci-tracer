import numpy as np

def bc(ya: np.ndarray, yb: np.ndarray, R1: float, R2: float, Ppl: float, Ppr: float) -> np.ndarray:
    """
    Boundary conditions for a Yb-doped double-clad fiber laser BVP.
    
    At z=0:
        ya[0] - Ppl = 0  (forward pump equals injected left pump)
        ya[3] - R1 * ya[2] = 0  (backward signal reflected from forward signal)
    At z=L:
        yb[1] - Ppr = 0  (backward pump equals injected right pump)
        yb[2] - R2 * yb[3] = 0  (forward signal reflected from backward signal)
    
    Parameters:
        ya : np.ndarray
            State vector at z=0: [Pp+, Pp-, Ps+, Ps-]
        yb : np.ndarray
            State vector at z=L: [Pp+, Pp-, Ps+, Ps-]
        R1 : float
            Reflectivity of input mirror (at z=0)
        R2 : float
            Reflectivity of output mirror (at z=L)
        Ppl : float
            Injected pump power at left end (forward direction)
        Ppr : float
            Injected pump power at right end (backward direction)
    
    Returns:
        np.ndarray
            Residuals of the boundary conditions (4 elements).
    """
    return np.array([
        ya[0] - Ppl,            # forward pump at z=0 equals left pump
        ya[3] - R1 * ya[2],     # backward signal = R1 * forward signal at z=0
        yb[1] - Ppr,            # backward pump at z=L equals right pump
        yb[2] - R2 * yb[3]      # forward signal = R2 * backward signal at z=L
    ])