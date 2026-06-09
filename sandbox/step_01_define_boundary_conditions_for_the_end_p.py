import numpy as np

def bc(ya: np.ndarray, yb: np.ndarray, Ppl: float, Ppr: float, R1: float, R2: float) -> np.ndarray:
    """
    Boundary conditions for an end-pumped fiber laser.
    ya: values at z=0 [forward_pump, backward_pump, forward_signal, backward_signal]
    yb: values at z=L [forward_pump, backward_pump, forward_signal, backward_signal]
    Ppl: launched forward pump power at z=0
    Ppr: launched backward pump power at z=L
    R1: reflectivity of mirror at z=0 for signal
    R2: reflectivity of mirror at z=L for signal
    Returns array of residuals that should be zero.
    """
    return np.array([
        ya[0] - Ppl,                 # forward pump at left = Ppl
        yb[1] - Ppr,                # backward pump at right = Ppr
        ya[3] - R1 * ya[2],         # backward signal at left = R1 * forward signal at left
        yb[2] - R2 * yb[3]          # forward signal at right = R2 * backward signal at right
    ])