import numpy as np
import cmath

# Permittivity of free space e^2/(eV·µm) (in units: e^2 / (eV * µm))
EPS0 = 55.26349406

def sqrt_positive_imag(z):
    """Return sqrt(z) with Im(sqrt(z)) > 0, handling branch cut."""
    w = cmath.sqrt(z)
    if w.imag < 0:
        w = -w
    return w