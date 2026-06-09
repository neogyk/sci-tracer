import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union

def q_cal(
    p: Union[Tuple[float, float], NDArray[np.float64]],
    b_c: Union[Tuple[float, float], NDArray[np.float64]],
    det_d: float,
    p_s: float,
    wl: float
) -> NDArray[np.float64]:
    """
    Compute momentum transfer vector Q from detector pixel coordinates.
    
    Parameters
    ----------
    p : (x, y) coordinates of the detector pixel (in pixels, can be fractional).
    b_c : (x, y) coordinates of the beam center (in pixels).
    det_d : sample-to-detector distance (same unit as wavelength, e.g., Angstroms).
    p_s : pixel size (same unit as det_d, e.g., Angstroms per pixel).
    wl : wavelength of incident radiation (same unit as det_d).
    
    Returns
    -------
    Q : numpy array of shape (3, 1) representing the momentum transfer vector
        with components (Qx, Qy, Qz) in 1/wavelength unit (e.g., 1/Angstrom).
    """
    # Convert inputs to numpy arrays
    p = np.asarray(p, dtype=np.float64)
    b_c = np.asarray(b_c, dtype=np.float64)
    
    # Compute pixel offsets relative to beam center, scaled to physical distance
    dx = (p[0] - b_c[0]) * p_s
    dy = (p[1] - b_c[1]) * p_s
    
    # Norm of the scattered vector from sample to detector pixel
    norm = np.sqrt(det_d**2 + dx**2 + dy**2)
    
    # Wave number
    k = 1.0 / wl
    
    # Scattered wavevector (pointing from sample to detector pixel)
    ks = np.array([det_d, -dx, -dy]) / norm * k
    
    # Incident wavevector (along +x direction)
    ki = np.array([k, 0.0, 0.0])
    
    # Momentum transfer: Q = ks - ki
    Q = ks - ki
    
    # Reshape to (3, 1)
    return Q.reshape(3, 1)