import numpy as np

def detector_pixel_vector(p: tuple[int, int], b_c: tuple[float, float], det_d: float, p_s: float) -> np.ndarray:
    """
    Compute the vector from sample to pixel in detector coordinates.

    Parameters
    ----------
    p : tuple of ints
        (x, y) pixel coordinates (integer indices).
    b_c : tuple of floats
        (xc, yc) beam center pixel coordinates (floats).
    det_d : float
        Sample-to-detector distance in mm.
    p_s : float
        Pixel size in mm.

    Returns
    -------
    np.ndarray
        3-element numpy array [dx, dy, det_d] representing the vector from
        sample to pixel in detector coordinates (units mm).
    """
    dx = (p[0] - b_c[0]) * p_s
    dy = (p[1] - b_c[1]) * p_s
    return np.array([dx, dy, det_d], dtype=np.float64)