import numpy as np
from typing import List, Tuple

def compute_Q_vectors(
    peaks: List[Tuple[float, float, int]],
    detector_params: dict
) -> List[np.ndarray]:
    """
    Compute momentum transfer vectors Q (in 1/Å) for a list of peaks.
    
    Parameters
    ----------
    peaks : list of (x_pixel, y_pixel, frame_number)
        Detector pixel coordinates and frame number (frame number is not used in Q computation but passed for compatibility).
    detector_params : dict
        Dictionary with keys:
        - 'wavelength' : float (Å)
        - 'distance'   : float (mm) sample-detector distance
        - 'pix_size_x' : float (mm) pixel size in x direction
        - 'pix_size_y' : float (mm) pixel size in y direction
        - 'beam_center_x' : float (pixels)
        - 'beam_center_y' : float (pixels)
        It is assumed that the detector is flat and perpendicular to the incident beam.
        The incident beam direction is along +z (from sample to detector).
        k_in = (0, 0, 2π/λ)
        k_out is from sample to pixel.
        Q = k_out - k_in.
    
    Returns
    -------
    list of numpy arrays, each of shape (3,) containing the Q vector in Å⁻¹.
    """
    wavelength = detector_params['wavelength']
    distance = detector_params['distance']
    pix_x = detector_params['pix_size_x']
    pix_y = detector_params['pix_size_y']
    cx = detector_params['beam_center_x']
    cy = detector_params['beam_center_y']
    
    k_in = np.array([0.0, 0.0, 2.0 * np.pi / wavelength])
    
    Q_vectors = []
    for x_pixel, y_pixel, _ in peaks:
        # Convert pixel coordinates to detector Cartesian coordinates (mm)
        x = (x_pixel - cx) * pix_x
        y = (y_pixel - cy) * pix_y
        z = distance
        
        # Vector from sample to pixel (lab frame)
        r = np.array([x, y, z])
        norm = np.linalg.norm(r)
        if norm == 0:
            raise ValueError("Pixel at beam center? norm = 0")
        
        k_out = (2.0 * np.pi / wavelength) * (r / norm)
        Q = k_out - k_in
        Q_vectors.append(Q)
    
    return Q_vectors