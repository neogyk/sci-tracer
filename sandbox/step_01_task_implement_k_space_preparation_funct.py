import numpy as np
from numpy.fft import fftfreq

def prepare_operator(N, q0):
    """
    Prepare the squared wavenumber grid and the linear operator for the 
    Swift-Hohenberg equation in Fourier space.

    Parameters:
        N (int): Number of grid points in each spatial dimension.
        q0 (float): Critical wavenumber.

    Returns:
        tuple: (K_sq, L_op) where K_sq is the squared wavenumber grid and 
               L_op = -(1 - q0**(-2) * K_sq)**2.
    """
    kx = 2 * np.pi * fftfreq(N)
    ky = 2 * np.pi * fftfreq(N)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K_sq = KX**2 + KY**2
    L_op = -(1 - q0**(-2) * K_sq)**2
    return K_sq, L_op