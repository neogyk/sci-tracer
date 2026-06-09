import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

def prepare_operator(N: int, q0: float):
    kx = 2 * np.pi * fftfreq(N)
    ky = 2 * np.pi * fftfreq(N)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K_sq = KX**2 + KY**2
    L_op = -(1 - q0**(-2) * K_sq)**2
    return K_sq, L_op

def solve_SH(u: np.ndarray, dt: float, T: float, N: int, epsilon: float, q0: float) -> np.ndarray:
    _, L_op = prepare_operator(N, q0)
    num_steps = int(T // dt)
    for _ in range(num_steps):
        # Nonlinear step in real space
        u_nl = u + dt * (epsilon * u - u**3)
        # Linear step in Fourier space
        u_hat = fft2(u_nl)
        u_hat = u_hat * np.exp(dt * L_op)
        u = np.real(ifft2(u_hat))
    return u