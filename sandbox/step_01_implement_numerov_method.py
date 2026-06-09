import numpy as np

def Numerov(x: np.ndarray, En: float, u0: float, up0: float, step: float) -> np.ndarray:
    n = len(x)
    u = np.zeros(n)
    u[0] = u0
    h = step
    h2 = h**2
    f0 = f_x(x[0], En)
    u[1] = u0 + h * up0 + 0.5 * h2 * f0 * u0  # Taylor expansion for second point
    f = f_x(x, En)
    for i in range(1, n-1):
        u[i+1] = (2 * (1 - 5 * h2 / 12 * f[i]) * u[i] - (1 + h2 / 12 * f[i-1]) * u[i-1]) / (1 + h2 / 12 * f[i+1])
    return u