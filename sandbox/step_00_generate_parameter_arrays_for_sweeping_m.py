import numpy as np

# Constants (should be provided externally; here we set example values)
N = 100  # number of grid points along each parameter direction
t2 = 0.1  # next-nearest-neighbor coupling constant

# Generate parameter arrays
m_ratio_values = np.linspace(-3, 3, N)
phi_values = np.linspace(-np.pi, np.pi, N)
m_values = m_ratio_values * t2