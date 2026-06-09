import numpy as np

# Set up energy grid and boundary conditions
energies = np.arange(0, Emax + Estep/2, Estep)  # inclusive of Emax
x_out = 10 * np.sqrt(Emax)                     # integration limit
parity = 0                                      # 0 for even states (0 nodes)
# Initial conditions for even states: u(0)=1, u'(0)=0
u_init = (1.0, 0.0)