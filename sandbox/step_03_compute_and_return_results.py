import numpy as np

# Compute acceptance rate
accp_rate = n_accp / (n_eq + n_prod)

# Long-range correction for energy
rho = N / (L**3)
U_tail = 8 * np.pi * N * rho * epsilon * ((sigma/r_c)**9 / 9 - (sigma/r_c)**3 / 3)
E_array_corrected = E_array + U_tail

# Excess chemical potential from Widom insertion
if insertion_count > 0:
    mu_ext = -T * np.log(mu_ext_sum / insertion_count)
else:
    mu_ext = 0.0

# Return results
results = (accp_rate, E_array_corrected, mu_ext)