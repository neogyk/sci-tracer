import numpy as np

def total_energy(lattice):
    N = lattice.shape[0]
    total = 0
    for i in range(N):
        for j in range(N):
            total += site_energy(lattice, i, j)
    return total / 2