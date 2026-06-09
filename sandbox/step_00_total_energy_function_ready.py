import numpy as np

def energy(lattice):
    N = lattice.shape[0]
    total = 0.0
    for i in range(N):
        for j in range(N):
            neighbors = [((i-1)%N, j), (i, (j+1)%N), ((i+1)%N, j), (i, (j-1)%N)]
            for ni, nj in neighbors:
                total += -1.0 * lattice[i,j] * lattice[ni,nj]
    return total / 2.0  # each bond counted twice