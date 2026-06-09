def metropolis_sweep(lattice, T):
    N = lattice.shape[0]
    for i in range(N):
        for j in range(N):
            s = lattice[i,j]
            neighbors = [((i-1)%N, j), (i, (j+1)%N), ((i+1)%N, j), (i, (j-1)%N)]
            dE = 2 * s * sum(lattice[ni,nj] for ni,nj in neighbors)
            if dE < 0 or np.random.random() < np.exp(-dE/T):
                lattice[i,j] = -s
    return lattice