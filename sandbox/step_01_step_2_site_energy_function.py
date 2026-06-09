def site_energy(lattice, i, j):
    """Compute energy of a single spin site given the lattice and coupling J=1."""
    J = 1
    neighbors = neighbor_list(i, j, lattice.shape[0])
    return -J * lattice[i, j] * sum(lattice[ni, nj] for ni, nj in neighbors)