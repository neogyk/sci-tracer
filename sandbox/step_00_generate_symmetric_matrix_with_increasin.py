def init_matrix(dim, noise):
    import numpy as np
    A = np.diag(np.arange(1, dim+1, dtype=float))
    for i in range(dim):
        for j in range(dim):
            A[i][j] *= np.random.normal() * noise
    A = (A + A.T) / 2
    return A