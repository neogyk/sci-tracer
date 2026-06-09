def forces(N, xyz, L, sigma, epsilon, rc):
    '''Calculate net forces on each particle from LJ pair interactions with minimum image convention.'''
    import numpy as np
    f_xyz = np.zeros((N, 3))
    for i in range(N):
        for j in range(i+1, N):
            dr = xyz[j] - xyz[i]
            dr = dr - L * np.round(dr / L)   # minimum image vector
            r = np.linalg.norm(dr)
            if r < rc:
                r2 = r * r
                r6 = r2 * r2 * r2
                r12 = r6 * r6
                f_mag = 4 * epsilon * (12 * sigma**12 / (r12 * r2) - 6 * sigma**6 / (r6 * r2))
                f_vec = f_mag * dr / r
                f_xyz[i] += f_vec
                f_xyz[j] -= f_vec
    return f_xyz