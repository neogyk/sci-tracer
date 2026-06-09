import numpy as np
def Bmat(pa):
    a, b, c, alpha, beta, gamma = pa
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)
    # Direct lattice vectors in Cartesian (standard orientation)
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
    c1 = c * np.cos(beta)
    c2 = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    c3 = c * np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma))**2)
    a3 = np.array([c1, c2, c3])
    # Volume
    V = np.dot(a1, np.cross(a2, a3))
    # Reciprocal lattice vectors (without 2π)
    a_star = np.cross(a2, a3) / V
    b_star = np.cross(a3, a1) / V
    c_star = np.cross(a1, a2) / V
    # Orthonormal basis aligned with reciprocal axes
    u = a_star / np.linalg.norm(a_star)
    w = np.cross(a_star, b_star)
    w = w / np.linalg.norm(w)
    v = np.cross(w, u)
    # B matrix elements: row i = basis vector, col j = reciprocal vector
    B = np.array([
        [np.dot(a_star, u), np.dot(b_star, u), np.dot(c_star, u)],
        [np.dot(a_star, v), np.dot(b_star, v), np.dot(c_star, v)],
        [np.dot(a_star, w), np.dot(b_star, w), np.dot(c_star, w)]
    ])
    return B