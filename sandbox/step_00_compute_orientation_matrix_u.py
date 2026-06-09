import numpy as np

def Umat(t_c: tuple[np.ndarray, np.ndarray, np.ndarray],
         t_g: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Compute the orientation matrix U that maps the crystal Cartesian basis t_c
    to the goniometer basis t_g.

    :param t_c: tuple of three 3x1 numpy arrays representing an orthonormal,
                right-handed crystal basis.
    :param t_g: tuple of three 3x1 numpy arrays representing an orthonormal,
                right-handed goniometer basis.
    :return: 3x3 orthogonal matrix U with determinant +1 satisfying
             t_g[i] = U @ t_c[i] for each i.
    """
    # Stack column vectors to form 3x3 matrices
    T_c = np.column_stack(t_c)
    T_g = np.column_stack(t_g)

    # Initial estimate (pure rotation if bases are orthonormal)
    U = T_g @ T_c.T

    # Enforce orthogonality via SVD (polar decomposition)
    U_svd, _, Vt = np.linalg.svd(U)
    U_orth = U_svd @ Vt

    # Ensure right-handed orientation (det = +1)
    if np.linalg.det(U_orth) < 0:
        U_orth[:, -1] *= -1

    return U_orth