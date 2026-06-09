import numpy as np

def grad_div(
    A_x: np.ndarray,
    A_y: np.ndarray,
    A_z: np.ndarray,
    delta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the gradient of divergence of a 3D vector field on a uniform
    Cartesian grid using second-order finite difference stencils.

    Parameters
    ----------
    A_x, A_y, A_z : np.ndarray
        Components of the vector field, each of shape (nx, ny, nz) with
        nx >= 3, ny >= 3, nz >= 3.
    delta : float
        Grid spacing (uniform in all directions).

    Returns
    -------
    grad_div_x, grad_div_y, grad_div_z : np.ndarray
        Components of ∇(div A), same shape as inputs. Boundary values
        are set to zero.
    """
    nx, ny, nz = A_x.shape
    inv_delta2 = 1.0 / (delta * delta)
    inv_4delta2 = 1.0 / (4.0 * delta * delta)

    # Pure second derivatives (three‑point central)
    # ∂²A_x/∂x² : interior indices i = 1..nx-2, j = 1..ny-2, k = 1..nz-2
    d2Ax_dx2 = (
        A_x[2:, 1:-1, 1:-1] - 2.0 * A_x[1:-1, 1:-1, 1:-1] + A_x[:-2, 1:-1, 1:-1]
    ) * inv_delta2
    # ∂²A_y/∂y²
    d2Ay_dy2 = (
        A_y[1:-1, 2:, 1:-1] - 2.0 * A_y[1:-1, 1:-1, 1:-1] + A_y[1:-1, :-2, 1:-1]
    ) * inv_delta2
    # ∂²A_z/∂z²
    d2Az_dz2 = (
        A_z[1:-1, 1:-1, 2:] - 2.0 * A_z[1:-1, 1:-1, 1:-1] + A_z[1:-1, 1:-1, :-2]
    ) * inv_delta2

    # Mixed second derivatives (four‑point formula)
    # ∂²A_y/∂x∂y
    d2Ay_dxdy = (
        A_y[2:, 2:, 1:-1] - A_y[:-2, 2:, 1:-1] -
        A_y[2:, :-2, 1:-1] + A_y[:-2, :-2, 1:-1]
    ) * inv_4delta2
    # ∂²A_z/∂x∂z
    d2Az_dxdz = (
        A_z[2:, 1:-1, 2:] - A_z[:-2, 1:-1, 2:] -
        A_z[2:, 1:-1, :-2] + A_z[:-2, 1:-1, :-2]
    ) * inv_4delta2
    # ∂²A_x/∂y∂x (symmetry: ∂²A_x/∂y∂x = ∂²A_x/∂x∂y)
    d2Ax_dydx = (
        A_x[2:, 2:, 1:-1] - A_x[:-2, 2:, 1:-1] -
        A_x[2:, :-2, 1:-1] + A_x[:-2, :-2, 1:-1]
    ) * inv_4delta2
    # ∂²A_z/∂y∂z
    d2Az_dydz = (
        A_z[1:-1, 2:, 2:] - A_z[1:-1, :-2, 2:] -
        A_z[1:-1, 2:, :-2] + A_z[1:-1, :-2, :-2]
    ) * inv_4delta2
    # ∂²A_x/∂z∂x
    d2Ax_dzdx = (
        A_x[2:, 1:-1, 2:] - A_x[:-2, 1:-1, 2:] -
        A_x[2:, 1:-1, :-2] + A_x[:-2, 1:-1, :-2]
    ) * inv_4delta2
    # ∂²A_y/∂z∂y
    d2Ay_dzdy = (
        A_y[1:-1, 2:, 2:] - A_y[1:-1, :-2, 2:] -
        A_y[1:-1, 2:, :-2] + A_y[1:-1, :-2, :-2]
    ) * inv_4delta2

    # Initialize output arrays (boundaries will remain zero)
    grad_div_x = np.zeros_like(A_x)
    grad_div_y = np.zeros_like(A_y)
    grad_div_z = np.zeros_like(A_z)

    # Fill interior values
    grad_div_x[1:-1, 1:-1, 1:-1] = d2Ax_dx2 + d2Ay_dxdy + d2Az_dxdz
    grad_div_y[1:-1, 1:-1, 1:-1] = d2Ax_dydx + d2Ay_dy2 + d2Az_dydz
    grad_div_z[1:-1, 1:-1, 1:-1] = d2Ax_dzdx + d2Ay_dzdy + d2Az_dz2

    return grad_div_x, grad_div_y, grad_div_z