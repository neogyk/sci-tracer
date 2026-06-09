import numpy as np

def check_constraint(E_x: np.ndarray, E_y: np.ndarray, E_z: np.ndarray, delta: float) -> float:
    """
    Compute the L2 norm of the divergence of the electric field on a uniform Cartesian grid.
    Uses second-order finite differences (central interior, one-sided at boundaries).
    
    Args:
        E_x: 3D array of x-component of electric field, shape (N,N,N)
        E_y: 3D array of y-component of electric field, shape (N,N,N)
        E_z: 3D array of z-component of electric field, shape (N,N,N)
        delta: uniform grid spacing
    
    Returns:
        float: L2 norm of constraint violation = sqrt( sum( (∇·E)^2 ) * delta^3 )
    """
    dEx_dx = np.gradient(E_x, delta, axis=0, edge_order=2)
    dEy_dy = np.gradient(E_y, delta, axis=1, edge_order=2)
    dEz_dz = np.gradient(E_z, delta, axis=2, edge_order=2)
    divE = dEx_dx + dEy_dy + dEz_dz
    constraint_violation = np.sqrt(np.sum(divE**2) * delta**3)
    return constraint_violation