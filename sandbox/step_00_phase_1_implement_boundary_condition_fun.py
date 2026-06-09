import numpy as np

def add_dirichlet_bc(grid: np.ndarray, time_index: int, bc: np.ndarray = np.array([])) -> np.ndarray:
    """
    Apply Dirichlet boundary conditions to a given time slice of a 3D temperature grid,
    excluding corner grid points.

    Parameters
    ----------
    grid : np.ndarray
        3D array of shape (Nt, Ny, Nx) representing temperature over time and space.
    time_index : int
        The time index (0 <= time_index < Nt) at which to apply boundary conditions.
    bc : np.ndarray, optional
        Array of shape (M, 3) with columns (i, j, T). Only non-corner boundary cells
        are updated. Default is empty array.

    Returns
    -------
    np.ndarray
        The same grid object, modified in-place, with boundary conditions applied.
    """
    if bc.size == 0:
        return grid

    Ny, Nx = grid.shape[1], grid.shape[2]

    for i, j, T in bc:
        # Skip if both i and j are corner indices
        if (i == 0 or i == Ny - 1) and (j == 0 or j == Nx - 1):
            continue
        grid[time_index, i, j] = T

    return grid