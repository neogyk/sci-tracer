import numpy as np
from typing import Tuple

def heat_equation(
    Nt: int,
    Nx: int,
    Ny: int,
    x_split: int,
    T1: float,
    alpha1: float,
    T2: float,
    alpha2: float,
    bc_dirichlet: np.ndarray,
    bc_neumann: np.ndarray
) -> np.ndarray:
    # Stability condition: dt <= 1/(4*max(alpha))
    alpha_max = max(alpha1, alpha2)
    dt = 1.0 / (4.0 * alpha_max)
    
    # Initialize grid
    temp = np.zeros((Ny, Nx), dtype=np.float64)
    for j in range(Nx):
        if j < x_split:
            temp[:, j] = T1
        else:
            temp[:, j] = T2
    
    # Store all time steps
    temp_grid = np.zeros((Nt, Ny, Nx), dtype=np.float64)
    
    # Apply Neumann BCs to initial condition (zero gradient on all boundaries)
    # Left edge
    for i in range(Ny):
        temp[i, 0] = temp[i, 1]
    # Right edge
    for i in range(Ny):
        temp[i, Nx-1] = temp[i, Nx-2]
    # Top edge
    for j in range(Nx):
        temp[0, j] = temp[1, j]
    # Bottom edge
    for j in range(Nx):
        temp[Ny-1, j] = temp[Ny-2, j]
    
    # Apply Dirichlet BCs to initial condition
    for bc in bc_dirichlet:
        i, j, val = int(bc[0]), int(bc[1]), bc[2]
        if 0 <= i < Ny and 0 <= j < Nx:
            temp[i, j] = val
    temp_grid[0] = temp.copy()
    
    # Time stepping
    for t in range(Nt-1):
        new_temp = temp.copy()
        
        # Update interior points
        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                # Select alpha based on material (column index)
                if j < x_split:
                    alpha = alpha1
                else:
                    alpha = alpha2
                # Laplace operator (unit spacing)
                laplacian = (temp[i-1,j] + temp[i+1,j] + temp[i,j-1] + temp[i,j+1] - 4*temp[i,j])
                new_temp[i,j] = temp[i,j] + dt * alpha * laplacian
        
        # Apply Neumann BCs (zero gradient on all boundaries)
        # Left edge
        for i in range(Ny):
            new_temp[i,0] = new_temp[i,1]
        # Right edge
        for i in range(Ny):
            new_temp[i,Nx-1] = new_temp[i,Nx-2]
        # Top edge
        for j in range(Nx):
            new_temp[0,j] = new_temp[1,j]
        # Bottom edge
        for j in range(Nx):
            new_temp[Ny-1,j] = new_temp[Ny-2,j]
        
        # Apply Dirichlet BCs
        for bc in bc_dirichlet:
            i, j, val = int(bc[0]), int(bc[1]), bc[2]
            if 0 <= i < Ny and 0 <= j < Nx:
                new_temp[i,j] = val
        
        temp_grid[t+1] = new_temp.copy()
        temp = new_temp
    
    return temp_grid

if __name__ == '__main__':
    # Verification test
    result = heat_equation(
        Nt=5, Nx=4, Ny=3, x_split=1,
        T1=25.0, alpha1=0.1, T2=100.0, alpha2=0.05,
        bc_dirichlet=np.array([[0,0,30.0]]),
        bc_neumann=np.array([])
    )
    print('Output shape:', result.shape)
    print('Contains NaN:', np.any(np.isnan(result)))
    print('Contains Inf:', np.any(np.isinf(result)))
    print('All finite:', np.all(np.isfinite(result)))