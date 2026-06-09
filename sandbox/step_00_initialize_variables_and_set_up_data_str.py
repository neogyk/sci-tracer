import numpy as np

def initialize_grid() -> np.ndarray:
    """Initializes an 8x8 grid with numbers from 1 to 64."""
    return np.arange(1, 65).reshape((8, 8))

def compute_M(grid: np.ndarray) -> int:
    """
    Computes the sum M based on the given grid arrangement.
    The value of M is defined as the sum of absolute differences of adjacent numbers.
    
    :param grid: 8x8 numpy array with numbers arranged in the grid
    :return: calculated M as an integer
    """
    M = 0
    rows, cols = grid.shape
    
    # Calculate the sum of absolute differences for horizontal and vertical neighbors
    for i in range(rows):
        for j in range(cols):
            if j < cols - 1:  # Check right neighbor
                M += abs(grid[i, j] - grid[i, j + 1])
            if i < rows - 1:  # Check below neighbor
                M += abs(grid[i, j] - grid[i + 1, j])
    
    return M

# Initialize the grid
grid = initialize_grid()

# Define the function that computes M for this arrangement
M_value = compute_M(grid)

# Output the initial grid and its M value for confirmation
grid, M_value