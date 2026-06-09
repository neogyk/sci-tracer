import numpy as np

def checkerboard_pattern() -> np.ndarray:
    """
    Arrange the integers 1 to 64 in an 8x8 grid such that the grid 
    follows a checkerboard pattern, maximizing the absolute differences
    between adjacent numbers.
    
    Returns:
        np.ndarray: An 8x8 numpy array representing the grid.
    """
    grid_size: int = 8
    grid: np.ndarray = np.zeros((grid_size, grid_size), dtype=int)

    # Fill the grid with a checkerboard pattern.
    # Alternating large and small numbers for maximal difference.
    numbers = list(range(1, grid_size * grid_size + 1))
    numbers.sort(reverse=True)  # Start with largest numbers first

    for i in range(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 0:
                grid[i, j] = numbers.pop(0)
            else:
                grid[i, j] = numbers.pop()

    return grid

if __name__ == "__main__":
    grid = checkerboard_pattern()
    print(grid)