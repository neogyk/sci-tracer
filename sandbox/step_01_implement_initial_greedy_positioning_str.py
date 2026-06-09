import numpy as np

def checkerboard_pattern() -> np.ndarray:
    """
    Arrange integers 1 to 64 in an 8x8 checkerboard pattern
    that maximizes adjacency differences in the grid.

    Returns:
        np.ndarray: An 8x8 numpy array with integers arranged.
    """
    # Create an 8x8 numpy array
    grid = np.zeros((8, 8), dtype=int)

    # Fill the grid in a checkerboard pattern
    numbers = np.arange(1, 65).reshape(8, 8)
    grid[::2, ::2] = numbers[::2, ::2]  # Fill even rows and even columns
    grid[1::2, 1::2] = numbers[1::2, 1::2]  # Fill odd rows and odd columns

    # Fill the remaining spots
    grid[::2, 1::2] = numbers[::2, 1::2]
    grid[1::2, ::2] = numbers[1::2, ::2]

    return grid

if __name__ == "__main__":
    checkerboard_grid = checkerboard_pattern()
    print("Checkerboard Grid:\n", checkerboard_grid)