from typing import List, Tuple
import numpy as np

# Define the grid size and numbers to be used
GRID_SIZE: int = 8
NUMBERS: List[int] = list(range(1, GRID_SIZE * GRID_SIZE + 1))

def initialize_grid() -> np.ndarray:
    """
    Initializes the grid with a checkerboard pattern for maximizing differences in adjacent cells.
    """
    grid: np.ndarray = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    index: int = 0
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            # Calculate checkerboard index by swapping two adjacent rows one by one
            checkerboard_rule: bool = (row + col) % 2 == 0
            pos: int = index if checkerboard_rule else index + 1
            grid[row, col] = NUMBERS[pos]
            index += 1
            if index >= len(NUMBERS) or pos >= len(NUMBERS) - 1:
                break
    return grid

def compute_M(grid: np.ndarray) -> int:
    """
    Computes the sum M based on the differences between integers in adjacent cells.
    In this context, adjacencies include horizontally and vertically adjacent cells only.
    """
    def edge_weight(r1: int, c1: int, r2: int, c2: int) -> int:
        return abs(grid[r1, c1] - grid[r2, c2])

    M: int = 0
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if col + 1 < GRID_SIZE:
                M += edge_weight(row, col, row, col + 1)
            if row + 1 < GRID_SIZE:
                M += edge_weight(row, col, row + 1, col)
    
    return M

# Initialize the grid using the initial greedy positioning strategy
grid = initialize_grid()
# Compute the initial sum M for this arrangement
initial_M: int = compute_M(grid)
# Record and print the result
print(f"Initial sum M: {initial_M}")