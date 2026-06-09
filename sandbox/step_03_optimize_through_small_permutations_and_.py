import numpy as np
from itertools import permutations

def compute_m(grid: np.ndarray) -> int:
    total_sum = 0
    rows, cols = grid.shape
    # Compute differences for horizontal and vertical adjacencies
    for i in range(rows):
        for j in range(cols):
            if j < cols - 1:
                total_sum += abs(grid[i, j] - grid[i, j + 1])
            if i < rows - 1:
                total_sum += abs(grid[i, j] - grid[i + 1, j])
    return total_sum

def optimize_grid(grid: np.ndarray, iterations: int = 1000) -> np.ndarray:
    max_m = compute_m(grid)
    best_grid = grid.copy()
    cell_indices = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])]
    
    for _ in range(iterations):
        # Choose two random positions to swap
        (x1, y1), (x2, y2) = np.random.choice(len(cell_indices), size=2, replace=False)
        # Swap the positions
        grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1]
        current_m = compute_m(grid)
        
        # If the new arrangement improves M, update best configuration
        if current_m > max_m:
            max_m = current_m
            best_grid = grid.copy()
        else:
            # Swap back if no improvement
            grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1]

    return best_grid

# Example usage:
# Initialization of a checkerboard pattern initial grid
initial_grid = np.zeros((8, 8), dtype=int)
numbers = np.arange(1, 65)

# Fill grid in a checkerboard pattern
checkerboard_indices = [(i, j) for i in range(8) for j in range(i % 2, 8, 2)]
for idx, val in zip(checkerboard_indices, numbers):
    initial_grid[idx] = val

# Fill remaining grid positions
empty_indices = [(i, j) for i in range(8) for j in range(8) if initial_grid[i, j] == 0]
for idx, val in zip(empty_indices, numbers[len(checkerboard_indices):]):
    initial_grid[idx] = val

# Optimize the grid
optimized_grid = optimize_grid(initial_grid)

# Calculate maximum M after optimization
max_m = compute_m(optimized_grid)
print("Maximum M:", max_m, "Modulo 1000:", max_m % 1000)