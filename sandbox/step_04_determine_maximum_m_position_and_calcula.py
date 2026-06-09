from typing import List
import numpy as np

def compute_M(grid: np.ndarray) -> int:
    """Compute M, the sum of absolute differences of adjacent numbers."""
    M = 0
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            # Check right
            if j < cols - 1:
                M += abs(grid[i][j] - grid[i][j + 1])
            # Check down
            if i < rows - 1:
                M += abs(grid[i][j] - grid[i + 1][j])
    return M

def find_optimal_M_modulo() -> int:
    """Find the optimal M and return it modulo 1000."""
    # Initial greedy checkerboard positioning
    grid = np.zeros((8, 8), dtype=int)
    nums = list(range(1, 65))
    nums.sort(reverse=True)
    
    # Arrange in checkerboard pattern
    index = 0
    for i in range(8):
        for j in range(8):
            grid[i, j] = nums[index]
            index += 1
    
    # Initial computation of M
    initial_M = compute_M(grid)
    
    # Optimize using small permutations
    optimal_M = initial_M
    for _ in range(1000):  # Iterative permutations
        for x in range(8):
            for y in range(8):
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 8 and 0 <= ny < 8:
                        # Swap and check
                        grid[x, y], grid[nx, ny] = grid[nx, ny], grid[x, y]
                        current_M = compute_M(grid)
                        if current_M > optimal_M:
                            optimal_M = current_M
                        else:
                            # Swap back
                            grid[x, y], grid[nx, ny] = grid[nx, ny], grid[x, y]

    # Return M modulo 1000
    return optimal_M % 1000

if __name__ == "__main__":
    optimal_M_modulo = find_optimal_M_modulo()
    print(optimal_M_modulo)