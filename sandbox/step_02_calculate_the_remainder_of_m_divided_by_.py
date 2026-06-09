from typing import List, Tuple
import numpy as np

def calculate_max_sum(grid: np.ndarray) -> int:
    def calculate_total_difference(grid: np.ndarray) -> int:
        rows, cols = grid.shape
        total_sum = 0
        for i in range(rows):
            for j in range(cols):
                if i > 0:
                    total_sum += abs(grid[i, j] - grid[i - 1, j])
                if i < rows - 1:
                    total_sum += abs(grid[i, j] - grid[i + 1, j])
                if j > 0:
                    total_sum += abs(grid[i, j] - grid[i, j - 1])
                if j < cols - 1:
                    total_sum += abs(grid[i, j] - grid[i, j + 1])
        return total_sum // 2  # Each pair is counted twice

    # Example checkerboard pattern maximizes differences
    def create_checkerboard_pattern() -> np.ndarray:
        # Create a pattern like a checkerboard
        n = 8
        # Place alternating high and low values
        pattern = np.zeros((n, n), dtype=int)
        place_high = True
        numbers = list(range(1, n*n + 1))
        
        for i in range(n):
            for j in range(n):
                if place_high:
                    pattern[i, j] = numbers.pop()
                else:
                    pattern[i, j] = numbers.pop(0)
                place_high = not place_high
            if n % 2 == 0:  # if n is even, swap for every new row
                place_high = not place_high
        return pattern
    
    grid = create_checkerboard_pattern()
    max_sum = calculate_total_difference(grid)
    return max_sum

# Calculate the remainder of the maximum sum divided by 1000
def max_sum_modulo() -> int:
    max_sum = calculate_max_sum(np.zeros((8, 8)))
    result = max_sum % 1000
    return result

# Run the calculation
max_sum_modulo_result = max_sum_modulo()
max_sum_modulo_result