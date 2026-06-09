from typing import List

def compute_sum_of_differences(grid: List[List[int]]) -> int:
    """
    Computes the sum of absolute differences between adjacent cells
    in the grid both row-wise and column-wise.

    :param grid: An 8x8 list of lists containing integers.
    :return: The sum of absolute differences in the grid.
    """
    total_sum_of_differences = 0
    n = len(grid)

    # Calculate row-wise differences
    for row in range(n):
        for col in range(n - 1):
            total_sum_of_differences += abs(grid[row][col] - grid[row][col + 1])

    # Calculate column-wise differences
    for col in range(n):
        for row in range(n - 1):
            total_sum_of_differences += abs(grid[row][col] - grid[row + 1][col])

    return total_sum_of_differences

# Example execution on a hypothetical 8x8 grid
example_grid = [
    [1, 64, 2, 63, 3, 62, 4, 61],
    [64, 1, 63, 2, 62, 3, 61, 4],
    [5, 60, 6, 59, 7, 58, 8, 57],
    [60, 5, 59, 6, 58, 7, 57, 8],
    [9, 56, 10, 55, 11, 54, 12, 53],
    [56, 9, 55, 10, 54, 11, 53, 12],
    [13, 52, 14, 51, 15, 50, 16, 49],
    [52, 13, 51, 14, 50, 15, 49, 16]
]

# Calling the function
result = compute_sum_of_differences(example_grid)
print("Total sum of absolute differences:", result)