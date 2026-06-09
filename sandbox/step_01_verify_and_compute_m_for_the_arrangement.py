import numpy as np

def compute_total_sum_of_differences(grid: np.ndarray) -> int:
    """
    Compute the total sum of absolute differences for an 8x8 grid.
        
    :param grid: An 8x8 numpy array of integers.
    :return: The total sum of absolute differences.
    """
    total_sum = 0

    # Calculate total sum of differences row-wise
    for i in range(8):
        for j in range(7):  # Only compare with next element
            total_sum += abs(grid[i, j] - grid[i, j + 1])
    
    # Calculate total sum of differences column-wise
    for i in range(7):  # Only compare with the next row
        for j in range(8):
            total_sum += abs(grid[i, j] - grid[i + 1, j])

    return total_sum

def main():
    # Example pattern that approximates optimal differences
    # Checkered pattern where large differences are ensured
    pattern = np.array([
        [1, 64, 2, 63, 3, 62, 4, 61],
        [64, 1, 63, 2, 62, 3, 61, 4],
        [5, 60, 6, 59, 7, 58, 8, 57],
        [60, 5, 59, 6, 58, 7, 57, 8],
        [9, 56, 10, 55, 11, 54, 12, 53],
        [56, 9, 55, 10, 54, 11, 53, 12],
        [13, 52, 14, 51, 15, 50, 16, 49],
        [52, 13, 51, 14, 50, 15, 49, 16],
    ])

    result = compute_total_sum_of_differences(pattern)
    print(f"Total sum of differences: {result}")

if __name__ == "__main__":
    main()