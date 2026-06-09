from typing import List, Tuple

def is_loop_coverage_valid(grid_size: int, loops: List[List[Tuple[int, int]]]) -> bool:
    """
    Verify whether a combination of loops can cover the entire grid without leaving any gaps or overlaps.

    :param grid_size: The size of the grid, assumed to be grid_size x grid_size.
    :param loops: A list of loops, where each loop is a list of cell coordinates (x, y) covered by that loop.
    :return: True if the loops cover the entire grid with each cell exactly once, False otherwise.
    """
    # Initialize a grid coverage map
    coverage_map = [[0] * grid_size for _ in range(grid_size)]
    
    # Mark each loop's cells in the coverage map
    for loop in loops:
        for x, y in loop:
            # Mark each cell as covered
            coverage_map[x][y] += 1
    
    # Check if each cell in the grid is covered exactly once
    for row in coverage_map:
        for cell in row:
            if cell != 1:
                return False

    return True

# Example test case to verify functionality:
# We assume a simple 2x2 grid for demonstration. The loops perfectly covering this grid are defined as follows:
example_loops = [
    [(0, 0), (0, 1), (1, 1), (1, 0)]  # One loop covering the entire grid
]
grid_size_example = 2  # Example for a 2x2 grid

# Expected output: True, since the single loop covers the entire grid.
print(is_loop_coverage_valid(grid_size_example, example_loops))  # This should output: True