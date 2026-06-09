import math
from math import comb

def calculate_catalan_number(n: int) -> int:
    """Calculate the nth Catalan number."""
    return comb(2 * n, n) // (n + 1)

def calculate_square_root_of_path_count(grid_size: int) -> float:
    """
    Compute the square root of the number of valid lattice paths from (0,0) to (n,n) 
    constrained by y >= x.
    
    :param grid_size: The size of the grid `n` for the path from (0, 0) to (n, n).
    :return: The square root of the number of valid paths.
    """
    # Calculate the nth Catalan number
    path_count = calculate_catalan_number(grid_size)
    
    # Calculate and return the square root of the path count
    sqrt_path_count = math.sqrt(path_count)
    return sqrt_path_count

# Example usage
grid_size = 10
sqrt_of_paths = calculate_square_root_of_path_count(grid_size)
print(f"Square root of the number of valid paths for grid size {grid_size}: {sqrt_of_paths}")