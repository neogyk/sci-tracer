import math
from sympy import binomial

def calculate_catalan_number(n: int) -> int:
    """
    Calculate the nth Catalan number using the formula:
    C_n = 1 / (n + 1) * (2n choose n)
    """
    return binomial(2 * n, n) // (n + 1)

def calculate_square_root_of_path_count(grid_size: int) -> float:
    """
    Calculate the square root of the number of valid lattice paths for a given grid size.
    """
    # Assume precondition: N (path count) is calculated as grid_size-th Catalan number
    N = calculate_catalan_number(grid_size)
    
    # Compute the square root
    sqrt_N = math.sqrt(N)
    
    return sqrt_N

# Example usage:
grid_size = 10
sqrt_path_count = calculate_square_root_of_path_count(grid_size)
print(f"The square root of the path count for a grid size {grid_size} is {sqrt_path_count}")