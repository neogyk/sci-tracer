from sympy import binomial

# Define a function to calculate the nth Catalan number
def catalan_number(n: int) -> int:
    """
    Calculate the nth Catalan number using the binomial coefficient.
    
    The formula for the nth Catalan number is:
    C_n = (1 / (n + 1)) * binomial(2n, n)

    Args:
    n (int): The index of the Catalan number to calculate.

    Returns:
    int: The nth Catalan number.
    """
    return binomial(2 * n, n) // (n + 1)

# Grid size is 10, therefore we need the 10th Catalan number
grid_size: int = 10
num_paths: int = catalan_number(grid_size)

print(f"Number of valid paths for a grid of size {grid_size}: {num_paths}")