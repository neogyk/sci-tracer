from itertools import permutations
from typing import List

def calculate_max_absolute_sum(grid_size: int) -> int:
    """ Calculate maximum sum of absolute differences of adjacent entries in a grid. """
    max_sum = 0
    numbers = list(range(1, grid_size * grid_size + 1))
    
    # Generate permutations
    for perm in permutations(numbers):
        # Calculate the sum of absolute differences in the permutation
        current_sum = sum(abs(perm[i] - perm[i-1]) for i in range(1, grid_size * grid_size))
        
        # Check if this sum is a new maximum
        if current_sum > max_sum:
            max_sum = current_sum
    
    return max_sum

def max_m_mod_1000() -> int:
    """ Computes the maximum M value modulo 1000. """
    grid_size = 8
    max_m = calculate_max_absolute_sum(grid_size)
    return max_m % 1000

# Main execution
if __name__ == "__main__":
    result = max_m_mod_1000()
    print(f"Max M modulo 1000: {result}")