from itertools import permutations
from typing import List, Tuple

# Define the grid size and the number of loops required
GRID_SIZE = 10
NUM_LOOPS = 5

def generate_valid_dimensions() -> List[Tuple[int, int]]:
    """
    Generate valid (a, b) dimensions for loops.

    Returns:
        List[Tuple[int, int]]: A list of valid (a, b) pairs such that each pair
                                satisfies the main loop criteria and can partition
                                the grid without exceeding loop count.
    """
    valid_dimensions = []
    
    # Iterate over potential values of a and b
    for a in range(2, GRID_SIZE + 1):
        for b in range(2, GRID_SIZE + 1):
            perimeter = 2 * a + 2 * b - 4
            if perimeter <= GRID_SIZE:
                valid_dimensions.append((a, b))
    
    return valid_dimensions

def is_grid_covered(loop_dims: List[Tuple[int, int]]) -> bool:
    """
    Check if a given set of loops completely covers the 10x10 grid without overlap.

    Args:
        loop_dims (List[Tuple[int, int]]): The loop dimensions to check.

    Returns:
        bool: True if the grid is completely covered and loops do not overlap, False otherwise.
    """
    total_covered = sum((a * b) for a, b in loop_dims)
    
    # Check coverage and loop overlap constraints
    if total_covered == GRID_SIZE * GRID_SIZE and len(loop_dims) == NUM_LOOPS:
        return True
    return False

def count_valid_arrangements() -> int:
    """
    Counts the number of valid configurations of loops that cover the grid.

    Returns:
        int: The number of valid partitions of the grid.
    """
    valid_dimensions = generate_valid_dimensions()
    valid_configs_count = 0

    # Generate permutations of the valid dimensions to find valid combinations
    for perm in permutations(valid_dimensions, NUM_LOOPS):
        if is_grid_covered(perm):
            valid_configs_count += 1

    return valid_configs_count

# Main execution
if __name__ == "__main__":
    result = count_valid_arrangements()
    print(f"The number of valid configurations of loops that partition the 10x10 grid is: {result}")