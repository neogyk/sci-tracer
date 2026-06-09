from typing import List
import random

def initialize_grid() -> List[List[int]]:
    """
    Initializes an 8x8 grid with integers 1 through 64 placed randomly.
    
    Returns:
        List[List[int]]: A randomly shuffled 8x8 grid containing integers from 1 to 64.
    """
    # Create a list of integers from 1 to 64
    numbers = list(range(1, 65))
    # Shuffle the list to ensure random placement
    random.shuffle(numbers)
    # Populate the 8x8 grid
    grid = [numbers[i*8:(i+1)*8] for i in range(8)]
    return grid

# Test initialization
grid = initialize_grid()
for row in grid:
    print(row)