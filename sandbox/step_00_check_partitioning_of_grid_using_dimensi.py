from typing import List, Tuple
from itertools import permutations

def is_valid_configuration(grid_size: int, loops: List[Tuple[int, int]]) -> bool:
    # Prepare an empty grid
    grid = [[0] * grid_size for _ in range(grid_size)]

    # Function to fill the grid with loop
    def fill_loop(x: int, y: int, a: int, b: int) -> bool:
        # Mark the loop perimeter in the grid. Returns False if overlaps
        try:
            # Top and bottom
            for i in range(x, x + a):
                if grid[y][i] == 1 or grid[y + b - 1][i] == 1:
                    return False
                grid[y][i] = grid[y + b - 1][i] = 1
            # Left and right
            for j in range(y, y + b):
                if grid[j][x] == 1 or grid[j][x + a - 1] == 1:
                    return False
                grid[j][x] = grid[j][x + a - 1] = 1
        except IndexError:
            return False
        return True

    # Check all permutations of loops in the grid
    for loop_order in permutations(loops):
        # Reset grid for each permutation
        grid = [[0] * grid_size for _ in range(grid_size)]
        success = True
        
        for x in range(grid_size):
            for y in range(grid_size):
                if not fill_loop(x, y, *loop_order[0]):
                    success = False
                    break
            if not success:
                break
        
        # If successfully filled with current permutation, return True
        if all(all(cell == 1 for cell in row) for row in grid):
            return True

    return False

# Example to run
grid_size = 10
possible_loops = [(2, 3), (3, 2), (3, 3), (4, 2)]  # Example valid dimension loops
valid_configurations = is_valid_configuration(grid_size, possible_loops)
print(valid_configurations)