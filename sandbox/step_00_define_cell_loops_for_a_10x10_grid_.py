from typing import List, Tuple, Set

def generate_loops(grid_size: int) -> List[Set[Tuple[int, int]]]:
    """
    Generate all possible loops within a grid of specified size.

    Args:
    grid_size (int): Size of the grid, e.g., 10 for a 10x10 grid.

    Returns:
    List[Set[Tuple[int, int]]]: A list of sets where each set contains tuples
                                representing coordinates of a loop in the grid.
    """
    def is_valid_position(x: int, y: int) -> bool:
        """Check if position is within the grid boundaries."""
        return 0 <= x < grid_size and 0 <= y < grid_size
    
    def generate_single_loop(current_loop: Set[Tuple[int, int]], start_pos: Tuple[int, int]):
        """Recursively generate loops starting from a given position."""
        if len(current_loop) > 1 and start_pos == next(iter(current_loop)):
            loops.append(current_loop)
            return
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (start_pos[0] + dx, start_pos[1] + dy)
            if new_pos not in current_loop and is_valid_position(*new_pos):
                generate_single_loop(current_loop | {new_pos}, new_pos)
    
    loops = []
    for x in range(grid_size):
        for y in range(grid_size):
            generate_single_loop({(x, y)}, (x, y))
    
    return loops

# Example usage to generate loops for a 10x10 grid
grid_size = 10
loops = generate_loops(grid_size)
print(f"Generated {len(loops)} loops for a {grid_size}x{grid_size} grid.")