from typing import List, Tuple

def generate_valid_loop_dimensions(max_perimeter: int) -> List[Tuple[int, int]]:
    valid_dimensions = []

    # Iterate over possible dimensions for a and b
    for a in range(2, 21): # inclusive of 2 and exclusive of 21
        for b in range(2, 21): # inclusive of 2 and exclusive of 21
            # Calculate perimeter 2a + 2b - 4 and check if it's within the constraint
            perimeter = 2 * a + 2 * b - 4
            if perimeter <= max_perimeter:
                valid_dimensions.append((a, b))
    
    return valid_dimensions

# Example usage
max_perimeter = 40  # Assume the maximum allowable perimeter for loop is 40
valid_dimensions = generate_valid_loop_dimensions(max_perimeter)
print(valid_dimensions)