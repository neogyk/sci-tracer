from typing import List, Tuple

def generate_loop_configurations() -> List[Tuple[int, int]]:
    # List to store valid pairs (a, b)
    valid_pairs = []
    
    # Check all combinations of a and b between 2 and 10
    for a in range(2, 11):
        for b in range(2, 11):
            # Compute perimeter of the rectangle
            perimeter = 2 * a + 2 * b - 4
            # Check if this perimeter divides 100
            if perimeter > 0 and 100 % perimeter == 0:
                # Add the pair to the list of valid pairs
                valid_pairs.append((a, b))
    
    return valid_pairs

# Example usage:
valid_loop_configurations = generate_loop_configurations()
print(valid_loop_configurations)