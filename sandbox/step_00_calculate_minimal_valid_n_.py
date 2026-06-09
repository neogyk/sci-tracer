from typing import List

def compute_minimal_valid_n() -> List[int]:
    # List to store the smallest n values that satisfy the conditions
    smallest_n_values = []

    # Constraints derived from the problem
    for r in range(7, 100):  # assuming r will not need to exceed 100
        for b in range(7, 100):  # assuming b will not need to exceed 100
            # Derived equation 2b = r
            if 2 * b == r:
                n = r + b
                smallest_n_values.append(n)

    # Sort smallest_n_values to get the five smallest values
    smallest_n_values.sort()

    # Return the first five smallest n values as per the task requirement
    return smallest_n_values[:5]

# Compute and print the result
print(compute_minimal_valid_n())