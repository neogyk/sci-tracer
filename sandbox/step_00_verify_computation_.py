def find_minimal_n_values() -> list[int]:
    # Constraints from the algebra derivation
    # 2b = r and b >= 7, r >= 7, r + b = n
    minimal_n_values = []
    
    # Iterate over possible b values starting from 7
    for b in range(7, 20):  # Let's assume a reasonable boundary based on initial constraints
        # Calculate r using derived relation 2b = r
        r = 2 * b
        n = r + b  # Calculate n for current r, b
        
        # Add to the list of minimal n values
        minimal_n_values.append(n)

    # Sort the list to find the smallest 5 n values
    minimal_n_values.sort()
    
    # Select the 5 smallest values
    smallest_5_values = minimal_n_values[:5]
    
    return smallest_5_values


def verify_minimal_n_values() -> int:
    # Compute the sum of the smallest 5 n values
    smallest_5_values = find_minimal_n_values()
    sum_smallest_5 = sum(smallest_5_values)
    return sum_smallest_5


# Output the verified sum of the smallest 5 n values
verified_sum = verify_minimal_n_values()
verified_sum