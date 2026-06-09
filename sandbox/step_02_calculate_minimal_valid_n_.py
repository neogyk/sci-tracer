from typing import List, Tuple

def calculate_minimal_n_candidates() -> List[Tuple[int, int, int]]:
    r = 7
    b_candidates = []
    
    # Calculate b_values according to derived relationship r = 2b
    while 2 * r >= 7:
        b = 2 * r
        if b >= 7:
            n = r + b
            b_candidates.append((n, r, b))
        r += 1
    
    # Sort by n to get the minimum values
    b_candidates.sort(key=lambda x: x[0])
    return b_candidates

def sum_of_five_least_n_values() -> int:
    candidates = calculate_minimal_n_candidates()
    
    # Return the sum of the five least n values
    return sum(n for n, _, _ in candidates[:5])

# Example usage:
if __name__ == "__main__":
    minimal_n_result = calculate_minimal_n_candidates()
    minimal_sum = sum_of_five_least_n_values()
    print("Minimal valid n candidates (n, r, b):", minimal_n_result)
    print("Sum of the five least n values:", minimal_sum)