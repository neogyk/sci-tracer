import numpy as np

def compute_geometric_sum() -> float:
    # Constants
    a: float = 1/9  # First term
    r: float = 0.1  # Common ratio

    # Precondition
    assert -1 < r < 1, "The common ratio must be between -1 and 1 for convergence."

    # Calculate the sum S of the infinite geometric series
    S: float = a / (1 - r)

    return S

# Compute the sum
sum_S = compute_geometric_sum()
print(f"The sum S = {sum_S}")