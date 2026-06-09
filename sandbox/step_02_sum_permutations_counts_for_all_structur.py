from typing import Dict
import numpy as np

def sum_permutation_counts(cycle_structure_counts: Dict[str, int]) -> int:
    """
    Sums the number of permutations for all valid cycle structures to get the total count.

    Parameters:
    - cycle_structure_counts: dict
        A dictionary where keys are cycle structures represented as strings and values are the counts
        of permutations for those structures.

    Returns:
    - int
        The total number of permutations that satisfy the cycle structures.
    """
    # Use numpy to sum the counts of permutations
    total_permutations = np.sum(list(cycle_structure_counts.values()))
    return total_permutations

# Example cycle structure counts for demonstration purposes
cycle_structure_counts = {
    '(6)': 120,   # Assuming there are 120 permutations with cycle structure (6)
    '(3, 3)': 20, # Assuming there are 20 permutations with cycle structure (3, 3)
    '(2, 2, 2)': 15 # Assuming there are 15 permutations with cycle structure (2, 2, 2)
}

# Calculate total number of suitable permutations
total = sum_permutation_counts(cycle_structure_counts)
print(f"Total number of suitable permutations: {total}")