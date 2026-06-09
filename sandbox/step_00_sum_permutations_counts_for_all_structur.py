import numpy as np

def sum_permutations_counts() -> int:
    # Permutation counts for each valid cycle structure
    permutation_counts = [120, 90, 45]
    
    # Sum the number of permutations for all valid cycle structures
    total_permutations = np.sum(permutation_counts)
    
    return total_permutations

total = sum_permutations_counts()
total