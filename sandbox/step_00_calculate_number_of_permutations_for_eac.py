from sympy.combinatorics import Permutation

def factorial(n: int) -> int:
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def count_permutations_by_cycle_structure(n: int, cycle_structure: tuple) -> int:
    """
    Calculate the number of permutations of n elements with a given cycle structure.
    
    :param n: Total number of elements in the set.
    :param cycle_structure: The cycle structure as a tuple of integers.
    :return: Number of permutations with the given cycle structure.
    """
    # Calculate the number of permutations for the given cycle structure
    total_permutations = factorial(n)
    cycle_lengths_count = {length: cycle_structure.count(length) for length in cycle_structure}
    
    divisor = 1
    for cycle_length, count in cycle_lengths_count.items():
        divisor *= factorial(count) * (cycle_length ** count)
    
    return total_permutations // divisor

# Define the valid cycle structures for permutations of order dividing 6
valid_cycle_structures = [(6,), (3, 3), (2, 2, 2)]

# Calculate number of permutations for each cycle structure
n = 6
permutation_counts = {}
for cycle_structure in valid_cycle_structures:
    permutation_counts[cycle_structure] = count_permutations_by_cycle_structure(n, cycle_structure)

permutation_counts