from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import SymmetricGroup
from typing import List, Tuple

def count_permutations_for_cycle_structures(n: int, cycle_structures: List[Tuple[int, ...]]) -> List[int]:
    """
    Counts the permutations of cycle structures dividing the order 6.

    :param n: The size of the permutation set.
    :param cycle_structures: List of cycle structures to count permutations for.
    :return: A list of permutation counts for each cycle structure.
    """
    
    # Initialize the symmetric group of order n
    S_n = SymmetricGroup(n)

    def number_of_permutations(cycle_type: Tuple[int, ...]) -> int:
        """
        Calculate number of permutations for a specific cycle type.

        :param cycle_type: The cycle structure tuple.
        :return: The number of permutations with that cycle structure.
        """
        from math import factorial
        from collections import Counter

        size = sum(cycle_type)
        if size != n:
            return 0
        
        # Count the repetitions of each cycle length
        frequency = Counter(cycle_type)
        # Calculate the product of factorials of the cycle occurrences
        denominator = 1
        for cycle_length, count in frequency.items():
            denominator *= (factorial(count) * cycle_length**count)
        
        # Total permutations given by n! divided by product of factorials of cycle lengths and repeated permutations
        return factorial(n) // denominator

    results = [number_of_permutations(cycle_type) for cycle_type in cycle_structures]
    
    return results


# Define the cycle structures that divide 6
valid_cycle_structures = [(6,), (3, 3), (2, 2, 2)]

# Calculate the number of permutations for each cycle structure
permutation_counts = count_permutations_for_cycle_structures(6, valid_cycle_structures)

# Print results
for structure, count in zip(valid_cycle_structures, permutation_counts):
    print(f"Cycle structure {structure} has {count} permutations.")