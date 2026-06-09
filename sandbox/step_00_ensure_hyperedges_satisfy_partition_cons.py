from itertools import combinations
from typing import List, Set, Tuple

def check_hyperedges_partition(hyperedges: List[Set[int]], max_partition_size: int) -> bool:
    """
    Check that no subset of the hyperedges can form a partition larger than the specified size.

    :param hyperedges: A list of hyperedges, each is a set of vertices.
    :param max_partition_size: Maximum allowed size for any subset that forms a partition.
    :return: True if no subset of hyperedges forms a partition larger than max_partition_size, False otherwise.
    """
    total_vertices = set.union(*hyperedges)
    
    # Iterate over all possible subsets of hyperedges
    for r in range(1, len(hyperedges) + 1):
        for subset in combinations(hyperedges, r):
            union_subset = set.union(*subset)
            if len(union_subset) > max_partition_size:
                return False
    return True

# Define some example hyperedges, assuming the example constraints are already satisfied
hyperedges_example = [
    {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12},
    {13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24},
    {25, 26, 27}, {28, 29, 30}, {31, 32, 33}, {34, 35, 36}
]

max_partition_size = 20

# Check the example hyperedges
is_valid = check_hyperedges_partition(hyperedges_example, max_partition_size)
print("No partition larger than max size:", is_valid)