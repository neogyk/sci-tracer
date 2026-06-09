from itertools import permutations
from typing import List, Set, Tuple

def valid_cousin_pair(s: int, t: int) -> bool:
    """Check if a given pair is a valid cousin pair."""
    return abs(t - s) == 1

def enumerate_cousins(S: Set[int]) -> List[Tuple[int]]:
    """Return all valid configurations for cousin sets T given set S."""
    cousins = []
    possible_elements = set(x + 1 for x in S).union(x - 1 for x in S)
    
    # Generate all possible sets of the same size as S using permutations
    for perm in permutations(possible_elements, len(S)):
        if all(valid_cousin_pair(s, t) for s, t in zip(S, perm)):
            cousins.append(perm)

    return cousins

def find_smallest_n_for_4040_cousins() -> int:
    """Find the smallest n where the number of valid cousin sets T is 4040."""
    n = 1
    while True:
        # Attempt to construct set S with n integers
        S = set(range(1, n + 1))
        cousins = enumerate_cousins(S)
        if len(cousins) == 4040:
            return n
        n += 1

# Main execution to find the smallest n
smallest_n = find_smallest_n_for_4040_cousins()
print(f"The smallest n such that set S has 4040 cousins is: {smallest_n}")