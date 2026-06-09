from typing import FrozenSet, List, Tuple


def enumerate_allowed_duplicate_pairs(n: int = 6, visible_indices: FrozenSet[int] = frozenset({2, 4})) -> List[Tuple[int, int]]:
    """
    Enumerate allowed duplicate index pairs (i, j) with i < j under event E:
    - For any i in visible_indices, no later duplicate is allowed (i.e., (i, j) is forbidden for all j > i).
    - Indices not in visible_indices place no restriction on later duplicates for the purpose of E.

    Args:
        n: Total number of rolls (indices are 1..n).
        visible_indices: Indices whose stickers must be visible (i.e., cannot repeat later).

    Returns:
        A list of allowed (i, j) pairs with i < j.
    """
    allowed: List[Tuple[int, int]] = []
    for i in range(1, n):
        for j in range(i + 1, n + 1):
            # If i is constrained to be visible, (i, j) duplicates are forbidden.
            if i in visible_indices:
                continue
            allowed.append((i, j))
    return allowed


if __name__ == "__main__":
    pairs = enumerate_allowed_duplicate_pairs()
    # Validate against the expected set per the research plan
    expected = {
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (3, 4),
        (3, 5),
        (3, 6),
        (5, 6),
    }
    assert set(pairs) == expected, f"Mismatch: got {set(pairs)}, expected {expected}"
    print("Allowed pairs:", sorted(pairs))
    print("Count:", len(pairs))