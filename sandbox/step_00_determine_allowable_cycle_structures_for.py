from typing import List


def divisors(n: int) -> List[int]:
    """Return all positive divisors of n."""
    return [d for d in range(1, n + 1) if n % d == 0]


def cycle_type_partitions(n: int, k: int) -> List[List[int]]:
    """
    Enumerate all valid cycle-type partitions (as sorted nondecreasing lists of parts)
    for permutations on n elements satisfying π^k = id.
    A cycle-type is valid iff every cycle length divides k and the parts sum to n.
    """
    allowed_parts: List[int] = sorted(d for d in divisors(k) if d <= n)

    results: List[List[int]] = []

    def backtrack(remaining: int, start_idx: int, current: List[int]) -> None:
        if remaining == 0:
            results.append(current.copy())
            return
        for i in range(start_idx, len(allowed_parts)):
            part = allowed_parts[i]
            if part > remaining:
                break
            current.append(part)
            backtrack(remaining - part, i, current)  # allow reuse of the same part
            current.pop()

    backtrack(n, 0, [])
    return results


def main() -> None:
    n: int = 6
    k: int = 6

    allowed_lengths: List[int] = [d for d in divisors(k) if d <= n]
    partitions: List[List[int]] = cycle_type_partitions(n, k)

    print(f"Allowed cycle lengths (dividing {k}): {allowed_lengths}")
    print("Valid cycle-type partitions of 6 using {1,2,3,6}:")
    for p in partitions:
        print(p)

    # Optional verification against the enumerated list in the research plan
    expected: List[List[int]] = [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 2],
        [1, 1, 2, 2],
        [2, 2, 2],
        [1, 1, 1, 3],
        [3, 3],
        [1, 2, 3],
        [6],
    ]
    assert set(map(tuple, partitions)) == set(map(tuple, expected)), "Partition set mismatch"


if __name__ == "__main__":
    main()