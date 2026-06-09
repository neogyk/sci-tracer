from typing import Iterator, Sequence, Tuple


def compositions(n: int) -> Iterator[Tuple[int, ...]]:
    """
    Generate all ordered compositions of n into positive integers as tuples.
    For n == 0, yields a single empty tuple ().
    """
    assert n >= 0, "n must be non-negative"
    if n == 0:
        yield ()
        return
    for first in range(1, n + 1):
        for rest in compositions(n - first):
            yield (first,) + rest


def op_result(seq: Sequence[int]) -> int:
    """
    Compute a1 ∘ a2 ∘ ... ∘ ak with the rule:
      - subtract if current accumulator is odd and next term is even
      - otherwise add
    """
    assert len(seq) > 0, "Sequence must be non-empty"
    acc: int = seq[0]
    for b in seq[1:]:
        if (acc % 2 == 1) and (b % 2 == 0):
            acc = acc - b
        else:
            acc = acc + b
    return acc


def count_sequences(total: int) -> int:
    """
    Count ordered compositions of 'total' whose left-to-right ∘-evaluation equals 0.
    """
    count: int = 0
    for seq in compositions(total):
        if not seq:  # skip the empty composition (occurs only when total == 0)
            continue
        if op_result(seq) == 0:
            count += 1
    return count


if __name__ == "__main__":
    res: int = count_sequences(12)
    assert res == 157, f"Expected 157, got {res}"
    print(res)