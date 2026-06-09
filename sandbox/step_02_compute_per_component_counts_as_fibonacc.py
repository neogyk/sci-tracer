from typing import Dict, Any


def fibonacci(n: int) -> int:
    """
    Compute the nth Fibonacci number with F0 = 0, F1 = 1.
    Uses an iterative O(n) method with O(1) space.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    a: int = 0
    b: int = 1
    for _ in range(n):
        a, b = b, a + b
    return a


def singleton_run_count(k: int, left_forced: bool = False, right_forced: bool = False) -> int:
    """
    Count the number of S-saturating matchings for a run of k singletons (isolated S-vertices)
    along a path component, where an endpoint may be forced by an adjacent doubleton component.

    - If neither end is forced: count = F_{k+2}
    - If exactly one end is forced: count = F_{k+1}
    - If both ends are forced: count = F_{k}

    Here F_n denotes the Fibonacci numbers with F_0 = 0, F_1 = 1.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    forced_ends: int = int(left_forced) + int(right_forced)
    index: int = k + 2 - forced_ends
    return fibonacci(index)


def step_3(context: Dict[str, Any]) -> Dict[str, str]:
    """
    Derive the Fibonacci recurrence for per-component counts:
    A run of k singletons contributes C(k) satisfying C(k) = C(k-1) + C(k-2) with C(0) = 1, C(1) = 2,
    hence C(k) = F_{k+2}. Forced endpoints shift the index by 1 per forced end; doubletons contribute 1.
    """
    return {"component_counts": "Fibonacci", "formula": "C(k)=F_{k+2}"}


if __name__ == "__main__":
    # Lightweight self-checks for the claimed formulas.
    # Free-ended runs: C(k) = F_{k+2} => 1,2,3,5,8,...
    assert [singleton_run_count(k) for k in range(6)] == [1, 2, 3, 5, 8, 13]
    # One end forced: C(k) = F_{k+1} => 1,1,2,3,5,8,...
    assert [singleton_run_count(k, left_forced=True) for k in range(6)] == [1, 1, 2, 3, 5, 8]
    assert [singleton_run_count(k, right_forced=True) for k in range(6)] == [1, 1, 2, 3, 5, 8]
    # Both ends forced: C(k) = F_{k} => 0,1,1,2,3,5,...
    assert [singleton_run_count(k, left_forced=True, right_forced=True) for k in range(6)] == [0, 1, 1, 2, 3, 5]