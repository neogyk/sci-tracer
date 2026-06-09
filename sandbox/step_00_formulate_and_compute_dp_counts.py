from __future__ import annotations

import math
from typing import Tuple, TypedDict


class Result(TypedDict):
    N: int
    sqrtN: float


def dp_count(n: int = 10) -> Tuple[int, int]:
    """
    Compute the number of constrained paths using a 2-state DP with transfer:
      [f_i^0]   [1 2] [f_{i+1}^0]
      [f_i^1] = [1 1] [f_{i+1}^1]
    Base at i = n: f_n^0 = 1, f_n^1 = 1.

    Args:
        n: number of columns in the 2-row strip.

    Returns:
        Tuple (f_0^0, f_0^1)
        - f_0^0 is the total number of valid paths starting from (0,0) to (n,1)
        - f_0^1 is the count when starting from top at column 0
    """
    f0: int = 1  # f_n^0
    f1: int = 1  # f_n^1
    for _ in range(n):
        f0, f1 = (f0 + 2 * f1, f0 + f1)
    return f0, f1


if __name__ == "__main__":
    N, top = dp_count(10)
    sqrtN: float = math.sqrt(N)
    # Known values for n=10 under these rules
    assert N == 8119 and top == 5741
    result: Result = {"N": N, "sqrtN": sqrtN}
    print(result)