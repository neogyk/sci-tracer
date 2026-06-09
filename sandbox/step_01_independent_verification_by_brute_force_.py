from __future__ import annotations

from functools import lru_cache
from typing import Literal


def enum_count(n: int = 10) -> int:
    @lru_cache(maxsize=None)
    def ways(i: int, row: Literal[0, 1]) -> int:
        if i == n:
            return 1  # row==1 already at B; row==0 one final vertical up
        if row == 0:
            # right on bottom; diagonal to top; up then right on top
            return ways(i + 1, 0) + ways(i + 1, 1) + ways(i + 1, 1)
        else:
            # right on top; down then right on bottom
            return ways(i + 1, 1) + ways(i + 1, 0)

    return ways(0, 0)


def main() -> None:
    bfN: int = enum_count(10)
    assert bfN == 8119
    print({'bfN': bfN})


if __name__ == "__main__":
    main()