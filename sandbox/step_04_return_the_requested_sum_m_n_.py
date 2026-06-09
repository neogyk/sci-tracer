from typing import Final
from math import gcd


COS_NUM: Final[int] = 29
COS_DEN: Final[int] = 36


def is_coprime(a: int, b: int) -> bool:
    return gcd(abs(a), abs(b)) == 1


def sum_if_coprime(m: int, n: int) -> int:
    if not is_coprime(m, n):
        raise ValueError("m and n must be coprime.")
    return m + n


def main() -> int:
    return sum_if_coprime(COS_NUM, COS_DEN)


if __name__ == "__main__":
    print(main())