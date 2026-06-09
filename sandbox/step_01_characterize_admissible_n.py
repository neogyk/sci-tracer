from typing import List
import math


def is_prime(n: int) -> bool:
    """
    Check primality of n (n >= 0).
    """
    if n <= 1:
        return False
    if n <= 3:
        return True  # 2, 3
    if n % 2 == 0 or n % 3 == 0:
        return False
    i: int = 5
    step: int = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += step
        step = 6 - step
    return True


def is_composite(n: int) -> bool:
    """
    Return True iff n is composite.
    """
    return n > 1 and not is_prime(n)


def is_prime_square(m: int) -> bool:
    """
    Return True iff m = p^2 for some prime p.
    """
    if m <= 0:
        return False
    r: int = math.isqrt(m)
    return r * r == m and is_prime(r)


def has_unequal_factor_pair_ge2_scan(m: int) -> bool:
    """
    Directly check whether m has a factorization m = x*y with x, y >= 2 and x != y.
    """
    if m < 4:
        return False
    root: int = math.isqrt(m)
    for d in range(2, root + 1):
        if m % d == 0:
            other: int = m // d
            if other >= 2 and d != other:
                return True
    return False


def representable_with_a_neq_b(n: int) -> bool:
    """
    Characterization:
    n is representable as n = a + b + ab with integers a, b >= 1 and a != b
    iff m = n + 1 is composite and not equal to p^2 for a prime p.
    """
    m: int = n + 1
    return is_composite(m) and not is_prime_square(m)


def main() -> None:
    # Verify the equivalence m has unequal factor pair <=> m composite and not prime square
    for m in range(2, 10_000):
        scan: bool = has_unequal_factor_pair_ge2_scan(m)
        criterion: bool = is_composite(m) and not is_prime_square(m)
        assert scan == criterion, f"Mismatch for m={m}: scan={scan}, criterion={criterion}"

    # Demonstrate on n in [0, 100]
    valid_ns: List[int] = [n for n in range(0, 101) if representable_with_a_neq_b(n)]
    print(f"Count of n in [0, 100] representable as a + b + ab with a != b: {len(valid_ns)}")
    print(f"Sample valid n (first 15): {valid_ns[:15]}")
    print(f"Sample valid n (last 15): {valid_ns[-15:]}")


if __name__ == "__main__":
    main()