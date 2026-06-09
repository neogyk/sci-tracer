from __future__ import annotations
from typing import Dict
from math import prod


def factorize(n: int) -> Dict[int, int]:
    """Return the prime factorization of n as a dict {prime: exponent}.

    Uses trial division. Requires n >= 2.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    factors: Dict[int, int] = {}
    # Factor out 2s
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2
    # Odd factors
    p: int = 3
    while p * p <= n:
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
        p += 2
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def value_from_factors(factors: Dict[int, int]) -> int:
    """Reconstruct the integer from its prime factorization dict."""
    return prod(p ** e for p, e in factors.items())


def main() -> None:
    n: int = 17017
    factors: Dict[int, int] = factorize(n)
    # Verify the factorization multiplies back to n
    assert value_from_factors(factors) == n

    print(f"n = {n}")
    print(f"prime factorization = {factors}")

    # Confirm the expected factorization explicitly
    expected: Dict[int, int] = {7: 1, 11: 1, 13: 1, 17: 1}
    assert factors == expected, f"Unexpected factorization: {factors}"

    # Factorization of n**17 has exponents scaled by 17
    factors_n17: Dict[int, int] = {p: e * 17 for p, e in factors.items()}
    # Quick consistency check
    assert value_from_factors(factors_n17) == n ** 17
    print(f"prime factorization of n**17 = {factors_n17}")


if __name__ == "__main__":
    main()