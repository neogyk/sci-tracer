from math import isqrt
from typing import List, Tuple


def sieve_primes_upto(n: int) -> List[int]:
    """
    Return a list of all prime numbers <= n using the Sieve of Eratosthenes.
    """
    if n < 2:
        return []
    sieve: List[bool] = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    p: int = 2
    while p * p <= n:
        if sieve[p]:
            step_start: int = p * p
            sieve[step_start : n + 1 : p] = [False] * (((n - step_start) // p) + 1)
        p += 1
    return [i for i, is_prime in enumerate(sieve) if is_prime]


def prime_squares_leq(limit: int) -> Tuple[List[int], int]:
    """
    Enumerate prime squares <= limit and return (list_of_squares, count).
    """
    max_prime: int = isqrt(limit)
    primes: List[int] = sieve_primes_upto(max_prime)
    squares: List[int] = [p * p for p in primes if p * p <= limit]
    return squares, len(squares)


if __name__ == "__main__":
    limit_val: int = 101
    squares_list, count_prime_squares = prime_squares_leq(limit_val)
    print(f"Prime squares <= {limit_val}: {squares_list}")
    print(f"Count: {count_prime_squares}")