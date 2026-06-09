from typing import Tuple


def count_even_odd_inclusive(n: int) -> Tuple[int, int]:
    """
    Count evens and odds in the inclusive range [0, n].
    Returns (even_count, odd_count).
    """
    # If n is even: there are n//2 odds and n//2 + 1 evens (including 0)
    # If n is odd: there are (n+1)//2 odds and (n+1)//2 evens
    if n % 2 == 0:
        return (n // 2 + 1, n // 2)
    else:
        half: int = (n + 1) // 2
        return (half, half)


def count_valid_parity_combinations(max_exp_inclusive: int = 17) -> int:
    """
    Using parity logic in (Z/12Z)^× = {1,5,7,11} with exponent 2,
    count triples (a, b, d) in [0..max_exp_inclusive]^3 such that
    7^a * 11^b * 17^d ≡ 5 (mod 12).

    Parity mapping:
      - 7^a ≡ 1 if a even else 7
      - 11^b ≡ 1 if b even else 11
      - 17^d ≡ 1 if d even else 5

    Valid patterns for product 5:
      - d odd, a even, b even  (since 5 * 1 * 1 = 5)
      - d even, a odd, b odd   (since 1 * 7 * 11 = 5)
    """
    even_count, odd_count = count_even_odd_inclusive(max_exp_inclusive)

    pattern1: int = odd_count * even_count * even_count  # d odd, a even, b even
    pattern2: int = even_count * odd_count * odd_count  # d even, a odd, b odd

    return pattern1 + pattern2


def brute_force_check(max_exp_inclusive: int = 17) -> int:
    """
    Brute-force verification by direct modular exponentiation.
    """
    total: int = 0
    for a in range(max_exp_inclusive + 1):
        for b in range(max_exp_inclusive + 1):
            for d in range(max_exp_inclusive + 1):
                prod: int = (pow(7, a, 12) * pow(11, b, 12) * pow(17, d, 12)) % 12
                if prod == 5:
                    total += 1
    return total


if __name__ == "__main__":
    count_formula: int = count_valid_parity_combinations(17)
    count_bruteforce: int = brute_force_check(17)
    assert count_formula == count_bruteforce, f"Mismatch: {count_formula=} vs {count_bruteforce=}"
    print(count_formula)  # Expected: 1458