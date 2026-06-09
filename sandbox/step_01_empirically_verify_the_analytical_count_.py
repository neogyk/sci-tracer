from itertools import product


def count_valid_palindromes() -> int:
    """
    Empirically verify the analytical count of palindromes with no zeros and digit sum 13.
    Only odd-length palindromes are possible. For half-length n in 1..6, enumerate
    all half-tuples (digits 1..9), compute S=sum(half), and middle digit m=13-2*S.
    Count when m is an odd digit between 1 and 9 inclusive.
    """
    count: int = 0
    for n in range(1, 7):
        for half in product(range(1, 10), repeat=n):
            s: int = sum(half)
            m: int = 13 - 2 * s
            if 1 <= m <= 9 and (m % 2 == 1):
                count += 1
    return count


def main() -> None:
    result: int = count_valid_palindromes()
    assert result == 62, f"Expected 62, got {result}"
    print(result)


if __name__ == "__main__":
    main()