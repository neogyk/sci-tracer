from typing import List


def first_valid_ns(k: int, min_n: int = 20, modulus: int = 8, residue: int = 6) -> List[int]:
    """
    Enumerate the first k integers n such that:
      - n >= min_n
      - n ≡ residue (mod modulus)
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if modulus <= 0:
        raise ValueError("modulus must be a positive integer")
    if not (0 <= residue < modulus):
        raise ValueError("residue must satisfy 0 <= residue < modulus")

    # Smallest n >= min_n with n % modulus == residue
    diff: int = (residue - (min_n % modulus)) % modulus
    first: int = min_n + diff

    return [first + i * modulus for i in range(k)]


def main() -> None:
    terms: List[int] = first_valid_ns(k=5, min_n=20, modulus=8, residue=6)
    total: int = sum(terms)

    # Outputs:
    # First five n: [22, 30, 38, 46, 54]
    # Sum: 190
    print(terms)
    print(total)


if __name__ == "__main__":
    main()