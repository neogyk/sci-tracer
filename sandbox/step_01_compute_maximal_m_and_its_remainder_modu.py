from typing import Tuple


def sum_range(a: int, b: int) -> int:
    """
    Inclusive sum of consecutive integers from a to b.
    """
    if a > b:
        raise ValueError("Lower bound a must be <= upper bound b.")
    n: int = b - a + 1
    return (a + b) * n // 2


def compute_M_and_remainder_mod_1000() -> Tuple[int, int]:
    """
    Compute the maximal M for the 8x8 grid assignment described and M mod 1000.
    Returns:
        (M, M % 1000)
    """
    # High partition (B): assign larger labels to larger degrees
    SB4: int = sum_range(47, 64)  # 18 values
    SB3: int = sum_range(35, 46)  # 12 values
    SB2: int = sum_range(33, 34)  # 2 values

    # Low partition (W): assign smaller labels to larger degrees
    SW4: int = sum_range(1, 18)    # 18 values
    SW3: int = sum_range(19, 30)   # 12 values
    SW2: int = sum_range(31, 32)   # 2 values

    M: int = 4 * SB4 + 3 * SB3 + 2 * SB2 - (4 * SW4 + 3 * SW3 + 2 * SW2)
    remainder: int = M % 1000
    return M, remainder


if __name__ == "__main__":
    M_val, rem = compute_M_and_remainder_mod_1000()
    print(M_val)
    print(rem)