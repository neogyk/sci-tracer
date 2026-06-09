from typing import Tuple


def derive_lambert_series_representation() -> Tuple[str, str]:
    """
    Derive the Lambert-series representation for
      S = sum_{k>=1} 1/(10^k - 1)
    using the identity (Lambert series)
      sum_{k>=1} x^k/(1 - x^k) = sum_{n>=1} d(n) x^n, for |x| < 1,
    with x = 1/10, which yields
      S = sum_{n>=1} d(n) * 10^{-n}.
    Then split
      10^{100} S = A_100 + R_100
    with
      A_100 = sum_{n=1}^{100} d(n) 10^{100-n}   (an integer),
      R_100 = sum_{m=1}^{\\infty} d(100+m) 10^{-m}   (a fractional tail).
    Returns a pair of human-readable strings encoding these statements.
    """
    s_as_digits: str = "S = sum_{n>=1} d(n) * 10^{-n}"
    a100_r100: str = (
        "10^{100} S = sum_{n=1}^{100} d(n) 10^{100-n} + sum_{m=1}^{\\infty} d(100+m) 10^{-m} "
        "=: A_100 + R_100, with A_100 integer and R_100 fractional."
    )
    return s_as_digits, a100_r100


if __name__ == "__main__":
    S_as_digits, A100_R100 = derive_lambert_series_representation()
    # These variables contain the formal statements requested by the task.
    print(S_as_digits)
    print(A100_R100)