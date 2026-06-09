from typing import Dict


def tau(n: int) -> int:
    """
    Compute the number of positive divisors of n (also known as d(n) or tau(n)).
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    cnt: int = 1
    x: int = n
    p: int = 2
    while p * p <= x:
        e: int = 0
        while x % p == 0:
            x //= p
            e += 1
        if e > 0:
            cnt *= (e + 1)
        p = 3 if p == 2 else p + 2  # check 2, then odd numbers
    if x > 1:
        cnt *= 2
    return cnt


def compute_A100_mod_1000() -> int:
    """
    Compute A_100 modulo 1000 where
    A_100 = sum_{n=1}^{100} d(n) * 10^{100-n}.
    Modulo 1000, only n in {98, 99, 100} contribute.
    """
    vals: Dict[int, int] = {98: tau(98), 99: tau(99), 100: tau(100)}
    result: int = (100 * vals[98] + 10 * vals[99] + vals[100]) % 1000
    return result


if __name__ == "__main__":
    ans: int = compute_A100_mod_1000()
    print(ans)