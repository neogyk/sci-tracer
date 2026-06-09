from typing import List, Tuple


def precompute_factorials(p: int) -> Tuple[List[int], List[int]]:
    """
    Precompute factorials and inverse factorials modulo prime p up to p-1.
    """
    fac: List[int] = [1] * p
    for i in range(1, p):
        fac[i] = (fac[i - 1] * i) % p

    invfac: List[int] = [1] * p
    # Fermat's little theorem for inverse of fac[p-1]
    invfac[p - 1] = pow(fac[p - 1], p - 2, p)
    for i in range(p - 2, -1, -1):
        invfac[i] = (invfac[i + 1] * (i + 1)) % p
    return fac, invfac


def nCk_mod_p_small(n: int, k: int, p: int, fac: List[int], invfac: List[int]) -> int:
    """
    Compute C(n, k) mod p for 0 <= n, k <= p-1 using precomputed factorials.
    """
    if k < 0 or k > n:
        return 0
    return (fac[n] * invfac[k] % p) * invfac[n - k] % p


def nCk_mod_p_lucas(N: int, K: int, p: int, fac: List[int], invfac: List[int]) -> int:
    """
    Compute C(N, K) mod p using Lucas' theorem with base p digit decomposition.
    """
    res: int = 1
    n_val: int = N
    k_val: int = K
    while n_val > 0 or k_val > 0:
        n_i: int = n_val % p
        k_i: int = k_val % p
        if k_i > n_i:
            return 0
        res = (res * nCk_mod_p_small(n_i, k_i, p, fac, invfac)) % p
        n_val //= p
        k_val //= p
    return res


def main() -> None:
    # Parameters
    p: int = 503
    m: int = p - 1  # 502
    n: int = 10000

    # Precompute factorials mod p
    fac, invfac = precompute_factorials(p)

    # Aggregate S_r = sum_{k ≡ r mod m} C(n, k) (mod p)
    S: List[int] = [0] * m
    for k in range(n + 1):
        ck: int = nCk_mod_p_lucas(n, k, p, fac, invfac)
        r: int = k % m
        S[r] = (S[r] + ck) % p

    zeros: List[int] = [r for r in range(m) if S[r] % p == 0]
    count_zeros: int = len(zeros)

    print(count_zeros, zeros[:5], zeros[-5:])

    # Verify the expected property
    assert count_zeros == 39, f"Expected 39 zeros, got {count_zeros}"
    assert zeros == list(range(463, 502)), f"Unexpected zero positions: {zeros}"


if __name__ == "__main__":
    main()