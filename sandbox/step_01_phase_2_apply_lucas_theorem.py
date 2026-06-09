from typing import List, Tuple

def base_503_representation(n: int) -> List[int]:
    """
    Convert a number to its base-503 representation.
    
    Args:
    n (int): The number to convert.
    
    Returns:
    List[int]: A list of digits in base-503 representation.
    """
    representation = []
    while n > 0:
        representation.append(n % 503)
        n //= 503
    return representation

def lucas_congruence(bin_coefficient: Tuple[int, int]) -> int:
    """
    Apply Lucas' Theorem to calculate the congruence of a binomial coefficient.

    Args:
    bin_coefficient (Tuple[int, int]): A tuple representing (m, n) for binomial coefficient C(m, n).

    Returns:
    int: The congruence modulo prime 503.
    """
    m, n = bin_coefficient
    m_digits = base_503_representation(m)
    n_digits = base_503_representation(n)

    # Apply Lucas' Theorem only if the lengths match, fill with 0s if needed
    length_diff = abs(len(m_digits) - len(n_digits))
    if len(m_digits) < len(n_digits):
        m_digits.extend([0] * length_diff)
    else:
        n_digits.extend([0] * length_diff)

    # Calculate the product of binomial coefficients of corresponding digits modulo 503
    congruence = 1
    for mi, ni in zip(m_digits, n_digits):
        if mi < ni:
            return 0
        congruence *= binomial_coefficient(mi, ni) % 503
        congruence %= 503
    return congruence

def binomial_coefficient(m: int, n: int) -> int:
    """
    Calculate binomial coefficient C(m, n).

    Args:
    m (int): Top number of the binomial coefficient.
    n (int): Bottom number of the binomial coefficient.

    Returns:
    int: The binomial coefficient value.
    """
    if m < n:
        return 0
    if n == 0 or m == n:
        return 1
    num = 1
    denom = 1
    for i in range(1, n + 1):
        num *= m
        denom *= i
        m -= 1
    return num // denom

def check_congruences_for_r(r: int, max_val: int) -> List[int]:
    """
    Determine the congruence of binomial coefficients for a given r.

    Args:
    r (int): The current value of r being checked.
    max_val (int): The maximum value, i.e., 10000.

    Returns:
    List[int]: List of congruence results for each term modulo 503.
    """
    congruences = []
    m = 0
    while 502 * m + r <= max_val:
        congruence = lucas_congruence((max_val, 502 * m + r))
        congruences.append(congruence)
        m += 1
    return congruences

def check_all_congruences() -> List[Tuple[int, List[int]]]:
    """
    Calculate congruences for each r from 0 to 501.

    Returns:
    List[Tuple[int, List[int]]]: A list of tuples with each r and its corresponding congruence results.
    """
    max_val = 10000
    all_congruences = []
    for r in range(503):
        congruences = check_congruences_for_r(r, max_val)
        all_congruences.append((r, congruences))
    return all_congruences