from typing import List

def base_503_representation(n: int) -> List[int]:
    """Convert a number n into its base-503 representation."""
    digits = []
    while n > 0:
        digits.append(n % 503)
        n //= 503
    return digits

def lucas_theorem_binomial_congruence(m: int, n: int, p: int) -> int:
    """
    Calculate the congruence \\(\\binom{m}{n}\\) mod p using Lucas' Theorem.
    
    Parameters:
    m (int): The upper index of the binomial coefficient.
    n (int): The lower index of the binomial coefficient.
    p (int): A prime number (here, 503).
    
    Returns:
    int: The congruence of the binomial coefficient mod p.
    """
    m_digits = base_503_representation(m)
    n_digits = base_503_representation(n)
    
    result = 1
    for m_i, n_i in zip(m_digits, n_digits):
        # If n_i > m_i, then the binomial is 0 mod p
        if n_i > m_i:
            return 0
        # Multiply with binomial coefficient of the digits mod p
        result *= factorial(m_i) // (factorial(n_i) * factorial(m_i - n_i))
        result %= p
        
    return result

def factorial(k: int) -> int:
    """Compute the factorial of k."""
    result = 1
    for i in range(2, k+1):
        result *= i
    return result

def calculate_congruence_10000_mod_503_for_r(r: int, p: int = 503) -> int:
    """
    Calculate the sum \\(\\sum_{m \\ge 0} \\binom{10000}{502m + r}\\) mod 503.
    
    Parameters:
    r (int): The offset in the index.
    p (int): A prime number (here it defaults to 503).
    
    Returns:
    int: The calculated congruence for the specified r.
    """
    total_congruence = 0
    m = 0
    while True:
        n = 502 * m + r
        if n > 10000:
            break
        congruence = lucas_theorem_binomial_congruence(10000, n, p)
        total_congruence += congruence
        total_congruence %= p
        m += 1
    return total_congruence