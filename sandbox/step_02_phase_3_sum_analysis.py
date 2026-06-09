def base_503_representation(number: int) -> list[int]:
    """
    Converts a given number into its base-503 representation.
    
    :param number: The number to be converted.
    :return: A list of digits representing the number in base-503.
    """
    digits = []
    while number > 0:
        digits.append(number % 503)
        number //= 503
    return digits


def lucas_congruences(m: int, n: int, prime: int) -> bool:
    """
    Determines if binomial coefficient C(m, n) is 0 modulo a given prime using Lucas' Theorem.
    
    :param m: The upper index of the binomial coefficient.
    :param n: The lower index of the binomial coefficient.
    :param prime: The prime number for modulus.
    :return: True if C(m, n) is 0 modulo `prime`, False otherwise.
    """
    m_digits = base_503_representation(m)
    n_digits = base_503_representation(n)
    
    for md, nd in zip(m_digits, n_digits):
        if md < nd:
            return True
    return False


def analyze_sums(max_value: int, prime: int) -> int:
    """
    Analyzes which sums S_r = sum(C(max_value, 502*m + r)) are multiples of `prime`.

    :param max_value: The maximum value of m in the binomial sum.
    :param prime: The given prime number for modulus.
    :return: The count of r for which S_r is divisible by `prime`.
    """
    count_divisible = 0
    for r in range(503):
        sr_congruent_zero = True
        
        for m in range((max_value // 502) + 1):
            if not lucas_congruences(max_value, 502 * m + r, prime):
                sr_congruent_zero = False
                break
        
        if sr_congruent_zero:
            count_divisible += 1

    return count_divisible


# Execute the analysis based on the task
if __name__ == "__main__":
    max_value = 10000
    prime = 503
    result = analyze_sums(max_value, prime)
    print("Count of S_r divisible by {}: {}".format(prime, result))