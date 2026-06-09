from sympy import factorint
from itertools import combinations

def generate_divisors_from_factors(factors: dict[int, int]) -> list[int]:
    """
    Generate all divisors from the prime factorization.

    :param factors: A dictionary containing prime factors and their exponents.
    :return: A list of all divisors.
    """
    divisors = [1]
    for prime, exp in factors.items():
        current_divisors = []
        for divisor in divisors:
            for e in range(1, exp * 17 + 1):  # Factor is raised to the power of 17
                current_divisors.append(divisor * (prime ** e))
        divisors.extend(current_divisors)
    return sorted(divisors)

# Prime factorization of 17017
prime_factors = factorint(17017)

# Get all divisors of 17017^17
divisors = generate_divisors_from_factors(prime_factors)
print(f"Number of divisors: {len(divisors)}")  # Only for verification purposes