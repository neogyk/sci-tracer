from sympy import factorint
from sympy.ntheory.factor_ import divisors

def prime_factors(n: int) -> dict[int, int]:
    """Factors an integer into its prime components."""
    return factorint(n)

def calculate_divisors(exponentiated_factors: dict[int, int]) -> list[int]:
    """Calculates divisors of a number given its prime factorization with specific exponents."""
    base_divisors = divisors(exponentiated_factors)
    return base_divisors

# Step 1: Factor 17017
factors_17017 = prime_factors(17017)

# Step 2: Calculate divisors of 17017^17
# Create the factor dictionary for 17017^17 by multiplying each factor's exponent by 17
exponentiated_factors = {p: exp * 17 for p, exp in factors_17017.items()}
divisors_17017_17 = calculate_divisors(exponentiated_factors)

divisors_17017_17