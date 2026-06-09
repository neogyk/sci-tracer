from math import gcd

# Assume the distance 'd' obtained from the previous step (solving equations) is expressed as a fraction (numerator/denominator)
numerator: int = 30  # example numerator from the solved equations
denominator: int = 24  # example denominator from the solved equations

def simplify_fraction(numer: int, denom: int) -> tuple[int, int]:
    common_divisor = gcd(numer, denom)
    return numer // common_divisor, denom // common_divisor

# Simplifying the fraction
simplified_numerator, simplified_denominator = simplify_fraction(numerator, denominator)

# Output the values
simplified_numerator, simplified_denominator