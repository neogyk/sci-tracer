from math import gcd

def simplify_fraction(numerator: int, denominator: int) -> tuple[int, int]:
    """
    Function to simplify a fraction and return it as a tuple (numerator, denominator)
    """
    common_divisor = gcd(numerator, denominator)
    return numerator // common_divisor, denominator // common_divisor

def calculate_m_plus_n(numerator: int, denominator: int) -> int:
    """
    Function to calculate the value of m + n given that the fraction is m/n in its simplest form
    """
    # Simplify the fraction
    num, denom = simplify_fraction(numerator, denominator)
    
    # Return the sum of the numerator and denominator
    return num + denom

# Example usage:
# Assuming the fraction that represents the distance is 275/396
numerator = 275
denominator = 396

# Calculating m + n
m_plus_n = calculate_m_plus_n(numerator, denominator)
m_plus_n