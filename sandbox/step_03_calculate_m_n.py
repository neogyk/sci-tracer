from fractions import Fraction
from typing import Union

def calculate_m_plus_n(r: Union[int, float, Fraction]) -> int:
    """
    Converts the radius r expressed as a simple fraction and computes the sum m+n
    where r = m/n.

    :param r: Radius represented either as an int, float, or Fraction
    :return: Sum of m and n where r = m/n
    """
    if not isinstance(r, Fraction):
        r = Fraction(r).limit_denominator()
    
    m = r.numerator
    n = r.denominator
    
    return m + n

# Example usage:
radius_fraction = 3 / 5  # Example radius expressed as a fraction
result = calculate_m_plus_n(radius_fraction)
print(result)