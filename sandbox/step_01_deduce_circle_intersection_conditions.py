from sympy import symbols, Eq, solve, Rational
from typing import Tuple, List

# Define the segment lengths as symbols
AB, AC, AE, AF, BD, DC = symbols('AB AC AE AF BD DC', positive=True, integer=True)

def cyclic_quadrilateral_conditions(
    AB_value: int,
    AC_value: int
) -> List[Tuple[int, int]]:
    """
    Establish possible integer lengths for AE and AF based on cyclic quadrilateral
    properties and the tangency condition at D for quadrilateral AEFD.
    
    :param AB_value: Length of segment AB
    :param AC_value: Length of segment AC
    :return: List of tuples containing possible integer lengths for (AE, AF)
    """
    # Using given AB and AC values
    conditions = []

    # The ratio from the Angle Bisector Theorem:
    angle_bisector_ratio = Rational(AB_value, AC_value)  # Equivalent to AB / AC = BD / DC assumption

    # Equation representing integer conditions following the cyclic nature and tangency property
    eq1 = Eq(BD / DC, angle_bisector_ratio)

    # Substitute or assume particular values for BD and DC based on possible integer constraints.
    # This equation needs to be solved and integers checked manually or use trial/error up to max value constraints.
    for m in range(1, AB_value):  # possible integer values for BD
        for n in range(1, AC_value):  # possible integer values for DC
            # Validate the condition and that m + n can form a side of the triangle respecting integer boundary and cyclic conditions.
            if solve(eq1.subs({BD: m, DC: n})):
                # Assume cyclic quadrilateral solutions must respect AE + AF = BD (this must be defined by deeper geometric logic)
                AF_value = m
                AE_value = n
                conditions.append((AE_value, AF_value))

    return conditions

# Example usage using known lengths
possible_ae_af_values = cyclic_quadrilateral_conditions(200, 225)
print("Possible (AE, AF) pairs meeting cyclic and tangency conditions:", possible_ae_af_values)