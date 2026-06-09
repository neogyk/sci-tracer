from sympy import symbols, Rational, eq, solve, sqrt
from sympy.core.numbers import Integer

# Define the symbols
theta = symbols('theta', real=True)
m, n = symbols('m n', integer=True, positive=True)

# Given conditions:
cos_theta = symbols('cos_theta', real=True)
given_AB_prime = Rational(4, 3)

# Rotated coordinates for A' and B':
A_prime = (cos_theta, sqrt(1 - cos_theta**2))
B_prime = (cos_theta + 1, sqrt(1 - cos_theta**2))  # Since B starts at (1, 0) and rotates similarly

# Calculate distance AB' as sqrt((x2 - x1)^2 + (y2 - y1)^2)
AB_prime_expr = sqrt((B_prime[0] - 0)**2 + (B_prime[1] - 0)**2)

# Set the condition for distance AB' = 4/3
distance_condition = eq(AB_prime_expr, given_AB_prime)

# Solve for cos(theta)
cos_solution = solve(distance_condition, cos_theta)

# Extract one solution and convert to Rational for gcd check
cos_theta_value = cos_solution[0].simplify()

# Verify gcd(m, n) = 1 by finding m and n such that cos_theta_value = m / n
m_n_ratio = Rational(cos_theta_value).limit_denominator()
m_value = m_n_ratio.numerator
n_value = m_n_ratio.denominator

from sympy import gcd
gcd_check = gcd(m_value, n_value) == Integer(1)

{
    "cos_theta_value": cos_theta_value,
    "gcd_is_one": gcd_check,
    "m": m_value,
    "n": n_value
}