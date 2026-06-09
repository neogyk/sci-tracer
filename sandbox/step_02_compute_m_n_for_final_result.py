from sympy import symbols, Eq, solve
from math import gcd

# Define variables
r = symbols('r')
d = symbols('d')

# Set up the equations
# Patrick's speed: r, Tanya's speed: r + 2, Jose's speed: r + 9
# Patrick, Tanya, and Jose arrive simultaneously, so we can set up d = r*t, d = (r+2)*t, d = (r+9)*t

patrick_time = d / r
tanya_time = d / (r + 2)
jose_time = d / (r + 9)

# Solve the equation patrick_time = tanya_time and tanya_time = jose_time

eq1 = Eq(patrick_time, tanya_time)
eq2 = Eq(tanya_time, jose_time)

# Solve for distance d using simultaneous equations
solution_d = solve((eq1, eq2), (d, r))

# Extract the distance as an exact fraction
distance_fraction = solution_d[d].simplify()

# Reduce fraction to its lowest terms, and calculate m + n
numerator, denominator = distance_fraction.as_numer_denom()
numerator = numerator // gcd(numerator, denominator)
denominator = denominator // gcd(numerator, denominator)

m_plus_n = numerator + denominator

m_plus_n