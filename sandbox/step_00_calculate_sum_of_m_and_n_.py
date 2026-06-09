from sympy import symbols, Eq, solve, Rational
from math import gcd

# Define symbols
v, D = symbols('v D', positive=True, real=True)

# Travel times for each person using D = rt (t = D/r)
patrick_time = D / v
tanya_time = D / (v + 2)
jose_time = D / (v + 9)

# Equations based on given information
eq_tanya = Eq(tanya_time, patrick_time - 1/6)
eq_jose = Eq(jose_time, tanya_time - 3/4)

# Solve for D in terms of v using the above equations
solutions = solve([eq_tanya, eq_jose], (v, D))
D_value = solutions[0][1]  # Extract the value of D from the solutions

# Express D as a fraction
D_fraction = Rational(D_value).limit_denominator()
m, n = D_fraction.numerator, D_fraction.denominator

# Ensure m and n are coprime
def are_coprime(a: int, b: int) -> bool:
    return gcd(a, b) == 1

# Check if m and n are coprime and calculate m + n
if are_coprime(m, n):
    result = m + n
else:
    raise ValueError("m and n are not coprime which is unexpected")

result