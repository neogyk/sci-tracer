from sympy import symbols, Eq, solve
from math import gcd

# Define the symbolic variable for Patrick's speed
v = symbols('v')

# Equations setup
# Let total time for Patrick be T hours, the total distance D is the same for all three:
# Patrick's equation: D = v * T
# Tanya's equation: D = (v + 2) * (T - 1)
# Jose's equation: D = (v + 9) * (T - 2)

# Substitute D from Patrick's equation in the others
# (v * T) = ((v + 2) * (T - 1)) = ((v + 9) * (T - 2))

eq1 = Eq(v * symbols('T'), (v + 2) * (symbols('T') - 1))
eq2 = Eq(v * symbols('T'), (v + 9) * (symbols('T') - 2))

# Solving the system of equations
solutions = solve((eq1, eq2), (v, symbols('T')))

# Retrieving the relevant solution for v and calculating D with respect to T
patrick_speed = solutions[0][0]
total_distance = patrick_speed * solutions[0][1]

# Simplifying the distance in form m/n
m = total_distance.as_numer_denom()[0]
n = total_distance.as_numer_denom()[1]

# To its simplest fraction form
gcd_value = gcd(m, n)
m_simplified = m // gcd_value
n_simplified = n // gcd_value

# Calculate m + n as the final result
final_result = m_simplified + n_simplified

final_result