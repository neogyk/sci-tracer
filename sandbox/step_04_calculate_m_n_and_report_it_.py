from sympy import symbols, Eq, solve, gcd

# Setup variables for the speeds of the travelers and the distance
v, x = symbols('v x', positive=True)

# Formulate equations based on each traveler's speed and start time
eq_patrick = Eq(x / v + 1, x / (v + 2))
eq_tanya = Eq(x / (v + 2) + 1, x / (v + 9))

# Solve the system of equations to find the distance x
solutions = solve([eq_patrick, eq_tanya], (x, v))
distance = solutions[x]

# Extract and verify the distance from the solution, express as fraction
m, n = distance.as_numer_denom()
assert gcd(m, n) == 1

# Calculate m + n and report it
result = m + n
result