from sympy import symbols, Eq, solve

# Define the symbols for the speeds and distance
v, x = symbols('v x', positive=True)

# Equations formulated based on travel times for each contestant
eq_patrick = Eq(x / v + 1, x / (v + 2))
eq_tanya = Eq(x / (v + 2) + 1, x / (v + 9))

# Solving the system of equations for x and v
solutions = solve([eq_patrick, eq_tanya], (x, v))
solutions