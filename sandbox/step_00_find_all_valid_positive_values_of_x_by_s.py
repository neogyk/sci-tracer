from sympy import symbols, Eq, solve, log

# Define the variable and equation
x = symbols('x', positive=True, real=True)
equation = Eq(x**(log(x, 2026)/20), 26 * x)

# Solve the equation for x
solutions = solve(equation, x)

solutions