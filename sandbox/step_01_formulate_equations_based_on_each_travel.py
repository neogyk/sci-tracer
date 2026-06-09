from sympy import symbols, Eq

# Initialize symbolic variables
v = symbols('v', positive=True)
x = symbols('x', positive=True)

# Formulate equations
eq_patrick = Eq(x / v + 1, x / (v + 2))
eq_tanya = Eq(x / (v + 2) + 1, x / (v + 9))