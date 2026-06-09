import numpy as np
from sympy import symbols, Eq, solve

# Define symbols
x, r = symbols('x r')

# Circle equation (x-4)^2 + (y-39)^2 = r^2
# Parabola equation 2y = x^2 - 8x + 12
# From parabola, solve for y
y = (x**2 - 8*x + 12) / 2

# Substituting y in circle equation and simplifying
circle_eq = Eq((x - 4)**2 + ((y - 39)**2), r**2)

# Expanding and simplifying the expressions
expanded_circle_eq = circle_eq.expand()

# Resulting equation to find intersection, in terms of x and r
intersection_equation = expanded_circle_eq.simplify()

# Output the equation
intersection_equation