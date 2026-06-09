from math import comb
from sympy import Eq, solve, symbols

# Define symbols for red and blue marbles
r, b = symbols('r b', integer=True, positive=True)

# Define the equations for probabilities of drawing 4 and 5 red marbles
prob_4_red = comb(r, 4) * comb(b, 1)
prob_5_red = comb(r, 5)

# Simplify the condition of equal probabilities
prob_eq = Eq(prob_4_red, prob_5_red)

# Solve the equation to express the relationship between the number of red (r) and blue (b) marbles
relationship = solve(prob_eq, b)

# Return the relationship derived
relationship