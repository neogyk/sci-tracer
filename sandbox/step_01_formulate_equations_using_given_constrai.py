from sympy import symbols, Eq

# Define variables for side and diagonal lengths
x, y1, y2, y3 = symbols('x y1 y2 y3')

# Construct equations for the sums of squares of side lengths
eq_side_squares = Eq(5 * x**2, 308)

# Construct equation for the sums of squares of diagonal lengths
eq_diagonal_squares = Eq(y1**2 + y2**2 + y3**2, 800)

eq_side_squares, eq_diagonal_squares