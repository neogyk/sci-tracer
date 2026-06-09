from sympy import symbols, Eq, solve, simplify, Rational

# Define symbols
x, r = symbols('x r', real=True)

# Circle centered at (4, 39) and radius r
circle_equation = (x - 4)**2 + (Rational((x**2 - 8*x + 12), 2) - 39)**2 - r**2

# Solve for x to find conditions where circle and parabola are tangent
discriminant = simplify(Eq(circle_equation.as_poly().discriminant(x), 0))
tangency_rs = solve(discriminant, r)

# Sum all possible real values of r that satisfy the tangency condition
sum_of_rs = sum(tangency_rs)

# Output the sum of all valid r values
sum_of_rs