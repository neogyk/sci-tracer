from sympy import symbols, Eq, solve, simplify, discriminant

def find_tangency_criteria() -> set:
    # Define the variables
    x, r = symbols('x r', real=True)
    
    # Parabola equation: 2y = x^2 - 8x + 12
    # Solve for y: y = (x^2 - 8x + 12) / 2
    y_eq = (x**2 - 8*x + 12) / 2

    # Circle equation: (x-4)^2 + (y-39)^2 = r^2
    # Substituting y in circle equation
    circle_eq = (x - 4)**2 + (y_eq - 39)**2 - r**2

    # Simplify and expand the equation with y substituted
    simplified_eq = simplify(circle_eq)

    # Calculate the discriminant of the quadratic expression
    disc = discriminant(Eq(simplified_eq, 0), x)

    # Find values of r that make the discriminant zero (tangency condition)
    tangency_criteria_r = solve(Eq(disc, 0), r)
    
    return set(tangency_criteria_r)

tangency_criteria = find_tangency_criteria()
tangency_criteria