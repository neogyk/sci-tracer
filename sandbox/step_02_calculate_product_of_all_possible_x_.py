from sympy import symbols, solve, Eq, log

def calculate_product_of_solutions() -> float:
    # Define the variable
    x = symbols('x', positive=True, real=True)

    # Define the equation: sqrt(20)^log_2026(x) = 26 * x
    # Transformed solvable algebraic equation based on earlier steps
    # Let's assume we already have the possible solutions [solution1, solution2, ...]
    solutions = solve(Eq(x**(log(x, 2026)/20), 26 * x), x)

    # Calculate the product of all positive solutions
    product_of_x = 1
    for sol in solutions:
        if sol.is_real and sol > 0:
            product_of_x *= sol

    return float(product_of_x)

product = calculate_product_of_solutions()
product