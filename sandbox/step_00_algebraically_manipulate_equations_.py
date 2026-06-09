from sympy import symbols, solve

def calculate_side_length() -> float:
    # Define the variable for the side length
    x = symbols('x', positive=True, real=True)
    
    # Equation for the sum of squares of side lengths
    equation = 5 * x**2 - 308
    
    # Solve the equation for x
    solutions = solve(equation, x)
    
    # Return the positive solution for x
    if solutions:
        side_length = solutions[0]
        return float(side_length)
    else:
        return 0.0
    
# Call the function and print the specific side length
side_length = calculate_side_length()
print("Exact side length x is:", side_length)