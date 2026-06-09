from sympy import symbols, solve, Eq, sqrt

# Define the variable for the side length
x = symbols('x', positive=True, real=True)

# Set up the equation as per the problem statement
# The sum of the squares of the sides is given by: 5*x^2 = 308
equation = Eq(5 * x**2, 308)

# Solve the equation for x
solution = solve(equation, x)

# Extract the positive solution for the side length
side_length = solution[0]

# Output the side length
side_length