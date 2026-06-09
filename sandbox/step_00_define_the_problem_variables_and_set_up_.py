from sympy import symbols, solve, Eq

# Define the variables
r, d = symbols('r d', positive=True, real=True)

# Speeds for Patrick, Tanya, and Jose
speed_patrick = r
speed_tanya = r + 2
speed_jose = r + 9

# Time = Distance / Speed; However, all times are equivalent
# Set the equations based on equal travel times and equal distances
eq_patrick = Eq(d / speed_patrick, d / speed_tanya)
eq_tanya = Eq(d / speed_tanya, d / speed_jose)

# Solve the equations
solution_patrick = solve(eq_patrick, d)
solution_tanya = solve(eq_tanya, d)

# The solutions are the same for distance, let's store them
distance_eq = solution_patrick[0]

# Output the distance equation to verify the setup
distance_eq