from sympy import symbols, Eq, solve

# Declare symbols for the roots
alpha, beta, gamma = symbols('alpha beta gamma', real=True)

# Using Vieta's formulas for the roots of the polynomial:
# If the original polynomial is x^3 + ax^2 + bx + c = 0 with roots alpha, beta, gamma
# Vieta gives us the following equations:
#   alpha + beta + gamma = -a
#   alpha*beta + beta*gamma + gamma*alpha = b
#   alpha * beta * gamma = -c

# Now let's compute the polynomial in terms of alpha^2, beta^2, gamma^2
# Define new transformations for the roots
alpha_squared, beta_squared, gamma_squared = alpha**2, beta**2, gamma**2

# Using Vieta's formulas on the polynomial with roots alpha^2, beta^2, gamma^2:
#   alpha^2 + beta^2 + gamma^2 = -a'
#   alpha^2*beta^2 + beta^2*gamma^2 + gamma^2*alpha^2 = b'
#   alpha^2 * beta^2 * gamma^2 = -c'

# Sum of squares can be computed as:
sum_squares = alpha_squared + beta_squared + gamma_squared

# Express the sum of squares in terms of Vieta's formulas for alpha, beta, gamma
# alpha^2 + beta^2 + gamma^2 = (alpha + beta + gamma)^2 - 2*(alpha*beta + beta*gamma + gamma*alpha)
vieta1 = alpha + beta + gamma
vieta2 = alpha*beta + beta*gamma + gamma*alpha

sum_squares_expr = vieta1**2 - 2 * vieta2

# Define equations for the new polynomial coefficients
# These would normally be expressed in terms of parameters a, b of the original cubic polynomial
eq1 = Eq(sum_squares_expr, symbols('a_prime'))

# Solve for a_prime in terms of the original Vieta parameters
solution = solve(eq1, symbols('a_prime'))

# The equations for a simplified cubic in terms of alpha squared transform
print("Expression for new polynomial coefficient a':", solution)