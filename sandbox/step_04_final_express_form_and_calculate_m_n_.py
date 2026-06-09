from sympy import sqrt

# Previously calculated side length square sum
side_len_square_sum = 308

# Calculate length of one side assuming all are equal in a regular pentagon
x_square = side_len_square_sum / 5
x = sqrt(x_square)

# Calculate perimeter
perimeter = 5 * x

# Calculate square of the perimeter
perimeter_square = perimeter**2

# perimeter_square should be represented as m * sqrt(n), let's simplify it
# In this case, since perimeter is already a simplified expression based on the constraints
# Let's assume the expression perimeter_square can be rewritten directly if needed

# However, perimeter_square here is sqrt(1540)
m = 5 ** 2
n = x_square

m_n_sum = m + n

# Results
print(f"m = {m}, n = {n}, m + n = {m_n_sum}")