import numpy as np
from sympy import symbols, cos, solve, Rational

def calculate_cos_theta():
    # Define known coordinates and distance
    # B is initially at origin (0, 0) and A at (1, 0) on a 2D plane
    A = np.array([1, 0])  # A(1, 0)
    B = np.array([0, 0])  # B(0, 0)

    # A' is the coordinates of A after rotation around B by angle theta
    # B' is the coordinates of B after rotation around A' by angle -theta
    # Let d be the distance AB' = 4/3 which is given
    d = Rational(4, 3)

    # The distance formula gives (B' coordinates to be determined in previous tasks)
    # (A'_x - B'_x)**2 + (A'_y - B'_y)**2 = d**2

    # Using the law of cosines: 
    # c^2 = a^2 + b^2 - 2ab*cos(theta)
    # where c = d, a = 1, b = 1 (since both are distance 1 from origin)
    # d^2 = 1 + 1 - 2 * 1 * 1 * cos(theta)

    # Define the symbol for theta and set up the equation
    theta = symbols('theta')
    equation = (d)**2 - (1 + 1 - 2 * cos(theta))

    # Solve for cos(theta)
    cos_theta_solution = solve(equation, cos(theta))
    
    # cos_theta_solution is in form [solution1, solution2] choose real value if necessary
    cos_theta = cos_theta_solution[0]  # assuming first solution is valid, need more context

    # Assuming cos_theta = m/n, find m and n
    m, n = cos_theta.as_numer_denom()

    return m + n

# Execute the function
result = calculate_cos_theta()
print(result)