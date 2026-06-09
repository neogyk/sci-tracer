from sympy import symbols, Eq, solve, Rational

# Define symbols
v_p, v_t, v_j = symbols('v_p v_t v_j')  # Speeds of Patrick, Tanya, Jose
t = symbols('t', positive=True)         # Time when they all met
D = symbols('D')                        # Total distance they traveled

# Given conditions
v_p = 90  # Patrick's speed is 90 km/h
v_t = v_p - 10  # Tanya's speed is 10 less than Patrick's
v_j = v_p - 20  # Jose's speed is 20 less than Patrick's

# Time relations
# Patrick starts at t=0, Tanya at t=2 hours, Jose at t=3 hours
t_p = t
t_t = t - 2
t_j = t - 3

# Equations based on D = vt
eq_p = Eq(v_p * t_p, D)
eq_t = Eq(v_t * t_t, D)
eq_j = Eq(v_j * t_j, D)

# Solve the equations to find common time
common_time_val = solve((eq_p, eq_t, eq_j, Eq(D, v_p * t_p)), (t, D))

# Extract D and t from the solution
total_distance = common_time_val[0][1]

# Ensure distance is in fractional form m/n
total_distance_fraction = Rational(total_distance).cancel()

# Extract numerator and denominator
m = total_distance_fraction.numerator
n = total_distance_fraction.denominator

# Calculate sum m + n
m_plus_n = m + n

m_plus_n