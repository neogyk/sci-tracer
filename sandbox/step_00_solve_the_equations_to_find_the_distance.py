from sympy import symbols, Eq, solve, Rational

# Define variables
v, D = symbols('v D', real=True, positive=True)
pat_time = 20 / 60  # converting 20 minutes to hours
tan_time = 15 / 60  # converting 15 minutes to hours
jos_time = 12 / 60  # converting 12 minutes to hours

# Establish equations based on d = rt
eq_patrick = Eq(D, v * pat_time)
eq_tanya = Eq(D, (v + 2) * tan_time)
eq_jose = Eq(D, (v + 9) * jos_time)

# Solve equations for D in terms of v
d_expr_p = solve(eq_patrick, D)[0]
d_expr_t = solve(eq_tanya, D)[0]
d_expr_j = solve(eq_jose, D)[0]

# Equate the expressions to solve for v
eq_v1 = Eq(d_expr_p, d_expr_t)
eq_v2 = Eq(d_expr_p, d_expr_j)

# Solve the system of equations for v
v_solution = solve((eq_v1, eq_v2), v)

# Substitute back to find D
D_value = d_expr_p.subs(v, v_solution[0])

# Express D as a fraction m/n
D_fraction = Rational(D_value).limit_denominator()
m, n = D_fraction.as_numer_denom()

{"code": f"# The distance D from school to park expressed as a fraction m/n is {D_fraction}.\n# The values of m and n are {m} and {n} respectively."}