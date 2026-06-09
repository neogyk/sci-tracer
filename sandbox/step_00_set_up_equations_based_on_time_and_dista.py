from sympy import symbols, Eq

# Define symbolic variables
v, D, T = symbols('v D T', positive=True, integer=True)

# Setup equations based on the problem statement
patrick_time = T
tanya_time = T - 1
jose_time = T - 2

# Patrick's equation for distance
patrick_eq = Eq(D, v * patrick_time)

# Tanya's equation (Tanya is 2 km/hr faster than Patrick)
tanya_eq = Eq(D, (v + 2) * tanya_time)

# Jose's equation (Jose is 7 km/hr faster than Tanya or 9 km/hr faster than Patrick)
jose_eq = Eq(D, (v + 9) * jose_time)

# Equations representing travel dynamics are formulated: patrick_eq, tanya_eq, jose_eq
equations = [patrick_eq, tanya_eq, jose_eq]