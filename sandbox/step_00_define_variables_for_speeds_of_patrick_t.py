from sympy import symbols

# Define the symbolic variable for Patrick's speed
v = symbols('v', integer=True)

# Define Tanya's speed as v + 2
tanya_speed = v + 2

# Define Jose's speed as (v + 2) + 7 = v + 9
jose_speed = v + 9

# The speeds of Patrick, Tanya, and Jose are defined symbolically
(v, tanya_speed, jose_speed)