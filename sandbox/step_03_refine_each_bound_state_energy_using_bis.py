import numpy as np

# Assuming the following are already defined:
# - x: numpy array of spatial grid (1D)
# - solve_schrodinger(x, E, u_init): returns wavefunction u (numpy array)
# - bounds: list of (E_low, E_high) tuples, in order of increasing energy

tolerance = 1e-6
refined_energies = []

for i, (a, b) in enumerate(bounds):
    # Determine parity from the index (0 -> even, 1 -> odd, 2 -> even, ...)
    parity = i % 2
    if parity == 0:
        u_init = (1.0, 0.0)   # even: u(0)=1, u'(0)=0
    else:
        u_init = (0.0, 1.0)   # odd: u(0)=0, u'(0)=1

    # Bisection interval
    E_low, E_high = a, b

    # Evaluate wavefunction at endpoints to ensure sign change
    u_low = solve_schrodinger(x, E_low, u_init)
    u_high = solve_schrodinger(x, E_high, u_init)
    tail_low = u_low[-1]
    tail_high = u_high[-1]

    if tail_low * tail_high >= 0:
        # No sign change; interval may be invalid – skip or fallback
        # In practice, the sign change should exist for bound states
        continue

    while E_high - E_low > tolerance:
        E_mid = (E_low + E_high) / 2.0
        u_mid = solve_schrodinger(x, E_mid, u_init)
        tail_mid = u_mid[-1]

        if tail_low * tail_mid < 0:
            # Root lies in [E_low, E_mid]
            E_high = E_mid
            tail_high = tail_mid
        else:
            # Root lies in [E_mid, E_high]
            E_low = E_mid
            tail_low = tail_mid

    refined_energies.append((E_low + E_high) / 2.0)

# refined_energies now contains the converged energy for each bound state