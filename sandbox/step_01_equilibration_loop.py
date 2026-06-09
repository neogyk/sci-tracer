# Equilibration loop: perform n_eq Monte Carlo steps without recording data
for step in range(n_eq):
    # Select a random particle
    i = np.random.randint(N)
    
    # Store old position
    old_pos = positions[i].copy()
    
    # Propose a new position by random displacement
    new_pos = old_pos + move_magnitude * (np.random.rand(3) - 0.5)
    new_pos = wrap(new_pos, L)  # apply periodic boundary conditions
    
    # Compute energy difference (delta_U = U_new - U_old)
    delta_E = E_i(new_pos, positions, sigma, epsilon, L, r_c) - E_i(old_pos, positions, sigma, epsilon, L, r_c)
    
    # Metropolis acceptance criterion
    if np.random.rand() < np.exp(-delta_E / T):
        positions[i] = new_pos
        n_accp += 1  # count accepted move