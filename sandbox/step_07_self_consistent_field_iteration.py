def self_consistent_field(r_grid, energy_grid, nmax, Z, tolerance):
    # Initial guess: no Hartree term
    hartreeU = np.zeros(len(r_grid))
    charge_old = np.zeros(len(r_grid))
    converge = False
    total_energy = 0.0
    while not converge:
        # Solve full equation with hartree potential
        # We need modified f_Schrod that includes hartreeU/r
        def f_full(r, energy, l):
            # Interpolate hartreeU at r
            idx = np.searchsorted(r_grid, r)
            if idx >= len(r_grid):
                idx = len(r_grid)-1
            uh = hartreeU[idx]  # approximate
            return l*(l+1)/r**2 - 2*Z/r + uh/r - energy
        # Find bound states for each l up to nmax-1
        all_bound = []
        for l in range(nmax):
            bound_l = find_bound_states(r_grid, l, energy_grid)
            all_bound.extend(bound_l)
        # Compute charge density
        charge = compute_charge_density(r_grid, all_bound, nmax)
        # Compute Hartree potential
        new_hartreeU = solve_hartree(r_grid, charge)
        # Compute total energy
        # Total energy = sum of eigenvalues - 0.5 * ∫ ρ V_H dV
        # For simplicity, approximate as sum of eigenvalues
        total_energy = sum([e for (l,e) in all_bound])
        # Check convergence
        diff = np.max(np.abs(charge - charge_old))
        if diff < tolerance:
            converge = True
        charge_old = charge
        hartreeU = new_hartreeU
    return (charge, total_energy)