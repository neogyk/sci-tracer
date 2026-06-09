import numpy as np
from typing import Any

# Python code for the production loop of a Monte Carlo simulation of Lennard-Jones fluid.
# Assumes the following variables/functions are defined in the outer scope:
#   N: int - number of particles
#   L: float - box length
#   sigma, epsilon, r_c: float - LJ parameters
#   T: float - temperature in reduced units (kT/epsilon)
#   move_magnitude: float - maximum displacement per move
#   n_prod: int - number of production steps
#   insertion_freq: int - frequency of Widom insertions (every insertion_freq steps)
#   positions: np.ndarray of shape (N, 3) - particle positions (already equilibrated)
#   E_total: float - current total configurational energy (intermolecular, without tail correction)
#   E_array: np.ndarray of shape (n_prod,) - array to store energy per step (already allocated)
#   n_accp: int - global counter for accepted moves (incremented here)
# Functions assumed:
#   wrap(pos: np.ndarray, box_len: float) -> np.ndarray : wraps coordinates into [0, box_len)
#   E_i(target_pos: np.ndarray, all_positions: np.ndarray, sigma: float, epsilon: float,
#       L: float, r_c: float) -> float : computes interaction energy of a particle at target_pos
#       with all other particles in all_positions (excluding itself if target_pos coincides)
#   Widom_insertion(positions: np.ndarray, sigma: float, epsilon: float, L: float,
#                   r_c: float) -> float : computes interaction energy of a randomly inserted
#       test particle with all system particles

# Initialize accumulation variables
mu_ext_sum: float = 0.0
insertion_count: int = 0

for step in range(n_prod):
    # --- Metropolis move attempt ---
    i: int = np.random.randint(N)                     # choose random particle
    old_pos: np.ndarray = positions[i].copy()
    # Generate trial displacement
    displacement: np.ndarray = move_magnitude * (np.random.rand(3) - 0.5)
    new_pos: np.ndarray = old_pos + displacement
    new_pos = wrap(new_pos, L)                        # apply periodic boundary conditions

    # Energy change for particle i
    delta_E: float = (E_i(new_pos, positions, sigma, epsilon, L, r_c) -
                      E_i(old_pos, positions, sigma, epsilon, L, r_c))

    # Metropolis acceptance criterion
    if np.random.random() < np.exp(-delta_E / T):
        positions[i] = new_pos                        # accept move
        E_total += delta_E                            # update total energy incrementally
        n_accp += 1                                   # increment acceptance counter

    # Record total configurational energy after the step (without tail correction)
    E_array[step] = E_total

    # --- Widom insertion test ---
    if step % insertion_freq == 0:
        E_insert: float = Widom_insertion(positions, sigma, epsilon, L, r_c)
        mu_ext_sum += np.exp(-E_insert / T)
        insertion_count += 1

# After the loop, mu_ext_sum and insertion_count are ready for post‑processing.
# The excess chemical potential can be computed as:
#   if insertion_count > 0:
#       mu_excess = -T * np.log(mu_ext_sum / insertion_count)
#   else:
#       mu_excess = 0.0
# The corrected energy array (with long‑range tail correction) will be computed later.