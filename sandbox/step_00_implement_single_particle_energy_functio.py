import math

def dist(r1, r2, L):
    """
    Compute the minimum image distance between two points in a periodic cubic box of side L.
    """
    dx = r1[0] - r2[0]
    dy = r1[1] - r2[1]
    dz = r1[2] - r2[2]
    # Apply periodic boundary conditions
    if dx > L/2.0:
        dx -= L
    elif dx < -L/2.0:
        dx += L
    if dy > L/2.0:
        dy -= L
    elif dy < -L/2.0:
        dy += L
    if dz > L/2.0:
        dz -= L
    elif dz < -L/2.0:
        dz += L
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def E_ij(d, sigma, epsilon):
    """
    Lennard-Jones potential between two particles at distance d.
    """
    if d == 0.0:
        # Avoid division by zero; self-interaction should be excluded elsewhere
        return 0.0
    sr = sigma / d
    sr6 = sr**6
    sr12 = sr6 * sr6
    return 4.0 * epsilon * (sr12 - sr6)

def E_i(r, positions, L, sigma, epsilon):
    """
    Compute total Lennard-Jones potential energy of a particle at position r
    with all other particles in the positions list/array.
    """
    total = 0.0
    for other in positions:
        d = dist(r, other, L)
        # Exclude self-interaction: skip if distance is zero
        if d == 0.0:
            continue
        total += E_ij(d, sigma, epsilon)
    return total