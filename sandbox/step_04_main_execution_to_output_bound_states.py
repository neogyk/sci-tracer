import numpy as np
from scipy.integrate import simpson

def f_x(x, En):
    """Return f(x) = x^2 - En for the harmonic oscillator Schrödinger equation."""
    return x**2 - En

def Numerov(x, En, u0, up0, step):
    """
    Numerov integration for u'' = f(x) u.
    Returns u array for given x, energy En, initial conditions (u0, up0) and step size.
    """
    n = len(x)
    u = np.zeros(n)
    u[0] = u0
    h = step
    h2 = h**2
    # Taylor expansion for second point
    f0 = f_x(x[0], En)
    u[1] = u0 + h * up0 + 0.5 * h2 * f0 * u0
    f = f_x(x, En)
    for i in range(1, n - 1):
        # Numerov recurrence
        numerator = 2 * (1 - 5 * h2 / 12 * f[i]) * u[i] - (1 + h2 / 12 * f[i-1]) * u[i-1]
        denominator = 1 + h2 / 12 * f[i+1]
        u[i+1] = numerator / denominator
    return u

def Solve_Schrod(x, En, u_b, up_b, step):
    """Normalized wavefunction using Numerov and Simpson integration."""
    u = Numerov(x, En, u_b, up_b, step)
    norm = simpson(u**2, x)
    u_norm = u / np.sqrt(norm)
    return u_norm

def find_bound_states(x, Emax, Estep):
    """
    Shoot for bound states: scan energy from 0 to Emax.
    Returns list of (n_nodes, E) for found states.
    """
    xmax = x[-1]
    E_vals = np.arange(0, Emax, Estep)
    found = []
    for En in E_vals:
        u_norm = Solve_Schrod(x, En, 0.0, 1.0, x[1] - x[0])
        # Count sign changes (nodes)
        signs = np.sign(u_norm)
        n_nodes = np.sum(signs[1:] != signs[:-1])
        # Check if wavefunction dies at far right boundary (within tolerance)
        if abs(u_norm[-1]) < 1e-6:
            found.append((n_nodes, En))
    # Remove possible duplicates and sort by energy
    found = sorted(set(found), key=lambda x: x[1])
    return found

# Main execution
if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)
    Emax = 10.0
    Estep = 0.01
    bound_states = find_bound_states(x, Emax, Estep)
    print("Found bound states (nodes, energy):")
    for n, E in bound_states:
        print(f"n = {n}, E = {E:.6f}")