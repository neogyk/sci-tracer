import numpy as np
from scipy.integrate import solve_bvp
from typing import Callable, Tuple, Optional

# --- Physical and geometric parameters ---
L: float = 10.0                    # fiber length [m]
N: float = 1.0e25                  # total doping concentration [m^-3]
tau: float = 10.0e-3               # spontaneous lifetime [s]
sigma_ap: float = 2.0e-24          # pump absorption cross-section [m^2]
sigma_ep: float = 2.0e-24          # pump emission cross-section [m^2]
sigma_as: float = 5.0e-25          # signal absorption cross-section [m^2]
sigma_es: float = 5.0e-25          # signal emission cross-section [m^2]
gamma_p: float = 0.8               # pump overlap factor
gamma_s: float = 0.7               # signal overlap factor
alpha_p: float = 0.0               # pump background loss [1/m] (neglected)
alpha_s: float = 0.0               # signal background loss [1/m] (neglected)
h: float = 6.626e-34               # Planck constant [J·s]
c: float = 3.0e8                   # speed of light [m/s]
lambda_p: float = 980e-9           # pump wavelength [m]
lambda_s: float = 1550e-9          # signal wavelength [m]
A_core: float = np.pi * (5.0e-6)**2  # core area [m^2]

# Pump and signal saturation powers
Ppsat: float = (h * c / lambda_p) * A_core / (tau * (sigma_ap + sigma_ep))
Pssat: float = (h * c / lambda_s) * A_core / (tau * (sigma_as + sigma_es))

# Cavity mirror reflectivities
R1: float = 0.99       # high reflector at z=0
R2: float = 0.04       # output coupler at z=L (4% output)

# Pump powers at the ends
Ppl: float = 1.0       # forward pump power at z=0 [W]
Ppr: float = 0.0       # backward pump power at z=L [W]


def ode(z: float, y: np.ndarray, Ppsat: float, Pssat: float, N: float,
        sigma_ap: float, sigma_ep: float, sigma_as: float, sigma_es: float,
        gamma_p: float, gamma_s: float, alpha_p: float, alpha_s: float) -> np.ndarray:
    """
    ODE system for four fields:
        y[0] = forward pump power  (Pp^+)
        y[1] = backward pump power (Pp^-)
        y[2] = forward signal power (Ps^+)
        y[3] = backward signal power (Ps^-)
    Returns dydz array.
    """
    Pp_f, Pp_b, Ps_f, Ps_b = y

    # Total pump and signal power at this point
    Pp_total = Pp_f + Pp_b
    Ps_total = Ps_f + Ps_b

    # Normalised population inversion n2 (fraction)
    # n2 = (sigma_ap * Pp_total / (Ppsat * gamma_p) + sigma_as * Ps_total / (Pssat * gamma_s)) /
    #       (1/tau + (sigma_ap+sigma_ep) * Pp_total / (Ppsat * gamma_p) + (sigma_as+sigma_es) * Ps_total / (Pssat * gamma_s))
    # Actually using the standard form: n2 = (R_ap + R_as) / (1/tau + R_ep + R_es)
    # where R_ap = sigma_ap * gamma_p * Pp_total / (h*nu_p * A_core) etc.
    # But simplified with saturation powers:
    # R_ap = sigma_ap * Pp_total / (Ppsat * tau)  (if gamma factors included properly)
    # Let's use the consistent dimensionless form:
    # n2 = (sigma_ap*Pp_total/(gamma_p*Ppsat) + sigma_as*Ps_total/(gamma_s*Pssat)) /
    #      (1/(tau*N) + (sigma_ap+sigma_ep)*Pp_total/(gamma_p*Ppsat) + (sigma_as+sigma_es)*Ps_total/(gamma_s*Pssat))
    # But because N is already incorporated? Actually the rate equation for population density n2 (m^-3) is:
    # dn2/dt = (sigma_ap*I_p/(h*nu_p) + sigma_as*I_s/(h*nu_s))*n1 - (sigma_ep*I_p/(h*nu_p) + sigma_es*I_s/(h*nu_s) + 1/tau)*n2
    # n1 = N - n2. In steady state, solve for n2/N.
    # Using intensity = gamma*P/A_core.
    # So define: Gamma_p = gamma_p/A_core, Gamma_s = gamma_s/A_core.
    # R_ap = sigma_ap * gamma_p * Pp / (h*nu_p * A_core)
    # R_ep = sigma_ep * gamma_p * Pp / (h*nu_p * A_core)
    # Same for signal.
    # n2/N = (R_ap + R_as) / (R_ap + R_ep + R_as + R_es + 1/tau)
    # We'll compute directly in m^-3.

    # Intensities (actually photon flux)
    Ip = gamma_p * Pp_total / (h * c / lambda_p * A_core)   # photon flux for pump [1/(m^2 s)]
    Is = gamma_s * Ps_total / (h * c / lambda_s * A_core)   # photon flux for signal

    R_ap = sigma_ap * Ip
    R_ep = sigma_ep * Ip
    R_as = sigma_as * Is
    R_es = sigma_es * Is

    n2_over_N = (R_ap + R_as) / (R_ap + R_ep + R_as + R_es + 1.0/tau)
    n2 = n2_over_N * N   # excited population density [m^-3]

    # Gain coefficients
    gp = gamma_p * (sigma_ep * n2 - sigma_ap * (N - n2))   # net pump gain per unit length
    gs = gamma_s * (sigma_es * n2 - sigma_as * (N - n2))   # net signal gain per unit length

    # Loss (background) added
    alpha_p_inv = alpha_p
    alpha_s_inv = alpha_s

    # Derivatives: dP/dz = (g - alpha) * P, with sign for forward/backward
    dPp_f_dz = (gp - alpha_p_inv) * Pp_f
    dPp_b_dz = -(gp - alpha_p_inv) * Pp_b
    dPs_f_dz = (gs - alpha_s_inv) * Ps_f
    dPs_b_dz = -(gs - alpha_s_inv) * Ps_b

    return np.array([dPp_f_dz, dPp_b_dz, dPs_f_dz, dPs_b_dz])


def bc(ya: np.ndarray, yb: np.ndarray, Ppl: float, Ppr: float, R1: float, R2: float) -> np.ndarray:
    """
    Boundary conditions for a Fabry-Pérot cavity.
    ya: values at z=0, yb: values at z=L.
    Residuals should be zero.
    """
    # Forward pump at z=0 injected: ya[0] - Ppl = 0
    # Backward pump at z=L: no injection: yb[1] - Ppr = 0
    # Backward signal at z=0 multiplied by R1 equals forward signal at z=0: ya[3] - R1*ya[2] = 0
    # Forward signal at z=L multiplied by R2 equals backward signal at z=L: yb[2] - R2*yb[3] = 0
    return np.array([
        ya[0] - Ppl,
        yb[1] - Ppr,
        ya[3] - R1 * ya[2],
        yb[2] - R2 * yb[3]
    ])


# --- Prepare initial mesh and guess ---
n_mesh: int = 50
z_mesh: np.ndarray = np.linspace(0, L, n_mesh)

# Initial guess for y: uniform distribution (e.g., constant values)
# For pump, we start with forward pump decaying roughly. For signal, use a small constant.
Pp_init_guess: float = Ppl * np.exp(-1.0)   # some decay
Ps_init_guess: float = 1e-3                 # small signal
y_init: np.ndarray = np.zeros((4, n_mesh))
y_init[0, :] = Pp_init_guess
y_init[1, :] = Ppr * 0.1
y_init[2, :] = Ps_init_guess
y_init[3, :] = Ps_init_guess * 0.1  # small backward

# --- Solve BVP ---
sol = solve_bvp(
    lambda z, y: ode(z, y, Ppsat, Pssat, N, sigma_ap, sigma_ep, sigma_as, sigma_es,
                     gamma_p, gamma_s, alpha_p, alpha_s),
    lambda ya, yb: bc(ya, yb, Ppl, Ppr, R1, R2),
    z_mesh,
    y_init,
    tol=1e-6,
    max_nodes=5000
)

if not sol.success:
    raise RuntimeError(f"BVP solver failed: {sol.message}")

# --- Extract results ---
z_fine = np.linspace(0, L, 200)
y_fine = sol.sol(z_fine)

# Output power transmitted through output mirror at z=L
# Forward signal at z=L transmitted through mirror R2: P_out = (1 - R2) * Ps^+(L)
P_out: float = (1.0 - R2) * y_fine[2, -1]
print(f"Output power: {P_out:.6f} W")

# Population inversion along fiber (n2/N fraction)
# Compute from the same formula using the solved powers
Pp_total_fine = y_fine[0, :] + y_fine[1, :]
Ps_total_fine = y_fine[2, :] + y_fine[3, :]

Ip_fine = gamma_p * Pp_total_fine / (h * c / lambda_p * A_core)
Is_fine = gamma_s * Ps_total_fine / (h * c / lambda_s * A_core)

R_ap_fine = sigma_ap * Ip_fine
R_ep_fine = sigma_ep * Ip_fine
R_as_fine = sigma_as * Is_fine
R_es_fine = sigma_es * Is_fine

n2_over_N_fine = (R_ap_fine + R_as_fine) / (R_ap_fine + R_ep_fine + R_as_fine + R_es_fine + 1.0/tau)

# Print some stats
print(f"Maximum inversion fraction: {np.max(n2_over_N_fine):.4f}")
print(f"Minimum inversion fraction: {np.min(n2_over_N_fine):.4f}")