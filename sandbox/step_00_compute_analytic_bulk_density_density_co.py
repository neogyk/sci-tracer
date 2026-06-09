import numpy as np

def D_b_qz_analy(qz, D0, bg_eps, q, d):
    # Vacuum permittivity conversion constant: e^2/(epsilon0) = 1/55.26349406 eV*um
    eps0_conv = 55.26349406
    # Compute V_q in meV*Angstrom^2
    V_q = (2 * np.pi * 1e7) / (bg_eps * q * eps0_conv)
    # Compute S(q_z)
    qd = q * d
    # Avoid overflow for large qd
    if qd > 500:
        S = 0.0  # effectively zero because sinh/cosh ratio tends to 0
    else:
        S = np.sinh(qd) / (np.cosh(qd) - np.cos(qz * d))
    # Dyson equation in Fourier space
    denom = 1.0 - V_q * S * D0
    D_b_qz = D0 / denom
    return D_b_qz