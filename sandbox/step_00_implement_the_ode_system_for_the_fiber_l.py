import numpy as np

def f(z, y, Pssat, Ppsat, N, sigma_ap, sigma_ep, sigma_as, sigma_es, gamma_p, alpha_p, gamma_s, alpha_s):
    Pp_tot = y[0] + y[1]
    Ps_tot = y[2] + y[3]
    num = (sigma_ap/(sigma_ap+sigma_ep)) * (Pp_tot/Ppsat) + (sigma_as/(sigma_as+sigma_es)) * (Ps_tot/Pssat)
    den = (Pp_tot/Ppsat) + 1 + (Ps_tot/Pssat)
    n2 = num / den
    # Forward pump derivative
    dPp_f = -gamma_p * (sigma_ap * N - (sigma_ap+sigma_ep) * N * n2) * y[0] - alpha_p * y[0]
    # Backward pump derivative
    dPp_b = gamma_p * (sigma_ap * N - (sigma_ap+sigma_ep) * N * n2) * y[1] + alpha_p * y[1]
    # Forward signal derivative
    dPs_f = gamma_s * ((sigma_es+sigma_as) * N * n2 - sigma_as * N) * y[2] - alpha_s * y[2]
    # Backward signal derivative
    dPs_b = -gamma_s * ((sigma_es+sigma_as) * N * n2 - sigma_as * N) * y[3] + alpha_s * y[3]
    return np.array([dPp_f, dPp_b, dPs_f, dPs_b])