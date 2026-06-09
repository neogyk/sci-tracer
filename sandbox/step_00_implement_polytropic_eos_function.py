def eos_press_from_rho(rho: float, eos_Gamma: float, eos_kappa: float) -> float:
    return eos_kappa * (rho ** eos_Gamma)