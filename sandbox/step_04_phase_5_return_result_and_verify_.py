# Phase 5: Verify and return the Raman intensity
# Ensure I_omega is a real, non-negative finite float
assert isinstance(I_omega, (float, np.floating)) or (isinstance(I_omega, complex) and abs(I_omega.imag) < 1e-12), "I_omega must be essentially real"
I_omega_real = float(np.real(I_omega))  # use real part if complex
assert I_omega_real >= 0 and np.isfinite(I_omega_real), "Raman intensity must be non-negative and finite"
return I_omega_real