import numpy as np

# Example inputs (replace as needed):
# v_det from previous step: vector from sample to pixel in detector coordinates
v_det = np.array([1.0, 2.0, 1000.0])  # mm
# D: detector orientation matrix (3x3) built from yaw, pitch, roll
D = np.eye(3)  # placeholder identity; actual matrix from detector angles
# wl: X‑ray wavelength in Ångström
wl = 1.0  # Å

# Compute lab‑frame vector
v_lab = D @ v_det

# Normalise and scale to magnitude 1/λ
norm_v = np.linalg.norm(v_lab)
ks = (1.0 / wl) * v_lab / norm_v

# ks is now the scattered wavevector in units 1/Å