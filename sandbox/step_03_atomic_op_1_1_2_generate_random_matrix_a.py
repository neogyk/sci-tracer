import numpy as np

# Assuming n and num_eigenvalues are defined in the outer scope
np.random.seed(42)  # Ensure reproducibility
V, _ = np.linalg.qr(np.random.randn(n, num_eigenvalues + 2))