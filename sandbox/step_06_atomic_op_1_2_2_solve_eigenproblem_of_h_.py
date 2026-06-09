# Atomic op 1.2.2: Solve eigenproblem of H
theta, y = np.linalg.eigh(H)
# Ensure ascending order (eigh already returns sorted, but explicit)
idx = np.argsort(theta)
theta = theta[idx]
y = y[:, idx]