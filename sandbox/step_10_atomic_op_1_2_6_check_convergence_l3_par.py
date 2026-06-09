norms = np.linalg.norm(residuals, axis=0)
for i in range(len(theta)):
    if norms[i] < threshold and theta[i] not in converged:
        converged.append(theta[i])