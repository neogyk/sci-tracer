import numpy as np

class Slater:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def value(self, configs: np.ndarray) -> np.ndarray:
        """
        Compute the value of the wave function ψ = exp(-α r₁) * exp(-α r₂).
        
        Args:
            configs: numpy array of shape (nconf, 2, 3) with electron positions.
        
        Returns:
            numpy array of shape (nconf,) containing ψ for each configuration.
        """
        # Radial distances for both electrons: shape (nconf, 2)
        r = np.linalg.norm(configs, axis=-1)
        # ψ = exp(-α r₁) * exp(-α r₂)
        val = np.exp(-self.alpha * r[:, 0]) * np.exp(-self.alpha * r[:, 1])
        return val