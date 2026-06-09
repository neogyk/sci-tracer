import numpy as np

class Slater:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def laplacian(self, configs: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(configs, axis=-1) + 1e-12
        lap = -2.0 * self.alpha / r + self.alpha**2
        return lap  # shape (nconf,2)

    def kinetic(self, configs: np.ndarray) -> np.ndarray:
        lap = self.laplacian(configs)
        kin = -0.5 * np.sum(lap, axis=1)
        return kin