import numpy as np

class Slater:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def value(self, configs: np.ndarray) -> np.ndarray:
        # ψ = exp(-α r₁) exp(-α r₂)
        r = np.linalg.norm(configs, axis=-1)
        val = np.exp(-self.alpha * r[:, 0]) * np.exp(-self.alpha * r[:, 1])
        return val

    def gradient(self, configs: np.ndarray) -> np.ndarray:
        # (∇ψ)/ψ = (-α r̂₁, -α r̂₂)
        # configs shape: (nconf, 2, 3)
        r = np.linalg.norm(configs, axis=-1, keepdims=True) + 1e-12  # avoid division by zero
        grad = -self.alpha * configs / r
        return grad  # shape (nconf, 2, 3)

    def laplacian(self, configs: np.ndarray) -> np.ndarray:
        # (∇²ψ)/ψ per electron: -2α/r + α²
        r = np.linalg.norm(configs, axis=-1) + 1e-12
        lap = -2.0 * self.alpha / r + self.alpha ** 2
        return lap  # shape (nconf, 2)

    def kinetic(self, configs: np.ndarray) -> np.ndarray:
        # -½ Σₑ (∇²ψ/ψ)ₑ
        lap = self.laplacian(configs)
        kin = -0.5 * np.sum(lap, axis=1)
        return kin  # shape (nconf,)