import numpy as np

class Slater:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def laplacian(self, configs: np.ndarray) -> np.ndarray:
        # configs shape: (nconf, 2, 3)
        # Compute radial distances r for each electron, with a small epsilon to avoid division by zero
        r = np.linalg.norm(configs, axis=-1) + 1e-12  # shape (nconf, 2)
        # (∇²ψ)/ψ = -2α/r + α² for each electron
        lap = -2.0 * self.alpha / r + self.alpha ** 2   # shape (nconf, 2)
        return lap

# Optional self-test for verification
if __name__ == "__main__":
    model = Slater(alpha=1.0)
    configs = np.random.randn(10, 2, 3)
    out = model.laplacian(configs)
    assert out.shape == (10, 2), f"Expected shape (10,2), got {out.shape}"
    print("Laplacian shape OK")