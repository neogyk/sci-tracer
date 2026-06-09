import numpy as np

class Slater:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def value(self, configs: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(configs, axis=-1)
        val = np.exp(-self.alpha * r[:, 0]) * np.exp(-self.alpha * r[:, 1])
        return val

    def gradient(self, configs: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(configs, axis=-1, keepdims=True) + 1e-12
        grad = -self.alpha * configs / r
        return grad

    def laplacian(self, configs: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(configs, axis=-1) + 1e-12
        lap = -2.0 * self.alpha / r + self.alpha**2
        return lap

    def kinetic(self, configs: np.ndarray) -> np.ndarray:
        lap = self.laplacian(configs)
        kin = -0.5 * np.sum(lap, axis=1)
        return kin

def test_derivatives():
    np.random.seed(42)
    nconf = 10
    configs = np.random.randn(nconf, 2, 3) * 0.5  # random positions
    psi = Slater(alpha=1.0)

    # Analytical gradient and laplacian
    ana_grad = psi.gradient(configs)
    ana_lap = psi.laplacian(configs)

    # Numerical gradient of log psi
    eps = 1e-5
    num_grad = np.zeros_like(configs)
    for i in range(nconf):
        for j in range(2):
            for k in range(3):
                configs_plus = configs.copy()
                configs_plus[i, j, k] += eps
                configs_minus = configs.copy()
                configs_minus[i, j, k] -= eps
                psi_plus = psi.value(configs_plus)[i]
                psi_minus = psi.value(configs_minus)[i]
                num_grad[i, j, k] = (np.log(psi_plus) - np.log(psi_minus)) / (2*eps)

    # Numerical laplacian (∇²ψ)/ψ
    psi_center = psi.value(configs)
    num_lap = np.zeros((nconf, 2))
    for i in range(nconf):
        for j in range(2):
            second_der_sum = 0.0
            for k in range(3):
                configs_plus = configs.copy()
                configs_plus[i, j, k] += eps
                configs_minus = configs.copy()
                configs_minus[i, j, k] -= eps
                psi_plus = psi.value(configs_plus)[i]
                psi_minus = psi.value(configs_minus)[i]
                second_der = (psi_plus - 2*psi_center[i] + psi_minus) / (eps**2)
                second_der_sum += second_der
            num_lap[i, j] = second_der_sum / psi_center[i]

    # Compare
    atol = 1e-4
    max_grad_diff = np.max(np.abs(ana_grad - num_grad))
    max_lap_diff = np.max(np.abs(ana_lap - num_lap))
    print(f"Max gradient difference: {max_grad_diff:.6e}")
    print(f"Max laplacian difference: {max_lap_diff:.6e}")
    assert max_grad_diff < atol, f"Gradient difference too large: {max_grad_diff}"
    assert max_lap_diff < atol, f"Laplacian difference too large: {max_lap_diff}"
    print("All tests passed!")

if __name__ == "__main__":
    test_derivatives()