import numpy as np

class Slater:
    def __init__(self, alpha):
        self.alpha = alpha

    def value(self, configs):
        r = np.linalg.norm(configs, axis=-1)
        # r shape: (nconf, nelec)
        return np.exp(-self.alpha * np.sum(r, axis=-1))

    def gradient(self, configs):
        r = np.linalg.norm(configs, axis=-1, keepdims=True)
        grad = -self.alpha * configs / r
        # handle r = 0 by setting gradient to 0
        grad = np.where(r != 0, grad, 0.0)
        return grad

    def laplacian(self, configs):
        r = np.linalg.norm(configs, axis=-1)
        lap = -2.0 * self.alpha / r + self.alpha**2
        # handle r = 0 by setting laplacian contribution to 0
        lap = np.where(r != 0, lap, 0.0)
        return lap

    def kinetic(self, configs):
        lap = self.laplacian(configs)
        kin = -0.5 * np.sum(lap, axis=-1)
        return kin