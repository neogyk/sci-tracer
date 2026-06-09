import numpy as np

class Jastrow:
    """
    Jastrow wave function for two electrons: psi = exp(beta * |r1 - r2|)
    """
    def __init__(self, beta: float):
        """
        Args:
            beta: Jastrow parameter (non-zero float)
        """
        self.beta = beta

    def get_r_vec(self, configs: np.ndarray) -> np.ndarray:
        """
        Compute difference vector r1 - r2.
        
        Args:
            configs: shape (nconf, nelec, ndim) with nelec=2
        Returns:
            r_vec: shape (nconf, ndim)
        """
        return configs[:, 0, :] - configs[:, 1, :]

    def get_r_ee(self, configs: np.ndarray) -> np.ndarray:
        """
        Compute interelectronic distance |r1 - r2|.
        
        Args:
            configs: shape (nconf, nelec, ndim)
        Returns:
            r_ee: shape (nconf,)
        """
        r_vec = self.get_r_vec(configs)
        return np.linalg.norm(r_vec, axis=-1)

    def value(self, configs: np.ndarray) -> np.ndarray:
        """
        Evaluate wave function.
        
        Args:
            configs: shape (nconf, nelec, ndim)
        Returns:
            psi: shape (nconf,)
        """
        r_ee = self.get_r_ee(configs)
        return np.exp(self.beta * r_ee)

    def gradient(self, configs: np.ndarray) -> np.ndarray:
        """
        Compute gradient of log wave function: grad_i psi / psi.
        
        Args:
            configs: shape (nconf, nelec, ndim)
        Returns:
            grad: shape (nconf, nelec, ndim)
        """
        r_vec = self.get_r_vec(configs)      # (nconf, ndim)
        r_ee = self.get_r_ee(configs)        # (nconf,)
        factor = self.beta / r_ee            # (nconf,)
        # Add dimension for broadcasting: (nconf, 1, ndim) scaling
        factor_3d = factor[:, np.newaxis, np.newaxis]  # (nconf, 1, 1)
        # For electron 0: factor * r_vec; for electron 1: factor * (-r_vec)
        # Equivalent to stacking positive and negative parts
        grad_pos = factor[:, np.newaxis] * r_vec          # (nconf, ndim)
        grad_neg = -grad_pos                              # (nconf, ndim)
        return np.stack([grad_pos, grad_neg], axis=1)     # (nconf, 2, ndim)

    def laplacian(self, configs: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian of log wave function: del^2_i psi / psi.
        
        Args:
            configs: shape (nconf, nelec, ndim)
        Returns:
            lap: shape (nconf, nelec)
        """
        r_ee = self.get_r_ee(configs)               # (nconf,)
        lap_per_electron = (self.beta / r_ee) * (self.beta * r_ee + 2.0)  # (nconf,)
        # Same value for both electrons; shape (nconf, 2)
        return np.column_stack([lap_per_electron, lap_per_electron])