import numpy as np
from typing import Callable, Tuple

def derivatives(maxwell, fields: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
    """
    Compute time derivatives of the electromagnetic fields in Lorentz gauge.
    
    Args:
        maxwell: object with attributes:
            delta: grid spacing (float, uniform)
            nx, ny, nz: number of grid points along each axis
            inner_bc: dict specifying how to apply inner boundary conditions
            outer_bc: dict specifying outer boundary type (e.g., 'outgoing')
        fields: tuple of 7 arrays (E_x, E_y, E_z, A_x, A_y, A_z, phi)
            shape (nx, ny, nz)
    
    Returns:
        tuple of 7 arrays of same shape:
            (E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot)
    """
    E_x, E_y, E_z, A_x, A_y, A_z, phi = fields
    dx = maxwell.delta
    nx, ny, nz = maxwell.nx, maxwell.ny, maxwell.nz
    
    # ---- finite-difference operators (second-order) ----
    # 1. First derivative along axis (0=x,1=y,2=z)
    def d1(f: np.ndarray, axis: int, d: float) -> np.ndarray:
        """Second-order first derivative. Centred in interior, one-sided at boundaries."""
        grad = np.zeros_like(f)
        # interior: central difference
        slc = [slice(1, -1)] * 3
        slc[axis] = slice(1, -1)
        if axis == 0:
            grad[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2 * d)
        elif axis == 1:
            grad[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * d)
        else:  # axis=2
            grad[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * d)
        # boundaries: second-order one-sided
        # left boundary (index 0)
        if axis == 0:
            grad[0, :, :] = (-3*f[0,:,:] + 4*f[1,:,:] - f[2,:,:]) / (2*d)
            grad[-1,:,:] = (3*f[-1,:,:] - 4*f[-2,:,:] + f[-3,:,:]) / (2*d)
        elif axis == 1:
            grad[:,0,:] = (-3*f[:,0,:] + 4*f[:,1,:] - f[:,2,:]) / (2*d)
            grad[:,-1,:] = (3*f[:,-1,:] - 4*f[:,-2,:] + f[:,-3,:]) / (2*d)
        else:
            grad[:,:,0] = (-3*f[:,:,0] + 4*f[:,:,1] - f[:,:,2]) / (2*d)
            grad[:,:,-1] = (3*f[:,:,-1] - 4*f[:,:,-2] + f[:,:,-3]) / (2*d)
        return grad
    
    # 2. Second derivative (needed for Laplacian)
    def d2(f: np.ndarray, axis: int, d: float) -> np.ndarray:
        """Second-order second derivative."""
        d2f = np.zeros_like(f)
        # interior
        slc = [slice(1, -1)] * 3
        slc[axis] = slice(1, -1)
        if axis == 0:
            d2f[1:-1,:,:] = (f[2:,:,:] - 2*f[1:-1,:,:] + f[:-2,:,:]) / (d**2)
        elif axis == 1:
            d2f[:,1:-1,:] = (f[:,2:,:] - 2*f[:,1:-1,:] + f[:,:-2,:]) / (d**2)
        else:
            d2f[:,:,1:-1] = (f[:,:,2:] - 2*f[:,:,1:-1] + f[:,:,:-2]) / (d**2)
        # boundaries: second-order one-sided (Neumann not needed for Laplacian on boundaries? We'll use interior values there too)
        # For simplicity, use same one-sided formula as for d1 but careful: second derivative one-sided:
        # For right end: f''[-1] = (2f[-1] - 5f[-2] + 4f[-3] - f[-4]) / d^2 (second order)
        # But for Laplacian we only need interior; boundary values will be overwritten by BCs, so we can leave them zero or compute approximate.
        # We'll compute using the same d1 applied twice, but that introduces extra points. Alternatively, compute directly.
        # Here we only need Laplacian for interior points; for boundaries we will apply BCs later.
        # So we can keep d2f as computed for interior and set boundaries to zero (they will be corrected by BC function).
        return d2f
    
    # 3. Laplacian
    def laplacian(f: np.ndarray) -> np.ndarray:
        lap = d2(f, 0, dx) + d2(f, 1, dx) + d2(f, 2, dx)
        return lap
    
    # 4. Divergence of a vector (F_x, F_y, F_z)
    def divergence(Fx: np.ndarray, Fy: np.ndarray, Fz: np.ndarray) -> np.ndarray:
        div = d1(Fx, 0, dx) + d1(Fy, 1, dx) + d1(Fz, 2, dx)
        return div
    
    # 5. Gradient of a scalar
    def gradient(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gx = d1(f, 0, dx)
        gy = d1(f, 1, dx)
        gz = d1(f, 2, dx)
        return gx, gy, gz
    
    # ---- compute derivatives using evolution equations ----
    # A_i dot
    grad_phi_x, grad_phi_y, grad_phi_z = gradient(phi)
    A_x_dot = -E_x - grad_phi_x
    A_y_dot = -E_y - grad_phi_y
    A_z_dot = -E_z - grad_phi_z
    
    # φ dot
    phi_dot = -divergence(A_x, A_y, A_z)
    
    # E_i dot: -Laplacian(A_i) + gradient(divergence(A))
    lap_A_x = laplacian(A_x)
    lap_A_y = laplacian(A_y)
    lap_A_z = laplacian(A_z)
    divA = divergence(A_x, A_y, A_z)
    grad_divA_x, grad_divA_y, grad_divA_z = gradient(divA)
    
    E_x_dot = -lap_A_x + grad_divA_x
    E_y_dot = -lap_A_y + grad_divA_y
    E_z_dot = -lap_A_z + grad_divA_z
    
    # ---- apply boundary conditions ----
    # Convert to list for mutability
    derivs = [E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot]
    # Inner (symmetry) BCs: x=0, y=0, z=0
    # For x=0 (i=0): cylindrical symmetry: fields that are odd in x vanish at x=0 => time derivative zero.
    # E_x, A_x are odd in x -> set derivative to 0
    derivs[0][0, :, :] = 0.0   # E_x_dot at x=0
    derivs[3][0, :, :] = 0.0   # A_x_dot at x=0
    # For y=0 (j=0): similarly E_y, A_y odd -> derivative 0
    derivs[1][:, 0, :] = 0.0   # E_y_dot at y=0
    derivs[4][:, 0, :] = 0.0   # A_y_dot at y=0
    # For z=0 (k=0): mirror symmetry. E_z, A_z odd -> derivative 0; φ even -> derivative even? We set derivative = interior value? 
    # Actually, since φ is even, its z-derivative is odd at z=0 => 0. Ensure phi_dot at z=0 is 0.
    derivs[2][:, :, 0] = 0.0   # E_z_dot at z=0
    derivs[5][:, :, 0] = 0.0   # A_z_dot at z=0
    derivs[6][:, :, 0] = 0.0   # phi_dot at z=0 (since φ is even, its derivative is odd and must be zero)
    
    # Outer BCs: apply outgoing wave condition (Sommerfeld) at outer boundaries.
    # For each outer face, we set the derivative to -c * (outward normal derivative) where c=1.
    # We compute the normal derivative using a one-sided difference that is consistent with the interior.
    # For simplicity, we use first-order upwind: ∂_t f = - sign * (f - f_neighbor)/dx
    # But more accurate is to use the existing d1 at the boundary (one-sided) then set derivative accordingly.
    # We'll reuse the d1 function on each field at the boundaries.
    # Since we have the fields available, we can compute the boundary derivative for each field and set the derivative.
    # For each component in derivs list, we modify the outer boundaries.
    # We'll do it for all seven components.
    field_components = [E_x, E_y, E_z, A_x, A_y, A_z, phi]
    names = ['E_x', 'E_y', 'E_z', 'A_x', 'A_y', 'A_z', 'phi']
    for idx, (deriv, field) in enumerate(zip(derivs, field_components)):
        # x outer (right, i=nx-1): ∂_t f = -∂_x f   (outward normal is +x)
        # compute ∂_x f at i=nx-1 using one-sided backward difference (second-order)
        fx_right = (3*field[-1,:,:] - 4*field[-2,:,:] + field[-3,:,:]) / (2*dx)
        deriv[-1, :, :] = -fx_right
        # x outer left (i=0) but we already set inner BC for some components; for components not set to zero, apply outgoing? Actually left is inner boundary, already handled.
        # We'll only do right boundary for x.
        # y outer (j=ny-1): ∂_t f = -∂_y f
        fy_right = (3*field[:,-1,:] - 4*field[:,-2,:] + field[:,-3,:]) / (2*dx)
        deriv[:, -1, :] = -fy_right
        # z outer (k=nz-1): ∂_t f = -∂_z f
        fz_right = (3*field[:,:,-1] - 4*field[:,:,-2] + field[:,:,-3]) / (2*dx)
        deriv[:, :, -1] = -fz_right
    
    return tuple(derivs)