import numpy as np

def apply_nitsche_supg_boundary(A: np.ndarray, b: np.ndarray, N: int, h: float,
                                a: float, kappa: float, C: float, s_kappa: float) -> None:
    """
    Modify the global system matrix A and right-hand side b to enforce
    Dirichlet boundary condition u(1)=1 using Nitsche's method with SUPG
    stabilization boundary terms.

    Parameters
    ----------
    A : np.ndarray
        (N+1)x(N+1) global stiffness matrix, assembled without boundary treatment.
    b : np.ndarray
        Right-hand side vector of length N+1.
    N : int
        Number of elements (nodes = N+1). Last node index is N.
    h : float
        Uniform element length (m = 1/N).
    a : float
        Advection velocity (positive).
    kappa : float
        Diffusion coefficient.
    C : float
        Nitsche penalty parameter (positive constant).
    s_kappa : int
        Nitsche sign parameter, usually 1 or -1.
    """
    # Nitsche penalty term
    V_kappa = C / h * (1.0 + abs(s_kappa))
    A[N, N] += V_kappa
    b[N] += V_kappa * 1.0             # Dirichlet value u(1)=1

    # SUPG boundary integral term at x=1
    # Compute tau for the last element (element index N-1)
    Pe = a * h / (2.0 * kappa)             # element Peclet number
    if abs(a) > 1e-12:
        tau_last = h / (2.0 * abs(a)) * (1.0 / np.tanh(Pe) - 1.0 / Pe)
    else:
        tau_last = 0.0                     # pure diffusion

    # Contribution from SUPG term: ∫Γ τ a u' v ds
    # Since u' ≈ (u_N - u_{N-1})/h and v = shape function of node N at x=1,
    # we add to row N: + (τ a / h) * u_N - (τ a / h) * u_{N-1}
    if tau_last > 0.0:
        supg_coeff = tau_last * a / h
        A[N, N] += supg_coeff
        A[N, N-1] -= supg_coeff
    # b[N] is unchanged by this term (it depends on the unknown solution)