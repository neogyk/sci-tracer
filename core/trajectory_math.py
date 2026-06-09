from __future__ import annotations

import numpy as np


def _log_map(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Riemannian logarithmic map on the unit hypersphere.

    Returns the tangent vector at p that points toward q:
        v = log_p(q)
    such that exp_p(v) = q.
    """
    cos_angle = float(np.clip(np.dot(p, q), -1.0, 1.0))
    angle = np.arccos(cos_angle)
    if angle < 1e-7:
        return np.zeros_like(p)
    tangent = q - cos_angle * p
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-7:
        return np.zeros_like(p)
    return (angle / tangent_norm) * tangent


def _exp_map(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Riemannian exponential map on the unit hypersphere.

    Moves from p along the geodesic with initial tangent v:
        exp_p(v)  →  point on the sphere after travelling ‖v‖ radians.
    """
    norm_v = float(np.linalg.norm(v))
    if norm_v < 1e-7:
        return p
    return np.cos(norm_v) * p + np.sin(norm_v) * (v / norm_v)


def _christoffel_contraction(
    x: np.ndarray,
    v: np.ndarray,
    metric_fn,
    h: float = 1e-4,
) -> np.ndarray:
    """Contracted Christoffel term  Γ^k_{ij} v^i v^j  for diagonal metric g = diag(metric_fn(x)).

    For a diagonal metric g = (g₁, …, g_d) the non-zero symbols are:
        Γ^k_{kk} =  (∂_k g_k) / (2 g_k)
        Γ^k_{ik} =  (∂_i g_k) / (2 g_k)   i ≠ k
        Γ^k_{ii} = −(∂_k g_i) / (2 g_k)   i ≠ k

    The directional-derivative approximation (2 metric evaluations) gives:
        ∂g_k/∂x^i ≈ (dg_k/dv̂) · v̂^i

    which yields the vectorised contraction:
        Γ_v^k = [2 v^k (dg_k/dv̂ · ‖v‖) − v̂^k Σ_i(dg_i/dv̂ · (v^i)²)] / (2 g_k)
    """
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-8:
        return np.zeros_like(x)
    v_hat = v / v_norm

    g      = np.asarray(metric_fn(x),           dtype=np.float64)
    g_fwd  = np.asarray(metric_fn(x + h * v_hat), dtype=np.float64)
    g_bwd  = np.asarray(metric_fn(x - h * v_hat), dtype=np.float64)
    dg_dv  = (g_fwd - g_bwd) / (2.0 * h)       # ∂g_k/∂x · v̂  for every k, shape (d,)

    g_safe = g + 1e-8
    term1  = 2.0 * v * (dg_dv * v_norm)                      # (d,)
    term2  = v_hat * float(np.dot(dg_dv, v ** 2))            # (d,)  scalar broadcast
    return (term1 - term2) / (2.0 * g_safe)


def _curvature_contraction(
    x: np.ndarray,
    v: np.ndarray,
    metric_fn,
    h: float = 1e-4,
) -> np.ndarray:
    """Contracted Riemann curvature term  R^k_{abc} v^a v^b v^c  (approximate).

    Estimated as the directional derivative of the Christoffel contraction:
        R_v^k ≈ (Γ_v^k(x + h v̂) − Γ_v^k(x − h v̂)) / (2h) · ‖v‖
    """
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-8:
        return np.zeros_like(x)
    v_hat = v / v_norm

    gf = _christoffel_contraction(x + h * v_hat, v, metric_fn, h)
    gb = _christoffel_contraction(x - h * v_hat, v, metric_fn, h)
    return (gf - gb) / (2.0 * h) * v_norm


def prolongate_trajectory(
    embeddings: list[list[float]],
    dt: float = 1.0,
    fisher_diag: np.ndarray | None = None,
    method: str = "fisher",
    metric_fn = None,
) -> list[float]:
    """Predict the next trajectory point via geodesic prolongation.

    Two options (selected by *method*):

    ``"fisher"``  (default)
        Spherical log / exp map with optional Fisher-metric natural-gradient
        preconditioning of the tangent vector:
            v_ng = v / (g_ii + ε),  rescaled to ‖v_ng‖ = ‖v‖
            γ(t+1) = exp_{S_n}(dt · v_ng)

    ``"christoffel"``
        Second-order geodesic equation with Christoffel correction (requires
        *metric_fn*: callable np.ndarray → np.ndarray returning the FIM diagonal):
            x^k(t+dt) ≈ x^k + v^k dt
                         − ½ Γ^k_{ij} v^i v^j (dt)²    [Christoffel correction]
                         − ⅙ R^k_{abc} v^a v^b (dt)³   [curvature correction]
        The result is re-normalised to the unit sphere.
    """
    arr = np.array(embeddings, dtype=np.float64)
    if len(arr) < 2:
        norm = np.linalg.norm(arr[-1]) + 1e-12
        return (arr[-1] / norm).tolist()

    S_n    = arr[-1] / (np.linalg.norm(arr[-1])    + 1e-12)
    S_prev = arr[-2] / (np.linalg.norm(arr[-2])    + 1e-12)

    v = _log_map(S_n, S_prev)

    if method == "christoffel" and metric_fn is not None:
        # ── Option 2: Christoffel + curvature correction ──────────────────────
        gamma_v = _christoffel_contraction(S_n, v, metric_fn)
        r_v     = _curvature_contraction(S_n, v, metric_fn)

        x_new = S_n + v * dt - 0.5 * gamma_v * dt ** 2 - (1.0 / 6.0) * r_v * dt ** 3
        norm  = np.linalg.norm(x_new)
        return (x_new / (norm + 1e-12)).tolist()

    else:
        # ── Option 1: Fisher-preconditioned spherical geodesic ────────────────
        if fisher_diag is not None:
            g         = np.asarray(fisher_diag, dtype=np.float64)
            eps       = float(np.mean(g)) * 1e-3
            v_ng      = v / (g + eps)
            v_norm    = np.linalg.norm(v)
            v_ng_norm = np.linalg.norm(v_ng)
            if v_ng_norm > 1e-8:
                v = v_ng * (v_norm / v_ng_norm)

        return _exp_map(S_n, dt * v).tolist()

