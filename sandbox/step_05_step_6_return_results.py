import numpy as np
from typing import List, Tuple, Dict

def compute_B_matrix(a: float, b: float, c: float,
                     alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Compute the orthogonalization matrix B from lattice parameters.
    B maps reciprocal lattice coordinates (h,k,l) to Cartesian coordinates (2π/λ scaled).
    """
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)

    # Basis vectors in crystal Cartesian system (orthogonal, right-handed)
    ax = np.array([a, 0.0, 0.0])
    ay = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0.0])
    az = np.array([
        c * np.cos(beta_rad),
        c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad),
        c * np.sqrt(1.0 - np.cos(beta_rad)**2 -
                    ((np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad))**2)
    ])

    # Volume of unit cell
    V = np.dot(ax, np.cross(ay, az))

    # Reciprocal lattice vectors in Cartesian (factor 2π added later via numpy convention)
    a_star_vec = 2 * np.pi * np.cross(ay, az) / V
    b_star_vec = 2 * np.pi * np.cross(az, ax) / V
    c_star_vec = 2 * np.pi * np.cross(ax, ay) / V

    # B matrix: columns are reciprocal basis vectors
    B = np.column_stack((a_star_vec, b_star_vec, c_star_vec))
    return B


def generate_candidate_indices(Q: np.ndarray, B: np.ndarray,
                               tolerance: float = 0.01,
                               max_hkl: int = 5) -> List[Tuple[int, int, int]]:
    """
    Find all integer triples whose reciprocal lattice vector magnitude
    matches |Q|/(2π) within tolerance.
    """
    d_star = np.linalg.norm(Q) / (2 * np.pi)
    candidates: List[Tuple[int, int, int]] = []
    for h in range(-max_hkl, max_hkl + 1):
        for k in range(-max_hkl, max_hkl + 1):
            for l in range(-max_hkl, max_hkl + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                vec = B @ np.array([h, k, l], dtype=float)
                d_star_calc = np.linalg.norm(vec)
                if abs(d_star_calc - d_star) < tolerance:
                    candidates.append((h, k, l))
    return candidates


def compute_U_matrix(h1: Tuple[int, int, int], h2: Tuple[int, int, int],
                     Q1: np.ndarray, Q2: np.ndarray, B: np.ndarray) -> np.ndarray | None:
    """
    Compute orientation matrix U from two non‑collinear reflections.
    Returns None if the reflections are collinear.
    """
    # Crystal-frame orthogonal triad from reciprocal lattice vectors
    v1_c = B @ np.array(h1, dtype=float)
    v2_c = B @ np.array(h2, dtype=float)
    v3_c = np.cross(v1_c, v2_c)

    if np.linalg.norm(v3_c) < 1e-10:
        return None

    t1_c = v1_c / np.linalg.norm(v1_c)
    t2_c = v2_c / np.linalg.norm(v2_c)
    t3_c = v3_c / np.linalg.norm(v3_c)
    Tc = np.column_stack((t1_c, t2_c, t3_c))

    # Lab-frame orthogonal triad from measured Q vectors
    u1_l = Q1 / np.linalg.norm(Q1)
    u2_l = Q2 / np.linalg.norm(Q2)
    u3_l = np.cross(u1_l, u2_l)

    if np.linalg.norm(u3_l) < 1e-10:
        return None
    u3_l = u3_l / np.linalg.norm(u3_l)
    Tl = np.column_stack((u1_l, u2_l, u3_l))

    # U = Tl * Tc^T
    U = Tl @ Tc.T
    return U


def evaluate_deviation(U: np.ndarray, Q3: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute the integer indices for Q3 using the candidate orientation matrix U,
    and the deviation (Euclidean distance to nearest integer triple).
    """
    hkl_candidate = np.linalg.inv(B) @ U.T @ Q3
    hkl_int = np.round(hkl_candidate).astype(int)
    deviation = np.linalg.norm(hkl_candidate - hkl_int.astype(float))
    return hkl_int, deviation


def index_crystal(
    a: float, b: float, c: float,
    alpha: float, beta: float, gamma: float,
    Q1: np.ndarray, Q2: np.ndarray, Q3: np.ndarray,
    tolerance: float = 0.01,
    max_hkl: int = 5
) -> Tuple[np.ndarray, Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Index three Bragg peaks using the orientation matrix method.

    Parameters:
        a, b, c : unit cell lengths (Å)
        alpha, beta, gamma : unit cell angles (degrees)
        Q1, Q2, Q3 : momentum transfer vectors (lab frame) of three non‑coplanar reflections
        tolerance : d* matching tolerance (Å⁻¹)
        max_hkl : maximum |h|,|k|,|l| to search

    Returns:
        Tuple (U, h1, h2, h3) where
            U : 3×3 orientation matrix
            h1, h2 : indices of Q1 and Q2 (matched)
            h3 : indices of Q3 (computed from best U)
    """
    # Step 1: Compute B matrix
    B = compute_B_matrix(a, b, c, alpha, beta, gamma)

    # Step 2: Q vectors are provided as input (assumes prior computation from detector geometry)

    # Step 3: Generate candidate index sets for Q1 and Q2
    H1_candidates = generate_candidate_indices(Q1, B, tolerance, max_hkl)
    H2_candidates = generate_candidate_indices(Q2, B, tolerance, max_hkl)

    # Step 4: Build candidate orientation matrices
    candidates: List[Tuple[np.ndarray, Tuple[int, int, int], Tuple[int, int, int]]] = []
    for h1 in H1_candidates:
        for h2 in H2_candidates:
            U = compute_U_matrix(h1, h2, Q1, Q2, B)
            if U is not None:
                candidates.append((U, h1, h2))

    # Step 5: Evaluate with Q3 and select best
    best_U = None
    best_H1: Tuple[int, int, int] = (0, 0, 0)
    best_H2: Tuple[int, int, int] = (0, 0, 0)
    best_H3: Tuple[int, int, int] = (0, 0, 0)
    best_dev = np.inf

    for U, h1, h2 in candidates:
        hkl_int, dev = evaluate_deviation(U, Q3, B)
        if dev < best_dev:
            best_dev = dev
            best_U = U
            best_H1 = h1
            best_H2 = h2
            best_H3 = tuple(hkl_int)

    if best_U is None:
        raise RuntimeError("No valid orientation matrix found.")

    # Step 6: Return results
    return (best_U, best_H1, best_H2, best_H3)


# === Example usage (can be removed if not needed) ===
if __name__ == "__main__":
    # Lattice parameters (cubic aluminium, a=4.05 Å)
    a = b = c = 4.05
    alpha = beta = gamma = 90.0

    # Compute Q vectors for test reflections (1,0,0), (0,1,0), (1,1,1)
    B_test = compute_B_matrix(a, b, c, alpha, beta, gamma)
    Q1_true = B_test @ np.array([1.0, 0.0, 0.0]) / (2 * np.pi)  # scale to match (2π factor already in B)
    Q2_true = B_test @ np.array([0.0, 1.0, 0.0]) / (2 * np.pi)
    Q3_true = B_test @ np.array([1.0, 1.0, 1.0]) / (2 * np.pi)

    # Add small noise to simulate experimental data
    np.random.seed(42)
    noise_level = 0.005
    Q1 = Q1_true + np.random.normal(0, noise_level, 3)
    Q2 = Q2_true + np.random.normal(0, noise_level, 3)
    Q3 = Q3_true + np.random.normal(0, noise_level, 3)

    # Run indexing
    result = index_crystal(a, b, c, alpha, beta, gamma, Q1, Q2, Q3, tolerance=0.02, max_hkl=3)
    U_best, h1_best, h2_best, h3_best = result

    print("Best orientation matrix U:")
    print(U_best)
    print("\nIndices:")
    print(f"  Q1 -> {h1_best}")
    print(f"  Q2 -> {h2_best}")
    print(f"  Q3 -> {h3_best}")