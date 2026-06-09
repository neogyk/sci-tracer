import numpy as np
from typing import List, Tuple

def compute_candidate_U(
    H1: List[np.ndarray],
    H2: List[np.ndarray],
    Q1: np.ndarray,
    Q2: np.ndarray,
    B: np.ndarray,
    tol: float = 1e-10
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    For each pair (h1 in H1, h2 in H2), compute candidate U matrix.
    
    Parameters:
        H1: list of candidate index triples for Q1 (each as 3-element array)
        H2: list of candidate index triples for Q2
        Q1: measured momentum transfer vector for first reflection (3D array)
        Q2: measured momentum transfer vector for second reflection
        B: reciprocal space metric matrix (3x3)
        tol: tolerance for collinearity and orthogonality checks
        
    Returns:
        List of tuples (U, h1, h2) for valid candidates.
    """
    candidates = []
    
    # Ensure inputs are arrays
    Q1 = np.asarray(Q1, dtype=float)
    Q2 = np.asarray(Q2, dtype=float)
    B = np.asarray(B, dtype=float)
    
    # Helper: triads from two vectors
    def build_triad(v1, v2):
        # Orthonormalize: Gram-Schmidt
        e1 = v1 / np.linalg.norm(v1)
        v2_proj = np.dot(v2, e1) * e1
        e2 = v2 - v2_proj
        if np.linalg.norm(e2) < tol:
            raise ValueError("Vectors are collinear")
        e2 = e2 / np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        # Ensure right-handed
        if np.linalg.norm(e3) < tol:
            raise ValueError("Cross product zero")
        e3 = e3 / np.linalg.norm(e3)
        # Triad as columns
        return np.column_stack((e1, e2, e3))
    
    for h1_arr in H1:
        h1 = np.asarray(h1_arr, dtype=float).reshape(3)
        a1 = B @ h1
        
        for h2_arr in H2:
            h2 = np.asarray(h2_arr, dtype=float).reshape(3)
            a2 = B @ h2
            
            # Check non-collinearity
            cross_a = np.cross(a1, a2)
            if np.linalg.norm(cross_a) < tol:
                continue
            
            # Build triads
            try:
                T_crystal = build_triad(a1, a2)
                T_lab = build_triad(Q1, Q2)
            except ValueError:
                # Collinear vectors, skip
                continue
            
            # Compute U
            U = T_lab @ T_crystal.T
            
            # Orthogonality check
            if np.max(np.abs(U.T @ U - np.eye(3))) > tol:
                continue
            
            # Determinant should be +1 (right-handed)
            if np.linalg.det(U) < 1.0 - tol:  # allow small deviation
                continue
            
            # Valid candidate
            candidates.append((U, h1, h2))
    
    return candidates