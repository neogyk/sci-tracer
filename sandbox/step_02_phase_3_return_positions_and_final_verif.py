import numpy as np
from collections import Counter

def final_verify(seq: str, mat: np.ndarray, positions: list, scale: float = 0.8) -> list:
    """
    Filter positions by verifying that they achieve high KL divergence scores
    on the original (unshuffled) DNA sequence.

    Parameters
    ----------
    seq : str
        Original DNA sequence.
    mat : np.ndarray
        PWM matrix (not used directly in this function's scoring, but kept for signature consistency).
    positions : list
        List of candidate positions (0-indexed) from the shuffled-run analysis.
    scale : float, optional
        Fraction of the maximum score used as threshold (default: 0.8).

    Returns
    -------
    list
        Sorted list of positions that pass the original-sequence verification.
    """
    L = mat.shape[0]
    n = len(seq)
    uniform = 0.25

    # Compute KL divergence for each window in the original sequence
    scores_orig = np.zeros(n - L + 1)
    for i in range(n - L + 1):
        window = seq[i:i + L]
        cnt = Counter(window)
        emp = np.array([cnt['A'], cnt['C'], cnt['G'], cnt['T']], dtype=float) / L
        # Avoid log(0); the sum skips zero-emission components
        kl = sum(emp[j] * np.log2(emp[j] / uniform) for j in range(4) if emp[j] > 0)
        scores_orig[i] = kl

    # Set threshold based on maximum original score
    thresh = scale * np.max(scores_orig)
    # Keep only positions that are within range and meet threshold
    valid = [p for p in positions if p < len(scores_orig) and scores_orig[p] >= thresh]
    return sorted(valid)