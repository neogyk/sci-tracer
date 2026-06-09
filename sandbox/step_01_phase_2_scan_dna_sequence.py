import numpy as np
from collections import Counter
from typing import List

def scan_sequence(seq: str, mat: np.ndarray, scale: float = 0.8, num_runs: int = 100) -> List[int]:
    """
    Scan a DNA sequence using Kullback–Leibler divergence between empirical base frequencies
    in a sliding window and a uniform background distribution.
    
    Parameters:
    seq (str): DNA sequence (uppercase letters only).
    mat (np.ndarray): Normalized PWM of shape (L, 4) (not used in KL computation, only for window length).
    scale (float): Fraction of maximum score to use as threshold.
    num_runs (int): Number of random shuffles to average over.
    
    Returns:
    List[int]: 0‑based positions where the average KL divergence exceeds the threshold.
    """
    np.random.seed(42)  # for reproducibility
    n = len(seq)
    L = mat.shape[0]
    uniform = 0.25
    
    def kl_div(p: np.ndarray) -> float:
        """Kullback–Leibler divergence D_KL(p || uniform)."""
        return sum(p[i] * np.log2(p[i] / uniform) for i in range(4) if p[i] > 0)
    
    def run_scan(s: str) -> np.ndarray:
        """Compute KL divergence scores for all windows of length L in s."""
        scores = np.zeros(n - L + 1)
        for i in range(n - L + 1):
            window = s[i:i+L]
            cnt = Counter(window)
            emp = np.array([cnt['A'], cnt['C'], cnt['G'], cnt['T']], dtype=float) / L
            scores[i] = kl_div(emp)
        return scores
    
    all_scores = []
    for _ in range(num_runs):
        shuffled = list(seq)
        np.random.shuffle(shuffled)
        shuffled = ''.join(shuffled)
        all_scores.append(run_scan(shuffled))
    
    avg_scores = np.mean(all_scores, axis=0)
    thresh = scale * np.max(avg_scores)
    positions = np.where(avg_scores >= thresh)[0].tolist()
    return positions