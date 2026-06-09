import pandas as pd
import numpy as np

def load_motif_from_df(data: pd.DataFrame) -> np.ndarray:
    """
    Takes a pandas DataFrame with columns 'A','C','G','T' (counts),
    applies Laplace smoothing (add 1), L1-normalizes each row,
    and returns a numpy array of shape (n, 4).
    """
    # Extract the count matrix as float
    mat = data[['A', 'C', 'G', 'T']].values.astype(float)
    
    # Laplace smoothing: add 1 to each count
    mat += 1.0
    
    # L1 normalization: divide each row by its sum
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = mat / row_sums
    
    return mat