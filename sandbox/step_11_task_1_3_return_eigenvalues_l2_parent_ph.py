import numpy as np

def extract_converged_eigenvalues(converged, num_eigenvalues):
    """
    Extract the requested number of converged eigenvalues from the list.
    Assumes converged list has at least num_eigenvalues elements.
    Returns a 1D numpy array of length num_eigenvalues containing the lowest converged eigenvalues.
    """
    return np.array(sorted(converged)[:num_eigenvalues])