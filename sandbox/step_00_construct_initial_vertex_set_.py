import numpy as np

def create_vertex_set() -> np.ndarray:
    """
    Create a vertex set of size 64 labeled from 1 to 64.
    
    Returns:
        np.ndarray: A 1D array with vertex labels from 1 to 64.
    """
    V: np.ndarray = np.arange(1, 65, dtype=int)
    return V

# Initialize the vertex set
vertex_set = create_vertex_set()