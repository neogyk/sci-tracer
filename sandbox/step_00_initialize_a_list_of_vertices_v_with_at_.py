import numpy as np

def create_vertex_set() -> np.ndarray:
    """
    Generates a list of integers from 1 to 64 to be the vertex set of the hypergraph.

    Returns:
        np.ndarray: A numpy array representing the vertex set V = [1, ..., 64].
    """
    V: np.ndarray = np.arange(1, 65)
    return V

# Example usage
vertex_set = create_vertex_set()
print("Vertex Set V:", vertex_set)