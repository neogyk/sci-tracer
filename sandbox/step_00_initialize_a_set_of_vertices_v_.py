# Initialize a set of vertices V.
# Each vertex is uniquely labeled from 1 to 64.

def initialize_vertices() -> set[int]:
    V: set[int] = set(range(1, 65))
    return V

# Run the function to create the vertex set
vertices = initialize_vertices()