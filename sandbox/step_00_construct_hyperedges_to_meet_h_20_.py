from typing import List, Set

def construct_hyperedges(num_vertices: int = 64, max_hyperedges: int = 20) -> List[Set[int]]:
    """
    Construct hyperedges for a hypergraph with at most 20 edges, 
    each having limited overlap, given a vertex set of size least 64.
    
    :param num_vertices: Number of vertices in the vertex set (minimum 64)
    :param max_hyperedges: Maximum number of hyperedges to create (default 20)
    :return: List of hyperedges as sets of vertices
    """
  
    # Initialize the vertex set V
    V = set(range(1, num_vertices + 1))
  
    # Construct hyperedges
    H = []
    current_vertex = 1
    vertices_per_edge = num_vertices // max_hyperedges
    
    for i in range(max_hyperedges):
        # Create each hyperedge
        edge = set()    
        
        # Add vertices to the edge, ensuring limited overlap
        for _ in range(vertices_per_edge):
            edge.add(current_vertex)
            current_vertex += 1
            
            # Wrap around if we exceed the number of vertices
            if current_vertex > num_vertices:
                current_vertex = 1
            
        H.append(edge)
    
    return H

# Example usage
hyperedges = construct_hyperedges()
for idx, edge in enumerate(hyperedges):
    print(f"Edge {idx + 1}: {sorted(edge)}")