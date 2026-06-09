import random
from typing import Set, List, Tuple

def create_vertex_set() -> Set[int]:
    """Create a vertex set V of size 64, labeled from 1 to 64."""
    V = set(range(1, 65))
    return V

def initialize_edge_set() -> Set[Tuple[int]]:
    """Initialize an empty edge collection H."""
    H = set()
    return H

def populate_edge_set(V: Set[int], max_edges: int = 20, max_partition_size: int = 20) -> Set[Tuple[int]]:
    """Populate edge set H with hyperedges ensuring constraints: 
    - Add up to `max_edges` hyperedges
    - Cover all vertices
    - Ensure partition size validity
    """
    def is_valid_partition(H: Set[Tuple[int]]) -> bool:
        for subset in subsets:
            if len(subset) > max_partition_size and all(any(v in edge for edge in H) for v in subset):
                return False
        return True

    def cover_all_vertices(H: Set[Tuple[int]], V: Set[int]) -> bool:
        covered_vertices = set(v for edge in H for v in edge)
        return covered_vertices == V

    H = initialize_edge_set()
    subsets = list(V)

    while len(H) < max_edges:
        edge_size = random.randint(1, max_partition_size)
        edge = tuple(random.sample(V, edge_size))
        H.add(edge)

        if not is_valid_partition(H) or not cover_all_vertices(H, V):
            H.remove(edge)  # Remove the edge if constraints are not met

    return H

# Construct the hypergraph
V = create_vertex_set()
H = populate_edge_set(V)

# Output the result
print("Vertex set V:", V)
print("Edge set H:", H)