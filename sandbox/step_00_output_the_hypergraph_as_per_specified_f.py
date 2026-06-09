from typing import List, Set

def generate_and_print_hypergraph(vertices: List[int], hyperedges: List[Set[int]]) -> str:
    """
    Generates a string representation of a hypergraph and prints it.

    Args:
    vertices (List[int]): List of vertex identifiers.
    hyperedges (List[Set[int]]): List of hyperedges, each represented as a set of vertex identifiers.

    Returns:
    str: A string representation of the hypergraph.
    """
    output_lines = []

    # Format the vertices
    vertices_str = ', '.join(map(str, vertices))
    output_lines.append(f"Vertices: {vertices_str}")

    # Format the hyperedges
    output_lines.append("Hyperedges:")
    for i, edge in enumerate(hyperedges, start=1):
        edge_str = ', '.join(map(str, sorted(edge)))
        output_lines.append(f"  Edge {i}: {{{edge_str}}}")

    # Combine all lines into a single output string
    output_string = "\n".join(output_lines)

    # Print the result
    print(output_string)

    return output_string

# Example usage with dummy hypergraph data
V = list(range(1, 65))  # Vertices: 1 to 64

# Example hyperedge configuration based on constraints (Ensure |H| <= 20, no partition > 20)
H = [
    {i, i + 1, i + 2} for i in range(1, 62, 3)
]  # Simple consecutive triplet edges; adjust as per real requirements

# Call function to generate and print hypergraph
generate_and_print_hypergraph(V, H)