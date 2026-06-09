def create_vertex_set() -> set[int]:
    """
    Create a set of vertices labeled from 1 to 64.

    Returns:
        set[int]: A set containing integers from 1 to 64 inclusive.
    """
    V: set[int] = set(range(1, 65))
    return V

# Example usage
vertex_set = create_vertex_set()
print(vertex_set)