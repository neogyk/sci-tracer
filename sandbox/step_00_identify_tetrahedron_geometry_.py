from typing import List, Tuple

def isosceles_triangle_faces(edges: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """
    Given a list of edge tuples representing the edges of a tetrahedron, identify and
    return the list of isosceles triangle face configurations based on the provided edge lengths.

    :param edges: A list of tuples, each tuple has 3 integers representing the sides of a triangle.
    :return: A list of tuples representing the face configurations that form isosceles triangles.
    """
    isosceles_faces = []

    for face in edges:
        a, b, c = face
        if a == b or b == c or c == a:
            isosceles_faces.append(face)

    return isosceles_faces

# Example usage (given that the edges are defined, replace with correct data):
tetrahedron_edges = [
    (3, 3, 2), 
    (3, 2, 2), 
    (2, 3, 3), 
    (2, 2, 2)
]

# Identify the isosceles triangle faces based on the tetrahedron's given edges.
isosceles_faces = isosceles_triangle_faces(tetrahedron_edges)
print("Isosceles Triangle Faces:", isosceles_faces)