from typing import Tuple
import numpy as np

def compute_circumcenter(vertices: Tuple[Tuple[float, float, float], ...]) -> Tuple[float, float, float]:
    """
    Calculate the circumcenter of a tetrahedron given its vertices.
    :param vertices: A tuple of four tuples, each containing the x, y, and z coordinates of a vertex.
    :return: A tuple containing the x, y, and z coordinates of the circumcenter.
    """
    # Convert vertices to a numpy array for easier manipulation
    A, B, C, D = [np.array(v) for v in vertices]

    # Calculate midpoints of segments connecting A, B, C, and D
    mid_AB = (A + B) / 2
    mid_AC = (A + C) / 2
    mid_AD = (A + D) / 2

    # Create vectors from the midpoints to the opposite vertices
    vector_AB_D = D - mid_AB
    vector_AC_B = B - mid_AC
    vector_AD_C = C - mid_AD

    # Use normal vectors based on cross products to find orthogonal bisector planes' normals
    normal_plane_ABC = np.cross(B - A, C - A)
    normal_plane_ABD = np.cross(B - A, D - A)
    normal_plane_ACD = np.cross(C - A, D - A)

    normals = np.array([normal_plane_ABC, normal_plane_ABD, normal_plane_ACD])
    points_on_planes = np.array([mid_AB, mid_AC, mid_AD])

    # Solve the system of linear equations to determine the circumcenter
    # Ax = b -> x = inv(A)b
    b = np.array([np.dot(normal, point) for normal, point in zip(normals, points_on_planes)])
    circumcenter = np.linalg.solve(normals, b)

    return tuple(circumcenter)

# Example usage:
# vertices = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
# circumcenter = compute_circumcenter(vertices)
# print("Circumcenter:", circumcenter)