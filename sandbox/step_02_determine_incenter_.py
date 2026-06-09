from typing import Tuple
import numpy as np

def calculate_incenter(vertices: np.ndarray) -> np.ndarray:
    """
    Calculate the incenter of a tetrahedron based on provided vertices using tangency conditions.

    Parameters:
    vertices (np.ndarray): A 4x3 array representing the vertices of the tetrahedron.

    Returns:
    np.ndarray: The coordinates of the incenter.
    """
    # Calculate face areas and inradii based on given vertices
    def face_area_and_inradius(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> Tuple[float, float]:
        # Using Heron's formula for area
        a = np.linalg.norm(v2 - v1)
        b = np.linalg.norm(v3 - v2)
        c = np.linalg.norm(v1 - v3)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        inradius = area / s
        return area, inradius
    
    areas = []
    inradii = []

    for face_indices in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
        v1, v2, v3 = vertices[face_indices]
        area, inradius = face_area_and_inradius(v1, v2, v3)
        areas.append(area)
        inradii.append(inradius)
    
    # Calculate weighted centroid (by face area) for incenter
    inradii_sum = sum(inradii)
    incenter = sum((inr * np.mean(vertices[face_indices, :], axis=0) for inr, face_indices in zip(inradii, [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]))) / inradii_sum
    
    return incenter

# Example usage
# vertices: A, B, C, D
vertices = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, np.sqrt(3)/2, 0.0],
    [0.5, np.sqrt(3)/6, np.sqrt(6)/3]])
incenter = calculate_incenter(vertices)
print(f"Incenter coordinates: {incenter}")