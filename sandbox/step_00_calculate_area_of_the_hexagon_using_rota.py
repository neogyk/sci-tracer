from typing import Tuple
import numpy as np
from scipy.spatial import ConvexHull

def calculate_hexagon_area(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float],
                           A_p: Tuple[float, float], B_p: Tuple[float, float], C_p: Tuple[float, float]) -> float:
    """
    Calculate the area of hexagon using given vertices after rotation.
    
    A, B, C are the initial vertices of the triangle.
    A_p, B_p, C_p are the vertices after rotation.
    
    Uses ConvexHull from scipy to simplify the calculation of the polygon area.
    """
    vertices = np.array([A, B, C, A_p, B_p, C_p])

    # Utilize ConvexHull to find the area of the resulting polygon
    hull = ConvexHull(vertices)
    area = hull.volume  # In 2D, volume attribute holds the area
    return area

# Example coordinates of rotated points could be passed here for testing
A = (5, 0)
B = (-4, 8)
C = (-3, -6)
A_p = (6, 3)
B_p = (-2, 7)
C_p = (-4, -5)

hexagon_area = calculate_hexagon_area(A, B, C, A_p, B_p, C_p)
print(f"The area of the hexagon is: {hexagon_area:.2f} square units.")