import numpy as np

def calculate_triangle_coordinates(a: float, b: float, c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Cartesian coordinates of the vertices of a triangle with sides a, b, and c,
    where the circumcenter is at the origin.

    :param a: Length of side A.
    :param b: Length of side B.
    :param c: Length of side C.
    :return: Coordinates of the vertices A, B, and C.
    """
    # Using the law of cosines to calculate angles
    cos_A = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
    cos_B = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    cos_C = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    
    sin_A = np.sqrt(1 - cos_A ** 2)
    sin_B = np.sqrt(1 - cos_B ** 2)
    # sin_C is not needed for the placement as C is at origin (0,0)
    
    # Place the first corner at the origin
    C = np.array([0.0, 0.0])
    
    # Place the second corner at (c, 0) along the x-axis
    B = np.array([c, 0.0])
    
    # Place the third corner using calculated cosines and sines
    A = np.array([b * cos_C, b * sin_C])
    
    return A, B, C

# Define the lengths of the sides of the triangle
length_a = 13.0
length_b = 14.0
length_c = 15.0

# Calculate the triangle coordinates
triangle_vertices = calculate_triangle_coordinates(length_a, length_b, length_c)
print(f"Coordinates of vertices A, B, C: {triangle_vertices}")