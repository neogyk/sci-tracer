from typing import Tuple

def reflect_point_over_line(BC_line: Tuple[float, float, float], centroid: Tuple[float, float]) -> Tuple[float, float]:
    """
    Reflects a point G (centroid) over a line BC.

    :param BC_line: A tuple (a, b, c) representing the line in the form ax + by + c = 0
    :param centroid: A tuple (G_x, G_y) representing the coordinates of the centroid
    :return: A tuple (D_x, D_y) representing the coordinates of the reflected point
    """
    a, b, c = BC_line
    G_x, G_y = centroid

    # Calculating the perpendicular distance from the centroid to the line
    denominator = a**2 + b**2
    G_dist = (a * G_x + b * G_y + c) / denominator

    # Reflecting the centroid over the line
    D_x = G_x - 2 * a * G_dist
    D_y = G_y - 2 * b * G_dist
    
    return D_x, D_y