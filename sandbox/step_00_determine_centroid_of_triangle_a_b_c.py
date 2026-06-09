from typing import Tuple

def calculate_centroid(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> Tuple[float, float]:
    """
    Calculate the centroid G of triangle ABC.

    Parameters:
    A (Tuple[float, float]): The coordinates of point A (A_x, A_y).
    B (Tuple[float, float]): The coordinates of point B (B_x, B_y).
    C (Tuple[float, float]): The coordinates of point C (C_x, C_y).

    Returns:
    Tuple[float, float]: The coordinates of the centroid G (G_x, G_y).
    """
    G_x = (A[0] + B[0] + C[0]) / 3
    G_y = (A[1] + B[1] + C[1]) / 3
    return (G_x, G_y)