from typing import Tuple

def reflect_point_across_line(G: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> Tuple[float, float]:
    """
    Reflects point G across line segment BC and returns the coordinates of the reflected point D.

    :param G: Tuple of floats representing the coordinates of point G (Gx, Gy).
    :param B: Tuple of floats representing the coordinates of point B (Bx, By).
    :param C: Tuple of floats representing the coordinates of point C (Cx, Cy).
    :return: Tuple of floats representing the coordinates of point D.
    """
    Gx, Gy = G
    Bx, By = B
    Cx, Cy = C
    
    # Calculate the slope of the line BC
    if Cx != Bx:
        slope_bc = (Cy - By) / (Cx - Bx)
    else:
        slope_bc = float('inf')
    
    # Find the perpendicular slope
    if slope_bc != 0 and slope_bc != float('inf'):
        perp_slope = -1 / slope_bc
    else:
        perp_slope = 0 if slope_bc == float('inf') else float('inf')
        
    # If BC is a vertical line
    if slope_bc == float('inf'):
        Dx = 2 * Bx - Gx
        Dy = Gy
        return Dx, Dy

    # Equation of line BC: y = slope_bc * (x - Bx) + By
    # Equation of perpendicular line at G: y = perp_slope * (x - Gx) + Gy
    
    # Find the intersection of the perpendicular from G and line BC
    if perp_slope != float('inf'):
        x_intersect = (slope_bc * Bx - perp_slope * Gx + Gy - By) / (slope_bc - perp_slope)
        y_intersect = slope_bc * (x_intersect - Bx) + By
    else:  # If perpendicular is a vertical line (happens when BC is horizontal)
        x_intersect = Gx
        y_intersect = slope_bc * (x_intersect - Bx) + By

    # Reflect G over line BC using the midpoint formula
    Dx = 2 * x_intersect - Gx
    Dy = 2 * y_intersect - Gy
    
    return Dx, Dy

# Example inputs
G = (1.0, 2.0)
B = (0.0, 0.0)
C = (4.0, 0.0)

# Reflect point G across line BC
D = reflect_point_across_line(G, B, C)
print(D)