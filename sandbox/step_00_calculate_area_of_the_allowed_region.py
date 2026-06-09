import math

def compute_area_of_circle(radius: float) -> float:
    """Compute the area of a circle with a given radius."""
    return math.pi * radius ** 2

# Define the radius of region T
radius_T: float = 158.0

# Compute the area of the region T
area_T: float = compute_area_of_circle(radius_T)

# Output the area
area_T