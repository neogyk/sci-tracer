import math

def calculate_area_of_allowed_region(radius: int) -> float:
    """
    Calculate the area of the region where the sphere can be centered.
    
    Args:
    radius (int): The radius of the circle
    
    Returns:
    float: The area of the circle
    """
    return math.pi * radius**2

# Given radius of the allowed region
radius = 158
area_of_allowed_region = calculate_area_of_allowed_region(radius)
area_of_allowed_region