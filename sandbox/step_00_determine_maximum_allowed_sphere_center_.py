def calculate_maximum_distance(hemisphere_radius: int, sphere_radius: int) -> int:
    """
    Calculate the maximum distance from the center of the hemisphere to the
    center of the sphere within the hemisphere constraint.

    :param hemisphere_radius: The radius of the hemisphere.
    :param sphere_radius: The radius of the sphere.
    :return: The maximum allowed distance between the centers.
    """
    return hemisphere_radius - sphere_radius

# Given values
hemisphere_radius = 200
sphere_radius = 42

# Calculate maximum distance
d = calculate_maximum_distance(hemisphere_radius, sphere_radius)
d  # Expected output: 158