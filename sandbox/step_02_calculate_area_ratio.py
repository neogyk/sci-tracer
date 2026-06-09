def calculate_area_ratio(radius_T: float, radius_disk: float) -> float:
    """
    Calculate the area ratio of region T to the entire disk.

    Parameters:
    radius_T (float): Radius of the region T.
    radius_disk (float): Radius of the entire disk.

    Returns:
    float: The ratio of areas (A_T / A_disk).
    """
    ratio: float = (radius_T / radius_disk) ** 2
    return ratio

# Example use
radius_region_T: float = 158
radius_disk: float = 200
area_ratio: float = calculate_area_ratio(radius_region_T, radius_disk)
print(area_ratio)  # Expected: 0.6241 (i.e., equivalent to 6241/10000)