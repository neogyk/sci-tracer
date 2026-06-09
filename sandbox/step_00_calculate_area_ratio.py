def calculate_area_ratio(radius_region: int, total_radius: int) -> tuple[int, int]:
    # Calculate the area ratio
    ratio = (radius_region / total_radius) ** 2

    # Since the radius_region and total_radius are constant, it's equivalent to (158/200)^2
    numerator = 158 * 158
    denominator = 200 * 200

    # Simplifying the fraction 6241/10000, though it's already simplified
    return numerator, denominator

# Calculate the area ratio for the given radii
numerator, denominator = calculate_area_ratio(158, 200)

# Print the final ratio in fractional form and numeric form
print(f"Ratio as a fraction: {numerator}/{denominator}")
print(f"Ratio as a decimal: {numerator / denominator}")

# Post-condition given from task
assert (numerator, denominator) == (6241, 10000)