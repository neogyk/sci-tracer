def compute_modulo_1000(total_sum: int) -> int:
    """
    Compute the remainder of total_sum divided by 1000.

    Parameters:
    total_sum (int): The computed total maximum sum of differences.

    Returns:
    int: The remainder when total_sum is divided by 1000.
    """
    return total_sum % 1000

# Example usage:
# Let's assume we have a computed total sum of differences M
M: int = 12345  # Replace with actual computed sum of differences
result: int = compute_modulo_1000(M)
print(result)