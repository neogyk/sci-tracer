def sum_numerator_denominator(numerator: int, denominator: int) -> int:
    """
    Sums the numerator and denominator of a given fraction.
    
    Args:
    numerator (int): The numerator of the fraction.
    denominator (int): The denominator of the fraction.
    
    Returns:
    int: The sum of the numerator and denominator.
    """
    return numerator + denominator

# Given values for the task
numerator: int = 6241
denominator: int = 10000

# Calculate the sum of the numerator and denominator
p_q_sum: int = sum_numerator_denominator(numerator, denominator)