from typing import List

def filter_divisors_modulo(divisors: List[int]) -> List[int]:
    """
    Filter out the divisors that satisfy the condition of leaving a remainder of 5
    when divided by 12.
    
    Args:
    divisors (List[int]): List of divisors to filter.

    Returns:
    List[int]: List of divisors that leave a remainder of 5 when divided by 12.
    """
    return [d for d in divisors if d % 12 == 5]