from typing import List

def filter_divisors_modulo(divisors: List[int], mod: int = 12, remainder: int = 5) -> List[int]:
    """
    Filter divisors that leave a specific remainder when divided by a given modulus.

    :param divisors: List of divisors to be filtered.
    :param mod: The modulus for division.
    :param remainder: The remainder condition for filtering.
    :return: Filtered divisors fulfilling the modular condition.
    """
    filtered_divisors = [divisor for divisor in divisors if divisor % mod == remainder]
    return filtered_divisors

# Example usage (this line would not be included in the module but is shown for illustration):
# divisors_list = [some_list_of_divisors]
# filtered = filter_divisors_modulo(divisors_list)