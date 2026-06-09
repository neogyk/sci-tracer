from sympy import factorint

def factor_number(number: int) -> dict[int, int]:
    """
    Factors the given number and returns a dictionary of prime factors.

    Parameters:
    number (int): The number to be factored.

    Returns:
    dict[int, int]: A dictionary containing the prime factors and their exponents.
    """
    # Precondition: Verify input is a positive integer
    if number <= 0:
        raise ValueError("The number must be a positive integer.")

    # Factor the number using sympy's factorint function
    prime_factors = factorint(number)

    # Postcondition: Return the dictionary of prime factors
    return prime_factors

# Example usage for 17017
number_to_factor = 17017
prime_factors_result = factor_number(number_to_factor)
prime_factors_result  # This will contain the prime factors of 17017