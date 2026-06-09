from sympy import factorint
from typing import Dict

def count_positive_divisors(product: int) -> int:
    # Factorize the product into its prime factors
    factors: Dict[int, int] = factorint(product)
    
    # Calculate the number of positive divisors
    divisors_count: int = 1
    for exponent in factors.values():
        divisors_count *= (exponent + 1)
    
    return divisors_count

# Example usage with a product obtained from a previous step
product_of_x_values: int = 13860  # This value should be the product from previous calculations
divisors_count: int = count_positive_divisors(product_of_x_values)
divisors_count