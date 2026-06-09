import numpy as np

def compute_remainder() -> int:
    # First term and common ratio for the series
    a: float = 1 / 9
    r: float = 1 / 10
    
    # Sum of the infinite geometric series
    S: float = a / (1 - r)  # S = 1 / 81 is the expected result

    # Compute 10^100
    ten_to_hundred: int = pow(10, 100)
    
    # Calculate the floor division of 10^100 by 81
    floor_division_result: int = ten_to_hundred // 81
    
    # Compute the remainder when the floor division result is divided by 1000
    remainder: int = floor_division_result % 1000
    
    return remainder

# Get the result
remainder_result = compute_remainder()
print(remainder_result)