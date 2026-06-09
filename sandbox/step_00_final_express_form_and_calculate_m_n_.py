import math
from typing import Tuple

def express_perimeter_as_m_sqrt_n(perimeter_squared: int) -> Tuple[int, int, int]:
    # Simplify the argument to the form m * sqrt(n)
    # Rewrite perimeter_squared as A*sqrt(B), then find m and n
    A = int(math.sqrt(perimeter_squared))  # Find integer part
    B = perimeter_squared // (A * A)     # Calculate remaining factor
    
    m = A
    n = B
    
    return m, n, m + n

# Example calculation
perimeter_squared = 1540  # Derived from 5 squares each side
m, n, m_plus_n = express_perimeter_as_m_sqrt_n(perimeter_squared)
m, n, m_plus_n