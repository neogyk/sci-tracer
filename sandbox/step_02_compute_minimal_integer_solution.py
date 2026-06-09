def find_minimal_x(k: int) -> int:
    for x in range(1, 10000):  # Arbitrarily large upper bound
        y = 125 * k - 2 * x   # From the equation P1 = 2x + y = 125k
        if y > 0 and y.is_integer():  # y must be a positive integer
            ai_ic_plus_y = 6 * k  # From equation P2 = AI + IC + y = 6k
            if ai_ic_plus_y > y:
                ai_ic = ai_ic_plus_y - y
                if ai_ic > 0 and ai_ic.is_integer():  # ai_ic must be valid and integer
                    # Check further constraints specific to triangle structure if any
                    # (...) This might involve checking combinations of AI and IC if specific
                    # type of isosceles triangle or conditions are to be met
                    return x
    return -1  # Return -1 if no valid solution is found

# Example usage:
k_value = 1  # This is a place-holder; actual k would be determined from additional context
minimal_x = find_minimal_x(k_value)
minimal_x