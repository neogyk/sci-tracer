def compute_possible_BC_values(AB: int, AC: int) -> int:
    from math import gcd

    # According to the Angle Bisector Theorem: BD/DC = AB/AC
    # We are given AB = 200, AC = 225, hence BD/DC = 200/225
    # Simplify 200/225
    ratio_gcd = gcd(200, 225)
    simplified_AB = 200 // ratio_gcd
    simplified_AC = 225 // ratio_gcd

    # Compute integer solutions for BD and DC that sum to BC
    sum_BC = 0
    for m in range(1, 1001): # arbitrary large number for calculating possible BD values
        n = m * simplified_AC // simplified_AB
        if m * simplified_AC % simplified_AB == 0:  # Check for integer n
            sum_BC += m + n

    return sum_BC

# Example usage
sum_of_BC = compute_possible_BC_values(200, 225)
sum_of_BC