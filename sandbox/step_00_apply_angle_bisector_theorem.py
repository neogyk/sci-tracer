from math import gcd
from typing import Tuple, List

def compute_bd_dc_ratios(ab: int, ac: int) -> List[Tuple[int, int]]:
    """
    Calculates possible integer values for segment lengths BD and DC
    given the angle bisector theorem ratio BD/DC = AB/AC.

    :param ab: Length of side AB
    :param ac: Length of side AC
    :return: List of tuples containing possible (BD, DC) pairs
    """
    # Determine the ratio BD/DC = AB/AC
    ratio_num, ratio_den = ab, ac

    # Calculate the greatest common divisor of the ratio
    gcd_ratio = gcd(ratio_num, ratio_den)

    # Simplify the ratio
    ratio_num //= gcd_ratio
    ratio_den //= gcd_ratio

    # Find the integer pairs (m * ratio_num, m * ratio_den) such that they are integers
    # We assume BD + DC must be an integer, so BD/DC = ratio_num/ratio_den holds.
    possible_pairs = []
    for m in range(1, (ab + ac) // gcd_ratio + 1):
        bd = m * ratio_num
        dc = m * ratio_den
        possible_pairs.append((bd, dc))
    
    return possible_pairs

# Example usage:
ab = 200
ac = 225
bd_dc_pairs = compute_bd_dc_ratios(ab, ac)
print(bd_dc_pairs)