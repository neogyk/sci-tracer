from sympy import Rational
from math import factorial

def calculate_conditional_probability() -> Rational:
    # Given the 6 stickers, 1 through 6.
    # We are counting the permutations such that all even-numbered stickers are visible, and exactly one face remains blank.

    # There are 3 even-numbered stickers (2, 4, 6), each must be on a different face.
    # This implies that these 3 even stickers need to be placed on any 3 of the 6 faces,
    # so we choose 3 faces from 6 for the odd-numbered stickers (1, 3, 5), leaving 3 for evens.
    
    # Total ways to arrange all stickers when all even stickers are visible
    total_permutations = factorial(6)

    # Number of permutations with all evens visible and exactly one face blank
    # We need to arrange the 5 remaining stickers (since one face is blank).
    # There are C(6, 5) = 6 ways to choose 5 faces from 6 to place 5 stickers:
    perms_one_face_blank_with_evens_visible = factorial(5)

    # Conditional probability: (Exactly one face blank | All even-numbered visible)
    # P(Exactly one face blank | All even visible) = N(perms with one face blank & even visible) / N(all perms with even visible)
    probability = Rational(perms_one_face_blank_with_evens_visible, total_permutations)

    return probability

# Calculate the conditional probability
conditional_probability = calculate_conditional_probability()
print(conditional_probability)