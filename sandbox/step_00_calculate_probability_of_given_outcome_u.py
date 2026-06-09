from sympy import Rational

def calculate_conditional_probability() -> Rational:
    # Total permutations with all even-numbered stickers visible
    # This is equivalent to arranging the 3 even numbers on 3 different faces
    # and allowing any arrangement for the remaining odd numbers on the free faces
    total_permutations_with_even_visible = 3! * 3!  # Place even numbers first, then odd numbers on remaining faces

    # Count permutations where exactly one face remains blank given that all even numbers are visible
    # Choose one face to remain empty from 6, and distribute 3 even numbers on the rest
    # among 5 faces: Choose 3 faces, then arrange 3 even numbers; remaining can have odd numbers
    permutations_with_one_blank = 6 * (3! * 3!)
    
    # Conditional probability = Number of favorable outcomes / Total number of outcomes
    conditional_probability = Rational(permutations_with_one_blank, total_permutations_with_even_visible)
    
    return conditional_probability

# Example execution
probability = calculate_conditional_probability()
print(probability)  # Output the reduced fraction probability