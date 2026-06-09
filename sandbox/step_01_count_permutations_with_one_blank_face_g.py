from math import factorial
from sympy import Rational

def permutations_one_blank_face(visible_even_stickers: int, total_faces: int) -> int:
    """
    Count the number of permutations resulting in exactly one face blank,
    given all even-numbered stickers (visible_even_stickers) are visible on separate faces.

    :param visible_even_stickers: The count of visible even-numbered stickers (3 in this case).
    :param total_faces: Total number of faces (6 for a cube).
    :return: Number of configurations with exactly one face blank.
    """
    # Calculate number of ways to choose 1 face to leave blank, and distribute stickers on remaining faces
    remaining_faces = total_faces - 1  # Since one face is to be left blank
    # For visible even stickers
    configurations_even = factorial(remaining_faces) // factorial(remaining_faces - visible_even_stickers)
    # Choose 1 face out of all faces to be left blank
    face_choices = total_faces

    # Total configurations with one blank face and even stickers visible
    total_configurations = configurations_even * face_choices

    return total_configurations

# Given that even-numbered stickers are 2, 4, and 6
visible_even_stickers = 3  # Even-numbered stickers visible
total_faces = 6  # Total cube faces

# Calculate the total configurations
result = permutations_one_blank_face(visible_even_stickers, total_faces)
result