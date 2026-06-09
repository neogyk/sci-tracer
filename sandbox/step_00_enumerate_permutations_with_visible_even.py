from math import factorial
from itertools import permutations

def count_permutations_with_even_visible(num_stickers: int = 6) -> int:
    """
    Calculate the number of permutations where each even-numbered sticker appears on a separate face and is visible.

    The stickers are numbered from 1 to num_stickers, and even-numbered stickers must be visible.
    num_stickers must be even due to the problem constraints.

    :param num_stickers: Total number of stickers (default is 6 for a cube with 6 even-numbered stickers)
    :return: The total number of permutations where each even-numbered sticker is visible
    """
    # Total faces available on a cube
    num_faces = num_stickers

    if num_faces % 2 != 0:
        raise ValueError("The number of faces (stickers) must be even.")

    # Choose 3 faces for the 3 even-numbered stickers
    # and place these 3 stickers on these faces in 3! different ways
    even_faces_permutations = factorial(num_faces // 2)

    # Now, place all stickers in different arrangements so that each face has a sticker
    total_permutations = len(list(permutations(range(1, num_stickers + 1), num_stickers)))

    # return number of valid permutations with even numbered stickers positioned correctly
    return even_faces_permutations * total_permutations

# Total permutations with visible even-numbered stickers
total_configurations = count_permutations_with_even_visible()
total_configurations