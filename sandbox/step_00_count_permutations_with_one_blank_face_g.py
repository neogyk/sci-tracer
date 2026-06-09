from math import factorial

def count_one_blank_given_visible_evens() -> int:
    """
    Count the number of permutations where one face is blank given all even-numbered stickers are visible.

    Returns:
        int: Number of permutations with one blank face and all even stickers visible.
    """

    # Total number of stickers (1 through 6)
    n = 6
    
    # Even stickers are 2, 4, 6. Hence, 3 even stickers.
    num_even_stickers = 3

    # Out of 6 faces, choose 3 to place the even stickers (all should be visible)
    # 6 choose 3 = 20 ways
    ways_to_place_even = factorial(n) // (factorial(num_even_stickers) * factorial(n - num_even_stickers))

    # Once the even stickers are placed uniquely on 3 faces, the other 3 faces are initially blank.
    # We want exactly 1 face to remain blank, meaning 2 faces will have the odd-numbered stickers placed.

    # The process of placing odd-numbered stickers on these 2 faces:
    # - There are exactly 3 odd-numbered stickers (1, 3, 5), which can be placed in 3P2 ways to fill 2 non-blank faces.

    ways_to_place_odd = factorial(num_even_stickers) // factorial(num_even_stickers - 2)

    # Hence, final count of permutations with exactly 1 face blank = ways of placing even stickers * ways of filling faces with odd stickers
    permutations_with_one_blank_face = ways_to_place_even * ways_to_place_odd

    return permutations_with_one_blank_face

# Call the function and print the result
number_of_configurations = count_one_blank_given_visible_evens()
print(number_of_configurations)