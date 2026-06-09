def derive_relationships() -> str:
    """
    This function solves the given equation b(r-4) = r(b-2) using algebraic manipulation,
    specifically focusing on simplifying the expression to find relationships between
    the number of red (r) and blue (b) marbles.

    Returns:
        str: A string representation of the derived solution "2b = r".
    """
    # Start with the given equation: b(r - 4) = r(b - 2)
    # Expand both sides:
    # br - 4b = rb - 2r
    # Since 'br' appears on both sides, we can rearrange and simplify to:
    # br - rb = 4b - 2r
    # 0 = 4b - 2r

    # Rearrange the simplified equation to find the relationship between r and b:
    # 2r = 4b -> r = 2b

    # Return the derived relationship
    return "2b = r"

# Call the function and print result for verification
relationship = derive_relationships()
print(relationship)