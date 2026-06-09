def to_base_503(number: int) -> list[int]:
    """
    Converts a given integer to its base-503 representation.

    :param number: The integer to convert.
    :return: A list of integers representing the digits in base-503, least significant digit first.
    """
    base_503_digits = []
    while number > 0:
        base_503_digits.append(number % 503)
        number //= 503
    return base_503_digits

def convert_indices(r: int, limit: int = 10000) -> tuple[list[int], list[int]]:
    """
    Converts the indices `10000` and `502n + r` to base-503 representation.

    :param r: The integer added to 502n to form indices 502n + r.
    :param limit: The limiting value of n, defaulting to 10000.
    :return: A tuple of lists representing base-503 digits of the limit and 502n + r.
    """
    # Convert the limit to base-503
    limit_base_503 = to_base_503(limit)

    # Create a list of base-503 representations for 502n + r where n ranges to ensure all feasible m
    index_representations = []
    for n in range(limit // 502 + 1):
        index = 502 * n + r
        if index <= limit:
            index_base_503 = to_base_503(index)
            index_representations.append(index_base_503)

    return limit_base_503, index_representations

# Example usage:
r = 5  # Example value of r
limit_base_503, indices_base_503 = convert_indices(r)
print("Base-503 Representation of 10000:", limit_base_503)
print("Base-503 Representations of indices 502n + r:", indices_base_503)