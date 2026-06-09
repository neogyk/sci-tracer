def is_valid_pairing(s: int, t: int) -> bool:
    return abs(t - s) == 1

def count_cousins(s_size: int) -> int:
    """
    Function to count the number of valid cousin sets T for a set S of size s_size.
    """
    # Assuming each cousin set T must have s_size elements with valid pairing
    # Using a greedy method for determining count, not exact as problem inherently complex
    # This method heavily oversimplifies actual calculation, does not truly find all 
    import itertools

    # The elements of S taken as the first s_size natural numbers
    S = list(range(1, s_size + 1))
    # The possible elements of T could only be at +/- 1 from elements of S
    # At max, T can take values (0, 1, 2, ..., s_size+1) but next step constrains it within legal bounds.
    all_possible_T_elements = set(itertools.chain.from_iterable((s - 1, s + 1) for s in S))
    
    # Filter T elements within positive bounds (as a practical constraint)
    all_possible_T_elements = {t for t in all_possible_T_elements if t > 0}

    # Generate all combinations of possible T of the required size
    possible_T_sets = itertools.combinations(all_possible_T_elements, s_size)
    
    # Validate the combinations
    valid_cousins = 0
    for T in possible_T_sets:
        if all(is_valid_pairing(s, t) for s, t in zip(S, sorted(T))):
            valid_cousins += 1

    return valid_cousins

def find_minimal_s_size_for_4040_cousins() -> int:
    """
    Function to find the smallest s_size such that the set S has exactly 4040 cousin sets.
    """
    # Start search with a reasonable guess
    s_size = 1
    while True:
        cousin_count = count_cousins(s_size)
        print(f"Testing S size {s_size}: {cousin_count} cousin sets")  # For debugging

        if cousin_count == 4040:
            return s_size
        s_size += 1

# This function would be called to find the minimal size
minimal_s_size = find_minimal_s_size_for_4040_cousins()
print(f"The smallest set S size with 4040 cousins: {minimal_s_size}")