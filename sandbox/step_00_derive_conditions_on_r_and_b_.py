def derive_conditions():
    # Convert the given equation b(r-4) = r(b-2) to the form to express r in terms of b
    # b(r-4) = r(b-2) translates to:
    # br - 4b = rb - 2r
    # Simplify to get all terms in one side:
    # br - rb + 2r - 4b = 0
    # Factor common terms:
    # r(b - 2) = 4b
    # Express r in terms of b:
    r = lambda b: 4 * b / (b - 2)

    # Validate the condition 2b = r
    condition = lambda b: 2 * b == r(b)
    
    # Values of b should be such that the resultant r is also an integer.
    def find_valid_b_and_r_values():
        min_b = 7
        valid_b_and_r = []

        # Iterate over possible b values
        # Check if r calculated is integer and satisfy conditions
        for b in range(min_b, 100):  # Arbitrary range, can adjust
            try:
                potential_r = r(b)
                if potential_r.is_integer():
                    valid_r = int(potential_r)
                    if condition(b) and valid_r >= 7:
                        valid_b_and_r.append((b, valid_r))
            except ZeroDivisionError:
                continue
        
        return valid_b_and_r

    return find_valid_b_and_r_values()

# Example call to derive conditions
valid_pairs = derive_conditions()
print(valid_pairs)  # List of valid (b, r) pairs