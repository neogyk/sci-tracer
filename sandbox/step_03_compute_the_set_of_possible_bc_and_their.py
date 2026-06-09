from typing import Set, List


def compute_bc_values(valid_m: Set[int]) -> Set[int]:
    """
    Compute the set of possible BC values given valid m values, where BC = 85 * m.
    """
    return {85 * m for m in valid_m}


def sum_values(values: Set[int]) -> int:
    """
    Sum a set of integer values.
    """
    return sum(values)


if __name__ == "__main__":
    valid_m: Set[int] = {2, 3, 4}
    bc_values: Set[int] = compute_bc_values(valid_m)
    total_sum: int = sum_values(bc_values)

    # Optional sanity checks based on the research plan
    assert bc_values == {170, 255, 340}, "Computed BC set does not match expected values."
    assert total_sum == 765, "Sum of BC values does not match expected total."

    # Output
    sorted_bc_values: List[int] = sorted(bc_values)
    print(f"BC values: {sorted_bc_values}")
    print(f"Sum: {total_sum}")