from typing import List, Tuple

def calculate_possible_values(r_start: int, b_start: int) -> List[Tuple[int, int, int]]:
    valid_values = []
    r = r_start
    while True:
        # Calculate b using derived relationship 2b = r
        b = r // 2
        if r >= r_start and b >= b_start and 2 * b == r:
            valid_values.append((r, b, r + b))
        r += 1
        # Stop condition when r exceeds a reasonable limit
        if r > 100:
            break
    return valid_values

def sum_of_five_smallest_n(valid_values: List[Tuple[int, int, int]]) -> int:
    # Sort the tuples (r, b, n) by the value of n
    valid_values.sort(key=lambda x: x[2])
    # Extract the smallest five n values and compute their sum
    five_smallest = [n for _, _, n in valid_values[:5]]
    return sum(five_smallest)

def main() -> int:
    # Start with minimum valid r and b values as per constraints: r >= 7, b >= 7
    r_start = 14  # 2b = r, hence minimum valid r respecting this case is 14 (2*7)
    b_start = 7
    # Calculate possible values
    valid_values = calculate_possible_values(r_start, b_start)
    # Calculate the sum of the five smallest n
    result = sum_of_five_smallest_n(valid_values)
    return result

# Execute the main function
result = main()
print(result)