from typing import List, Tuple
from itertools import product

def generate_palindromes(length: int) -> List[Tuple[int]]:
    """Generates all palindromes of a given length using digits 1 to 9."""
    half_length = (length + 1) // 2
    palindromes = []
    
    # Generate all possible half-length combinations without 0
    for half in product(range(1, 10), repeat=half_length):
        full_palindrome = half + half[::-1][length % 2:]
        palindromes.append(full_palindrome)
    
    return palindromes

def count_palindromes_with_digit_sum(digit_sum: int) -> int:
    """Counts valid palindromes whose digits sum up to `digit_sum`."""
    count = 0
    
    # Considering different lengths of palindromes
    for length in range(1, 8):  # Palindromes with length 1 to 7
        palindromes = generate_palindromes(length)
        for palindrome in palindromes:
            if sum(palindrome) == digit_sum:
                count += 1
    
    return count

# Example usage:
palindrome_count = count_palindromes_with_digit_sum(13)
print(f"Total number of palindromes with digit sum 13: {palindrome_count}")