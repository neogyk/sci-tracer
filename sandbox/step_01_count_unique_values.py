from typing import Set

def count_unique_results() -> int:
    results: Set[int] = set()
    
    for a in range(1, 15):
        for b in range(a + 1, 15):  # ensure a != b
            n = (a + 1) * (b + 1) - 1
            if n <= 100:
                results.add(n)
    
    return len(results)

# Call the function to get the number of unique results
count_unique_values = count_unique_results()