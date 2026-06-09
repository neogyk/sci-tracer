from typing import Set

def find_distinct_integers() -> Set[int]:
    results: Set[int] = set()
    for a in range(1, 15):
        for b in range(a + 1, 15):
            n = (a + 1) * (b + 1) - 1
            if n <= 100:
                results.add(n)
    return results

# You can call the function to find and list the distinct integers
distinct_values = find_distinct_integers()
print(distinct_values)