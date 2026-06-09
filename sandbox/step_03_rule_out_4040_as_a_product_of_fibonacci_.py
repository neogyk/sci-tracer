from typing import List, Tuple, Dict, Any
import math


def fibonacci_numbers_up_to(n: int) -> List[int]:
    """
    Generate the list of Fibonacci numbers up to and including n.
    Returns the full sequence starting [1, 1, 2, 3, ...] up to <= n.
    """
    if n < 1:
        return []
    fib: List[int] = [1, 1]
    while fib[-1] <= n:
        fib.append(fib[-1] + fib[-2])
    # Last element exceeded n; remove it
    if fib and fib[-1] > n:
        fib.pop()
    return fib


def can_form_product_from_fibs(target: int, allowed_fibs: List[int]) -> bool:
    """
    Decide if 'target' can be written as a product of numbers from 'allowed_fibs' (each >= 2),
    allowing repetitions. Backtracking with pruning by not exceeding 'target'.
    """
    # Filter to factors >= 2 and <= target
    factors: List[int] = [f for f in allowed_fibs if 2 <= f <= target]
    factors.sort()

    found: bool = False

    def dfs(i: int, prod: int) -> None:
        nonlocal found
        if found:
            return
        if prod == target:
            found = True
            return
        if prod > target or i == len(factors):
            return
        # Option 1: skip current factor
        dfs(i + 1, prod)
        # Option 2: take current factor repeatedly while not exceeding target
        p: int = prod
        while p * factors[i] <= target:
            p *= factors[i]
            dfs(i + 1, p)

    dfs(0, 1)
    return found


def check_impossibility_for_4040() -> Dict[str, Any]:
    """
    Implements two independent checks that 4040 cannot be expressed
    as a product of Fibonacci numbers > 1:
      1) A number-theoretic screen: 4040 = 2^3 * 5 * 101. Any product of Fibonacci numbers
         (all <= 4040) that equals 4040 would require at least one Fibonacci factor divisible by 101.
         We verify none exist among Fibonacci numbers up to 4040.
      2) A constructive exhaustive search over products of Fibonacci factors <= 4040.
    Returns a dictionary summarizing the result.
    """
    target: int = 4040
    fibs_up_to_target: List[int] = fibonacci_numbers_up_to(target)
    allowed_fibs: List[int] = [f for f in fibs_up_to_target if f >= 2]

    # Quick necessary condition: there must exist a Fibonacci number <= target divisible by 101
    has_fib_divisible_by_101: bool = any((f % 101 == 0) for f in allowed_fibs)

    # Exhaustive product search (with repetition)
    can_factor: bool = can_form_product_from_fibs(target, allowed_fibs)

    return {
        "target": target,
        "allowed_fibs": allowed_fibs,
        "has_fib_divisible_by_101": has_fib_divisible_by_101,
        "can_factor": can_factor,
    }


if __name__ == "__main__":
    result: Dict[str, Any] = check_impossibility_for_4040()
    print("Allowed Fibonacci numbers (>=2) up to 4040:", result["allowed_fibs"])
    print("Exists Fibonacci <= 4040 divisible by 101?:", result["has_fib_divisible_by_101"])
    print("Can 4040 be expressed as a product of Fibonacci numbers > 1?:", result["can_factor"])
    # Assert the impossibility
    assert result["has_fib_divisible_by_101"] is False
    assert result["can_factor"] is False