from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class ReparamResult:
    """Container for the reparameterization result.

    Fields:
        a, b: original positive integers with a != b
        n: value of a + b + a*b
        x, y: reparameterized factors a+1 and b+1 respectively
    """
    a: int
    b: int
    n: int
    x: int
    y: int


def reparameterize(a: int, b: int) -> ReparamResult:
    """Reparameterize n = a + b + ab into n + 1 = (a + 1)(b + 1).

    Preconditions (validated):
      - a, b are integers with a >= 1, b >= 1, and a != b.

    Postcondition (ensured):
      - Let x = a + 1 and y = b + 1, then x, y >= 2, x != y, and n + 1 = x * y.

    Args:
        a: Positive integer (>= 1).
        b: Positive integer (>= 1), distinct from a.

    Returns:
        ReparamResult with computed n, x, y satisfying the postcondition.

    Raises:
        ValueError: If a or b are not positive or if a == b.
    """
    if a < 1 or b < 1:
        raise ValueError("Inputs a and b must be positive integers (>= 1).")
    if a == b:
        raise ValueError("Inputs a and b must be distinct (a != b).")

    x: Final[int] = a + 1
    y: Final[int] = b + 1
    n: Final[int] = a + b + a * b

    # Postcondition checks
    if x < 2 or y < 2:
        raise AssertionError("Postcondition failed: x and y must be >= 2.")
    if x == y:
        raise AssertionError("Postcondition failed: x and y must be unequal.")
    if n + 1 != x * y:
        raise AssertionError("Postcondition failed: n + 1 must equal x * y.")

    return ReparamResult(a=a, b=b, n=n, x=x, y=y)


if __name__ == "__main__":
    # Minimal demonstration
    example = reparameterize(1, 2)
    # Expect: a=1, b=2 -> n=1+2+2=5, x=2, y=3, and 5+1 = 2*3
    print(example)