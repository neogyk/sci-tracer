from typing import TypedDict
import math


class SqrtResult(TypedDict):
    sqrtN_exact: str
    sqrtN_numeric: float


def compute_sqrt_of_N(N: int = 8119) -> SqrtResult:
    return {"sqrtN_exact": f"sqrt({N})", "sqrtN_numeric": math.sqrt(N)}


if __name__ == "__main__":
    result: SqrtResult = compute_sqrt_of_N()
    print(result)