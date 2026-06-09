from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class QSqrt5:
    """
    Represents numbers of the form a + b*sqrt(5) with integer coefficients.
    """
    const: int  # a
    sqrt5: int  # b

    def __add__(self, other: "QSqrt5") -> "QSqrt5":
        return QSqrt5(self.const + other.const, self.sqrt5 + other.sqrt5)

    def scale(self, k: int) -> "QSqrt5":
        return QSqrt5(self.const * k, self.sqrt5 * k)

    def to_float(self) -> float:
        return float(self.const + self.sqrt5 * (5 ** 0.5))


def compute_perimeter_square_and_sum() -> Tuple[QSqrt5, int, int, int]:
    """
    Using:
      S0 = 308
      S1 = 92(√5 + 1)
      S2 = 246(√5 − 1)
    Compute:
      P^2 = S0 + 2(S1 + S2) = 676√5
      Identify m = 676, n = 5 (squarefree), and return m + n.
    Returns:
      (P2_as_QSqrt5, m, n, m_plus_n)
    """
    S0: QSqrt5 = QSqrt5(308, 0)
    S1: QSqrt5 = QSqrt5(92, 92)      # 92*1 + 92*√5
    S2: QSqrt5 = QSqrt5(-246, 246)   # -246*1 + 246*√5

    S1_plus_S2: QSqrt5 = S1 + S2                     # (-154, 338)
    P2: QSqrt5 = S0 + S1_plus_S2.scale(2)            # (0, 676) => 676√5

    m: int = P2.sqrt5  # 676
    n: int = 5
    m_plus_n: int = m + n  # 681

    return P2, m, n, m_plus_n


if __name__ == "__main__":
    P2_val, m_val, n_val, result = compute_perimeter_square_and_sum()
    # Outputs:
    print(f"P^2 = {m_val}*sqrt({n_val})")
    print(f"m + n = {result}")