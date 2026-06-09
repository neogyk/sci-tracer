from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Final


@dataclass(frozen=True)
class EquiangularPentagonConstants:
    S0: int          # Sum of squares of side lengths
    cos72: float     # cos(72°) = (√5 − 1)/4
    cos36: float     # cos(36°) = (√5 + 1)/4


def setup_constants() -> EquiangularPentagonConstants:
    """
    Set up constants for an equiangular pentagon:
    - S0 = Σ s_i^2 = 308
    - cos72 = cos(72°) = (√5 − 1)/4
    - cos36 = cos(36°) = (√5 + 1)/4
    """
    root5: float = sqrt(5.0)
    cos72: float = (root5 - 1.0) / 4.0
    cos36: float = (root5 + 1.0) / 4.0
    S0: Final[int] = 308
    return EquiangularPentagonConstants(S0=S0, cos72=cos72, cos36=cos36)


if __name__ == "__main__":
    consts = setup_constants()
    print(consts)