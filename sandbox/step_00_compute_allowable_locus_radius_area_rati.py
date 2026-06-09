from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from math import sqrt
from typing import TypedDict


class AreaRatio(TypedDict):
    numerator: int
    denominator: int


class Results(TypedDict):
    R: int
    a: int
    locus_radius_squared: int
    locus_radius: float
    area_ratio: AreaRatio
    p_plus_q: int


def compute_results(R: int, a: int) -> Results:
    """
    Compute:
    - locus_radius_squared = (R - a)^2 - a^2
    - locus_radius = sqrt(locus_radius_squared)
    - area_ratio = locus_radius_squared / R^2, reduced
    - p_plus_q = numerator + denominator of the reduced area ratio
    """
    r2: int = (R - a) ** 2 - a ** 2
    r: float = sqrt(r2)
    ratio: Fraction = Fraction(r2, R ** 2).limit_denominator()
    p_plus_q: int = ratio.numerator + ratio.denominator

    return Results(
        R=R,
        a=a,
        locus_radius_squared=r2,
        locus_radius=r,
        area_ratio=AreaRatio(numerator=ratio.numerator, denominator=ratio.denominator),
        p_plus_q=p_plus_q,
    )


if __name__ == "__main__":
    R_input: int = 200
    a_input: int = 42
    results: Results = compute_results(R_input, a_input)
    print(json.dumps(results, separators=(",", ":"), sort_keys=True))