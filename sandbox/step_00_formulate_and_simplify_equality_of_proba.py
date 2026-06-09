from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Tuple

import sympy as sp


@dataclass(frozen=True)
class DerivationResult:
    eq_counts: str
    ratio_equation: str
    linear_relation: str
    r_in_terms_of_n: str
    b_in_terms_of_n: str
    congruence: Tuple[int, int]  # (residue, modulus), meaning n ≡ residue (mod modulus)


def derive_linear_relation_and_congruence() -> DerivationResult:
    # Symbols (integers)
    r, b, n = sp.symbols("r b n", integer=True)

    # 1) Equality of hypergeometric counts (proportional to probabilities)
    eq_counts = sp.Eq(sp.binomial(r, 4) * sp.binomial(b, 3), sp.binomial(r, 5) * sp.binomial(b, 2))

    # 2) Reduce to ratio of binomials
    lhs_ratio = sp.simplify(sp.binomial(r, 4) / sp.binomial(r, 5))  # -> 5/(r-4)
    rhs_ratio = sp.simplify(sp.binomial(b, 2) / sp.binomial(b, 3))  # -> 3/(b-2)
    ratio_equation = sp.Eq(lhs_ratio, rhs_ratio)

    # Sanity checks for the identities used
    assert sp.simplify(lhs_ratio - 5 / (r - 4)) == 0
    assert sp.simplify(rhs_ratio - 3 / (b - 2)) == 0

    # 3) Cross-multiply to obtain a linear relation
    #    5/(r-4) = 3/(b-2)  =>  5(b-2) = 3(r-4)
    linear_relation = sp.Eq(5 * (b - 2), 3 * (r - 4))

    # 4) Substitute b = n - r and solve for r in terms of n
    relation_in_n = sp.Eq(5 * (n - r - 2), 3 * (r - 4))
    r_expr = sp.simplify(sp.solve(relation_in_n, r)[0])  # (5n + 2)/8
    b_expr = sp.simplify(n - r_expr)  # (3n - 2)/8

    # 5) Extract integrality condition for r: denominator must divide the numerator
    #    r = (a*n + c) / m  =>  a*n + c ≡ 0 (mod m)  =>  n ≡ (-c) * a^{-1} (mod m) if gcd(a, m) = 1
    num, den = sp.fraction(sp.together(r_expr))  # num = 5*n + 2, den = 8
    num_expanded = sp.expand(num)
    a = int(num_expanded.coeff(n))  # 5
    c = int(num_expanded.subs(n, 0))  # 2
    m = int(den)  # 8

    if gcd(a, m) != 1:
        raise ValueError("Unexpected: gcd(a, m) != 1; cannot form a unique congruence class.")

    inv_a = pow(a, -1, m)  # modular inverse of a modulo m
    residue = (-c * inv_a) % m  # n ≡ residue (mod m)

    return DerivationResult(
        eq_counts=str(eq_counts),
        ratio_equation=str(ratio_equation),
        linear_relation=str(linear_relation),
        r_in_terms_of_n=str(r_expr),
        b_in_terms_of_n=str(b_expr),
        congruence=(residue, m),
    )


if __name__ == "__main__":
    result = derive_linear_relation_and_congruence()
    print("Equality of counts:", result.eq_counts)
    print("Ratio equation:", result.ratio_equation)
    print("Linear relation:", result.linear_relation)
    print("r in terms of n:", result.r_in_terms_of_n)
    print("b in terms of n:", result.b_in_terms_of_n)
    residue, modulus = result.congruence
    print(f"Congruence for integrality: n ≡ {residue} (mod {modulus})")