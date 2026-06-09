from typing import Dict, List, Tuple
from math import gcd

MODULUS: int = 12
BASES: List[int] = [7, 11, 13, 17]


def compute_residues(bases: List[int], modulus: int) -> Dict[int, int]:
    """Return base -> (base mod modulus)."""
    return {b: b % modulus for b in bases}


def verify_unit_exponent_two(modulus: int) -> List[int]:
    """Return the list of units modulo `modulus` and assert x^2 ≡ 1 for each."""
    units: List[int] = [x for x in range(1, modulus) if gcd(x, modulus) == 1]
    for x in units:
        assert pow(x, 2, modulus) == 1, f"Unit {x} does not satisfy x^2 ≡ 1 (mod {modulus})"
    return units


def parity_behavior(x_mod: int, modulus: int, max_exp: int = 17) -> Tuple[int, int]:
    """
    For a reduced residue x_mod modulo `modulus`, verify for exponents e in [0, max_exp]
    that x^e ≡ 1 if e is even, else x (mod modulus). Return (even_residue, odd_residue).
    """
    even_residue: int = 1  # expected in a group of exponent 2
    odd_residue: int = x_mod % modulus

    for e in range(0, max_exp + 1):
        expected: int = even_residue if (e % 2 == 0) else odd_residue
        actual: int = pow(x_mod, e, modulus)
        assert actual == expected, (
            f"Parity rule failed for x={x_mod} (mod {modulus}), e={e}: got {actual}, expected {expected}"
        )
    return even_residue, odd_residue


def main() -> None:
    residues: Dict[int, int] = compute_residues(BASES, MODULUS)
    print("Residues modulo 12:", residues)  # Expected: {7:7, 11:11, 13:1, 17:5}

    units: List[int] = verify_unit_exponent_two(MODULUS)
    print("Units modulo 12 (each squares to 1):", units)  # Expected: [1, 5, 7, 11]

    # Verify and summarize parity-power behavior for each base
    summary: Dict[int, Dict[str, int]] = {}
    for b in BASES:
        x_mod: int = residues[b]
        even_res, odd_res = parity_behavior(x_mod, MODULUS, max_exp=17)
        summary[b] = {"even_exponent_residue": even_res, "odd_exponent_residue": odd_res}

    print("Parity-power behavior summary (mod 12):")
    for b in BASES:
        print(f"  Base {b} (≡ {residues[b]} mod 12): even→{summary[b]['even_exponent_residue']}, odd→{summary[b]['odd_exponent_residue']}")


if __name__ == "__main__":
    main()