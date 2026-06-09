from typing import Dict, List, Tuple


def compute_R(m: int) -> int:
    """
    Compute R = -425 + 17*m^2 for the Diophantine equation 8u - 9v = R.
    """
    return -425 + 17 * (m * m)


def feasible_pairs_for_m(m: int) -> List[Tuple[int, int]]:
    """
    For a given m in {1,2,3,4}, find all integer pairs (u, v) satisfying:
      - 8u - 9v = R where R = -425 + 17*m^2
      - 1 <= u <= 199, 1 <= v <= 224
      - u > 200 - 8m^2, v > 225 - 9m^2
      - v ≡ -R (mod 8) (used to enumerate candidates efficiently)

    Returns a list of (u, v) pairs.
    """
    R: int = compute_R(m)

    # Bounds derived from constraints
    u_min_strict: int = 200 - 8 * (m * m)
    v_min_strict: int = 225 - 9 * (m * m)
    u_min: int = max(1, u_min_strict + 1)  # strict > becomes >= with +1
    v_min: int = max(1, v_min_strict + 1)
    u_max: int = 199
    v_max: int = 224

    # Early exit if bounds are inconsistent
    if u_min > u_max or v_min > v_max:
        return []

    # v must satisfy v ≡ -R (mod 8)
    target_mod: int = (-R) % 8
    # First v >= v_min with v % 8 == target_mod
    first_v: int = v_min + ((target_mod - (v_min % 8)) % 8)

    solutions: List[Tuple[int, int]] = []
    for v in range(first_v, v_max + 1, 8):
        numerator: int = R + 9 * v
        # By construction numerator % 8 == 0, but safeguard anyway
        if numerator % 8 != 0:
            continue
        u: int = numerator // 8
        if u_min <= u <= u_max:
            solutions.append((u, v))

    return solutions


def main() -> None:
    ms: List[int] = [1, 2, 3, 4]
    results: Dict[int, List[Tuple[int, int]]] = {m: feasible_pairs_for_m(m) for m in ms}

    # Display results
    for m in ms:
        R = compute_R(m)
        sols = results[m]
        print(f"m={m}, R={R}, solutions count={len(sols)}")
        if sols:
            # Show a small sample including the first found solution
            sample_to_show = sols[:5]
            print(f"  sample solutions (u, v): {sample_to_show}")
        else:
            print("  no solutions")

    # For convenience, also list valid m values and a representative example for each
    valid_ms: List[int] = [m for m, sols in results.items() if sols]
    print(f"Valid m values: {valid_ms}")
    examples: Dict[int, Tuple[int, int]] = {m: results[m][0] for m in valid_ms}
    print(f"Example solutions by m: {examples}")


if __name__ == "__main__":
    main()