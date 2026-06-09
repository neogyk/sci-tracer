from typing import Tuple

def compute_final_counts(triple_count: int = 1458, c_choices: int = 18, modulus: int = 1000) -> Tuple[int, int]:
    """Compute total N and N modulo a given modulus.

    Args:
        triple_count: Number of (a,b,d) triples yielding residue 5.
        c_choices: Number of independent choices for exponent c.
        modulus: Modulus for remainder computation.

    Returns:
        A tuple (N, remainder) where N = triple_count * c_choices and remainder = N % modulus.
    """
    N: int = triple_count * c_choices
    remainder: int = N % modulus
    return N, remainder

if __name__ == "__main__":
    N, remainder = compute_final_counts()
    print(f"N = {N}")
    print(f"N mod 1000 = {remainder}")