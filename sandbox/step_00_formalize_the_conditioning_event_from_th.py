from typing import List, Sequence, Set, Tuple

IndexPair = Tuple[int, int]


def derive_even_visibility_constraints(
    n_rolls: int = 6, even_positions: Sequence[int] = (2, 4, 6)
) -> List[IndexPair]:
    """
    Formalize the conditioning event that the stickers with even indices are visible
    at the end of n_rolls. A sticker placed at time t is visible iff its face does not
    appear at any later time j > t. This induces inequality constraints of the form
    f_t != f_j for all j in {t+1, ..., n_rolls}.

    For n_rolls = 6 and even_positions = (2,4,6):
      - t = 2 contributes (2,3), (2,4), (2,5), (2,6)
      - t = 4 contributes (4,5), (4,6)
      - t = 6 contributes no constraints (last roll is always visible)

    Returns a list of index pairs (i, j) with 1 <= i < j <= n_rolls such that
    the conditioning event requires f_i != f_j.
    """
    if n_rolls <= 0:
        raise ValueError("n_rolls must be positive")

    for t in even_positions:
        if not (1 <= t <= n_rolls):
            raise ValueError("even_positions must be within 1..n_rolls")

    constraints: List[IndexPair] = []
    for t in even_positions:
        for j in range(t + 1, n_rolls + 1):
            constraints.append((t, j))
    return constraints


def event_E_constraints() -> List[IndexPair]:
    """
    Convenience wrapper for the specific case n_rolls = 6 and even stickers {2,4,6} visible.

    Encodes the constraints:
      f2 != f3, f4, f5, f6
      f4 != f5, f6
      (f6 is automatically visible, so it imposes no constraints)
    """
    return derive_even_visibility_constraints(6, (2, 4, 6))


def satisfies_event_E(faces: Sequence[int]) -> bool:
    """Return True iff a length-6 outcome sequence faces satisfies event E.

    Event E (even-numbered stickers visible):
      - f2 is distinct from f3, f4, f5, f6
      - f4 is distinct from f5, f6
      - f6 imposes no constraint (always visible)
    """
    if len(faces) != 6:
        raise ValueError("faces must have length 6")

    for (i, j) in event_E_constraints():
        if faces[i - 1] == faces[j - 1]:
            return False
    return True


def visible_positions(faces: Sequence[int]) -> Set[int]:
    """
    Compute the set of 1-based positions whose stickers are visible at the end of the sequence.
    Position t is visible iff faces[t-1] does not reappear in faces[t:].
    """
    n = len(faces)
    vis: Set[int] = set()
    for t in range(n):
        if faces[t] not in faces[t + 1 :]:
            vis.add(t + 1)
    return vis


if __name__ == "__main__":
    # Demonstration of the derived constraints for event E
    cons = event_E_constraints()
    print("Event E constraints (f_i != f_j):", cons)

    # Example sequences
    ex1 = [1, 2, 3, 4, 5, 6]  # satisfies E
    ex2 = [1, 2, 2, 4, 4, 6]  # violates E (2 repeats at 3; 4 repeats at 5)

    print("ex1 satisfies E:", satisfies_event_E(ex1), "; visible:", sorted(visible_positions(ex1)))
    print("ex2 satisfies E:", satisfies_event_E(ex2), "; visible:", sorted(visible_positions(ex2)))