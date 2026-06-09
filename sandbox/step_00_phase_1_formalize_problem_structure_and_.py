from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Sequence, Set, Tuple

# ---- Grid model ----

GRID_SIZE: int = 10
Coord = Tuple[int, int]  # (row, col), 0-based, 0 <= row,col < GRID_SIZE


# ---- Rectangle (loop) model ----

@dataclass(frozen=True)
class Rect:
    """
    Axis-aligned rectangle on the GRID_SIZE x GRID_SIZE board.
    Interpreted as a 'loop' occupying exactly the perimeter band
    (1-cell-thick ring) of the rectangle.

    Coordinates are 0-based and inclusive for the area covered.
    Constraints: height >= 2, width >= 2, and rectangle fully within the grid.
    """
    top: int
    left: int
    height: int
    width: int

    @property
    def bottom(self) -> int:
        return self.top + self.height - 1

    @property
    def right(self) -> int:
        return self.left + self.width - 1


# ---- Validation utilities ----

def validate_rect(rect: Rect, grid_size: int = GRID_SIZE) -> None:
    """
    Validate that:
    - height, width >= 2 (so the ring is well-defined)
    - rectangle is fully within the grid bounds [0, grid_size-1]
    Raises ValueError on violation.
    """
    if rect.height < 2 or rect.width < 2:
        raise ValueError(f"Invalid rectangle dimensions: height and width must be >= 2, got {rect.height}x{rect.width}.")
    if rect.top < 0 or rect.left < 0:
        raise ValueError(f"Rectangle position out of bounds (negative): top={rect.top}, left={rect.left}.")
    if rect.bottom >= grid_size or rect.right >= grid_size:
        raise ValueError(
            f"Rectangle exceeds grid bounds {grid_size}x{grid_size}: "
            f"rect bottom/right = ({rect.bottom},{rect.right})."
        )


# ---- Perimeter / ring utilities ----

def perimeter_cell_count(rect: Rect) -> int:
    """
    Number of cells in the 1-cell-thick rectangular ring.
    Formula: 2h + 2w - 4
    """
    return 2 * rect.height + 2 * rect.width - 4


def ring_cells(rect: Rect, grid_size: int = GRID_SIZE) -> FrozenSet[Coord]:
    """
    Compute the set of grid cells occupied by the 1-cell-thick rectangular ring.
    Ensures correctness via validation and length check.
    """
    validate_rect(rect, grid_size=grid_size)

    r0, c0, h, w = rect.top, rect.left, rect.height, rect.width
    r1, c1 = rect.bottom, rect.right

    cells: Set[Coord] = set()

    # Top and bottom rows of the ring
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))

    # Left and right columns of the ring
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))

    expected = perimeter_cell_count(rect)
    actual = len(cells)
    if actual != expected:
        # This should not happen for valid rectangles; defensive check
        raise RuntimeError(
            f"Perimeter cell count mismatch for rect={rect}: expected {expected}, got {actual}."
        )

    return frozenset(cells)


# ---- Partition / coverage utilities ----

def union_of_rings(rects: Sequence[Rect], grid_size: int = GRID_SIZE) -> FrozenSet[Coord]:
    """
    Compute the union of perimeter cells for a collection of rectangles.
    """
    union: Set[Coord] = set()
    for rect in rects:
        union |= set(ring_cells(rect, grid_size=grid_size))
    return frozenset(union)


def are_rings_pairwise_disjoint(rects: Sequence[Rect], grid_size: int = GRID_SIZE) -> bool:
    """
    Check that the perimeter-cell sets of the given rectangles are pairwise disjoint.
    """
    seen: Set[Coord] = set()
    for rect in rects:
        rcells = ring_cells(rect, grid_size=grid_size)
        # If any overlap with already seen cells -> not disjoint
        if any(cell in seen for cell in rcells):
            return False
        seen |= rcells
    return True


def covers_entire_grid(rects: Sequence[Rect], grid_size: int = GRID_SIZE) -> bool:
    """
    Check if the union of ring cells of the given rectangles covers the entire grid.
    """
    all_cells_count = grid_size * grid_size
    u = union_of_rings(rects, grid_size=grid_size)
    return len(u) == all_cells_count


def perimeter_sum(rects: Sequence[Rect]) -> int:
    """
    Sum of perimeter-cell counts across the given rectangles.
    """
    return sum(perimeter_cell_count(r) for r in rects)


# ---- Optional: simple sanity tests (executed only when run as script) ----

if __name__ == "__main__":
    # Basic validation of utilities
    outer = Rect(0, 0, 10, 10)
    assert perimeter_cell_count(outer) == 36
    outer_cells = ring_cells(outer)
    assert len(outer_cells) == 36

    # Example of a 2x10 strip (top two rows)
    strip_top = Rect(0, 0, 2, 10)
    assert perimeter_cell_count(strip_top) == 2 * 2 + 2 * 10 - 4  # equals 20
    assert len(ring_cells(strip_top)) == 20  # the entire 2x10 area

    # Disjointness check
    # Two non-overlapping 2x2 rings at corners
    r1 = Rect(0, 0, 2, 2)
    r2 = Rect(8, 8, 2, 2)
    assert are_rings_pairwise_disjoint([r1, r2]) is True

    # Overlapping rings (share corner cells)
    r3 = Rect(0, 0, 3, 3)
    r4 = Rect(1, 1, 3, 3)
    assert are_rings_pairwise_disjoint([r3, r4]) is False

    print("Sanity checks passed.")