from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Set, Tuple

BOARD_SIZE: int = 10
Orientation = Literal["H", "V"]  # H = horizontal strips (2 x d), V = vertical strips (d x 2)


@dataclass(frozen=True)
class Rect:
    """
    Axis-aligned rectangle specified by its top-left corner (row, col),
    height (h), and width (w). All sizes are in unit cells.
    """
    row: int
    col: int
    h: int
    w: int

    def perimeter_cells(self) -> Set[Tuple[int, int]]:
        """
        Return the set of cells (r, c) on the perimeter of this rectangle.
        Requires h >= 2 and w >= 2.
        """
        cells: Set[Tuple[int, int]] = set()
        r0, c0, h, w = self.row, self.col, self.h, self.w
        r1: int = r0 + h - 1
        c1: int = c0 + w - 1

        # Top and bottom rows
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
            cells.add((r1, c))

        # Left and right columns (excluding corners already added)
        for r in range(r0 + 1, r1):
            cells.add((r, c0))
            cells.add((r, c1))

        return cells

    def dims_valid(self) -> bool:
        return self.h >= 2 and self.w >= 2

    def in_bounds(self, n: int = BOARD_SIZE) -> bool:
        return (0 <= self.row < n) and (0 <= self.col < n) and (self.row + self.h <= n) and (self.col + self.w <= n)


def construct_partition(t: int, orientation: Orientation) -> List[Rect]:
    """
    Construct a candidate partition per the classification:
    - t nested square rings of sizes 10, 8, ..., 10 - 2*(t-1)
    - plus d/2 strips (d = 10 - 2*t) inside the d x d interior, oriented per 'orientation'.

    Returns exactly 5 rectangles (loops).
    """
    assert 0 <= t <= 5
    rects: List[Rect] = []

    # Add t nested square rings
    for k in range(t):
        s: int = BOARD_SIZE - 2 * k  # size of the k-th ring
        rects.append(Rect(row=k, col=k, h=s, w=s))

    # Interior size
    d: int = BOARD_SIZE - 2 * t
    # Add strips if interior is non-empty
    if d > 0:
        strips: int = d // 2
        if orientation == "H":
            # Horizontal strips: 2 x d, stacked from top of the interior
            for i in range(strips):
                r: int = t + 2 * i
                rects.append(Rect(row=r, col=t, h=2, w=d))
        elif orientation == "V":
            # Vertical strips: d x 2, stacked from left of the interior
            for i in range(strips):
                c: int = t + 2 * i
                rects.append(Rect(row=t, col=c, h=d, w=2))
        else:
            raise ValueError("Invalid orientation")
    # Sanity: should always be 5 rectangles
    assert len(rects) == 5, f"Expected 5 loops, got {len(rects)}"
    return rects


def validate_partition(rects: List[Rect], n: int = BOARD_SIZE) -> bool:
    """
    Validate that:
    - Each rectangle is in bounds and has dims >= 2.
    - The union of perimeter cells equals the full n x n grid.
    - Perimeters are pairwise disjoint (i.e., a perfect partition by perimeters).
    """
    # Check rectangle validity
    for r in rects:
        if not r.dims_valid() or not r.in_bounds(n):
            return False

    # Collect perimeter cells, checking for disjointness
    seen: Set[Tuple[int, int]] = set()
    for r in rects:
        perim = r.perimeter_cells()
        # No overlaps allowed
        if seen.intersection(perim):
            return False
        seen.update(perim)

    # Must cover the entire board
    return len(seen) == n * n


def canonical_signature(rects: List[Rect]) -> Tuple[Tuple[int, int, int, int], ...]:
    """
    Canonical signature for deduplication: sorted tuple of (row, col, h, w) for the 5 rectangles.
    """
    sig: List[Tuple[int, int, int, int]] = [(r.row, r.col, r.h, r.w) for r in rects]
    sig.sort()
    return tuple(sig)


def generate_canonical_partitions() -> List[List[Rect]]:
    """
    Generate all canonical partitions per classification, deduplicate, and validate.
    """
    partitions: Dict[Tuple[Tuple[int, int, int, int], ...], List[Rect]] = {}

    for t in range(0, 6):
        d: int = BOARD_SIZE - 2 * t
        orientations: Iterable[Orientation]
        if d > 0:
            orientations = ("H", "V")
        else:
            # No interior: orientation irrelevant; generate once
            orientations = ("H",)

        for orient in orientations:
            rects = construct_partition(t, orient)
            if not validate_partition(rects, BOARD_SIZE):
                continue
            sig = canonical_signature(rects)
            partitions.setdefault(sig, rects)

    return list(partitions.values())


def main() -> None:
    parts = generate_canonical_partitions()
    count = len(parts)
    assert count == 10, f"Expected 10 partitions, found {count}"

    # Optional: print a compact summary
    print(f"Valid canonical partitions found: {count}")
    for idx, rects in enumerate(parts, start=1):
        desc = "; ".join(f"({r.row},{r.col},{r.h},{r.w})" for r in sorted(rects, key=lambda x: (x.row, x.col, x.h, x.w)))
        print(f"{idx:2d}: {desc}")


if __name__ == "__main__":
    main()