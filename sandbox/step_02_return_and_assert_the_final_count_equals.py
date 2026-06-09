from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Set, Tuple

# Types
Cell = Tuple[int, int]


GRID_SIZE: int = 10


@dataclass(frozen=True)
class Rectangle:
    r: int  # top row (0-indexed)
    c: int  # left col (0-indexed)
    h: int  # height (rows), >= 2
    w: int  # width (cols), >= 2

    def perimeter_cells(self) -> FrozenSet[Cell]:
        s: Set[Cell] = set()
        for dr in range(self.h):
            for dc in range(self.w):
                rr: int = self.r + dr
                cc: int = self.c + dc
                if dr in (0, self.h - 1) or dc in (0, self.w - 1):
                    s.add((rr, cc))
        return frozenset(s)

    def is_within_grid(self, n: int = GRID_SIZE) -> bool:
        return (
            0 <= self.r <= n - 2
            and 0 <= self.c <= n - 2
            and 2 <= self.h <= n
            and 2 <= self.w <= n
            and self.r + self.h <= n
            and self.c + self.w <= n
        )

    def perimeter_length(self) -> int:
        return 2 * self.h + 2 * self.w - 4


def make_square_ring(offset: int, size: int) -> Rectangle:
    return Rectangle(r=offset, c=offset, h=size, w=size)


def make_interior_strips(origin: int, d: int, orientation: str) -> List[Rectangle]:
    # d is side length of the remaining interior square
    # orientation: 'H' (horizontal 2 x d strips) or 'V' (vertical d x 2 strips)
    if d == 0:
        return []
    rects: List[Rectangle] = []
    if orientation == "H":
        for k in range(d // 2):
            rects.append(Rectangle(r=origin + 2 * k, c=origin, h=2, w=d))
    elif orientation == "V":
        for k in range(d // 2):
            rects.append(Rectangle(r=origin, c=origin + 2 * k, h=d, w=2))
    else:
        raise ValueError("orientation must be 'H' or 'V'")
    return rects


def build_partition(t: int, orientation: str) -> List[Rectangle]:
    # t nested square rings of sizes 10, 8, 6, ...
    rects: List[Rectangle] = []
    for j in range(t):
        size: int = GRID_SIZE - 2 * j
        rects.append(make_square_ring(offset=j, size=size))
    d: int = GRID_SIZE - 2 * t
    rects.extend(make_interior_strips(origin=t, d=d, orientation=orientation))
    assert len(rects) == 5, "Each partition must consist of exactly 5 loops"
    return rects


def partition_signature(rects: List[Rectangle]) -> FrozenSet[FrozenSet[Cell]]:
    # Unlabeled multiloop signature as frozenset of each loop's perimeter cell-set
    return frozenset(r.perimeter_cells() for r in rects)


def validate_partition(rects: List[Rectangle]) -> bool:
    # Validate rectangle sizes and bounds
    for r in rects:
        if not r.is_within_grid(GRID_SIZE):
            return False

    # Accumulate coverage and check disjointness
    seen: dict[Cell, int] = {}
    perim_sum: int = 0
    for r in rects:
        cells = r.perimeter_cells()
        perim_sum += len(cells)
        # Also cross-check with length formula
        if len(cells) != r.perimeter_length():
            return False
        for cell in cells:
            seen[cell] = seen.get(cell, 0) + 1

    # Must cover the entire grid exactly once
    all_cells: Set[Cell] = {(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)}
    if set(seen.keys()) != all_cells:
        return False
    if any(v != 1 for v in seen.values()):
        return False
    # Optional: sum of perimeters equals total cells
    if perim_sum != GRID_SIZE * GRID_SIZE:
        return False
    return True


def enumerate_partitions() -> List[List[Rectangle]]:
    results: List[List[Rectangle]] = []
    seen_sigs: Set[FrozenSet[FrozenSet[Cell]]] = set()
    for t in range(0, 6):
        d: int = GRID_SIZE - 2 * t
        # Orientation choices; for d == 0 or d == 2, H and V coincide (degenerate), keep one
        if d in (0, 2):
            orientations: List[str] = ["H"]
        else:
            orientations = ["H", "V"]
        for orient in orientations:
            rects = build_partition(t, orient)
            if not validate_partition(rects):
                raise AssertionError("Generated invalid partition")
            sig = partition_signature(rects)
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                results.append(rects)
    return results


def main() -> None:
    partitions = enumerate_partitions()
    count: int = len(partitions)
    assert count == 10, f"Expected 10 partitions, found {count}"
    print(count)


if __name__ == "__main__":
    main()