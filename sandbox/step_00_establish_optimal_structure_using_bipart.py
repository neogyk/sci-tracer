from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple, Set
import random

Node = Tuple[int, int]


def build_grid(n: int = 8) -> Tuple[List[Node], List[Tuple[Node, Node]]]:
    """
    Build the n x n grid graph (vertices are (r,c) with 0-based indices),
    and return (nodes, edges). Edges are undirected and listed once.
    """
    nodes: List[Node] = [(r, c) for r in range(n) for c in range(n)]
    edges: List[Tuple[Node, Node]] = []
    for r in range(n):
        for c in range(n):
            if r + 1 < n:
                edges.append(((r, c), (r + 1, c)))
            if c + 1 < n:
                edges.append(((r, c), (r, c + 1)))
    return nodes, edges


def bipartition(nodes: Iterable[Node]) -> Tuple[Set[Node], Set[Node]]:
    """
    Checkerboard bipartition by parity of r+c.
    Returns (black, white) where black = {(r,c): (r+c)%2==1}, white otherwise.
    """
    black: Set[Node] = set()
    white: Set[Node] = set()
    for r, c in nodes:
        if (r + c) % 2 == 1:
            black.add((r, c))
        else:
            white.add((r, c))
    return black, white


def degrees(nodes: Iterable[Node], edges: Iterable[Tuple[Node, Node]]) -> Dict[Node, int]:
    """
    Compute degree of each node from the edges list.
    """
    deg: Dict[Node, int] = {v: 0 for v in nodes}
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def degree_histogram_for_partition(part: Set[Node], deg: Mapping[Node, int]) -> Dict[int, int]:
    """
    Count how many vertices in 'part' have each degree.
    """
    hist: Dict[int, int] = {}
    for v in part:
        d = deg[v]
        hist[d] = hist.get(d, 0) + 1
    return hist


def sum_abs_edge_diffs(labels: Mapping[Node, int], edges: Iterable[Tuple[Node, Node]]) -> int:
    """
    Compute M = sum over edges of |x_u - x_v|.
    """
    total = 0
    for u, v in edges:
        total += abs(labels[u] - labels[v])
    return total


def linearized_sum_under_threshold(
    labels: Mapping[Node, int],
    deg: Mapping[Node, int],
    black: Set[Node],
    white: Set[Node],
) -> int:
    """
    Under the threshold condition (all black labels > all white labels),
    |x_b - x_w| = x_b - x_w for every edge (b,w).
    Then M = sum_b deg(b)*x_b - sum_w deg(w)*x_w.
    This function returns the RHS linearized value.
    """
    sum_black = sum(deg[v] * labels[v] for v in black)
    sum_white = sum(deg[v] * labels[v] for v in white)
    return sum_black - sum_white


def threshold_assignment(
    black: Sequence[Node],
    white: Sequence[Node],
    low_start: int = 1,
    high_start: int = 33,
) -> Dict[Node, int]:
    """
    Deterministic threshold assignment:
    - Assign labels low_start .. low_start+len(white)-1 to white vertices.
    - Assign labels high_start .. high_start+len(black)-1 to black vertices.
    """
    labels: Dict[Node, int] = {}
    for i, v in enumerate(sorted(white)):
        labels[v] = low_start + i
    for i, v in enumerate(sorted(black)):
        labels[v] = high_start + i
    return labels


def random_threshold_assignment(
    black: Sequence[Node],
    white: Sequence[Node],
    seed: int | None = None,
) -> Dict[Node, int]:
    """
    Randomized threshold assignment (still respects the threshold):
    - A random permutation of 1..32 assigned to white,
    - A random permutation of 33..64 assigned to black.
    """
    rng = random.Random(seed)
    w_vals = list(range(1, len(white) + 1))
    b_vals = list(range(len(white) + 1, len(white) + len(black) + 1))
    rng.shuffle(w_vals)
    rng.shuffle(b_vals)

    labels: Dict[Node, int] = {}
    w_nodes = list(white)
    b_nodes = list(black)
    rng.shuffle(w_nodes)
    rng.shuffle(b_nodes)
    for v, x in zip(w_nodes, w_vals):
        labels[v] = x
    for v, x in zip(b_nodes, b_vals):
        labels[v] = x
    return labels


def verify_linearization(
    labels: Mapping[Node, int],
    edges: Iterable[Tuple[Node, Node]],
    deg: Mapping[Node, int],
    black: Set[Node],
    white: Set[Node],
) -> bool:
    """
    Verify M_abs == linearized_sum_under_threshold for the given threshold-respecting labels.
    """
    # Ensure threshold condition holds
    max_white = max(labels[v] for v in white)
    min_black = min(labels[v] for v in black)
    assert max_white < min_black, "Threshold condition not satisfied."

    M_abs = sum_abs_edge_diffs(labels, edges)
    M_lin = linearized_sum_under_threshold(labels, deg, black, white)
    return M_abs == M_lin


def main() -> None:
    # 1) Build the 8x8 grid and its bipartition
    nodes, edges = build_grid(8)
    black, white = bipartition(nodes)
    assert len(black) == 32 and len(white) == 32, "Each partition must have 32 vertices on an 8x8 grid."

    # 2) Compute degrees and degree multiplicities per partition
    deg = degrees(nodes, edges)
    hist_black = degree_histogram_for_partition(black, deg)
    hist_white = degree_histogram_for_partition(white, deg)

    # Expected counts per partition on 8x8 grid: {4: 18, 3: 12, 2: 2}
    expected = {4: 18, 3: 12, 2: 2}
    assert hist_black == expected, f"Black partition degree counts mismatch: {hist_black} != {expected}"
    assert hist_white == expected, f"White partition degree counts mismatch: {hist_white} != {expected}"

    print("Bipartition established on 8x8 grid:")
    print(f"- |Black| = {len(black)}, |White| = {len(white)}")
    print(f"- Degree multiplicities per partition (deg:count) = {expected}")

    # 3) Verify linearization equality under threshold assignment
    det_labels = threshold_assignment(sorted(black), sorted(white))
    ok_det = verify_linearization(det_labels, edges, deg, black, white)
    print(f"Linearization equality holds for deterministic threshold assignment: {ok_det}")

    # 4) Additional randomized checks within each partition (still threshold)
    trials = 5
    all_ok = True
    for t in range(trials):
        rnd_labels = random_threshold_assignment(sorted(black), sorted(white), seed=1234 + t)
        ok = verify_linearization(rnd_labels, edges, deg, black, white)
        all_ok = all_ok and ok
    print(f"Linearization equality holds for {trials} randomized threshold assignments: {all_ok}")

    # 5) Report the linearized form explicitly
    print("Under any threshold assignment (all black labels > all white labels):")
    print("M = sum_{b in Black} deg(b)*x_b - sum_{w in White} deg(w)*x_w")
    print("Degree multiplicities within each partition are: 4×18, 3×12, 2×2.")


if __name__ == "__main__":
    main()