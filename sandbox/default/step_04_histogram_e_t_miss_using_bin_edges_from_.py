from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PipelineContext:
    results_yaml: str
    bin_edges: List[Tuple[float, float]]


def load_context(path: Path) -> PipelineContext:
    if not path.exists():
        raise FileNotFoundError(f"Context file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "bin_edges" not in data or not data["bin_edges"]:
        raise ValueError("No bin_edges in pipeline context")
    edges_raw = data["bin_edges"]
    # Validate and coerce to list of tuples[float, float]
    edges: List[Tuple[float, float]] = []
    for item in edges_raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid bin edge entry: {item}")
        lo, hi = float(item[0]), float(item[1])
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Non-finite bin edge: {item}")
        if hi <= lo:
            raise ValueError(f"Bin edge has non-positive width: {item}")
        edges.append((lo, hi))
    results_yaml = str(data.get("results_yaml", ""))
    return PipelineContext(results_yaml=results_yaml, bin_edges=edges)


def build_bins(edges: Sequence[Tuple[float, float]]) -> np.ndarray:
    # Use the union of all low/high edges and sort uniquely
    flat_edges = [e for pair in edges for e in pair]
    bins = np.array(sorted(set(float(x) for x in flat_edges)), dtype=float)
    if bins.size < 2:
        raise ValueError("Insufficient unique bin edges to form histogram bins")
    # Ensure strictly increasing
    if not np.all(np.diff(bins) > 0):
        raise ValueError("Bin edges are not strictly increasing")
    return bins


def histogram_met(met: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if met.ndim != 1:
        raise ValueError("MET array must be 1-dimensional")
    counts, edges = np.histogram(met, bins=bins)
    return counts.astype(np.int64, copy=False), edges


def save_hist(counts: np.ndarray, bins: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, counts=counts, bins=bins)


def main() -> None:
    ctx_path = Path("data") / "pipeline_context.json"
    sel_path = Path("data") / "selected.npz"
    out_path = Path("data") / "hist_raw.npz"

    ctx = load_context(ctx_path)
    bins = build_bins(ctx.bin_edges)

    if not sel_path.exists():
        raise FileNotFoundError(f"Selected events file not found: {sel_path}")
    sel = np.load(sel_path)
    if "met" not in sel:
        raise KeyError("selected.npz missing 'met' array")
    met = sel["met"].astype(float, copy=False)

    counts, edges = histogram_met(met, bins)
    save_hist(counts, edges, out_path)

    # Report number of bins (counts length)
    print(f"Histogrammed MET into {counts.size} bins")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)