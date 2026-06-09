from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Meta:
    sigma_fb: float
    lumi_fb: float


def load_n_gen(events_path: str = "data/events.npz") -> int:
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Missing events file: {events_path}")
    ev = np.load(events_path)
    n_gen = int(len(ev["met"]))
    if n_gen <= 0:
        raise ValueError(f"N_gen must be > 0, got {n_gen}")
    return n_gen


def parse_meta(meta_path: str = "sims/TChiWZ_550_200.dat") -> Meta:
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta file: {meta_path}")
    with open(meta_path, "r") as f:
        txt = f.read()

    # Support decimals and scientific notation
    num_re = r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)"

    sigma_fb_val: float | None = None
    lumi_fb_val: float | None = None

    # Try common keys for sigma
    for key in ("sigma_fb_effective_trilepton", "sigma_fb"):
        m = re.search(rf"{re.escape(key)}\s*:\s*{num_re}", txt)
        if m:
            sigma_fb_val = float(m.group(1))
            break

    # Lumi key
    m = re.search(rf"lumi_fb\s*:\s*{num_re}", txt)
    if m:
        lumi_fb_val = float(m.group(1))

    if sigma_fb_val is None or lumi_fb_val is None:
        raise RuntimeError(
            "Missing sigma_fb or lumi_fb in meta file. "
            f"Parsed sigma={sigma_fb_val}, lumi={lumi_fb_val}"
        )
    return Meta(sigma_fb=sigma_fb_val, lumi_fb=lumi_fb_val)


def load_counts(hist_path: str = "data/hist_raw.npz") -> NDArray[np.int64]:
    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"Missing histogram file: {hist_path}")
    hr = np.load(hist_path)
    counts = hr["counts"]
    # Ensure integer counts
    if not np.issubdtype(counts.dtype, np.integer):
        counts = counts.astype(np.int64, copy=False)
    return counts  # type: ignore[return-value]


def compute_yields(
    counts: NDArray[np.integer], sigma_fb: float, lumi_fb: float, n_gen: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
    if n_gen <= 0:
        raise ValueError("n_gen must be positive")
    w: float = (sigma_fb * lumi_fb) / float(n_gen)
    counts_f = counts.astype(np.float64, copy=False)
    yields: NDArray[np.float64] = counts_f * w
    unc: NDArray[np.float64] = np.sqrt(counts_f) * w
    return yields, unc, w


def save_yields(
    yields: NDArray[np.float64],
    unc: NDArray[np.float64],
    out_path: str = "data/yields_norm.npz",
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, yields=yields, unc=unc)


def main() -> None:
    n_gen = load_n_gen("data/events.npz")
    meta = parse_meta("sims/TChiWZ_550_200.dat")
    counts = load_counts("data/hist_raw.npz")
    yields, unc, w = compute_yields(counts, meta.sigma_fb, meta.lumi_fb, n_gen)
    save_yields(yields, unc, "data/yields_norm.npz")
    print(f"Normalized yields with w = {w:.6e}")
    print(f"Saved to data/yields_norm.npz; bins = {len(yields)}")


if __name__ == "__main__":
    main()