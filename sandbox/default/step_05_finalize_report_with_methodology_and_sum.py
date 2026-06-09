from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml
from numpy.typing import NDArray


def _load_uncertainties(
    pattern: str = "results/*_uncertainties.yaml",
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    files: List[str] = sorted(glob.glob(pattern))
    if not files:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
    path = files[0]
    with open(path, "r") as f:
        obj = yaml.safe_load(f)
    edges = np.array(obj.get("bin_edges", []), dtype=float)
    yields = np.array(obj.get("yields", []), dtype=float)
    sumw2 = np.array(obj.get("sumw2", []), dtype=float)
    return edges, yields, sumw2


def _read_sim_card(path: Path = Path("sims/cards.dat")) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return "Simulation card not found. Please verify sims/cards.dat exists."


def _compose_report(
    edges: NDArray[np.float64],
    yields: NDArray[np.float64],
    sumw2: NDArray[np.float64],
    card_text: str,
) -> str:
    lines: List[str] = []
    lines.append("## Methodology Summary")
    lines.append(
        "- Generator: vectorized two-body decays with isotropic angles; no ISR jets; on-shell W/Z; massless leptons."
    )
    lines.append(
        "- Detector: lepton pT smearing 1%; MET x/y Gaussian 10 GeV; no explicit jet reco; b-jet veto assumed to pass."
    )
    lines.append(
        "- Selection (approx CMS-SUS-16-034 WZ/ZZ region): on-Z OSSF (76-106 GeV), pT thresholds: W-lepton >20 GeV, Z-leptons >15 GeV, |η|<2.4/2.5; MT(ℓ_W,MET)>100 GeV; exactly three light leptons (by construction)."
    )
    lines.append(
        "- Normalization: σ=0.055 pb (assumed wino-like), L=35.9 fb^-1; BR(Z→e/μ)=0.0673, BR(W→e/μ)=0.214 applied in weights."
    )

    if edges.size > 0 and yields.size == edges.size - 1:
        lines.append("## E_T^miss Binning and Yields")
        # Compute relative MC uncertainty
        rel = np.zeros_like(yields, dtype=float)
        mask = yields > 0
        rel[mask] = np.sqrt(sumw2[mask]) / yields[mask]
        for i in range(edges.size - 1):
            lo = float(edges[i])
            hi = float(edges[i + 1])
            yv = float(yields[i])
            rv = float(rel[i]) if i < rel.size else float("nan")
            lines.append(f"- Bin [{lo:.0f}, {hi:.0f}) GeV: yield = {yv:.3f}, rel. MC unc = {rv:.3f}")

    lines.append("## Simulation Card")
    lines.append("