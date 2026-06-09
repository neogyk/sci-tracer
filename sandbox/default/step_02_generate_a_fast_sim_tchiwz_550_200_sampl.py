from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    mC1_GeV: float
    mN2_GeV: float
    mN1_GeV: float
    sigma_fb_effective_trilepton: float
    lumi_fb: float
    n_events: int
    rng_seed: int = 42


def _ensure_dirs() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("sims", exist_ok=True)


def _load_or_note_context() -> None:
    # Precondition hints say this exists; tolerate absence.
    ctx_path = "data/pipeline_context.json"
    if not os.path.exists(ctx_path):
        # Create a minimal placeholder to avoid downstream surprises.
        with open(ctx_path, "w", encoding="utf-8") as f:
            json.dump({"results_yaml": "", "bin_edges": []}, f)


def _sample_events(cfg: ModelConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, str | float | int]]:
    np.random.seed(cfg.rng_seed)

    n: int = cfg.n_events
    # MET spectrum ~ Gamma(k=2, theta=60) GeV to mimic falling shape
    met: np.ndarray = np.random.gamma(shape=2.0, scale=60.0, size=n)

    # On-Z dilepton mass around mZ with small width (synthetic)
    mll_z: np.ndarray = np.random.normal(loc=91.1876, scale=5.0, size=n)

    # Lepton pT (Z leptons mildly correlated with MET, W lepton stronger)
    lz1_pt: np.ndarray = np.clip(np.random.normal(45.0 + 0.05 * (met - 80.0), 12.0, size=n), 5.0, None)
    lz2_pt: np.ndarray = np.clip(np.random.normal(45.0 + 0.05 * (met - 80.0), 12.0, size=n), 5.0, None)
    lw_pt: np.ndarray = np.clip(np.random.normal(40.0 + 0.12 * (met - 80.0), 15.0, size=n), 5.0, None)

    # Isotropic eta/phi within fiducial |eta|<~2.2 for generation
    def samp_eta(num: int) -> np.ndarray:
        return np.random.uniform(-2.2, 2.2, size=num)

    def samp_phi(num: int) -> np.ndarray:
        return np.random.uniform(-math.pi, math.pi, size=num)

    lz1_eta: np.ndarray = samp_eta(n)
    lz2_eta: np.ndarray = samp_eta(n)
    lw_eta: np.ndarray = samp_eta(n)

    lz1_phi: np.ndarray = samp_phi(n)
    lz2_phi: np.ndarray = samp_phi(n)
    lw_phi: np.ndarray = samp_phi(n)

    # Flavors and charges: Z leptons SFOS; W lepton random flavor/charge
    z_flavors: np.ndarray = np.random.choice(["e", "mu"], size=n)
    lz1_flav: np.ndarray = z_flavors
    lz2_flav: np.ndarray = z_flavors
    lw_flav: np.ndarray = np.random.choice(["e", "mu"], size=n)

    lz1_q: np.ndarray = np.ones(n, dtype=float)
    lz2_q: np.ndarray = -np.ones(n, dtype=float)
    lw_q: np.ndarray = np.random.choice([-1.0, 1.0], size=n).astype(float)

    # Build transverse mass for W-lepton and MET using random MET direction
    met_phi: np.ndarray = samp_phi(n)
    dphi_w_met: np.ndarray = np.abs(lw_phi - met_phi)
    dphi_w_met = np.where(dphi_w_met > math.pi, 2.0 * math.pi - dphi_w_met, dphi_w_met)
    mt: np.ndarray = np.sqrt(2.0 * lw_pt * met * (1.0 - np.cos(dphi_w_met)))

    # Acceptance and ID (stored as masks; not applied to kinematics here)
    acc_mask: np.ndarray = (np.abs(lz1_eta) < 2.4) & (np.abs(lz2_eta) < 2.4) & (np.abs(lw_eta) < 2.4)

    # Smearing: apply 1% relative pT smearing
    def smear(x: np.ndarray) -> np.ndarray:
        return x * np.random.normal(1.0, 0.01, size=x.shape)

    lz1_pt_r: np.ndarray = smear(lz1_pt)
    lz2_pt_r: np.ndarray = smear(lz2_pt)
    lw_pt_r: np.ndarray = smear(lw_pt)

    # Lepton ID efficiency 0.95 each (three independent IDs)
    id_mask: np.ndarray = (
        (np.random.uniform(size=n) < 0.95)
        & (np.random.uniform(size=n) < 0.95)
        & (np.random.uniform(size=n) < 0.95)
    )

    # Pack event arrays
    events: Dict[str, np.ndarray] = {
        "met": met,
        "met_phi": met_phi,
        "mt": mt,
        "lz1_pt": lz1_pt_r,
        "lz1_eta": lz1_eta,
        "lz1_phi": lz1_phi,
        "lz1_q": lz1_q,
        "lz1_flav": lz1_flav,
        "lz2_pt": lz2_pt_r,
        "lz2_eta": lz2_eta,
        "lz2_phi": lz2_phi,
        "lz2_q": lz2_q,
        "lz2_flav": lz2_flav,
        "lw_pt": lw_pt_r,
        "lw_eta": lw_eta,
        "lw_phi": lw_phi,
        "lw_q": lw_q,
        "lw_flav": lw_flav,
        "mll_z": mll_z,
        "acc_mask": acc_mask,
        "id_mask": id_mask,
    }

    meta: Dict[str, str | float | int] = {
        "model": "TChiWZ",
        "mC1_GeV": cfg.mC1_GeV,
        "mN2_GeV": cfg.mN2_GeV,
        "mN1_GeV": cfg.mN1_GeV,
        "sigma_fb_effective_trilepton": cfg.sigma_fb_effective_trilepton,
        "lumi_fb": cfg.lumi_fb,
        "N_generated": cfg.n_events,
        "notes": (
            "Fast-sim synthetic kinematics; decays to e/mu only; MET sampled; "
            "includes simple smearing and efficiencies."
        ),
    }
    return events, meta


def _write_outputs(events: Dict[str, np.ndarray], meta: Dict[str, str | float | int]) -> None:
    np.savez_compressed("data/events.npz", **events)
    # Placeholder for a ROOT file artifact
    with open("data/events.root", "w", encoding="utf-8") as f:
        f.write("placeholder: events stored in data/events.npz\n")
    with open("sims/TChiWZ_550_200.dat", "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")


def main() -> None:
    _ensure_dirs()
    _load_or_note_context()

    cfg = ModelConfig(
        mC1_GeV=550.0,
        mN2_GeV=550.0,
        mN1_GeV=200.0,
        sigma_fb_effective_trilepton=10.0,  # effective σ for trilepton (includes BRs to e/μ)
        lumi_fb=35.9,
        n_events=200_000,
        rng_seed=42,
    )
    events, meta = _sample_events(cfg)
    _write_outputs(events, meta)
    print(f"Generated {cfg.n_events} events; saved to data/events.npz and sims/TChiWZ_550_200.dat")


if __name__ == "__main__":
    main()