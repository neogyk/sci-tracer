import os
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Config:
    m_parent: float = 550.0  # GeV (m(chi1+)/m(chi2^0))
    m_lsp: float = 200.0  # GeV (m(chi1^0))
    mZ: float = 91.1876  # GeV
    mW: float = 80.379  # GeV
    sigma_pb: float = 0.055  # pb
    lumi_fb: float = 35.9  # fb^-1
    BR_Z_ll: float = 0.0673  # Z -> e/mu (ee + mumu)
    BR_W_lnu: float = 0.214  # W -> e nu or mu nu
    n_events: int = 400_000
    seed: int = 12345
    smear_lep_rel: float = 0.01
    met_res_xy: float = 10.0  # GeV
    parent_pt_scale: float = 120.0  # GeV, exp pT distribution parameter
    out_path: str = "data/tchiwz_550_200_events.root"
    sims_card_path: str = "sims/cards.dat"


def _ensure_dirs() -> None:
    for d in ("data", "sims"):
        os.makedirs(d, exist_ok=True)


def rand_unit(n: int, rng: np.random.Generator) -> np.ndarray:
    u: np.ndarray = rng.uniform(-1.0, 1.0, size=n)
    phi: np.ndarray = rng.uniform(0.0, 2.0 * math.pi, size=n)
    s: np.ndarray = np.sqrt(np.clip(1.0 - u * u, 0.0, None))
    vx: np.ndarray = s * np.cos(phi)
    vy: np.ndarray = s * np.sin(phi)
    vz: np.ndarray = u
    return np.stack((vx, vy, vz), axis=1)


def lorentz_boost(p4: np.ndarray, beta: np.ndarray) -> np.ndarray:
    # p4 shape: (N, 4) with [E, px, py, pz]
    # beta shape: (N, 3)
    b2: np.ndarray = np.sum(beta * beta, axis=1)
    out: np.ndarray = np.zeros_like(p4)
    mask: np.ndarray = b2 > 0.0

    if np.any(mask):
        E: np.ndarray = p4[mask, 0]
        p: np.ndarray = p4[mask, 1:4]
        b: np.ndarray = beta[mask]
        b2m: np.ndarray = b2[mask]
        gamma: np.ndarray = 1.0 / np.sqrt(1.0 - b2m)
        bp: np.ndarray = np.sum(b * p, axis=1)
        Ep: np.ndarray = gamma * (E + bp)
        # p' = p + [((gamma-1)*(beta·p)/beta^2) + gamma*E] * beta
        factor1: np.ndarray = ((gamma - 1.0) / b2m) * bp  # shape (Nm,)
        factor2: np.ndarray = gamma * E  # shape (Nm,)
        coef: np.ndarray = (factor1 + factor2)[:, None]  # (Nm,1)
        pp: np.ndarray = p + coef * b
        out[mask, 0] = Ep
        out[mask, 1:4] = pp

    if np.any(~mask):
        out[~mask] = p4[~mask]

    return out


def two_body_pstar(m_par: np.ndarray, mA: float, mB: float) -> np.ndarray:
    # p* = sqrt((m^2 - (mA+mB)^2)*(m^2 - (mA-mB)^2)) / (2 m)
    msq: np.ndarray = np.square(m_par)
    term1: np.ndarray = np.clip(msq - (mA + mB) ** 2, 0.0, None)
    term2: np.ndarray = np.clip(msq - (mA - mB) ** 2, 0.0, None)
    lam: np.ndarray = term1 * term2
    with np.errstate(invalid="ignore"):
        pstar: np.ndarray = np.sqrt(lam) / (2.0 * np.where(m_par > 0.0, m_par, 1.0))
    pstar = np.nan_to_num(pstar)
    return pstar


def make_parent_pair(n: int, m: float, scale: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    pT: np.ndarray = rng.exponential(scale=scale, size=n)
    phi: np.ndarray = rng.uniform(0.0, 2.0 * math.pi, size=n)
    px: np.ndarray = pT * np.cos(phi)
    py: np.ndarray = pT * np.sin(phi)
    pz: np.ndarray = np.zeros(n)
    E: np.ndarray = np.sqrt(m * m + pT * pT)
    p4_C: np.ndarray = np.stack((E, px, py, pz), axis=1)
    p4_N2: np.ndarray = np.stack((E, -px, -py, pz), axis=1)
    return p4_C, p4_N2


def two_body_decay(parent_p4: np.ndarray, mA: float, mB: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n: int = parent_p4.shape[0]
    # Parent mass from p4
    mass_sq: np.ndarray = np.maximum(parent_p4[:, 0] ** 2 - np.sum(parent_p4[:, 1:4] ** 2, axis=1), 0.0)
    m_par: np.ndarray = np.sqrt(mass_sq)
    pstar: np.ndarray = two_body_pstar(m_par, mA, mB)
    EA: np.ndarray = np.sqrt(pstar * pstar + mA * mA)
    EB: np.ndarray = np.sqrt(pstar * pstar + mB * mB)

    u: np.ndarray = rand_unit(n, rng)
    pA_rf: np.ndarray = pstar[:, None] * u
    pB_rf: np.ndarray = -pA_rf

    p4A_rf: np.ndarray = np.concatenate((EA[:, None], pA_rf), axis=1)
    p4B_rf: np.ndarray = np.concatenate((EB[:, None], pB_rf), axis=1)

    beta: np.ndarray = parent_p4[:, 1:4] / parent_p4[:, 0:1]
    p4A_lab: np.ndarray = lorentz_boost(p4A_rf, beta)
    p4B_lab: np.ndarray = lorentz_boost(p4B_rf, beta)
    return p4A_lab, p4B_lab


def boson_decay_ll(p4V: np.ndarray, mV: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n: int = p4V.shape[0]
    Estar: float = 0.5 * mV
    pstar: float = 0.5 * mV
    u: np.ndarray = rand_unit(n, rng)
    p_rf: np.ndarray = pstar * u
    l1_rf: np.ndarray = np.concatenate((np.full((n, 1), Estar), p_rf), axis=1)
    l2_rf: np.ndarray = np.concatenate((np.full((n, 1), Estar), -p_rf), axis=1)
    beta: np.ndarray = p4V[:, 1:4] / p4V[:, 0:1]
    l1: np.ndarray = lorentz_boost(l1_rf, beta)
    l2: np.ndarray = lorentz_boost(l2_rf, beta)
    return l1, l2


def boson_decay_lnu(p4V: np.ndarray, mV: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    l, nu = boson_decay_ll(p4V, mV, rng)
    return l, nu


def smear_p4(p4: np.ndarray, rel_sigma: float, rng: np.random.Generator) -> np.ndarray:
    n: int = p4.shape[0]
    delta: np.ndarray = rng.normal(loc=0.0, scale=rel_sigma, size=(n, 1))
    p: np.ndarray = p4[:, 1:4] * (1.0 + delta)
    E: np.ndarray = np.sqrt(np.sum(p * p, axis=1))
    out: np.ndarray = np.zeros_like(p4)
    out[:, 0] = E
    out[:, 1:4] = p
    return out


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)
    _ensure_dirs()

    # Generate parents
    p4_C, p4_N2 = make_parent_pair(cfg.n_events, cfg.m_parent, cfg.parent_pt_scale, rng)
    # Two-body decays
    p4_W, p4_LSP_C = two_body_decay(p4_C, cfg.mW, cfg.m_lsp, rng)
    p4_Z, p4_LSP_N = two_body_decay(p4_N2, cfg.mZ, cfg.m_lsp, rng)
    # Boson decays (massless leptons)
    lW, nu = boson_decay_lnu(p4_W, cfg.mW, rng)
    lZ1, lZ2 = boson_decay_ll(p4_Z, cfg.mZ, rng)

    # Assign lepton flavors (0=e, 1=mu)
    flav_W: np.ndarray = rng.integers(0, 2, size=cfg.n_events, endpoint=False)
    flav_Z: np.ndarray = rng.integers(0, 2, size=cfg.n_events, endpoint=False)

    # Smear leptons
    lW_s: np.ndarray = smear_p4(lW, cfg.smear_lep_rel, rng)
    lZ1_s: np.ndarray = smear_p4(lZ1, cfg.smear_lep_rel, rng)
    lZ2_s: np.ndarray = smear_p4(lZ2, cfg.smear_lep_rel, rng)

    # MET from LSPs + neutrino + resolution
    met_x: np.ndarray = -(p4_LSP_C[:, 1] + p4_LSP_N[:, 1] + nu[:, 1])
    met_y: np.ndarray = -(p4_LSP_C[:, 2] + p4_LSP_N[:, 2] + nu[:, 2])
    met_x = met_x + rng.normal(0.0, cfg.met_res_xy, size=cfg.n_events)
    met_y = met_y + rng.normal(0.0, cfg.met_res_xy, size=cfg.n_events)

    # Weights: total expected events = sigma(pb)*lumi(fb^-1)*1e3; then apply leptonic BRs
    tri_br: float = cfg.BR_Z_ll * cfg.BR_W_lnu
    per_event_w: float = cfg.sigma_pb * cfg.lumi_fb * 1000.0 * tri_br / float(cfg.n_events)
    weights: np.ndarray = np.full(cfg.n_events, per_event_w, dtype=np.float64)

    # Save NPZ (with .root suffix) without numpy auto-appending .npz:
    # write to a file-like object so the exact filename is used.
    with open(cfg.out_path, "wb") as f:
        np.savez_compressed(
            f,
            lW=lW_s.astype(np.float64),
            lZ1=lZ1_s.astype(np.float64),
            lZ2=lZ2_s.astype(np.float64),
            flav_W=flav_W.astype(np.int8),
            flav_Z=flav_Z.astype(np.int8),
            met_x=met_x.astype(np.float64),
            met_y=met_y.astype(np.float64),
            weight=weights,
        )

    # Write simple simulation card
    with open(cfg.sims_card_path, "w", encoding="utf-8") as f:
        f.write("Model: TChiWZ\n")
        f.write(f"m(chi1+/chi2^0)={cfg.m_parent} GeV, m(chi1^0)={cfg.m_lsp} GeV\n")
        f.write(f"sigma(pp->chi1+ chi2^0)={cfg.sigma_pb} pb (assumed)\n")
        f.write(f"L={cfg.lumi_fb} fb^-1\n")
        f.write(f"BR(Z->e/mu)={cfg.BR_Z_ll}, BR(W->e/mu)={cfg.BR_W_lnu}, BR(3l)={tri_br}\n")
        f.write(
            "Generation: vectorized 2-body decays (on-shell W/Z), isotropic, "
            f"detector smearing {cfg.smear_lep_rel*100:.1f}% leptons, {cfg.met_res_xy} GeV MET resolution\n"
        )

    print(f"Generated {cfg.n_events} events and saved to {cfg.out_path}")


if __name__ == "__main__":
    main()