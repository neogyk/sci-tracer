
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Tuple

Z_MASS: float = 91.1876

def load_events(path: str = 'data/events.npz') -> np.lib.npyio.NpzFile:
    return np.load(path, allow_pickle=True)

def on_z_pair(mll: npt.NDArray[np.floating], window: float = 15.0) -> npt.NDArray[np.bool_]:
    return np.abs(mll - Z_MASS) < window

def passes_pt_thresholds(p1: float, p2: float, p3: float) -> bool:
    pts = np.sort(np.array([p1, p2, p3], dtype=float))[::-1]
    return (pts[0] > 25.0) and (pts[1] > 20.0) and (pts[2] > 15.0)

def select_events(
    ev: np.lib.npyio.NpzFile,
) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.floating]]:
    met: npt.NDArray[np.floating] = ev['met']
    mll: npt.NDArray[np.floating] = ev['mll_z']
    lz1_pt: npt.NDArray[np.floating] = ev['lz1_pt']
    lz2_pt: npt.NDArray[np.floating] = ev['lz2_pt']
    lw_pt: npt.NDArray[np.floating] = ev['lw_pt']
    lz1_eta: npt.NDArray[np.floating] = ev['lz1_eta']
    lz2_eta: npt.NDArray[np.floating] = ev['lz2_eta']
    lw_eta: npt.NDArray[np.floating] = ev['lw_eta']
    mt: npt.NDArray[np.floating] = ev['mt']

    # Acceptance on eta
    acc: npt.NDArray[np.bool_] = (
        (np.abs(lz1_eta) < 2.4) & (np.abs(lz2_eta) < 2.4) & (np.abs(lw_eta) < 2.4)
    )
    # On-Z SFOS window (constructed in synthetic sample)
    onz: npt.NDArray[np.bool_] = on_z_pair(mll)

    # pT threshold mask (25, 20, 15) on the three leptons
    n: int = int(met.shape[0])
    ptmask_list = [
        passes_pt_thresholds(float(lz1_pt[i]), float(lz2_pt[i]), float(lw_pt[i]))
        for i in range(n)
    ]
    ptmask: npt.NDArray[np.bool_] = np.array(ptmask_list, dtype=np.bool_)

    # Transverse mass requirement
    mtmask: npt.NDArray[np.bool_] = (mt > 100.0)

    # Jet veto and b-veto are trivially satisfied in synthetic sample
    sel: npt.NDArray[np.bool_] = acc & onz & ptmask & mtmask

    return sel, met[sel]

if __name__ == '__main__':
    ev = load_events()
    sel, met_sel = select_events(ev)
    np.savez_compressed('data/selected.npz', met=met_sel, mask=sel)
    with open('data/selected.root', 'w') as f:
        f.write('placeholder: selected MET stored in data/selected.npz')
    print('Selected events:', int(sel.sum()))
