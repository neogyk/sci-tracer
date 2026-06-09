from pathlib import Path
from typing import NoReturn


def write_selection_module() -> None:
    """Write the CMS-SUS-16-034 WZ/ZZ on-Z trilepton selection module to analysis/wz_zz_selection.py."""
    src: str = '''import os
import glob
import math
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import numpy as np


def _pt_eta_phi(p4: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # p4: (N,4) [E, px, py, pz]
    px, py, pz = p4[:, 1], p4[:, 2], p4[:, 3]
    pt = np.sqrt(px * px + py * py)
    p = np.sqrt(px * px + py * py + pz * pz)
    # protect division
    eta = 0.5 * np.log((p + pz + 1e-12) / (p - pz + 1e-12))
    phi = np.arctan2(py, px)
    return pt, eta, phi


def _inv_mass(p41: np.ndarray, p42: np.ndarray) -> np.ndarray:
    E = p41[:, 0] + p42[:, 0]
    px = p41[:, 1] + p42[:, 1]
    py = p41[:, 2] + p42[:, 2]
    pz = p41[:, 3] + p42[:, 3]
    m2 = np.maximum(E * E - (px * px + py * py + pz * pz), 0.0)
    return np.sqrt(m2)


def _mt(ptl: np.ndarray, phil: np.ndarray, met: np.ndarray, phim: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 * ptl * met * (1.0 - np.cos(phil - phim)))


def _find_results_yaml() -> str:
    ys = sorted(glob.glob('results/*.yaml'))
    if not ys:
        raise FileNotFoundError('No results/*.yaml template found')
    # Prefer files that mention wz/zz in name
    for y in ys:
        name = os.path.basename(y).lower()
        if 'wz' in name or 'zz' in name:
            return y
    return ys[0]


def _load_bin_edges(yaml_path: str) -> Tuple[np.ndarray, Tuple[str, object]]:
    import yaml  # local import to avoid hard dependency on writer

    with open(yaml_path, 'r') as f:
        obj = yaml.safe_load(f)

    # 1) {bin_edges: [..]}
    if isinstance(obj, dict) and 'bin_edges' in obj and isinstance(obj['bin_edges'], list):
        return np.array(obj['bin_edges'], dtype=float), ('bin_edges', None)

    # 2) bins: list of dicts with low/high or min/max
    def find_bins_dict(o, path: Tuple[str, ...] = ()) -> Optional[Tuple[Tuple[str, ...], list]]:
        if isinstance(o, dict):
            for k, v in o.items():
                res = find_bins_dict(v, path + (k,))
                if res is not None:
                    return res
        elif isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
            keys = o[0].keys()
            if (("low" in keys and "high" in keys) or ("min" in keys and "max" in keys)):
                return (path, o)
        return None

    res = find_bins_dict(obj)
    if res is not None:
        path, bins_list = res
        lows, highs = [], []
        for b in bins_list:
            lo = b.get('low', b.get('min'))
            hi = b.get('high', b.get('max'))
            lows.append(float(lo))
            highs.append(float(hi))
        # Construct edges from contiguous bins
        edges = [lows[0]] + highs
        return np.array(edges, dtype=float), ('bins_path', path)

    # 3) fallback: look for keys 'edges' under nested dicts
    def find_edges(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ('edges', 'binning', 'binEdges') and isinstance(v, list):
                    return np.array(v, dtype=float), ('path', (k,))
                r = find_edges(v)
                if r is not None:
                    return r
        return None

    r = find_edges(obj)
    if r is not None:
        return r[0], r[1]

    raise RuntimeError('Could not locate bin edges in YAML template; please ensure it contains binning info')


def _fill_yaml_with_yields(yaml_path: str, yields: Sequence[float]) -> None:
    import yaml  # local import
    with open(yaml_path, 'r') as f:
        obj = yaml.safe_load(f)

    updated = False

    # Case A: bins list of dicts with a 'value' or 'yield' field
    def set_bins_values(o, vals: Sequence[float]) -> None:
        nonlocal updated
        if isinstance(o, dict):
            for _, v in o.items():
                set_bins_values(v, vals)
        elif isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
            keys = o[0].keys()
            if (("low" in keys and "high" in keys) or ("min" in keys and "max" in keys)):
                for i, b in enumerate(o):
                    if 'value' in b:
                        b['value'] = float(vals[i])
                    elif 'yield' in b:
                        b['yield'] = float(vals[i])
                    else:
                        b['value'] = float(vals[i])
                updated = True

    set_bins_values(obj, yields)

    # Case B: top-level or nested 'yields' key aligned with length
    def set_yields(o, vals: Sequence[float]) -> None:
        nonlocal updated
        if isinstance(o, dict):
            for k in ('yields', 'values', 'counts'):
                if k in o and isinstance(o[k], list) and len(o[k]) == len(vals):
                    o[k] = [float(x) for x in vals]
                    updated = True
            for v in o.values():
                set_yields(v, vals)

    if not updated:
        set_yields(obj, yields)

    with open(yaml_path, 'w') as f:
        yaml.safe_dump(obj, f, sort_keys=False)

    if not updated:
        raise RuntimeError('Filled YAML but could not confirm values were set alongside bins; please inspect template structure')


def select_and_hist(sample_path: str, yaml_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Tolerate numpy .npz auto-extension when saving with arbitrary suffix (e.g., .root)
    if not os.path.exists(sample_path) and os.path.exists(sample_path + '.npz'):
        sample_path = sample_path + '.npz'

    # Robust fallback: try loading, and if it fails, try appending .npz
    try:
        data = np.load(sample_path, allow_pickle=True)
    except FileNotFoundError:
        alt = sample_path + '.npz'
        if os.path.exists(alt):
            data = np.load(alt, allow_pickle=True)
        else:
            raise

    lW = data['lW']
    lZ1 = data['lZ1']
    lZ2 = data['lZ2']
    met_x = data['met_x']
    met_y = data['met_y']
    w = data['weight']

    # Kinematics
    ptW, etaW, phiW = _pt_eta_phi(lW)
    ptZ1, etaZ1, phiZ1 = _pt_eta_phi(lZ1)
    ptZ2, etaZ2, phiZ2 = _pt_eta_phi(lZ2)
    mZcand = _inv_mass(lZ1, lZ2)
    met = np.sqrt(met_x * met_x + met_y * met_y)
    phim = np.arctan2(met_y, met_x)
    MT = _mt(ptW, phiW, met, phim)

    # Selection (approx CMS-SUS-16-034): on-Z, pT thresholds, eta, MT>100 GeV, b-veto assumed
    sel = (
        (mZcand > 76.0)
        & (mZcand < 106.0)
        & (ptW > 20.0)
        & (ptZ1 > 15.0)
        & (ptZ2 > 15.0)
        & (np.abs(etaW) < 2.4)
        & (np.abs(etaZ1) < 2.5)
        & (np.abs(etaZ2) < 2.5)
        & (MT > 100.0)
    )

    # Bin edges from YAML template
    edges, _ = _load_bin_edges(yaml_path)

    # Weighted histogram and sumw2
    met_sel = met[sel]
    w_sel = w[sel]
    counts, _ = np.histogram(met_sel, bins=edges, weights=w_sel)
    w2, _ = np.histogram(met_sel, bins=edges, weights=w_sel * w_sel)

    return edges, counts, w2


def run() -> None:
    # Locate inputs
    yml = _find_results_yaml()
    samples = sorted(glob.glob('data/*.root'))
    if not samples:
        raise FileNotFoundError('No data/*.root sample found')
    samp = samples[0]

    edges, counts, w2 = select_and_hist(samp, yml)

    # Write results back into template
    _fill_yaml_with_yields(yml, counts)

    # Also emit a companion file with uncertainties for reporting
    out_unc = yml.replace('.yaml', '_uncertainties.yaml')
    try:
        import yaml
        with open(out_unc, 'w') as f:
            yaml.safe_dump({'bin_edges': edges.tolist(), 'yields': counts.tolist(), 'sumw2': w2.tolist()}, f, sort_keys=False)
    except Exception:
        with open(out_unc, 'w') as f:
            f.write(str({'bin_edges': edges.tolist(), 'yields': counts.tolist(), 'sumw2': w2.tolist()}))

    print(f'Filled results in {yml} and wrote uncertainties to {out_unc}')


if __name__ == '__main__':
    run()
'''
    out_path: Path = Path('analysis') / 'wz_zz_selection.py'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(src)
    print('Wrote analysis/wz_zz_selection.py')


if __name__ == "__main__":
    write_selection_module()