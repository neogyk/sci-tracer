from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import List

import numpy as np
from numpy.typing import NDArray
import zipfile


def _load_module(module_path: Path, module_name: str = "wzsel") -> ModuleType:
    if not module_path.exists():
        raise FileNotFoundError(f"Analysis module not found at {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module at {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_single_file(pattern: str, description: str) -> Path:
    files: List[Path] = sorted(Path(".").glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {description} found matching pattern: {pattern}")
    # Prefer NPZ-like (zip) files when looking for .root samples saved via np.savez_compressed
    # This avoids selecting genuine ROOT files that numpy cannot load.
    if ".root" in pattern:
        zip_like = [f for f in files if zipfile.is_zipfile(f)]
        if zip_like:
            return zip_like[0]
    return files[0]


def validate_mc_uncertainties(
    analysis_module_path: Path = Path("analysis/wz_zz_selection.py"),
    sample_glob: str = "data/*.root",
    results_glob: str = "results/*.yaml",
    rel_threshold: float = 0.20,
) -> None:
    # Load analysis module
    mod: ModuleType = _load_module(analysis_module_path)

    # Locate inputs
    yml_path: Path = _find_single_file(results_glob, "results YAML template")
    sample_path: Path = _find_single_file(sample_glob, "data sample")

    # Recompute histogram and sumw2 via the analysis selection
    edges, counts, w2 = mod.select_and_hist(str(sample_path), str(yml_path))

    counts_arr: NDArray[np.float64] = np.asarray(counts, dtype=float)
    w2_arr: NDArray[np.float64] = np.asarray(w2, dtype=float)
    edges_arr: NDArray[np.float64] = np.asarray(edges, dtype=float)

    if counts_arr.shape != w2_arr.shape:
        raise ValueError(
            f"Shape mismatch between counts {counts_arr.shape} and w2 {w2_arr.shape}"
        )
    if edges_arr.size != counts_arr.size + 1:
        raise ValueError(
            f"Inconsistent binning: edges size {edges_arr.size} vs counts size {counts_arr.size}"
        )

    rel: NDArray[np.float64] = np.zeros_like(counts_arr, dtype=float)
    mask_pos: NDArray[np.bool_] = counts_arr > 0.0
    rel[mask_pos] = np.sqrt(w2_arr[mask_pos]) / counts_arr[mask_pos]

    failing_idx: NDArray[np.int64] = np.where(mask_pos & (rel > rel_threshold))[0]
    if failing_idx.size > 0:
        details: List[str] = []
        for i in failing_idx:
            lo: float = float(edges_arr[i])
            hi: float = float(edges_arr[i + 1])
            details.append(
                f"bin {i} [{lo:.1f}, {hi:.1f}) GeV: yield={counts_arr[i]:.6g}, "
                f"sqrt(sumw2)={np.sqrt(w2_arr[i]):.6g}, rel={rel[i]:.3f}"
            )
        msg = (
            f"Relative MC uncertainty exceeds {rel_threshold:.0%} in {failing_idx.size} bin(s):\n"
            + "\n".join(details)
        )
        raise RuntimeError(msg)

    print(
        f"Uncertainty check passed: all populated bins have ≤ {rel_threshold:.0%} relative MC uncertainty."
    )


if __name__ == "__main__":
    validate_mc_uncertainties()