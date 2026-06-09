import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError(f"PyYAML is required to parse results templates: {e}")

# Broaden type checks to support non-dict/list YAML node types (e.g., ruamel.yaml)
from collections.abc import Mapping, Sequence as SeqABC

Context = Dict[str, Any]
BinPair = Tuple[float, float]


def _is_sequence(obj: Any) -> bool:
    # Consider general sequences but exclude strings/bytes
    return isinstance(obj, SeqABC) and not isinstance(obj, (str, bytes))


def load_or_init_context(path: str) -> Context:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    # Initialize minimal context if missing (robustness)
    return {"results_yaml": "", "bin_edges": []}


def mentions_met(text: str) -> bool:
    pattern = re.compile(r"E[ _-]*T\^?miss|E\s*miss|MET", flags=re.IGNORECASE)
    return bool(pattern.search(text))


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def try_extract_bins_from_bins_list(bins_list) -> Optional[List[BinPair]]:
    if not bins_list:
        return None
    first = bins_list[0]
    if not isinstance(first, Mapping):
        return None
    key_pairs: List[Tuple[str, str]] = [
        ("low", "high"),
        ("lo", "hi"),
        ("min", "max"),
        ("edge_low", "edge_high"),
    ]
    # Find which key pair matches
    chosen_keys: Optional[Tuple[str, str]] = None
    for a, b in key_pairs:
        ok_all = True
        for it in bins_list:
            if not isinstance(it, Mapping) or (a not in it) or (b not in it):
                ok_all = False
                break
        if ok_all:
            chosen_keys = (a, b)
            break
    if chosen_keys is None:
        return None
    lo_key, hi_key = chosen_keys
    edges: List[BinPair] = []
    for it in bins_list:
        # it is a Mapping; use get
        lo = _as_float(it.get(lo_key))  # type: ignore[attr-defined]
        hi = _as_float(it.get(hi_key))  # type: ignore[attr-defined]
        if lo is None or hi is None:
            return None
        edges.append((lo, hi))
    return edges if edges else None


def extract_bins(obj: Any) -> Optional[List[BinPair]]:
    # Try direct mapping with 'bins'
    if isinstance(obj, Mapping):
        bins_obj = obj.get("bins")
        if _is_sequence(bins_obj):
            parsed = try_extract_bins_from_bins_list(bins_obj)  # type: ignore[arg-type]
            if parsed:
                return parsed
        # Recurse into mapping values
        for v in obj.values():
            res = extract_bins(v)
            if res:
                return res
    # Recurse into sequences (exclude strings)
    if _is_sequence(obj):
        for it in obj:  # type: ignore[assignment]
            res = extract_bins(it)
            if res:
                return res
    return None


def find_yaml_with_bins(yaml_files) -> Tuple[str, List[BinPair]]:
    # Try files that mention MET first
    for yf in yaml_files:
        with open(yf, "r") as fh:
            text = fh.read()
        if mentions_met(text):
            with open(yf, "r") as fh:
                docs = list(yaml.safe_load_all(fh))
            for d in docs:
                edges = extract_bins(d)
                if edges:
                    return yf, edges
    # Fallback: first YAML with any bins
    for yf in yaml_files:
        with open(yf, "r") as fh:
            docs = list(yaml.safe_load_all(fh))
        for d in docs:
            edges = extract_bins(d)
            if edges:
                return yf, edges
    raise RuntimeError("Could not extract bin edges from any YAML template in results/*.yaml")


def main() -> None:
    # Preconditions:
    # - results/ contains at least one YAML template
    # - data/pipeline_context.json exists (we handle missing by initializing)
    ctx_path = os.path.join("data", "pipeline_context.json")
    ctx = load_or_init_context(ctx_path)

    yaml_files = sorted(glob.glob(os.path.join("results", "*.yaml")))
    if not yaml_files:
        raise FileNotFoundError("No YAML template found in results/*.yaml")

    chosen_yaml, edges = find_yaml_with_bins(yaml_files)

    # Persist in context (use JSON-friendly lists)
    ctx["results_yaml"] = chosen_yaml
    ctx["bin_edges"] = [[float(lo), float(hi)] for lo, hi in edges]

    # Ensure data directory exists
    os.makedirs(os.path.dirname(ctx_path), exist_ok=True)
    with open(ctx_path, "w") as f:
        json.dump(ctx, f)

    print("Using results YAML:", chosen_yaml)
    print("Bin edges:", edges)


if __name__ == "__main__":
    main()