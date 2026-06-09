from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, TypedDict


class PipelineContext(TypedDict):
    results_yaml: str
    bin_edges: List[Tuple[float, float]]


def initialize_project_structure(base_dir: Path = Path(".")) -> None:
    """
    Initialize project directories and create the pipeline context file.

    - Creates directories: results/, analysis/, sims/, data/
    - Touches data/.keep
    - Writes data/pipeline_context.json with keys: results_yaml, bin_edges
    """
    # Create required directories
    dirs: List[Path] = [base_dir / "results", base_dir / "analysis", base_dir / "sims", base_dir / "data"]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Touch placeholder file to ensure data/ is tracked and present
    keep_file: Path = base_dir / "data" / ".keep"
    keep_file.touch(exist_ok=True)

    # Initialize context
    ctx: PipelineContext = {"results_yaml": "", "bin_edges": []}

    # Write context JSON
    ctx_path: Path = base_dir / "data" / "pipeline_context.json"
    with ctx_path.open("w", encoding="utf-8") as f:
        json.dump(ctx, f)

    print("Initialized directories and context")


if __name__ == "__main__":
    initialize_project_structure()