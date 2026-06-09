"""SciTracer core modules.

Public surface
--------------
SciTracer          — main pipeline orchestrator
VerificationReport — structured result from verify()
Paper              — a downloaded and pre-processed research paper

Usage
-----
    from core import SciTracer, VerificationReport
    tracer = SciTracer()
    results = tracer.run("How does attention work in transformers?")
    report  = tracer.verify(results["question"], results["research_plan"], plan_str)
    print(report.summary)
"""
from core.lm_setup import _setup_lm
from core.tool_registry import _TOOL_REGISTRY, _tool_descriptions_str, _tool_names_str
from core.print_helpers import (
    _parse_structure,
    _print_hoare,
    _print_kripke,
    _print_verification,
    #_print_knowledge_graph,
    _print_final_decomposition,
    _print_tree_node,
    _fmt_block,
)

__all__ = [
    # Primary public API
    "SciTracer",
    "_plan_to_str",
    # Setup helpers
    "_setup_lm",
    # Tool registry
    "_TOOL_REGISTRY",
    "_tool_descriptions_str",
    "_tool_names_str",
    "TrajectoryVectorDB",
    # Printing / output helpers
    "_parse_structure",
    "_print_hoare",
    "_print_kripke",
    "_print_verification",
    "_print_knowledge_graph",
    "_print_final_decomposition",
    "_print_tree_node",
    "_fmt_block",
    # DSPy signature (internal)
    "PlanRefinement",
]
