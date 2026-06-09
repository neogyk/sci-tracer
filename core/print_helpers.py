from __future__ import annotations

from typing import Any

from storage.knowledge_graph import KnowledgeGraph
from tools.kripke import KripkeStructure, VerificationResult, parse_kripke_from_llm


# ── Decomposition print helpers ───────────────────────────────────────────────

def _fmt_block(text: str, prefix: str, indent: str) -> str:
    """Wrap long text with a hanging indent; preserves existing newlines."""
    lines_out: list[str] = []
    for i, raw_line in enumerate(text.splitlines()):
        lead = prefix if i == 0 else indent
        # word-wrap each source line at column 100
        words = raw_line.split()
        cur = lead
        for w in words:
            if cur == lead or cur == indent:
                cur += w
            elif len(cur) + 1 + len(w) <= 100:
                cur += " " + w
            else:
                lines_out.append(cur)
                cur = indent + w
        lines_out.append(cur)
    return "\n".join(lines_out)


def _print_tree_node(
    node:         "HoareTreeNode",
    indent:       int,
    verified:     bool | None = None,   # True=verified  False=failed  None=unknown
    z3_verified:  bool = False,
) -> None:
    """Recursively print a decomposition tree node with verification status.

    Internal nodes: one-line breadcrumb (path + task + badge).
    Leaf nodes    : full {P}, C, {Q}, command_body, and verification badge.
    """
    pad  = "  " * indent
    step = node.step

    # ── Verification badge ────────────────────────────────────────────────────
    if verified is True:
        badge = "✓ VERIFIED"
    elif verified is False:
        badge = "✗ UNVERIFIED"
    elif z3_verified:
        badge = "✓ Z3"
    else:
        badge = "?"

    if node.is_leaf:
        atomicity_tag = f"[{node.atomicity or '?'}]"
        tool_tag = f"  ACTION: {step.tool}" if step.tool else "  ACTION: (none)"
        print(f"{pad}┌─ LEAF {atomicity_tag}  [{badge}]  path={node.path}  step_id={step.step}{tool_tag}")
        if step.tool_rationale:
            print(f"{pad}│  tool rationale: {step.tool_rationale}")
        print(f"{pad}│")
        print(f"{pad}│  {{P}}")
        for line in step.precondition.splitlines():
            print(f"{pad}│  {line}")
        print(f"{pad}│")
        print(f"{pad}│   C")
        for line in step.command.splitlines():
            print(f"{pad}│  {line}")
        print(f"{pad}│")
        print(f"{pad}│  {{Q}}")
        for line in step.postcondition.splitlines():
            print(f"{pad}│  {line}")
        if step.verification_conditions:
            print(f"{pad}│")
            print(f"{pad}│  VERIFICATION CONDITIONS")
            for vc in step.verification_conditions:
                print(f"{pad}│    [{vc.kind}] {vc.description}")
                print(f"{pad}│      {vc.assertion}")
        if step.completion_metric:
            print(f"{pad}│")
            print(f"{pad}│  COMPLETION METRIC")
            print(f"{pad}│    {step.completion_metric}")
        if step.precondition_resources:
            print(f"{pad}│")
            print(f"{pad}│  PRECONDITION RESOURCES")
            for pr in step.precondition_resources:
                status = "available" if pr.available else "ACQUIRE"
                print(f"{pad}│    [{status}] {pr.variable}")
                if pr.available:
                    print(f"{pad}│      source: {pr.source}")
                else:
                    print(f"{pad}│      acquisition: {pr.acquisition}")
                    if pr.tool:
                        print(f"{pad}│      tool: {pr.tool}")
        if step.pre_state:
            print(f"{pad}│")
            print(f"{pad}│  Σ_pre  (state before α)")
            for sv in step.pre_state:
                mut = "rw" if sv.mutable else "ro"
                print(f"{pad}│    [{mut}] {sv.name}: {sv.python_type}"
                      + (f"  # {sv.constraint}" if sv.constraint else ""))
        if step.action_sig:
            sig = step.action_sig
            print(f"{pad}│")
            print(f"{pad}│  α  {sig.name}")
            print(f"{pad}│    reads : {sig.reads}")
            print(f"{pad}│    writes: {sig.writes}")
            if sig.frame:
                print(f"{pad}│    frame : {sig.frame}")
            if sig.expression:
                print(f"{pad}│    expr  : {sig.expression[:100]}")
        if step.post_state:
            print(f"{pad}│")
            print(f"{pad}│  Σ_post (state after α)")
            for sv in step.post_state:
                mut = "rw" if sv.mutable else "ro"
                print(f"{pad}│    [{mut}] {sv.name}: {sv.python_type}"
                      + (f"  # {sv.constraint}" if sv.constraint else ""))
        if step.command_body:
            print(f"{pad}│")
            print(f"{pad}│  body:")
            for line in step.command_body.splitlines():
                print(f"{pad}│    {line}")
        print(f"{pad}└{'─' * (60 - indent * 2)}")
    else:
        print(f"{pad}▸ [{node.path}] depth={node.depth}  [{badge}]  «{step.task[:70]}»")
        for child in node.children:
            _print_tree_node(child, indent + 1, verified=verified, z3_verified=z3_verified)


def _print_final_decomposition(
    decomp_roots:       list,
    decomposition_mode: str,
    htn_network:        "HTNNetwork | None" = None,
    lean_results:       list[dict] | None = None,
    z3_result:          dict | None = None,
) -> None:
    """Print the complete decomposition forest with per-tree verification badges."""
    sep  = "═" * 70
    thin = "─" * 70

    # ── Build verification lookup ─────────────────────────────────────────────
    lean_by_step: dict[int, dict] = {r["step"]: r for r in (lean_results or [])}
    z3_pipe      = (z3_result or {}).get("z3_pipeline", {})
    z3_verified  = bool(z3_pipe.get("surviving", []))

    print(f"\n{sep}")
    print(f"  FINAL DECOMPOSITION FOREST  (mode={decomposition_mode})")
    print(sep)

    if decomposition_mode == "tree" and decomp_roots:
        all_leaves   = [n for root in decomp_roots for n in root.leaves()]
        atomic_n     = sum(1 for n in all_leaves if n.atomicity == "atomic")
        compound_n   = sum(1 for n in all_leaves if n.atomicity == "compound")
        branch_n     = sum(1 for n in all_leaves if n.atomicity == "branching")
        depths       = [n.depth for n in all_leaves]
        max_depth    = max(depths) if depths else 0

        # Count verified roots
        verified_roots = 0
        for root in decomp_roots:
            lr = lean_by_step.get(root.step.step, {})
            if lr.get("verified") or (not lr and z3_verified):
                verified_roots += 1

        print(f"\n  {len(decomp_roots)} top-level steps  →  {len(all_leaves)} leaf tasks")
        print(f"  atomic={atomic_n}  compound={compound_n}  branching={branch_n}"
              f"  max_leaf_depth={max_depth}")
        print(f"  verified={verified_roots}/{len(decomp_roots)} trees")
        print()

        for root_idx, root in enumerate(decomp_roots, 1):
            lr        = lean_by_step.get(root.step.step, {})
            if lr.get("verified"):
                tree_badge = "✓ LEAN VERIFIED"
                verified   = True
            elif "error" in lr and lr["error"] == "lean_unavailable":
                tree_badge = ("✓ Z3 VERIFIED" if z3_verified else "✗ Z3 UNVERIFIED") + " (plan)"
                verified   = z3_verified
            elif lr:
                tree_badge = f"✗ LEAN UNVERIFIED (attempts={lr.get('attempts', 1)})"
                verified   = False
            else:
                tree_badge = "✓ Z3 VERIFIED (plan)" if z3_verified else "✗ UNVERIFIED"
                verified   = z3_verified

            root_leaves = root.leaves()
            print(f"{thin}")
            print(f"  TREE [{root_idx}/{len(decomp_roots)}]  [{tree_badge}]")
            print(f"  Step {root.step.step}: {root.step.task}")
            print(f"  {{P}} {root.step.precondition[:100]}")
            print(f"  {{Q}} {root.step.postcondition[:100]}")
            print(f"  {len(root_leaves)} leaf task(s)")
            print(thin)
            _print_tree_node(root, indent=1, verified=verified, z3_verified=z3_verified)
            print()

    elif decomposition_mode == "htn" and htn_network is not None:
        ordered = htn_network.topological_sort()
        prims   = [t for t in ordered if t.is_primitive]
        comps   = [t for t in ordered if not t.is_primitive]

        print(f"\n  {len(prims)} primitive tasks in execution order")
        print(f"  {len(comps)} compound tasks remaining")
        print(f"  ordering_edges={len(htn_network.ordering)}"
              f"  causal_links={len(htn_network.causal_links)}"
              f"  is_solution={htn_network.is_solution()}")
        print()

        for i, t in enumerate(prims, 1):
            args_str = f"({', '.join(t.args)})" if t.args else ""
            print(f"{thin}")
            print(f"  [{i:02d}]  {t.name}{args_str}")
            print(thin)

            if t.preconditions:
                print(_fmt_block(" ∧ ".join(t.preconditions),
                                  prefix="  {P} ",
                                  indent ="       "))
            print()
            print(_fmt_block(t.command if t.command else "(no command)",
                              prefix="   C  ",
                              indent ="      "))
            print()
            if t.effects:
                print(_fmt_block(" ∧ ".join(t.effects),
                                  prefix="  {Q} ",
                                  indent ="       "))
            if t.command_body:
                print("\n  body:")
                for line in t.command_body.splitlines():
                    print(f"    {line}")
            print()

    else:
        print(f"\n  (no decomposition — flat plan)")

    print(f"\n{sep}\n")


# # ── Printing helpers ───────────────────────────────────────────────────────────

# def _print_knowledge_graph(kg: KnowledgeGraph, max_edges: int = 40) -> None:
#     print("\n── Knowledge Graph ──────────────────────────")
#     print(f"  Nodes : {len(kg.nodes)}")
#     print(f"  Edges : {len(kg)}")
#     if kg.edges:
#         print()
#         for r in kg.edges[:max_edges]:
#             print(f"  [{r.source}] --{r.type}--> [{r.target}]")
#         if len(kg) > max_edges:
#             print(f"  … {len(kg) - max_edges} more edges not shown")


def _print_hoare(hoare: Any) -> None:
    print("\n── Hoare Triple ─────────────────────────────")
    print(f"  {{P}} {hoare.precondition}")
    print(f"   C   {hoare.command}")
    print(f"  {{Q}} {hoare.postcondition}")


def _print_kripke(structure: KripkeStructure) -> None:
    print("\n── Kripke Structure M = (S, R, V) ───────────")
    print(f"  S (worlds)  : {', '.join(structure.worlds)}")
    print(f"  s0 (initial): {structure.initial}")
    print(f"  R (relation): {structure.relation}")
    print(f"  Props       : {', '.join(structure.propositions)}")
    for prop, worlds in structure.valuation.items():
        print(f"  V({prop:12}): {{{', '.join(sorted(worlds))}}}")


def _print_verification(results: list[VerificationResult], suggestion: str) -> None:
    print("\n── Verification Results ─────────────────────")
    for r in results:
        mark = "✓" if r.holds else "✗"
        line = f"  [{mark}] {r.property}"
        if r.witness:
            line += f"  — witness: {r.witness}"
        if r.path:
            line += f"  path: {' → '.join(r.path)}"
        if r.error:
            line += f"  ERROR: {r.error}"
        print(line)
    print(f"\n── Suggestion ───────────────────────────────")
    print(f"  {suggestion}")


# ── CLI entry point ────────────────────────────────────────────────────────────


def _parse_structure(formalization: dict[str, Any]) -> tuple[Any, KripkeStructure]:
    """Helper: parse KripkeStructure from a formalization dict."""
    kripke_pred = formalization["kripke"]
    structure = parse_kripke_from_llm(
        worlds_str=kripke_pred.worlds,
        initial=kripke_pred.initial_world,
        relation_str=kripke_pred.accessibility_relation,
        propositions_str=kripke_pred.atomic_propositions,
        valuation_str=kripke_pred.valuation,
    )
    return kripke_pred, structure
