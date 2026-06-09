"""Z3-based CTL model checker for finite Kripke structures.

Supports operators: AG, EF, AX, EX, AF, EG.

Install dependency:
    pip install z3-solver
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

try:
    from z3 import BoolVal, Int, Not, Or, Solver, sat
except ImportError as exc:
    raise ImportError("z3-solver is required: pip install z3-solver") from exc

from tools.kripke import KripkeStructure, VerificationResult


# ── Diagnostic data structures ────────────────────────────────────────────────

@dataclass
class WorldAnnotation:
    """A single world with its full proposition valuation."""
    world: str
    props_true:  list[str] = field(default_factory=list)
    props_false: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        t = ", ".join(self.props_true)  or "∅"
        f = ", ".join(self.props_false) or "∅"
        return f"{self.world}  [holds: {t}]  [fails: {f}]"


@dataclass
class PropertyDiagnostic:
    """Rich per-property failure report."""
    prop_str:        str                         # e.g. "AG(safe)"
    operator:        str                         # "AG" | "EF" | "AF" | "EG" | "AX" | "EX"
    atom:            str                         # atomic proposition name
    holds:           bool
    annotated_path:  list[WorldAnnotation] = field(default_factory=list)
    violation_world: str | None = None           # world where violation occurs
    # AG: reachable worlds NOT satisfying prop
    missing_worlds:  list[str] = field(default_factory=list)
    # AF: reachable worlds NOT in the AF fixed-point (can avoid prop forever)
    blocking_worlds: list[str] = field(default_factory=list)
    # EF: worlds where prop holds but are unreachable
    unreachable_goal_worlds: list[str] = field(default_factory=list)
    # EG: worlds on a prop-cycle (holds = always-satisfying cycle exists)
    cycle_worlds:    list[str] = field(default_factory=list)

    def to_str(self) -> str:
        lines: list[str] = []
        status = "HOLDS" if self.holds else "FAILS"
        lines.append(f"Property: {self.prop_str}  →  {status}")

        if self.annotated_path:
            lines.append("  Counterexample / witness path:")
            for ann in self.annotated_path:
                marker = " ← VIOLATION" if ann.world == self.violation_world else ""
                lines.append(f"    {ann}{marker}")

        if self.missing_worlds:
            lines.append(f"  Worlds missing '{self.atom}' (AG violation sites): "
                         f"{', '.join(self.missing_worlds)}")

        if self.blocking_worlds:
            lines.append(f"  Worlds that can avoid '{self.atom}' forever (AF blocked): "
                         f"{', '.join(self.blocking_worlds)}")
            lines.append(f"  Fix: add a termination / convergence condition at these steps.")

        if self.unreachable_goal_worlds:
            lines.append(f"  Worlds where '{self.atom}' holds but are UNREACHABLE: "
                         f"{', '.join(self.unreachable_goal_worlds)}")
            lines.append(f"  Fix: add a transition from a reachable state to one of these worlds.")

        if self.cycle_worlds:
            lines.append(f"  Worlds on an infinite '{self.atom}'-cycle: "
                         f"{', '.join(self.cycle_worlds)}")

        return "\n".join(lines)


@dataclass
class Z3DiagnosticReport:
    """Full diagnostic bundle for one verification run."""
    structure_summary: str
    diagnostics: list[PropertyDiagnostic] = field(default_factory=list)

    def to_str(self) -> str:
        parts = ["═" * 60, "Z3 DIAGNOSTIC REPORT", "═" * 60,
                 self.structure_summary, ""]
        for d in self.diagnostics:
            parts.append(d.to_str())
            parts.append("")
        return "\n".join(parts)

    @property
    def failed(self) -> list[PropertyDiagnostic]:
        return [d for d in self.diagnostics if not d.holds]

    @property
    def passed(self) -> list[PropertyDiagnostic]:
        return [d for d in self.diagnostics if d.holds]

_PROP_RE = re.compile(r"^(AG|EF|AX|EX|AF|EG)\((\w+)\)$", re.IGNORECASE)


def _parse_modal(s: str) -> tuple[str, str]:
    m = _PROP_RE.match(s.strip())
    if not m:
        raise ValueError(
            f"Cannot parse modal property {s!r}. "
            "Expected form: AG(prop), EF(prop), AX(prop), EX(prop), AF(prop), EG(prop)."
        )
    return m.group(1).upper(), m.group(2)


class Z3KripkeVerifier:
    """Verify CTL properties over a finite Kripke structure using the Z3 SMT solver.

    Z3 is used to find counterexamples / witnesses via SAT queries over the
    concrete world indices, giving formal SAT/UNSAT certificates even for
    finite structures and providing a foundation for symbolic extension.
    """

    def __init__(self, structure: KripkeStructure) -> None:
        self.s = structure
        self._idx = {w: i for i, w in enumerate(structure.worlds)}
        self._rev = {i: w for w, i in self._idx.items()}
        self._n = len(structure.worlds)
        self._init = self._idx[structure.initial]
        self._R: set[tuple[int, int]] = {
            (self._idx[a], self._idx[b])
            for a, b in structure.relation
            if a in self._idx and b in self._idx
        }
        self._V: dict[str, set[int]] = {
            p: {self._idx[w] for w in ws if w in self._idx}
            for p, ws in structure.valuation.items()
        }
        self._reachable: set[int] = {
            self._idx[w] for w in structure.reachable_from(structure.initial)
        }

    # ------------------------------------------------------------------
    # Z3 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _member_of(x_var, idx_set: set[int]):
        """Z3 formula: x_var ∈ idx_set."""
        return Or([x_var == i for i in idx_set]) if idx_set else BoolVal(False)

    @staticmethod
    def _not_member_of(x_var, idx_set: set[int]):
        """Z3 formula: x_var ∉ idx_set."""
        return Not(Or([x_var == i for i in idx_set])) if idx_set else BoolVal(True)

    def _sat_query(self, *clauses) -> tuple[bool, int | None]:
        """Run Solver.check() with the given clauses. Returns (is_sat, model_x_value)."""
        x = Int("x")
        solver = Solver()
        for c in clauses:
            solver.add(c(x))
        if solver.check() == sat:
            return True, solver.model()[x].as_long()
        return False, None

    def _world_result(self, prop_str: str, holds: bool, idx: int | None) -> VerificationResult:
        witness = self._rev[idx] if idx is not None else None
        path = self.s.path_to(witness) if witness else None
        return VerificationResult(property=prop_str, holds=holds, witness=witness, path=path)

    # ------------------------------------------------------------------
    # CTL operators
    # ------------------------------------------------------------------

    def _check_AG(self, prop: str) -> VerificationResult:
        """AG(prop) — prop holds in every reachable state.
        Refuted if ∃ reachable x. ¬prop(x) is SAT."""
        prop_set = self._V.get(prop, set())
        found, idx = self._sat_query(
            lambda x: self._member_of(x, self._reachable),
            lambda x: self._not_member_of(x, prop_set),
        )
        return self._world_result(f"AG({prop})", holds=not found, idx=idx if found else None)

    def _check_EF(self, prop: str) -> VerificationResult:
        """EF(prop) — some reachable state satisfies prop.
        Witnessed if ∃ reachable x. prop(x) is SAT."""
        prop_set = self._V.get(prop, set())
        found, idx = self._sat_query(
            lambda x: self._member_of(x, self._reachable),
            lambda x: self._member_of(x, prop_set),
        )
        return self._world_result(f"EF({prop})", holds=found, idx=idx if found else None)

    def _check_AX(self, prop: str) -> VerificationResult:
        """AX(prop) — prop holds in all immediate successors of the initial state."""
        succs = [b for a, b in self._R if a == self._init]
        prop_set = self._V.get(prop, set())
        violations = [self._rev[b] for b in succs if b not in prop_set]
        if violations:
            return VerificationResult(property=f"AX({prop})", holds=False, witness=violations[0])
        return VerificationResult(property=f"AX({prop})", holds=True)

    def _check_EX(self, prop: str) -> VerificationResult:
        """EX(prop) — some immediate successor of the initial state satisfies prop."""
        succs = [b for a, b in self._R if a == self._init]
        prop_set = self._V.get(prop, set())
        witnesses = [self._rev[b] for b in succs if b in prop_set]
        if witnesses:
            return VerificationResult(property=f"EX({prop})", holds=True, witness=witnesses[0])
        return VerificationResult(property=f"EX({prop})", holds=False)

    def _compute_AF(self, prop: str) -> set[int]:
        """Backward lfp: AF(prop) = lfp(λS. prop_worlds ∪ {s | ∀ succ t of s: t ∈ S})."""
        af: set[int] = set(self._V.get(prop, set()))
        changed = True
        while changed:
            changed = False
            for s in range(self._n):
                if s in af:
                    continue
                succs = [b for a, b in self._R if a == s]
                # s enters AF iff it has successors and all are already in AF
                if succs and all(b in af for b in succs):
                    af.add(s)
                    changed = True
        return af

    def _check_AF(self, prop: str) -> VerificationResult:
        """AF(prop) — prop is inevitably reached on all paths from the initial state."""
        af_set = self._compute_AF(prop)
        violations = self._reachable - af_set
        if violations:
            cx = self._rev[next(iter(violations))]
            return VerificationResult(
                property=f"AF({prop})", holds=False,
                witness=cx, path=self.s.path_to(cx),
            )
        return VerificationResult(property=f"AF({prop})", holds=True)

    def _compute_EG(self, prop: str) -> set[int]:
        """Backward gfp: EG(prop) = gfp(λS. prop_worlds ∩ {s | ∃ succ t of s: t ∈ S}).
        The fixed point contains all worlds from which an infinite prop-path exists
        (i.e. worlds on a cycle within the prop-worlds subgraph)."""
        eg: set[int] = set(self._V.get(prop, set()))
        changed = True
        while changed:
            changed = False
            # Remove worlds that have no successor remaining inside eg
            to_remove = {
                s for s in eg
                if not any(b in eg for _, b in self._R if _ == s)
            }
            if to_remove:
                eg -= to_remove
                changed = True
        return eg

    def _check_EG(self, prop: str) -> VerificationResult:
        """EG(prop) — exists a path from the initial state where prop holds globally."""
        eg_set = self._compute_EG(prop)
        witnesses = self._reachable & eg_set
        if witnesses:
            wx = self._rev[next(iter(witnesses))]
            return VerificationResult(
                property=f"EG({prop})", holds=True,
                witness=wx, path=self.s.path_to(wx),
            )
        return VerificationResult(property=f"EG({prop})", holds=False)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def _annotate_world(self, world: str) -> WorldAnnotation:
        """Annotate a world with all proposition truth values."""
        idx = self._idx.get(world)
        true_props  = [p for p, ws in self._V.items() if idx in ws]
        false_props = [p for p, ws in self._V.items() if idx not in ws]
        return WorldAnnotation(world=world, props_true=true_props, props_false=false_props)

    def _annotate_path(self, path: list[str] | None) -> list[WorldAnnotation]:
        if not path:
            return []
        return [self._annotate_world(w) for w in path]

    def _structure_summary(self) -> str:
        lines = ["Kripke Structure M = (S, R, V)"]
        lines.append(f"  Worlds (S):  {', '.join(self.s.worlds)}")
        lines.append(f"  Initial:     {self.s.initial}")
        lines.append(f"  Reachable:   {', '.join(self._rev[i] for i in sorted(self._reachable))}")
        lines.append(f"  Transitions (R):")
        for a, b in self.s.relation:
            lines.append(f"    {a} → {b}")
        lines.append(f"  Valuation (V):")
        for prop, worlds_idx in self._V.items():
            worlds_str = ", ".join(self._rev[i] for i in sorted(worlds_idx)) or "∅"
            lines.append(f"    {prop}: {{{worlds_str}}}")
        return "\n".join(lines)

    def _build_diagnostic(
        self,
        result: VerificationResult,
        operator: str,
        atom: str,
    ) -> PropertyDiagnostic:
        """Build a PropertyDiagnostic from a VerificationResult."""
        d = PropertyDiagnostic(
            prop_str=result.property,
            operator=operator,
            atom=atom,
            holds=result.holds,
            violation_world=result.witness if not result.holds else None,
            annotated_path=self._annotate_path(result.path),
            cycle_worlds=(
                [self._rev[i] for i in sorted(self._compute_EG(atom) & self._reachable)]
                if operator == "EG" else []
            ),
        )

        if operator == "AG" and not result.holds:
            prop_set = self._V.get(atom, set())
            d.missing_worlds = [
                self._rev[i] for i in sorted(self._reachable - prop_set)
            ]

        if operator == "AF" and not result.holds:
            af_set = self._compute_AF(atom)
            d.blocking_worlds = [
                self._rev[i] for i in sorted(self._reachable - af_set)
            ]

        if operator == "EF" and not result.holds:
            prop_set = self._V.get(atom, set())
            d.unreachable_goal_worlds = [
                self._rev[i] for i in sorted(prop_set - self._reachable)
            ]

        return d

    def diagnostic_report(self, results: list[VerificationResult]) -> Z3DiagnosticReport:
        """Build a full Z3DiagnosticReport from an already-computed verify() result list."""
        diagnostics: list[PropertyDiagnostic] = []
        for r in results:
            if r.error:
                continue
            try:
                op, atom = _parse_modal(r.property)
                diagnostics.append(self._build_diagnostic(r, op, atom))
            except ValueError:
                pass
        return Z3DiagnosticReport(
            structure_summary=self._structure_summary(),
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, properties: list[str]) -> list[VerificationResult]:
        """Verify a list of CTL modal properties. Returns one result per property."""
        dispatch = {
            "AG": self._check_AG, "EF": self._check_EF,
            "AX": self._check_AX, "EX": self._check_EX,
            "AF": self._check_AF, "EG": self._check_EG,
        }
        results: list[VerificationResult] = []
        for prop_str in properties:
            try:
                op, prop = _parse_modal(prop_str)
                results.append(dispatch[op](prop))
            except (ValueError, KeyError) as exc:
                results.append(VerificationResult(property=prop_str, holds=False, error=str(exc)))
        return results


def verify_kripke_z3(
    structure: KripkeStructure,
    properties: list[str],
    with_diagnostics: bool = False,
) -> list[VerificationResult] | tuple[list[VerificationResult], Z3DiagnosticReport]:
    """Verify CTL properties over a Kripke structure using Z3.

    Args:
        structure:        The Kripke structure M = (S, R, V).
        properties:       CTL formulas, e.g. ['AG(safe)', 'EF(goal)', 'AF(done)'].
        with_diagnostics: If True, also return a Z3DiagnosticReport with annotated
                          counterexample paths, missing/blocking/unreachable worlds,
                          and cycle detection suitable for passing to plan refiners.

    Returns:
        list[VerificationResult] when with_diagnostics=False (default).
        (list[VerificationResult], Z3DiagnosticReport) when with_diagnostics=True.
    """
    verifier = Z3KripkeVerifier(structure)
    results  = verifier.verify(properties)
    if with_diagnostics:
        return results, verifier.diagnostic_report(results)
    return results
