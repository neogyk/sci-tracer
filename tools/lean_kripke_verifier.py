"""Lean 4 verifier for Kripke structures.

Generates a self-contained .lean file encoding M = (S, R, V), defines CTL
operators as decidable Lean 4 functions, evaluates each property via #eval,
then runs the Lean compiler and parses the output.

Supported operators: AG, EF, AX, EX, AF, EG.

Lean 4 install:
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from tools.kripke import KripkeStructure, VerificationResult

_PROP_RE = re.compile(r"^(AG|EF|AX|EX|AF|EG)\((\w+)\)$", re.IGNORECASE)
_LEAN_FALLBACK = Path.home() / ".elan" / "bin" / "lean"


def _find_lean() -> str:
    lean = shutil.which("lean") or (str(_LEAN_FALLBACK) if _LEAN_FALLBACK.exists() else None)
    if lean is None:
        raise RuntimeError(
            "Lean 4 not found. Install via:\n"
            "  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh"
        )
    return lean


def _parse_modal(s: str) -> tuple[str, str]:
    m = _PROP_RE.match(s.strip())
    if not m:
        raise ValueError(
            f"Cannot parse modal property {s!r}. "
            "Expected form: AG(prop), EF(prop), AX(prop), EX(prop), AF(prop), EG(prop)."
        )
    return m.group(1).upper(), m.group(2)


def _lean_id(name: str) -> str:
    """Convert an arbitrary string to a valid Lean 4 identifier."""
    ident = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    return ("w" + ident) if ident and ident[0].isdigit() else ident


class LeanKripkeVerifier:
    """Generate and execute Lean 4 code to verify CTL properties over a Kripke structure."""

    def __init__(self, structure: KripkeStructure, lean_path: str | None = None) -> None:
        self.s = structure
        self._lean = lean_path or _find_lean()

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def generate_lean_code(self, properties: list[str]) -> str:
        """Return a Lean 4 source string that verifies all given CTL properties."""
        s = self.s
        id_map = {w: _lean_id(w) for w in s.worlds}
        init_id = id_map[s.initial]

        L: list[str] = []

        def ln(line: str = "") -> None:
            L.append(line)

        # ── World inductive type ────────────────────────────────────────
        ln("-- Auto-generated Kripke structure verification")
        ln("-- M = (S, R, V) encoded in Lean 4")
        ln()
        ln("inductive World where")
        for w in s.worlds:
            ln(f"  | {id_map[w]} : World")
        ln("  deriving DecidableEq, Repr, Inhabited")
        ln()

        # ── All worlds ──────────────────────────────────────────────────
        worlds_list = ", ".join(f".{id_map[w]}" for w in s.worlds)
        ln(f"def allWorlds : List World := [{worlds_list}]")
        ln()

        # ── Transition relation R : World → World → Bool ────────────────
        ln("def R : World → World → Bool")
        for a, b in s.relation:
            if a in id_map and b in id_map:
                ln(f"  | .{id_map[a]}, .{id_map[b]} => true")
        ln("  | _, _ => false")
        ln()

        # ── Valuation V : String → World → Bool ────────────────────────
        ln("def V (prop : String) (w : World) : Bool :=")
        if s.valuation:
            first_branch = True
            for prop, worlds_set in s.valuation.items():
                valid = [w for w in worlds_set if w in id_map]
                if not valid:
                    continue
                kw = "if" if first_branch else "else if"
                first_branch = False
                ln(f'  {kw} prop == "{prop}" then')
                ln("    match w with")
                for w in valid:
                    ln(f"    | .{id_map[w]} => true")
                ln("    | _ => false")
            ln("  else false")
        else:
            ln("  false")
        ln()

        # ── Reachability ────────────────────────────────────────────────
        # Expand the reachable set by one BFS layer.
        ln("private def expandReach (reached : List World) : List World :=")
        ln("  let new := allWorlds.filter (fun w =>")
        ln("    !reached.contains w && reached.any (fun u => R u w))")
        ln("  reached ++ new")
        ln()
        # Apply expandReach n times (n = |worlds|) — fixpoint reached in ≤ n steps.
        ln("def reachSet (init : World) : List World :=")
        ln("  allWorlds.foldl (fun acc _ => expandReach acc) [init]")
        ln()

        # ── CTL operators ───────────────────────────────────────────────
        ln("-- CTL modal operators")
        ln()

        # AG: prop holds in every reachable state
        ln("def checkAG (prop : String) (init : World) : Bool :=")
        ln("  (reachSet init).all (V prop)")
        ln()

        # EF: some reachable state satisfies prop
        ln("def checkEF (prop : String) (init : World) : Bool :=")
        ln("  (reachSet init).any (V prop)")
        ln()

        # AX: prop holds in all immediate successors of init
        ln("def checkAX (prop : String) (init : World) : Bool :=")
        ln("  allWorlds.all (fun w => !R init w || V prop w)")
        ln()

        # EX: some immediate successor of init satisfies prop
        ln("def checkEX (prop : String) (init : World) : Bool :=")
        ln("  allWorlds.any (fun w => R init w && V prop w)")
        ln()

        # AF: backward lfp — prop inevitably reached on all paths
        # afStep adds worlds whose every successor is already in the AF set.
        ln("private def afStep (af : List World) : List World :=")
        ln("  let new := allWorlds.filter (fun w =>")
        ln("    !af.contains w &&")
        ln("    allWorlds.any (fun s => R w s) &&")
        ln("    allWorlds.all (fun s => !R w s || af.contains s))")
        ln("  af ++ new")
        ln()
        ln("def checkAF (prop : String) (init : World) : Bool :=")
        ln("  -- Start from prop-worlds; expand backward until fixpoint (≤ n steps).")
        ln("  let base := allWorlds.filter (V prop)")
        ln("  let afSet := allWorlds.foldl (fun acc _ => afStep acc) base")
        ln("  (reachSet init).all afSet.contains")
        ln()

        # EG: backward gfp — exists an infinite path where prop holds globally.
        # egStep removes worlds that have no successor remaining in the EG set.
        ln("private def egStep (eg : List World) : List World :=")
        ln("  eg.filter (fun w => allWorlds.any (fun s => R w s && eg.contains s))")
        ln()
        ln("def checkEG (prop : String) (init : World) : Bool :=")
        ln("  -- Prune worlds that cannot sustain an infinite prop-path (≤ n steps).")
        ln("  let base := allWorlds.filter (V prop)")
        ln("  let egSet := allWorlds.foldl (fun acc _ => egStep acc) base")
        ln("  (reachSet init).any egSet.contains")
        ln()

        # ── Verification block ──────────────────────────────────────────
        ln("-- Verification results (read from stdout after running lean)")
        fn_map = {
            "AG": "checkAG", "EF": "checkEF",
            "AX": "checkAX", "EX": "checkEX",
            "AF": "checkAF", "EG": "checkEG",
        }
        ln("#eval do")
        for prop_str in properties:
            try:
                op, prop = _parse_modal(prop_str)
                fn = fn_map[op]
                var = f"r_{op.lower()}_{_lean_id(prop)}"
                ln(f'  let {var} := {fn} "{prop}" .{init_id}')
                ln(f'  IO.println s!"{prop_str}: {{{var}}}"')
            except (ValueError, KeyError):
                ln(f'  IO.println "PARSE_ERROR: {prop_str}"')

        return "\n".join(L) + "\n"

    # ------------------------------------------------------------------
    # Execution & parsing
    # ------------------------------------------------------------------

    def verify(self, properties: list[str]) -> list[VerificationResult]:
        """Generate Lean code, compile and run it, return one result per property."""
        code = self.generate_lean_code(properties)

        with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as f:
            f.write(code)
            lean_file = Path(f.name)

        try:
            proc = subprocess.run(
                [self._lean, str(lean_file)],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            lean_file.unlink(missing_ok=True)
            return [
                VerificationResult(property=p, holds=False, error="Lean timed out (120 s)")
                for p in properties
            ]

        lean_file.unlink(missing_ok=True)

        if proc.returncode != 0 and not proc.stdout.strip():
            err_summary = proc.stderr[:500].strip()
            return [
                VerificationResult(property=p, holds=False, error=err_summary)
                for p in properties
            ]

        return self._parse_output(proc.stdout, properties)

    @staticmethod
    def _parse_output(output: str, properties: list[str]) -> list[VerificationResult]:
        # Each line looks like:  AG(safe): true
        results_map: dict[str, bool] = {}
        for line in output.splitlines():
            m = re.match(r"^([A-Z]+\(\w+\)):\s*(true|false)", line.strip())
            if m:
                results_map[m.group(1)] = m.group(2) == "true"

        results: list[VerificationResult] = []
        for prop_str in properties:
            try:
                op, prop = _parse_modal(prop_str)
                key = f"{op}({prop})"
            except ValueError:
                key = prop_str.strip()

            if key in results_map:
                results.append(VerificationResult(property=key, holds=results_map[key]))
            else:
                results.append(VerificationResult(
                    property=key, holds=False,
                    error=f"No result for {key!r} in Lean output",
                ))
        return results


def verify_kripke_lean(
    structure: KripkeStructure,
    properties: list[str],
    lean_path: str | None = None,
) -> list[VerificationResult]:
    """Verify CTL properties over a Kripke structure using Lean 4.

    Generates a standalone .lean file, compiles it with the local lean binary,
    and parses the #eval output to extract true/false results.

    Args:
        structure: The Kripke structure M = (S, R, V).
        properties: CTL formulas, e.g. ['AG(safe)', 'EF(goal)', 'AF(done)'].
        lean_path: Optional path to the lean binary (auto-detected if None).

    Returns:
        One VerificationResult per property.

    Example:
        from tools.kripke import KripkeStructure
        from tools.lean_kripke_verifier import verify_kripke_lean

        m = KripkeStructure(
            worlds=['s0', 's1', 's2'],
            initial='s0',
            relation=[('s0', 's1'), ('s1', 's2'), ('s2', 's0')],
            propositions=['safe', 'goal'],
            valuation={'safe': {'s0', 's1'}, 'goal': {'s2'}},
        )
        for r in verify_kripke_lean(m, ['AG(safe)', 'EF(goal)']):
            print(r)
    """
    return LeanKripkeVerifier(structure, lean_path).verify(properties)
