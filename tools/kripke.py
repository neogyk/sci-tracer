"""Shared data structures for Kripke structure verification."""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field


@dataclass
class KripkeStructure:
    """Finite Kripke structure M = (S, R, V).

    S  — finite set of worlds/states
    R  — accessibility relation (set of directed edges)
    V  — valuation: proposition -> set of worlds where it holds
    """

    worlds: list[str]
    initial: str
    relation: list[tuple[str, str]]
    propositions: list[str]
    valuation: dict[str, set[str]]

    def successors(self, world: str) -> list[str]:
        return [b for a, b in self.relation if a == world]

    def reachable_from(self, start: str) -> set[str]:
        visited: set[str] = set()
        queue: deque[str] = deque([start])
        while queue:
            w = queue.popleft()
            if w not in visited:
                visited.add(w)
                queue.extend(s for s in self.successors(w) if s not in visited)
        return visited

    def path_to(self, target: str, start: str | None = None) -> list[str] | None:
        """BFS shortest path from start (default: initial world) to target."""
        origin = start or self.initial
        queue: deque[list[str]] = deque([[origin]])
        visited = {origin}
        while queue:
            path = queue.popleft()
            if path[-1] == target:
                return path
            for nxt in self.successors(path[-1]):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return None

    def satisfies(self, world: str, prop: str) -> bool:
        return world in self.valuation.get(prop, set())


@dataclass
class VerificationResult:
    property: str
    holds: bool
    witness: str | None = None    # world name serving as witness or counterexample
    path: list[str] | None = None  # path from initial to witness
    error: str | None = None

    def __str__(self) -> str:
        status = "HOLDS" if self.holds else "FAILS"
        parts = [f"{self.property}: {status}"]
        if self.witness:
            parts.append(f"witness={self.witness}")
        if self.path:
            parts.append(f"path={' -> '.join(self.path)}")
        if self.error:
            parts.append(f"error={self.error}")
        return ", ".join(parts)


def parse_kripke_from_llm(
    worlds_str: str,
    initial: str,
    relation_str: str,
    propositions_str: str,
    valuation_str: str,
) -> KripkeStructure:
    """Parse string outputs from KripkeFormalizer into a KripkeStructure.

    Handles common LLM output formats, e.g.:
      relation: {(s0,s1),(s1,s2)} or s0->s1, s1->s2
      valuation: {safe: {s0, s1}, goal: {s2}} or safe: s0, s1 | goal: s2
    """
    worlds = [w.strip() for w in worlds_str.split(",") if w.strip()]
    propositions = [p.strip() for p in propositions_str.split(",") if p.strip()]
    world_set = set(worlds)

    relation: list[tuple[str, str]] = []
    for m in re.finditer(r"\(?\s*(\w+)\s*[,\->]+\s*(\w+)\s*\)?", relation_str):
        a, b = m.group(1), m.group(2)
        if a in world_set and b in world_set:
            relation.append((a, b))

    valuation: dict[str, set[str]] = {}
    # Try {prop: {w1, w2}} format first
    for m in re.finditer(r"(\w+)\s*:\s*\{([^}]*)\}", valuation_str):
        prop = m.group(1).strip()
        ws = {w.strip() for w in m.group(2).split(",") if w.strip()} & world_set
        valuation[prop] = ws
    # Fallback: prop: w1, w2 | prop2: w3
    if not valuation:
        for seg in re.split(r"[|;]", valuation_str):
            if ":" in seg:
                prop, rest = seg.split(":", 1)
                ws = {w for w in re.findall(r"\b(\w+)\b", rest) if w in world_set}
                if prop.strip() and ws:
                    valuation[prop.strip()] = ws

    return KripkeStructure(
        worlds=worlds,
        initial=initial.strip(),
        relation=relation,
        propositions=propositions,
        valuation=valuation,
    )
