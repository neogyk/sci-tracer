from pydantic import Field
from pydantic import BaseModel, Field
from typing import Any, Callable, List, Optional

# ── Knowledge-Graph store (Neo4j) ─────────────────────────────────────────────
class Relationship(BaseModel):
    source: str = Field(description="The origin entity (subject)")
    target: str = Field(description="The destination entity (object)")
    type: str = Field(description="The type of relationship (verb/predicate, e.g., 'CAUSE', 'IMPLY')")

# ── Knowledge Graph ────────────────────────────────────────────────────────────
class GraphStore:
    """Neo4j-backed store for Problem–Recipe–Result clusters and ontology triples."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def store_prr(self, prr_data) -> None:
        """Store a Problem–Recipe–Result node cluster from a Hoare triple."""
        query = """
        MERGE (p:Problem {name:$problem})
        MERGE (r:Recipe  {name:$recipe})
        MERGE (res:Result {name:$result})
        MERGE (p)-[:SOLVED_BY]->(r)
        MERGE (r)-[:PRODUCES]->(res)
        """
        with self.driver.session() as session:
            session.run(
                query,
                problem=prr_data.problem,
                recipe=prr_data.recipe,
                result=prr_data.result,
            )

    def store_triples(self, triples: list) -> None:
        """Store a list of {'subject', 'predicate', 'object'} ontology triples."""
        with self.driver.session() as session:
            for t in triples:
                predicate = t["predicate"].upper().replace(" ", "_")
                query = f"""
                MERGE (s:Concept {{name:$subject}})
                MERGE (o:Concept {{name:$object}})
                MERGE (s)-[:{predicate}]->(o)
                """
                session.run(query, subject=t["subject"], object=t["object"])

    def _concept_neighbors(self, concept: str) -> list[dict]:
        """Return all direct (1-hop) ontology triples incident to *concept*."""
        cypher = """
        MATCH (s:Concept)-[r]->(o:Concept)
        WHERE s.name = $name OR o.name = $name
        RETURN s.name AS subject, type(r) AS predicate, o.name AS object
        """
        try:
            with self.driver.session() as session:
                records = session.run(cypher, name=concept)
                return [
                    {
                        "subject":   rec["subject"],
                        "predicate": rec["predicate"],
                        "object":    rec["object"],
                    }
                    for rec in records
                ]
        except Exception:
            return []

    def find_neighbors_mcts(
        self,
        concepts:     list[str],
        n_iterations: int   = 200,
        max_depth:    int   = 4,
        exploration_c: float = 1.414,   # √2 — standard UCB1 constant
        sim_depth:    int   = 2,
        score_fn:     Optional[Callable[[str], float]] = None,
    ) -> list[dict]:
        """Explore the ontology graph with Monte Carlo Tree Search.

        Algorithm (per seed concept)
        ----------------------------
        For each concept in *concepts* a separate MCTS tree is grown:

          Selection    — walk from the root using UCB1 until an unexpanded
                         node or a depth limit is reached.
          Expansion    — pick one unvisited 1-hop neighbor, add it as a child.
          Simulation   — random walk of *sim_depth* steps; reward =
                         score_fn(terminal concept).
          Backprop     — propagate reward and visit counts to all ancestors.

        score_fn(concept: str) → float ∈ [0, 1]
          Reward signal for the MCTS rollout.  When None, uniform reward 1.0
          is used (pure exploration).  Typically set to cosine similarity
          between the concept name embedding and the query embedding so that
          MCTS steers toward query-relevant parts of the ontology.

        Returns
        -------
        All triples discovered across all trees, deduplicated and sorted by
        the visit count of their most-visited endpoint (highest first).
        """
        if not concepts:
            return []

        _score = score_fn if score_fn is not None else (lambda _: 1.0)

        # Memoize 1-hop DB calls — each concept is queried at most once
        _nb_cache: dict[str, list[dict]] = {}

        def get_triples(c: str) -> list[dict]:
            if c not in _nb_cache:
                _nb_cache[c] = self._concept_neighbors(c)
            return _nb_cache[c]

        def get_neighbor_names(c: str) -> list[str]:
            seen: set[str] = set()
            names: list[str] = []
            for t in get_triples(c):
                other = t["object"] if t["subject"] == c else t["subject"]
                if other not in seen:
                    seen.add(other)
                    names.append(other)
            return names

        # Global triple table: (s, p, o) → best visit count seen for either endpoint
        collected: dict[tuple, dict] = {}

        def _harvest(node: _MCTSNode) -> None:
            """Collect all triples incident to visited nodes, keyed by visit weight."""
            for t in get_triples(node.concept):
                key = (t["subject"], t["predicate"], t["object"])
                if key not in collected or collected[key]["_w"] < node.visits:
                    collected[key] = {**t, "_w": node.visits}
            for ch in node.children:
                _harvest(ch)

        for seed in concepts:
            root = _MCTSNode(concept=seed)

            for _ in range(n_iterations):
                # ── Selection ─────────────────────────────────────────────────
                node  = root
                depth = 0
                path: list[_MCTSNode] = [node]

                while node.children and depth < max_depth:
                    child_names = {ch.concept for ch in node.children}
                    unexplored  = [n for n in get_neighbor_names(node.concept)
                                   if n not in child_names]
                    if unexplored:
                        break   # not fully expanded — expand here
                    node = node.best_child(exploration_c)
                    path.append(node)
                    depth += 1

                # ── Expansion ─────────────────────────────────────────────────
                if depth < max_depth:
                    child_names = {ch.concept for ch in node.children}
                    unexplored  = [n for n in get_neighbor_names(node.concept)
                                   if n not in child_names]
                    if unexplored:
                        chosen = random.choice(unexplored)
                        child  = _MCTSNode(concept=chosen, parent=node)
                        node.children.append(child)
                        node = child
                        path.append(node)

                # ── Simulation (random rollout) ────────────────────────────────
                sim_concept = node.concept
                for _ in range(sim_depth):
                    nbrs = get_neighbor_names(sim_concept)
                    if not nbrs:
                        break
                    sim_concept = random.choice(nbrs)
                reward = _score(sim_concept)

                # ── Backpropagation ────────────────────────────────────────────
                for n in path:
                    n.visits       += 1
                    n.total_reward += reward

            _harvest(root)

        # Sort by visit weight (descending) and strip internal key
        ranked = sorted(collected.values(), key=lambda t: t["_w"], reverse=True)
        return [
            {"subject": t["subject"], "predicate": t["predicate"], "object": t["object"]}
            for t in ranked
        ]

    def find_prr(self, keyword: str, limit: int = 5) -> list[dict]:
        """Return PRR clusters whose Problem or Recipe text contains *keyword*."""
        cypher = """
        MATCH (p:Problem)-[:SOLVED_BY]->(r:Recipe)-[:PRODUCES]->(res:Result)
        WHERE toLower(p.name) CONTAINS toLower($kw)
           OR toLower(r.name) CONTAINS toLower($kw)
        RETURN p.name AS problem, r.name AS recipe, res.name AS result
        LIMIT $limit
        """
        with self.driver.session() as session:
            records = session.run(cypher, kw=keyword, limit=limit)
            return [
                {"problem": rec["problem"], "recipe": rec["recipe"], "result": rec["result"]}
                for rec in records
            ]

class KnowledgeGraph:
    """Incrementally built graph of (source, type, target) triples from trajectory sections.

    Edges accumulate across calls to add(); the graph is queried by entity name
    and serialised to compact text for LLM context injection.
    """

    def __init__(self) -> None:
        self._edges: list[Relationship] = []

    def add(self, relationships: list[Relationship]) -> None:
        self._edges.extend(relationships)

    def reset(self) -> None:
        self._edges.clear()

    @property
    def nodes(self) -> set[str]:
        s: set[str] = set()
        for r in self._edges:
            s.add(r.source)
            s.add(r.target)
        return s

    @property
    def edges(self) -> list[Relationship]:
        return list(self._edges)

    def query(self, entity: str, n_results: int = 60, vector_name: str = "problem") -> list[Relationship]:
        """All edges that involve *entity* as source or target.

        n_results and vector_name are accepted for API compatibility with
        TrajectoryStore.query() — n_results caps the returned list size.
        """
        e = entity.lower()
        matches = [r for r in self._edges
                   if r.source.lower() == e or r.target.lower() == e]
        return matches[:n_results]

    def to_text(self, max_edges: int = 60) -> str:
        """Compact text form suitable for LLM context."""
        lines = [f"[{r.source}] --{r.type}--> [{r.target}]"
                 for r in self._edges[:max_edges]]
        if len(self._edges) > max_edges:
            lines.append(f"... ({len(self._edges) - max_edges} more edges not shown)")
        return "\n".join(lines)

    def serialize(self) -> list[dict]:
        """Return all edges as a list of plain dicts for pickling."""
        return [{"source": r.source, "type": r.type, "target": r.target}
                for r in self._edges]

    def restore(self, data: list[dict]) -> None:
        """Replace current edges with previously serialized data (no LLM call needed)."""
        self._edges = [
            Relationship(source=d["source"], type=d["type"], target=d["target"])
            for d in data
            if isinstance(d, dict) and "source" in d and "type" in d and "target" in d
        ]

    def __len__(self) -> int:
        return len(self._edges)

    def __repr__(self) -> str:
        return f"KnowledgeGraph(nodes={len(self.nodes)}, edges={len(self._edges)})"
