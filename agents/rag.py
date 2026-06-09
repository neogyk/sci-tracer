import itertools
import json
import math
import random
import uuid
import dspy
import threading
import tqdm
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional
from pydantic import BaseModel, Field

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, NamedVector
from tools.pub_search import arxiv_search, scolars_search, file_reader
from tools.pub_search import duckduckgo_pdf_search_tool
from storage.vector_db import VectorStore, TrajectoryStore
from storage.knowledge_graph import GraphStore

# ── Shared embedding model (singleton) ────────────────────────────────────────
# One model for the entire process — avoids loading bge-small + bge-large in parallel.
# bge-small-en-v1.5 is fast and good enough for retrieval; upgrade here if needed.

_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_EMBED_DIM   = 384

_embed_lock:  Optional[threading.Lock] = None
_embed_model: Optional[Any] = None


def _get_embed_model() -> Any:
    """Lazy-load the shared FastEmbed model (thread-safe singleton)."""
    global _embed_model, _embed_lock
    if _embed_lock is None:
        _embed_lock = threading.Lock()
    with _embed_lock:
        if _embed_model is None:
            from fastembed import TextEmbedding
            _embed_model = TextEmbedding(model_name=_EMBED_MODEL)
    return _embed_model


def shared_embed(text: str) -> list[float]:
    """Embed a single string using the shared model. Returns a plain Python list."""
    model = _get_embed_model()
    return list(model.embed([text]))[0].tolist()

class Trajectory(dspy.Signature):
    question:str = dspy.InputField(
        desc="The problem or question that is necessary to solve"
    )
    paragraph = dspy.InputField(desc="Part of the research article")
    problem: str = dspy.OutputField(
        desc="The formulation of the problem being solved in the paragraph or article"
    )
    method: str = dspy.OutputField(
        desc="The methodology, algorithm, or list of instructions the author used to solve the problem"
    )
    result: str = dspy.OutputField(desc="The result obtained by solving the problem")


# ── Hybrid retriever ───────────────────────────────────────────────────────────

@dataclass
class RetrievedContext:
    """Fused retrieval result from VectorStore and GraphStore.

    Fields
    ------
    vector_hits   — ordered list of Hoare-step payloads from Qdrant (highest score first).
    graph_triples — deduplicated ontology edges from the Neo4j concept neighbourhood.
    graph_prr     — Problem–Recipe–Result clusters from Neo4j matching the query keyword.
    """
    vector_hits:   list
    graph_triples: list
    graph_prr:     list

    def to_text(self, max_vector: int = 5, max_graph_triples: int = 40) -> str:
        """Render a single LLM-injectable context block from both retrieval legs."""
        parts: list[str] = []

        if self.vector_hits:
            parts.append("── SIMILAR HOARE STEPS (semantic search) ──")
            for i, hit in enumerate(self.vector_hits[:max_vector]):
                header = f"[{i + 1}] Step {hit.get('step', '?')}: {hit.get('task', '')}"
                parts.append(f"{header}\n{hit.get('text', '')}")

        if self.graph_triples:
            parts.append("── CONCEPT NEIGHBOURHOOD (graph expansion) ──")
            lines = [
                f"[{t['subject']}] --{t['predicate']}--> [{t['object']}]"
                for t in self.graph_triples[:max_graph_triples]
            ]
            if len(self.graph_triples) > max_graph_triples:
                lines.append(f"… ({len(self.graph_triples) - max_graph_triples} more edges not shown)")
            parts.append("\n".join(lines))

        if self.graph_prr:
            parts.append("── RELATED PROBLEM–RECIPE–RESULT CLUSTERS ──")
            for prr in self.graph_prr:
                parts.append(
                    f"Problem : {prr['problem']}\n"
                    f"Recipe  : {prr['recipe']}\n"
                    f"Result  : {prr['result']}"
                )

        return "\n\n".join(parts) if parts else ""


class HybridRetriever:
    """Retrieves and fuses context from VectorStore (Qdrant) and GraphStore (Neo4j).

    Retrieval strategy
    ------------------
    1. Vector leg          — query VectorStore for the top-k Hoare-step chunks nearest
                             to the query string using dense embeddings.
    2. Concept extraction  — collect entity names (subject / object) from the ontology
                             triples stored in each vector hit's payload.
    3. Graph leg (MCTS)    — explore the Neo4j concept graph via Monte Carlo Tree Search
                             seeded by the extracted concepts.  The UCB1 reward signal is
                             the cosine similarity between each concept's name embedding
                             and the query embedding, steering search toward query-relevant
                             sub-graphs rather than expanding uniformly.
    4. PRR clusters        — fetch Problem–Recipe–Result clusters from Neo4j that contain
                             the query keyword.
    5. Fusion              — deduplicated, visit-ranked graph triples are appended after
                             the ranked vector hits; serialised via RetrievedContext.to_text().

    Either store may be None; the retriever degrades gracefully to whichever is present.
    """

    def __init__(
        self,
        vector_store:     Optional[VectorStore] = None,
        graph_store:      Optional[GraphStore]  = None,
        k_vector:         int   = 5,
        mcts_iterations:  int   = 200,
        mcts_max_depth:   int   = 4,
        mcts_exploration: float = 1.414,   # √2
        mcts_sim_depth:   int   = 2,
    ) -> None:
        self.vectors          = vector_store
        self.graph            = graph_store
        self.k_vector         = k_vector
        self.mcts_iterations  = mcts_iterations
        self.mcts_max_depth   = mcts_max_depth
        self.mcts_exploration = mcts_exploration
        self.mcts_sim_depth   = mcts_sim_depth

    def retrieve(self, query: str) -> RetrievedContext:
        """Run both retrieval legs and return the fused RetrievedContext."""
        vector_hits   = self._vector_search(query)
        concepts      = self._extract_concepts(vector_hits)
        graph_triples = self._graph_neighbors_mcts(concepts, query)
        graph_prr     = self._graph_prr(query)
        return RetrievedContext(
            vector_hits=vector_hits,
            graph_triples=graph_triples,
            graph_prr=graph_prr,
        )

    # ── private helpers ────────────────────────────────────────────────────────

    def _vector_search(self, query: str) -> list:
        if self.vectors is None:
            return []
        results = self.vectors.search(query, k=self.k_vector)
        return [r.payload for r in results]

    def _extract_concepts(self, hits: list) -> list[str]:
        """Collect unique concept names from the triples embedded in vector-hit payloads."""
        seen:     set[str]  = set()
        concepts: list[str] = []
        for hit in hits:
            for triple in hit.get("triples", []):
                for key in ("subject", "object"):
                    name = triple.get(key, "").strip()
                    if name and name not in seen:
                        seen.add(name)
                        concepts.append(name)
        return concepts

    def _graph_neighbors_mcts(self, concepts: list[str], query: str = "") -> list:
        """Run MCTS on the ontology graph and return visit-ranked triples."""
        if self.graph is None or not concepts:
            return []
        score_fn = self._make_score_fn(query) if query else None
        return self.graph.find_neighbors_mcts(
            concepts,
            n_iterations  = self.mcts_iterations,
            max_depth     = self.mcts_max_depth,
            exploration_c = self.mcts_exploration,
            sim_depth     = self.mcts_sim_depth,
            score_fn      = score_fn,
        )

    def _make_score_fn(self, query: str) -> Callable[[str], float]:
        """Return a memoized cosine-similarity scorer for MCTS rollouts.

        Uses shared_embed so no separate model is loaded.
        Results are cached so each concept name is encoded at most once per
        retrieve() call, keeping repeated MCTS visits cheap.
        """
        cache: dict[str, float] = {}
        q_vec  = np.array(shared_embed(query), dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec)) + 1e-8

        def dense_scorer(concept: str) -> float:
            if concept in cache:
                return cache[concept]
            c_vec  = np.array(shared_embed(concept), dtype=np.float32)
            c_norm = float(np.linalg.norm(c_vec)) + 1e-8
            score  = max(0.0, float(np.dot(q_vec, c_vec)) / (q_norm * c_norm))
            cache[concept] = score
            return score

        return dense_scorer

    def _graph_prr(self, keyword: str) -> list:
        if self.graph is None:
            return []
        return self.graph.find_prr(keyword)


class RAG(dspy.Module):
    """
    The RAG Module provides the discovery trajectories for given idea, question or hypothesis.
    Please chunk the text of the document and decompose it into the sequence of the
    problem - method of the problem - solution and cosequences. This discovery trajectory for each of the paper
    will be used for the research planning.
    """

    def __init__(self, vector_db, store: Optional[TrajectoryStore] = None):
        #self.trajectories = dspy.Predict(Trajectory)
        #self.researcher = dspy.ChainOfThought("Trajectory->Plan")
        self.vector_db = vector_db
        self.store = store

    def make_trajectory(self, paper):
        trajectory = self.trajectory.forward(section)
        return trajectory

    def build_rag(self, prompt) -> list[tuple[str, str, str]]:
        """Search all configured sources and return (url, title, abstract) tuples."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        papers: list[tuple[str, str, str]] = []
        with ThreadPoolExecutor(max_workers=len(self.search_tools)) as executor:
            futures = {
                executor.submit(fn, prompt=prompt, n=2): fn.__name__
                for fn in self.search_tools
            }
            for future in as_completed(futures):
                source = futures[future]
                try:
                    results = future.result()
                    for r in results:
                        url = r[0] if len(r) > 0 else ""
                        title = r[1] if len(r) > 1 else ""
                        abstract = r[2] if len(r) > 2 else ""
                        if url:
                            papers.append((url, title, abstract))
                    print(f"  [{source}] {len(results)} results")
                except Exception as exc:
                    print(f"  [{source}] search failed: {exc}")
        return papers

    def sample_trajectory_points(self, starting_point: str, n_results: int = 5) -> list:
        """Sample trajectory points from Qdrant; returns problem+method only (not result)."""
        if self.store is None:
            return []
        results = self.store.query(starting_point, n_results=n_results, vector_name="problem")
        return [
            {"problem": r["problem"], "method": r["method"], "next_id": None}
            for r in results
        ]

    def retrieve_meta_point(self, prompt:str):
        #The meta point is Hoare triplets;
        #Add the embedding and retrieve it by embedding;
        self.vector_db.retrieve(prompt)
        return retrieved_points

    def retrieve_meta_path(self, prompt: str):
        return
    
    def retrive_point(self):
        points = self.vector_db.retrieve(prompt)
        return points

    def retrieve_code(self):
        code = self.vector_db.retrieve(prompt)
        return code