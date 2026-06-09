import uuid

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, NamedVector, Filter, FieldCondition, MatchValue
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional
from pydantic import BaseModel, Field
import threading
import tqdm as _tqdm

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


def shared_embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple strings in one model call. Much faster than N x shared_embed()."""
    if not texts:
        return []
    model = _get_embed_model()
    return [v.tolist() for v in model.embed(texts)]


class VectorStore:
    """In-memory Qdrant store for searchable text chunks (e.g. Hoare triple texts).

    Uses the shared_embed singleton — no extra model loaded here.
    """

    _EMBED_DIM = _EMBED_DIM   # same dimension as TrajectoryStore (shared model)

    def __init__(self) -> None:
        self.client = QdrantClient(":memory:")
        self.collection = "papers"
        # Use _ensure_collection (not recreate) so that restore_points() works
        # even when __init__ is called after points have been pre-loaded.
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self._EMBED_DIM, distance=Distance.COSINE),
            )

    def add_chunk(self, chunk: dict) -> None:
        embedding = shared_embed(chunk["text"])
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=chunk,
                )
            ],
        )

    def search(self, query: str, k: int = 5) -> list:
        q = shared_embed(query)
        return self.client.query_points(
            collection_name=self.collection,
            query=q,
            limit=k,
        ).points

    def count(self) -> int:
        """Number of indexed chunks."""
        return self.client.count(self.collection).count

    def dump_points(self) -> list[dict]:
        """Serialize all stored points as {id, vector, payload} dicts for pickling.

        Scrolls the entire collection — suitable for session persistence.
        """
        points: list[dict] = []
        offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            for p in batch:
                points.append({
                    "id":      str(p.id),
                    "vector":  p.vector if isinstance(p.vector, list) else list(p.vector),
                    "payload": p.payload or {},
                })
            if next_offset is None:
                break
            offset = next_offset
        return points

    def restore_points(self, points: list[dict]) -> None:
        """Re-insert pre-computed points without re-embedding.

        Called on session load to skip the expensive model.encode() calls.
        Only inserts points whose id is not already present.
        """
        if not points:
            return
        # Collect ids already in the store
        existing_ids: set[str] = set()
        if self.count() > 0:
            offset = None
            while True:
                batch, next_offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=256,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                for p in batch:
                    existing_ids.add(str(p.id))
                if next_offset is None:
                    break
                offset = next_offset

        to_insert = [
            PointStruct(id=p["id"], vector=p["vector"], payload=p.get("payload", {}))
            for p in points
            if p.get("id") and p.get("vector") and str(p["id"]) not in existing_ids
        ]
        if to_insert:
            self.client.upsert(collection_name=self.collection, points=to_insert)

class TrajectoryStore:
    """Persistent Qdrant store for (problem, method, result) trajectory triplets."""

    def __init__(self, path: str = "./qdrant_database", collection: str = "trajectories") -> None:
        self._client     = QdrantClient(path=path)
        self._collection = collection
        self._init_collection()

    def _init_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    "problem": VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
                    "method":  VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
                    "result":  VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
                },
            )

    def embed(self, text: str) -> list:
        return shared_embed(text)

    def add(self, trajectories: list) -> None:
        """Embed and store a list of trajectory predictions (.problem, .method, .result).

        All texts are embedded in a single batch call (3N -> 1 call).
        """
        if not trajectories:
            return

        # Collect all texts in one flat list: [p0, m0, r0, p1, m1, r1, ...]
        texts = []
        for entry in trajectories:
            texts.extend([entry.problem, entry.method, entry.result])

        vectors = shared_embed_batch(texts)

        points = []
        for i, entry in _tqdm.tqdm(
            enumerate(trajectories),
            total=len(trajectories),
            desc="indexing trajectories",
            unit="chunk",
        ):
            p_vec, m_vec, r_vec = vectors[i * 3], vectors[i * 3 + 1], vectors[i * 3 + 2]
            payload: dict = {"problem": entry.problem, "method": entry.method, "result": entry.result}
            if getattr(entry, "source", None):
                payload["source"] = entry.source
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={"problem": p_vec, "method": m_vec, "result": r_vec},
                payload=payload,
            ))

        self._client.upsert(collection_name=self._collection, points=points)

    def query(self, text: str, n_results: int = 5, vector_name: str = "problem") -> list:
        """Return top-n payloads most similar to *text* on the named vector."""
        count = self._client.count(self._collection).count
        if count == 0:
            return []
        emb = self.embed(text)
        results = self._client.query_points(
            collection_name=self._collection,
            query=emb,
            using=vector_name,
            limit=min(n_results, count),
        ).points
        return [r.payload for r in results]

    def query_hoare(self, embedding: list, n_results: int = 3) -> list:
        """Return formatted problem+method strings nearest to *embedding*."""
        count = self._client.count(self._collection).count
        if count == 0:
            return []
        results = self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            using="problem",
            limit=min(n_results, count),
        ).points
        return [
            f"Problem: {r.payload['problem']}\nMethod: {r.payload['method']}"
            for r in results
        ]

    def has_source(self, source: str) -> bool:
        """Return True if any stored point has payload['source'] == source."""
        result, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(result) > 0

    def count(self) -> int:
        return self._client.count(self._collection).count
