from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, NamedVector
import numpy as np
from transformers import AutoModel, AutoTokenizer


class TrajectoryVectorDB:
    """Persistent Qdrant store for trajectory and Hoare-triple embeddings.

    Two collections are maintained:
      raw_trajectories  — (problem, method, result) points from literature
      hoare_triplets    — {P} C {Q} triples from formalization

    Hoare triplets accumulate across runs and enrich the prolongation RAG
    for future sessions; trajectory points are populated fresh each run.
    """

    _MODEL_NAME  = "BAAI/bge-small-en-v1.5"
    _EMBED_DIM   = 384
    _TRAJ_COL    = "raw_trajectories"
    _HOARE_COL   = "hoare_triplets"

    def __init__(self, persist_dir: str = "./trajectory_db") -> None:
        try:
            self._client = QdrantClient(path=persist_dir)
        except RuntimeError as exc:
            if "already accessed" in str(exc):
                print(
                    f"  [TrajectoryVectorDB] '{persist_dir}' is locked by another process "
                    f"— falling back to in-memory storage. Hoare triplets will not persist."
                )
                self._client = QdrantClient(":memory:")
            else:
                raise
        self._hf_model: AutoModel | None = None
        self._ensure_collection(
            self._TRAJ_COL,
            ["problem", "method", "result"],
        )
        self._ensure_collection(
            self._HOARE_COL,
            ["precondition", "command", "postcondition"],
        )

    def _ensure_collection(self, name: str, vector_names: list) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if name not in existing:
            self._client.create_collection(
                collection_name=name,
                vectors_config={
                    vn: VectorParams(size=self._EMBED_DIM, distance=Distance.COSINE)
                    for vn in vector_names
                },
            )

    def _load_vocab_model(self) -> None:
        """Load the HuggingFace tokenizer + model once; shared by embed, unembed, and FIM."""
        self._unembed_tok = AutoTokenizer.from_pretrained(self._MODEL_NAME)
        self._hf_model    = AutoModel.from_pretrained(self._MODEL_NAME).eval()
        self._unembed_emb = self._hf_model.embeddings.word_embeddings.weight.detach()

    def embed(self, text: str) -> list[float]:
        """Sentence embedding via the shared HuggingFace model (mean-pool + L2 norm)."""
        import torch
        if self._hf_model is None:
            self._load_vocab_model()
        inputs = self._unembed_tok(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            out = self._hf_model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()        # (1, L, 1)
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)  # (1, d) mean pool
        emb  = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)       # L2 normalise
        return emb[0].tolist()

    def store_trajectory(self, traj: Any, traj_id: str | None = None) -> str:
        """Embed and persist a (problem, method, result) trajectory point."""
        tid = traj_id or str(uuid.uuid4())
        self._client.upsert(
            collection_name=self._TRAJ_COL,
            points=[PointStruct(
                id=tid,
                vector={
                    "problem": self.embed(traj.problem),
                    "method":  self.embed(traj.method),
                    "result":  self.embed(traj.result),
                },
                payload={
                    "problem": traj.problem,
                    "method":  traj.method,
                    "result":  traj.result,
                    "traj_id": tid,
                },
            )],
        )
        return tid

    def store_hoare(self, hoare: Any, triplet_id: str | None = None) -> str:
        """Embed and persist a Hoare triple {P} C {Q}."""
        tid = triplet_id or str(uuid.uuid4())
        self._client.upsert(
            collection_name=self._HOARE_COL,
            points=[PointStruct(
                id=tid,
                vector={
                    "precondition":  self.embed(hoare.precondition),
                    "command":       self.embed(hoare.command),
                    "postcondition": self.embed(hoare.postcondition),
                },
                payload={
                    "precondition":  hoare.precondition,
                    "command":       hoare.command,
                    "postcondition": hoare.postcondition,
                    "triplet_id":    tid,
                },
            )],
        )
        return tid

    def query_hoare(self, query_embedding: list[float], n_results: int = 5) -> list[str]:
        """Return formatted {P} C {Q} strings from the Hoare collection nearest to the query."""
        count = self._client.count(self._HOARE_COL).count
        if count == 0:
            return []
        results = self._client.query_points(
            collection_name=self._HOARE_COL,
            query=query_embedding,
            using="precondition",
            limit=min(n_results, count),
        ).points
        return [
            f"{{P}} {r.payload['precondition']}\nC {r.payload['command']}\n{{Q}} {r.payload['postcondition']}"
            for r in results
        ]

    def query_trajectories(
        self,
        query_embedding: list[float],
        role: str | None = None,
        n_results: int = 5,
    ) -> list[str]:
        """Return document texts from the trajectory collection nearest to the query.

        When *role* is given (one of 'problem', 'method', 'result') the search
        uses that named vector so each prolonged component retrieves matching texts.
        """
        count = self._client.count(self._TRAJ_COL).count
        if count == 0:
            return []
        vector_name = role if role in ("problem", "method", "result") else "problem"
        try:
            results = self._client.query_points(
                collection_name=self._TRAJ_COL,
                query=query_embedding,
                using=vector_name,
                limit=min(n_results, count),
            ).points
        except Exception:
            return []
        field = role if role in ("problem", "method", "result") else "problem"
        return [r.payload.get(field, "") for r in results]

    def unembed(self, embedding: list[float], top_k: int = 10) -> str:
        """Approximate textual representation of an embedding vector.

        Computes cosine similarity between the vector and every entry in the
        model's word-piece embedding matrix, then returns the top-k closest
        tokens as a space-separated string.  This gives an interpretable
        "semantic direction" for a prolonged trajectory point without querying
        the VectorDB corpus.
        """
        import torch
        if self._hf_model is None:
            self._load_vocab_model()

        vec = torch.tensor(embedding, dtype=torch.float32)
        sims = torch.nn.functional.cosine_similarity(
            vec.unsqueeze(0), self._unembed_emb
        )
        idxs = sims.topk(top_k).indices.tolist()
        tokens = self._unembed_tok.convert_ids_to_tokens(idxs)
        words = [t for t in tokens if not t.startswith("##") and not t.startswith("[")]
        return " | ".join(words)

    def hoare_count(self) -> int:
        return self._client.count(self._HOARE_COL).count

    def trajectory_count(self) -> int:
        return self._client.count(self._TRAJ_COL).count

