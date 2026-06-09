import dspy
from tools.pub_search import arxiv_search
from pydantic import BaseModel, Field
from storage.vector_db import TrajectoryStore
from storage.knowledge_graph import KnowledgeGraph
from typing import Any, List
from tools.file_reader import doc_reader, doc_reader_pages
from tools.pub_search import duckduckgo_pdf_search_tool
from dataclasses import dataclass


@dataclass
class _PaperTrajectory:
    """Minimal trajectory-shaped record for storing paper chunks in TrajectoryStore.

    TrajectoryStore.add() expects objects with .problem, .method, .result.
    We map Hoare triplet fields onto those slots so paper chunks are
    searchable via the same vector index as agent trajectories.
    """
    problem: str        # chunk text / precondition  -- what the section states
    method:  str        # Hoare command              -- what the section instructs
    result:  str        # Hoare postcondition        -- what the section concludes
    source:  str = ""   # originating file path or URL for deduplication

class HoareTriplet(BaseModel):
    precondition: str = Field(desc="Logical state before (P)")
    command: str = Field(desc="The action or code (C)")
    postcondition: str = Field(desc="Logical state after (Q)")

class Hoare(dspy.Signature):
    """Convert natural language requirements into a formal Hoare Triple {P} C {Q}."""
    section: str = dspy.InputField(desc="The human-readable requirement")
    hoare: HoareTriplet = dspy.OutputField(desc="")

class SearchPhrase(dspy.Signature):
    """Transform question into the search phrase for search"""
    question: str = dspy.InputField()
    search_phrase: str = dspy.OutputField(desc="The research problem formulation that is necessary to solve")
    topics: List[str] = dspy.OutputField(desc="Search phrase for the literature search that will help to find the paper")

class PaperRelevanceCheck(dspy.Signature):
    """Decide whether a research paper is relevant to a research question.
    Use the title and abstract only -- do not guess at full content.
    A paper is relevant if it directly addresses the research question,
    provides background theory, datasets, methods, or results that would
    inform the solution.  A paper is NOT relevant if it is only tangentially
    related, covers a completely different domain, or is a general survey
    with no specific connection to the question.
    """
    question: str = dspy.InputField(
        desc="The research question or problem being investigated"
    )
    title: str = dspy.InputField(
        desc="Title of the paper to evaluate"
    )
    abstract: str = dspy.InputField(
        desc="Abstract of the paper (may be empty if unavailable)"
    )
    verdict: str = dspy.OutputField(
        desc="Exactly 'relevant' or 'not_relevant'"
    )
    reason: str = dspy.OutputField(
        desc="One sentence explaining the verdict"
    )
    
# ── Supported file extensions for local ingestion ─────────────────────────
_SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".docx", ".tex"}


class KnowledgeConsructor(dspy.Module):
    def __init__(self, question: str, storage: TrajectoryStore, kg: KnowledgeGraph):
        self.searcher = dspy.Predict(SearchPhrase)
        self.hoare    = dspy.Predict(Hoare)
        self.storage  = storage
        self.kg       = kg

    def rephrase(self, prompt: str):
        return self.searcher(question=prompt)

    def discover_tools_from_trajectories(
        self,
        trajectories: list[Any],
        question: str,
        sample_size: int = 6,
    ) -> dict[str, str]:
        """Scan trajectory excerpts for tools/libraries/algorithms and extend _TOOL_REGISTRY."""
        if not trajectories:
            return {}

        ranked = sorted(trajectories, key=lambda t: len(t.method), reverse=True)
        sample = ranked[:sample_size]

        newly_found: dict[str, str] = {}
        for t in sample:
            text = f"Problem: {t.problem}\nMethod: {t.method}\nResult: {t.result}"
            try:
                pred = self.tool_discoverer(text=text, question=question)
                for tool in pred.tools:
                    name = tool.name.strip().lower().replace(" ", "_")
                    if not name or name in _TOOL_REGISTRY:
                        continue
                    desc = f"{tool.description}  Call: {tool.call_signature}"
                    _TOOL_REGISTRY[name] = desc
                    newly_found[name] = desc
                    print(f"  + Discovered tool: {name}")
                    print(f"      {tool.relevance}")
            except Exception as exc:
                print(f"  (tool discovery failed for one trajectory: {exc})")

        return newly_found


    def collect_literature(
        self,
        prompt: str,
        local_path: str | None = None,
    ) -> list:
        """Collect research papers either from a local folder or online databases.

        Args:
            prompt:     Research question used to contextualise each paper and,
                        when local_path is None, to drive the online search.
            local_path: Optional path to a local folder.  When provided every
                        supported file (pdf, md, txt, docx, tex) inside that
                        folder is read directly -- no network calls are made.

        Returns:
            List of stored chunk dicts (each written to self.storage).
        """
        if local_path is not None:
            print(f"  [kc] Source: local folder -> {local_path}")
            return self._collect_from_local(prompt, local_path)

        print("  [kc] Source: internet search (ArXiv + DuckDuckGo)")
        rephrased     = self.rephrase(prompt)
        search_phrase = rephrased.search_phrase

        res_arxiv = arxiv_search(search_phrase, 2)
        all_chunks: list = []
        for url, _abstract in res_arxiv:
            chunks = self._ingest_source(url, prompt)
            all_chunks.extend(chunks)
        return all_chunks

    def download_paper(self, url: str, dest_dir: str) -> str:
        """Download a PDF from *url* into *dest_dir* and return the local file path."""
        import urllib.request
        from pathlib import Path as _Path

        dest = _Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        filename = url.rstrip("/").split("/")[-1]
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        local_path = dest / filename
        print(f"  [kc] Downloading {url} -> {local_path}")
        urllib.request.urlretrieve(url, local_path)
        return str(local_path)

    def _collect_from_local(self, prompt: str, local_path: str) -> list:
        """Read every supported file in local_path, chunk, and store into vector DB.

        Returns list of stored _PaperTrajectory objects.
        """
        from pathlib import Path as _Path

        folder = _Path(local_path)
        files  = [
            f for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in _SUPPORTED_SUFFIXES
        ]
        if not files:
            print(f"  [kc] no supported files found in {local_path}")
            return []

        all_chunks: list = []
        for fp in files:
            chunks = self._ingest_source(str(fp), prompt) or []
            all_chunks.extend(chunks)
            print(f"  [kc] {fp.name} -> {len(chunks)} chunk(s) stored")

        return all_chunks

    # Section headings whose pages should not be stored in the trajectory DB.
    _SKIP_HEADINGS = {
        "acknowledgments", "acknowledgements", "acknowledgment",
        "references", "bibliography",
        "author contributions", "authors", "author list", "affiliations",
        "conflict of interest", "conflicts of interest",
        "funding", "funding information",
        "appendix", "supporting information", "supplementary material",
        "data availability", "ethics statement",
    }

    @classmethod
    def _is_skippable(cls, page_text: str) -> bool:
        """Return True if this page belongs to a section we do not want to index."""
        first_line = page_text.lstrip("#• ").split("\n")[0].strip().lower()
        return any(first_line.startswith(heading) for heading in cls._SKIP_HEADINGS)

    def _ingest_source(self, source: str, prompt: str) -> list:
        """Convert one file/URL to pages, generate Hoare triplets, and store.

        Each page is stored as a _PaperTrajectory:
          problem = page text        (searchable by RAG query on "problem" vector)
          method  = Hoare command    (what the page instructs to do)
          result  = Hoare postcondition (what the page concludes)
        """
        if self.storage.has_source(source):
            print(f"  [kc] {source}: already indexed, skipping")
            return []

        try:
            pages = doc_reader_pages(source)
        except Exception as exc:
            print(f"  [kc] doc_reader_pages failed for {source}: {exc}")
            return []

        # Filter out boilerplate sections
        chunks = [p for p in pages if p.strip() and not self._is_skippable(p)]
        print(f"  [kc] {source}: {len(pages)} page(s) -> {len(chunks)} after filtering")
        if not chunks:
            return []

        triplets: list = []
        for i, chunk_text in enumerate(chunks):
            try:
                pred = self.hoare.forward(section=chunk_text)
                triplets.append(pred.hoare)
            except Exception as exc:
                print(f"  [kc] hoare extraction failed (chunk {i}): {exc}")
                triplets.append(None)

        # Build trajectory records (order preserved)
        records: list[_PaperTrajectory] = [
            _PaperTrajectory(
                problem = chunks[i],
                method  = getattr(triplets[i], "command",       "") if triplets[i] else "",
                result  = getattr(triplets[i], "postcondition", "") if triplets[i] else "",
                source  = source,
            )
            for i in range(len(chunks))
        ]

        # Single batched upsert — embeddings computed in one model.embed() call
        try:
            self.storage.add(records)
        except Exception as exc:
            print(f"  [kc] storage.add failed for {source}: {exc}")
            return []

        return records

