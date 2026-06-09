from __future__ import annotations


# ── Tool registry ─────────────────────────────────────────────────────────────

_TOOL_REGISTRY: dict[str, str] = {
    "arxiv_search":   "Search the ArXiv preprint server for academic papers. "
                      "Call: arxiv_search(prompt: str, n: int) → list of (url, title, abstract).",
    "scholar_search": "Search Google Scholar for academic papers and citations. "
                      "Call: scolars_search(prompt: str, n: int) → list of (url, title, abstract).",
    "file_reader":    "Download and extract full text from a remote PDF or web URL. "
                      "Call: file_reader(url: str) → str (Markdown-formatted paper text).",
    "doc_reader":     "Parse and extract text from a local PDF, DOCX, TeX, or Markdown file. "
                      "Call: doc_reader(path: str) → str (structured Markdown text).",
    "z3_solver":      "Execute a self-contained Z3Py (SMT solver) program to verify logical "
                      "constraints, check satisfiability, or derive model values. "
                      "Call: exec(z3_python_code, {}) — captures stdout via redirect.",
    "lean_verifier":  "Verify Lean 4 theorem statements and proof terms for formal correctness. "
                      "Call: LeanKripkeVerifier().verify(structure, properties).",
    "python_exec":    "Execute arbitrary Python code for numerical computation, data processing, "
                      "plotting, or algorithm implementation. "
                      "Call: exec(python_code, namespace) — captures stdout via redirect.",
    "knowledge_graph": "Query the in-memory knowledge graph of (source, type, target) triples "
                       "extracted from literature sections. "
                       "Call: knowledge_graph.query(entity: str) → list[Relationship].",
    # ── search & retrieval tools (available during decomposition & refinement) ──
    "web_search":      "Search the open web via DuckDuckGo and return ranked results. "
                       "Call: web_search(query: str, n: int) → list[{title, url, snippet}]. "
                       "Use for finding tutorials, blog posts, documentation, or recent news "
                       "not indexed in academic databases.",
    "github_search":   "Search GitHub for repositories, code snippets, or topics. "
                       "Call: github_search(query: str, n: int, kind: str) → list[{name, url, description, stars, language}]. "
                       "kind ∈ {'repositories', 'code', 'topics'}. "
                       "Use to find reference implementations, datasets hosted on GitHub, "
                       "or open-source libraries for a specific algorithm.",
    "wiki_search":     "Search Wikipedia and return article summaries via the MediaWiki REST API. "
                       "Call: wiki_search(query: str, n: int) → list[{title, url, extract}]. "
                       "Use for background definitions, mathematical concepts, historical context, "
                       "or named entity disambiguation.",
    "rag_retriever":   "Query the hybrid RAG system (Qdrant vector store + Neo4j ontology graph) "
                       "for verified Hoare triples, method excerpts, and concept neighbourhoods "
                       "most relevant to the current step. "
                       "Call: rag_query(query: str, n: int, retriever=hybrid_retriever) → list[str]. "
                       "Use before writing any command C to ground it in prior verified steps.",
    # ── dspy-skills meta-tools ─────────────────────────────────────────────────
    "list_skills":        "Discover all available DSPy skills. "
                          "Call: list_skills() → list of skill names + one-line descriptions.",
    "activate_skill":     "Load full instructions and interface for a named skill. "
                          "Call: activate_skill(name: str) → skill spec dict.",
    "run_skill_script":   "Execute a bundled skill script in a sandboxed environment. "
                          "Call: run_skill_script(name: str, args: dict) → stdout str.",
    "read_skill_resource": "Read a skill's documentation or asset file. "
                           "Call: read_skill_resource(name: str, path: str) → str.",
}

def _tool_descriptions_str() -> str:
    """Format the tool registry as 'name: description' lines."""
    return "\n".join(f"{name}: {desc}" for name, desc in _TOOL_REGISTRY.items())

def _tool_names_str() -> str:
    """Comma-separated list of tool names."""
    return ", ".join(_TOOL_REGISTRY.keys())
