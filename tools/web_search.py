"""Lightweight search tools: web (DuckDuckGo), GitHub, Wikipedia, RAG query.

All functions use only the standard library + requests (already a dependency).
No API keys are required for basic usage.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import requests

_TIMEOUT = 15  # seconds


# ── Web search (DuckDuckGo Instant-Answer / HTML) ────────────────────────────

def web_search(query: str, n: int = 5) -> list[dict]:
    """Search the web via DuckDuckGo and return up to n results.

    Returns a list of dicts with keys: title, url, snippet.
    Falls back to an empty list on network errors.
    """
    results: list[dict] = []
    try:
        # DuckDuckGo HTML endpoint — no JS required, no key needed
        url = "https://html.duckduckgo.com/html/"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SciTracer/1.0)"}
        resp = requests.post(
            url,
            data={"q": query, "b": "", "kl": "us-en"},
            headers=headers,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        # Parse result links from the HTML response
        link_re  = re.compile(r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)<')
        snip_re  = re.compile(r'class="result__snippet"[^>]*>([^<]+)<')
        links    = link_re.findall(resp.text)
        snippets = snip_re.findall(resp.text)
        for i, (href, title) in enumerate(links[:n]):
            snippet = snippets[i] if i < len(snippets) else ""
            results.append({"title": title.strip(), "url": href, "snippet": snippet.strip()})
    except Exception:
        pass
    return results


# ── GitHub search (public REST API, no token required for low rate) ──────────

def github_search(
    query: str,
    n: int = 5,
    kind: str = "repositories",
) -> list[dict]:
    """Search GitHub for repositories or code.

    Parameters
    ----------
    query : str   — search query (GitHub search syntax supported)
    n     : int   — max results to return (GitHub API cap: 30 per page)
    kind  : str   — 'repositories', 'code', or 'topics'

    Returns a list of dicts.  For repositories: name, url, description,
    stars, language.  For code: name, url, repository, path.
    """
    kind = kind if kind in ("repositories", "code", "topics") else "repositories"
    api_url = f"https://api.github.com/search/{kind}"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    params  = {"q": query, "per_page": min(n, 30), "page": 1}
    results: list[dict] = []
    try:
        resp = requests.get(api_url, headers=headers, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data  = resp.json()
        items = data.get("items", [])
        if kind == "repositories":
            for item in items[:n]:
                results.append({
                    "name":        item.get("full_name", ""),
                    "url":         item.get("html_url", ""),
                    "description": item.get("description") or "",
                    "stars":       item.get("stargazers_count", 0),
                    "language":    item.get("language") or "",
                })
        elif kind == "code":
            for item in items[:n]:
                results.append({
                    "name":       item.get("name", ""),
                    "url":        item.get("html_url", ""),
                    "repository": item.get("repository", {}).get("full_name", ""),
                    "path":       item.get("path", ""),
                })
        else:
            for item in items[:n]:
                results.append({
                    "name":        item.get("name", ""),
                    "url":         f"https://github.com/topics/{item.get('name','')}",
                    "description": item.get("short_description") or "",
                })
    except Exception:
        pass
    return results


# ── Wikipedia search (MediaWiki REST API) ─────────────────────────────────────

def wiki_search(query: str, n: int = 5) -> list[dict]:
    """Search Wikipedia and return up to n article summaries.

    Uses the public MediaWiki REST API — no key required.
    Returns a list of dicts with keys: title, url, extract.
    """
    results: list[dict] = []
    try:
        # Step 1: opensearch to find matching titles
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action":   "opensearch",
            "search":   query,
            "limit":    min(n, 10),
            "namespace": 0,
            "format":   "json",
        }
        wiki_headers = {"User-Agent": "SciTracer/1.0 (research tool; python-requests) contact: open-source"}
        resp = requests.get(search_url, params=params, headers=wiki_headers, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        titles = data[1][:n] if len(data) > 1 else []
        urls   = data[3][:n] if len(data) > 3 else []

        # Step 2: fetch a short extract for each title
        for title, url in zip(titles, urls):
            try:
                safe = urllib.parse.quote(title.replace(" ", "_"))
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe}"
                sr = requests.get(summary_url, timeout=_TIMEOUT)
                extract = sr.json().get("extract", "") if sr.ok else ""
            except Exception:
                extract = ""
            results.append({
                "title":   title,
                "url":     url,
                "extract": extract[:800],
            })
    except Exception:
        pass
    return results


# ── RAG context query (callable tool wrapper) ─────────────────────────────────

def rag_query(query: str, n: int = 5, retriever=None) -> list[str]:
    """Query the hybrid RAG retriever and return context passages.

    Parameters
    ----------
    query     : str             — free-text query
    n         : int             — max results
    retriever : HybridRetriever — injected at call time from SciTracer context

    Returns a list of text passages (Hoare step summaries + ontology triples).
    """
    if retriever is None:
        return []
    try:
        ctx  = retriever.retrieve(query)
        text = ctx.to_text(max_vector=n, max_graph_triples=20)
        return [text] if text.strip() else []
    except Exception:
        return []
