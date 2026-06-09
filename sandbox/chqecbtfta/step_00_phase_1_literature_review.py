import arxiv

def arxiv_search(query: str, max_results: int = 10) -> list:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    return [result.title for result in search.results()]

# Query Arxiv for the specified topics
queries = [
    "quantum error-correcting codes",
    "quantum noise models",
    "lattice surgery quantum computing"
]

literature_found = {}
for query in queries:
    literature_found[query] = arxiv_search(query)
    
literature_found