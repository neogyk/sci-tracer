import arxiv

def arxiv_search(query: str, max_results: int):
    # Search for papers using the specified query on ArXiv
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = list(client.results(search))
    return papers

# Search ArXiv for papers on quantum error correction tailored to specific hardware
papers_on_qecc = arxiv_search('quantum error correction hardware specific', 10)

# Print the titles of the found papers
for paper in papers_on_qecc:
    print(paper.title, paper.entry_id)

# Output for post processing
papers_on_qecc_data = [(paper.title, paper.entry_id) for paper in papers_on_qecc]

# Check that the result is a list and contains tuples with titles and URLs
assert isinstance(papers_on_qecc_data, list), "Output should be a list"
assert all(isinstance(item, tuple) and len(item) == 2 for item in papers_on_qecc_data), "Each item should be a tuple with 2 elements: title and url"

print("postcondition OK: list of tuples with titles and URLs")