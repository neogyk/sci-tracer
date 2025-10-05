import wikipediaapi


class RAG:
    def __init__(self, top_k: int, year_limit: int, tree_depth: int):
        self.top_k = top_k
        self.year_limit = year_limit
        self.tree_depth = tree_depth
        return

    def _parse(self, doc_path: str):
        return

    def generate(self, prompt: str):
        return

    def search(self, prompt: str):
        # extract the topic, key-words, time, suggest the tool
        return

    async def search_tool(self, prompt: str):
        # add custom research tool
        return

    async def search_web(self, prompt: str):
        return

    async def search_arxiv(self, prompt: str):
        # Extract all arcticles from the arxiv, than go to refferences and extract the refferences.
        return

    async def search_wiki(self, prompt: str) -> tuple[str, str]:
        wiki_wiki = wikipediaapi.Wikipedia(
            user_agent="MyProjectName (merlin@example.com)",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            language="en",
        )
        page_py = wiki_wiki.page(prompt)
        url = page_py.fullurl
        return url, page_py.text

    async def process_doc(self):
        # identify the format of the document;
        # construct the
        return
