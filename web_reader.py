class WebSearchAgent:
    def __init__(self, llm_vendor: str, search_engine_vendor: str, memory_layer: str):
        self.llm_vendor = llm_vendor
        self.search_engine_vendor = search_engine_vendor
        self.memory_layer = memory_layer

    def port_search_engine(self, search_engine: str):
        # add the search engine with api for the document retrieving
        return

    def _analyze_prompt(self, doc, context: str):
        return

    def search_docs(self, prompt: str, context: str):
        # using the prompt retrieve the top-k relevant docs:
        # check if the document doesn't appear in the memory
        # based on context understand the relevanse of the document
        # return the docs
        return
