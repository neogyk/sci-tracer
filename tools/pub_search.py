import os
import requests
from datetime import datetime, timedelta
from IPython.display import Markdown
import json
from scholarly import scholarly
from crewai_tools import ArxivPaperTool
from semanticscholar import SemanticScholar
import socket
import sys
import fitz
import requests
import io

sys.path.append("/Users/leonid/Desktop/FusionAI/scitracer/tools")

# TODO add the webpage scrapper;
# Search top articles using search engine-> perform scrapping
# TODO store the results to the vector database
LIMIT_PAGE = 4


def arxiv_search(prompt: str, n: int):
    socket.setdefaulttimeout(60)

    tool = ArxivPaperTool(
        download_pdfs=False, save_dir="./arxiv_pdfs", use_title_as_filename=True
    )
    results = tool.run(search_query=prompt, max_results=n)
    res = set()
    for i in range(1, n + 1):
        try:
            url = results.split("\nPDF: ")[i].split("\n")[0]
            title = results.split("Title: ")[i].split("\n")[0]
            abstract = results.split("Summary: ")[i].split("\n")[0]
            res.add((url, title, abstract))
        except Exception:
            pass
    return res


def scolars_search(prompt: str, n: int):
    search_query = scholarly.search_pubs(prompt)
    res = set()
    for i in range(n):
        try:
            publication = next(search_query)
            title = publication["bib"]["title"]
            abstract = publication["bib"]["abstract"]
            url = publication.get("eprint_url")
            # TODO check the compatibility of the abstract
            # Check if the abstract is possible to fetch
            res.add((url, title, abstract))
        except Exception:
            pass
    return res


def file_reader(url):
    response = requests.get(url)
    response.raise_for_status()
    pdf_stream = io.BytesIO(response.content)
    my_pdf_doc = fitz.open(stream=pdf_stream, filetype="pdf")
    text = ""
    for page in my_pdf_doc:
        if page_id < LIMIT_PAGE:
            text += page.get_text()
            text += "##"
            page_id += 1
    return text
