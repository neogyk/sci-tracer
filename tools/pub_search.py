"""Publication search utilities.

Dependencies:
    pip install arxiv scholarly semanticscholar pymupdf
"""
import io
import socket
import time
import random
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from ddgs import DDGS

import arxiv
import fitz
import requests
from scholarly import scholarly
from crewai.tools import tool

LIMIT_PAGE = 99
class PDFSearchResult(BaseModel):
    title: str = Field(..., description="The title of the academic paper or document.")
    url: str = Field(..., description="The direct URL link to the PDF file.")
    snippet: str = Field(..., description="The text snippet containing context/abstract from the search page.")

def arxiv_search(prompt: str, n: int) -> set[tuple[str, str, str]]:
    """Search arXiv and return up to n results as (pdf_url, title, abstract) tuples."""
    socket.setdefaulttimeout(60)
    client = arxiv.Client()
    search = arxiv.Search(query=prompt, max_results=n)
    res: set[tuple[str, str, str]] = set()
    for paper in client.results(search):
        url = paper.pdf_url
        title = paper.title
        abstract = paper.summary
        if url:
            res.add((url, title, abstract))
    return res


def scolars_search(prompt: str, n: int) -> set[tuple[str, str, str]]:
    """Search Google Scholar and return up to n results as (pdf_url, title, abstract) tuples."""
    search_query = scholarly.search_pubs(prompt)
    res: set[tuple[str, str, str]] = set()
    for _ in range(n):
        try:
            publication = next(search_query)
            title = publication["bib"]["title"]
            abstract = publication["bib"].get("abstract", "")
            url = publication.get("eprint_url")
            res.add((url, title, abstract))
        except StopIteration:
            break
        except Exception:
            pass
    return res


def file_reader(url: str) -> str:
    """Download a PDF from url and return its text, sections separated by ##."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    pdf_stream = io.BytesIO(response.content)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    text = ""
    for page_id, page in enumerate(doc):
        if page_id >= LIMIT_PAGE:
            break
        text += page.get_text()
        text += "##"
    return text



#@tool("DuckDuckGo PDF Paper Searcher")
def search_pdf_papers(query: str, max_results: int = 5, max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    Executes a specialized DuckDuckGo search targeted strictly at retrieving PDF documents.
    
    Args:
        query (str): The search topic or hypothesis string.
        max_results (int): Total results to return.
        max_retries (int): Retries allowed under HTTP 429 rate limit exception states.
        
    Returns:
        List[Dict[str, Any]]: A verified list of parsed PDFSearchResult schemas.
    """
    # Enforce advanced operators to pull down academic/scientific PDFs exclusively
    sanitized_query = query.strip()
    if "filetype:pdf" not in sanitized_query.lower():
        sanitized_query = f"{sanitized_query} filetype:pdf"

    # User-Agent handling is abstracted cleanly inside the native DDGS engine context
    for attempt in range(max_retries):
        try:
            results_list = []
            with DDGS() as ddgs:
                results = ddgs.text(sanitized_query, max_results=10)
                for idx, result in enumerate(results):
                    if idx >= max_results:
                        break
                    
                    # Validate that the link points directly to a PDF
                    url = result.get("href", "")
                    if url.lower().endswith(".pdf") or "pdf" in url.lower():
                        validated_entry = PDFSearchResult(
                            title=result.get("title", "Untitled Document"),
                            url=url,
                            snippet=result.get("body", "")
                        )
                        results_list.append(validated_entry.model_dump())
                        
            return results_list
            
        except Exception as error_msg:
            # Handle rate-limit mitigation via exponential backoff with jitter
            if "429" in str(error_msg) or "RateLimit" in type(error_msg).__name__:
                sleep_duration = (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(sleep_duration)
                continue
            else:
                # Bubble up structural or internet socket failures immediately
                raise RuntimeError(f"DuckDuckGo search tool execution failure: {str(error_msg)}")
                
    return []

def duckduckgo_pdf_search_tool(search_prompt: str) -> str:
    """
    Useful for searching the web specifically for scientific papers, academic texts, 
    and comprehensive PDF documentation matching the search prompt.
    """
    try:
        raw_results = search_pdf_papers(query=search_prompt, max_results=5)
        if not raw_results:
            return f"No verified PDF targets found for query: '{search_prompt}'."
            
        # Format explicitly as an easy-to-parse string for LLM execution contexts
        formatted_output = "--- Verified PDF Search Results ---\n"
        for idx, item in enumerate(raw_results, 1):
            formatted_output += f"[{idx}] {item['title']}\n"
            formatted_output += f"    URL: {item['url']}\n"
            formatted_output += f"    Context: {item['snippet']}\n\n"
        return formatted_output
        
    except Exception as e:
        return f"Tool execution failed with error string: {str(e)}"