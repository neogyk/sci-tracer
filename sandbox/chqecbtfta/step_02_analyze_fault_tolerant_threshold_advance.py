import requests
from bs4 import BeautifulSoup
from typing import List

def scholar_search(query: str, num_results: int) -> List[str]:
    """
    Search for scholarly articles on a specified query using a
    hypothetical scholarly search engine and return a list of titles.

    Parameters:
    - query: The search query string.
    - num_results: The number of results to retrieve.

    Returns:
    - A list of article titles from the search.
    """
    base_url = "https://scholar.google.com/scholar"  # Hypothetical use
    params = {
        'q': query,
        'num': num_results
    }

    response = requests.get(base_url, params=params)
    
    if response.status_code == 429:
        print("Rate limit exceeded: Slow down requests or try again later.")
        return []  # Return an empty list or handle differently as needed

    if response.status_code != 200:
        raise Exception(f"Error occurred: {response.status_code}")

    soup = BeautifulSoup(response.content, 'html.parser')
    titles = [tag.get_text() for tag in soup.find_all('h3')[:num_results]]

    return titles

# Example usage
papers_on_fault_tolerance = scholar_search('fault tolerance thresholds quantum computing', 10)
print(papers_on_fault_tolerance)