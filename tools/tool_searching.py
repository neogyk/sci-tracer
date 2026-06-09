from github import Github


def search_github_repos(name: str, token: str = None, top_k: int = 5) -> list[dict]:
    g = Github(token) if token else Github()
    query = name + "language:python"
    repositories = g.search_repositories(query=query, sort="stars", order="desc")
    results = []
    for repo in repositories[:top_k]:
        results.append({
            "full_name": repo.full_name,
            "stars": repo.stargazers_count,
            "description": repo.description,
            "url": repo.html_url,
        })
    return results

def get_pypi(name:str):
    return


if __name__ == "__main__":
    results = search_github_repos("qiskit language:python")
    print("--- GitHub Search Results ---")
    for repo in results:
        print(f"⭐ {repo['stars']} | {repo['full_name']}")
        print(f"   Description: {repo['description']}")
        print(f"   URL: {repo['url']}\n")
