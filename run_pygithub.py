
from github import Github

# Initialize with your GitHub Personal Access Token
# g = Github("YOUR_GITHUB_TOKEN")
g = Github()  # Unauthenticated (strict rate limits)

# Search for repositories (e.g., Python projects matching 'fastapi')
query = "qiskit language:python"
repositories = g.search_repositories(query=query, sort="stars", order="desc")

# Print the top 5 results
print("--- GitHub Search Results ---")
for repo in repositories[:5]:
    print(f"⭐ {repo.stargazers_count} | {repo.full_name}")
    print(f"   Description: {repo.description}")
    print(f"   URL: {repo.html_url}\n")
