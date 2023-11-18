import requests
from datetime import datetime
from typing import List
from prompt_systematic_review.paperSource import Paper
import time


class SemanticScholarSource:
    """A class to represent a source of papers from Semantic Scholar."""

    searchBaseURL = "https://api.semanticscholar.org/graph/v1/paper/search"
    paperBaseURL = "https://api.semanticscholar.org/graph/v1/paper/"

    def getPapers(self, count: int, keyWords: List[str]) -> List[Paper]:
        all_papers = []
        max_retries = 5  # Maximum number of retries after hitting rate limit
        for keyword in keyWords:
            query = f'"{keyword}"'  # Query for each keyword
            retry_count = 0  # Count of current retries
            while retry_count < max_retries:
                try:
                    papers_data = self.bulkSearchPapers(query)[
                        :count
                    ]  # Fetch and limit papers for each keyword
                    for paper_data in papers_data:
                        open_access_pdf_url = None
                        if paper_data.get("openAccessPdf"):
                            open_access_pdf_url = paper_data["openAccessPdf"].get("url")

                        paper = Paper(
                            title=paper_data["title"],
                            firstAuthor=paper_data["authors"][0]["name"]
                            if paper_data["authors"]
                            else "",
                            url=open_access_pdf_url,
                            dateSubmitted=datetime.strptime(
                                paper_data["publicationDate"], "%Y-%m-%d"
                            ).date()
                            if paper_data.get("publicationDate")
                            else None,
                            keyWords=[keyword.lower()],  # Single keyword for each paper
                            abstract=paper_data.get("abstract", ""),
                        )
                        all_papers.append(paper)
                    break  # Break the loop if successful
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Rate limit error
                        print(f"Rate limit hit for keyword '{keyword}'. Retrying...")
                        retry_count += 1
                        time.sleep(1.1 * retry_count)  # Exponential back-off
                    else:
                        print(f"Error during API request for keyword '{keyword}': {e}")
                        break  # Break the loop on other types of errors
        return all_papers

    def bulkSearchPapers(self, query: str, token: str = None) -> List[dict]:
        bulkSearchURL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
        params = {
            "query": query,
            "fields": "title,authors,abstract,publicationDate,openAccessPdf",
            "limit": 1000,  # Adjust limit as required
        }
        if token:
            params["token"] = token

        response = requests.get(bulkSearchURL, params=params)
        response.raise_for_status()
        return response.json().get("data", [])


#####################################################################
##########################Testing####################################
#####################################################################
# Create an instance of the SemanticScholarSource class
semantic_scholar = SemanticScholarSource()

# Define some sample keywords and a count
keywords = ["deep learning", "prompt engineering"]
count = 5  # Number of papers to fetch for each keyword

# Fetch papers
papers = semantic_scholar.getPapers(count, keywords)

# Print the details of each paper
for paper in papers:
    print("Title:", paper.title)
    print("First Author:", paper.firstAuthor)
    print("URL:", paper.url or "Not Available")
    print("Date Submitted:", paper.dateSubmitted)
    print("Keywords:", paper.keywords)
    print("Abstract:", paper.abstract or "No Abstract Available")
    print("-" * 50)  # Separator for readability
