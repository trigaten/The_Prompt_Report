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
            papers = []
            offset = 0
            retry_count = 0  # Count of current retries
            while len(papers) < count and retry_count < max_retries:
                papers_data = []  # Initialize papers_data before the try block
                try:
                    papers_data = self.fetchPapersByKeyword(keyword, count - len(papers), offset)
                    for paper_data in papers_data:
                        paper = Paper(
                            title=paper_data['Title'],
                            firstAuthor=paper_data['First Author'],
                            url=paper_data.get('Open Access PDF URL'),
                            dateSubmitted=datetime.strptime(paper_data['Publication Date'], '%Y-%m-%d').date(),
                            keyWords=[keyword],
                            abstract=paper_data.get('Abstract')
                        )
                        papers.append(paper)
                        if len(papers) >= count:
                            break
                    offset += len(papers_data)
                    retry_count = 0  # Reset retry count after a successful fetch
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        print("Rate limit hit. Waiting before retrying...")
                        time.sleep(1.1 * (retry_count + 1))  # Exponential back-off
                        retry_count += 1
                    else:
                        raise e
                if len(papers_data) < count:  # Less papers returned than requested
                    break
            all_papers.extend(papers)
        return all_papers



    def fetchPapersByKeyword(self, keyword: str, count: int, offset: int) -> List[dict]:
        params = {
            "query": keyword,
            "offset": offset,
            "limit": count,
            "fields": "title,authors,abstract,publicationDate,openAccessPdf"
        }
        response = requests.get(self.searchBaseURL, params=params)
        response.raise_for_status()
        data = response.json()

        papers_data = []
        for paper in data.get("data", []):
            first_author = paper.get("authors", [{}])[0].get("name", "") if paper.get("authors") else ""
            abstract = paper.get("abstract") or ""
            publication_date = paper.get("publicationDate") or "1900-01-01"  # Default if not available

            paper_info = {
                "Title": paper.get("title"),
                "First Author": first_author,
                "Abstract": abstract,
                "Publication Date": publication_date,
                "Open Access PDF URL": paper.get("openAccessPdf", {}).get("url") if paper.get("openAccessPdf") else None
            }
            papers_data.append(paper_info)

        return papers_data



# This is assuming you have the SemanticScholarSource class defined as per the previous code

# Create an instance of the SemanticScholarSource class
semantic_scholar = SemanticScholarSource()

# Define some sample keywords and a count
keywords = ["deep learning", "Prompt Engineering"]
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
